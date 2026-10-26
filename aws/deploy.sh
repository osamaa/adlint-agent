#!/usr/bin/env bash
# Deploy AdLint to AWS: S3 bucket (uploads + outputs) + Lambda (arm64/Graviton, zip) + public HTTPS Function URL.
# Safe to re-run: existing resources are reused, the function code is updated.
#
# Needs: aws cli v1/v2 with credentials in the environment, python3 + pip, zip.
# Usage:  AWS_DEFAULT_REGION=eu-north-1 aws/deploy.sh
set -euo pipefail
cd "$(dirname "$0")/.."

AWS="${AWS_CLI:-aws}"
export AWS_PAGER=""
REGION="${AWS_DEFAULT_REGION:?set AWS_DEFAULT_REGION}"
NAME="${ADLINT_NAME:-adlint}"
ACCOUNT=$($AWS sts get-caller-identity --query Account --output text)
BUCKET="${ADLINT_BUCKET:-$NAME-$ACCOUNT-$REGION}"
ROLE="$NAME-lambda-role"
TAGS_KV="Project=opencv-contest"

echo "== account $ACCOUNT, region $REGION, bucket $BUCKET"

# ---- S3 bucket: private, encrypted (default), run files expire after 7 days
if ! $AWS s3api head-bucket --bucket "$BUCKET" 2>/dev/null; then
  $AWS s3api create-bucket --bucket "$BUCKET" --create-bucket-configuration LocationConstraint="$REGION"
fi
$AWS s3api put-public-access-block --bucket "$BUCKET" --public-access-block-configuration \
  BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=true,RestrictPublicBuckets=true
$AWS s3api put-bucket-tagging --bucket "$BUCKET" --tagging 'TagSet=[{Key=Project,Value=opencv-contest}]'
$AWS s3api put-bucket-lifecycle-configuration --bucket "$BUCKET" --lifecycle-configuration \
  '{"Rules":[{"ID":"expire-runs","Status":"Enabled","Filter":{"Prefix":"runs/"},"Expiration":{"Days":7}}]}'

# ---- IAM role: logs + read/write only under this bucket's runs/ prefix
if ! $AWS iam get-role --role-name "$ROLE" >/dev/null 2>&1; then
  $AWS iam create-role --role-name "$ROLE" --tags Key=Project,Value=opencv-contest \
    --assume-role-policy-document '{"Version":"2012-10-17","Statement":[{"Effect":"Allow","Principal":{"Service":"lambda.amazonaws.com"},"Action":"sts:AssumeRole"}]}' >/dev/null
  $AWS iam attach-role-policy --role-name "$ROLE" --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole
  echo "waiting for the new role to propagate..."; sleep 12
fi
$AWS iam put-role-policy --role-name "$ROLE" --policy-name "$NAME-s3-runs" --policy-document "{
  \"Version\":\"2012-10-17\",
  \"Statement\":[{\"Effect\":\"Allow\",\"Action\":[\"s3:GetObject\",\"s3:PutObject\"],\"Resource\":\"arn:aws:s3:::$BUCKET/runs/*\"}]}"
ROLE_ARN=$($AWS iam get-role --role-name "$ROLE" --query Role.Arn --output text)

# ---- code bundle -> S3 -> Lambda
aws/build_lambda.sh
CODE_KEY="artifacts/lambda-$(date +%Y%m%d%H%M%S).zip"
$AWS s3 cp dist/lambda.zip "s3://$BUCKET/$CODE_KEY" --only-show-errors

ENVIRONMENT="Variables={ADLINT_S3_BUCKET=$BUCKET,ADLINT_WORK_DIR=/tmp/adlint,ADLINT_MAX_UPLOAD_MB=4}"
if $AWS lambda get-function --function-name "$NAME" >/dev/null 2>&1; then
  $AWS lambda update-function-code --function-name "$NAME" --s3-bucket "$BUCKET" --s3-key "$CODE_KEY" >/dev/null
  $AWS lambda wait function-updated --function-name "$NAME"
  $AWS lambda update-function-configuration --function-name "$NAME" --environment "$ENVIRONMENT" --memory-size 3008 --timeout 300 >/dev/null
else
  $AWS lambda create-function --function-name "$NAME" --runtime python3.12 --architectures arm64 \
    --handler lambda_handler.handler --role "$ROLE_ARN" --memory-size 3008 --timeout 300 \
    --ephemeral-storage Size=1024 --code "S3Bucket=$BUCKET,S3Key=$CODE_KEY" \
    --environment "$ENVIRONMENT" --tags "$TAGS_KV" >/dev/null
fi
$AWS lambda wait function-updated --function-name "$NAME"
# keep only the bundle that is live
$AWS s3 rm "s3://$BUCKET/artifacts/" --recursive --exclude "*" --include "lambda-*.zip" --only-show-errors \
  --exclude "$(basename "$CODE_KEY")"
# cost guard: at most 3 runs at once
$AWS lambda put-function-concurrency --function-name "$NAME" --reserved-concurrent-executions 3 >/dev/null

# ---- public HTTPS endpoint
if ! $AWS lambda get-function-url-config --function-name "$NAME" >/dev/null 2>&1; then
  $AWS lambda create-function-url-config --function-name "$NAME" --auth-type NONE >/dev/null
  $AWS lambda add-permission --function-name "$NAME" --statement-id public-url \
    --action lambda:InvokeFunctionUrl --principal "*" --function-url-auth-type NONE >/dev/null
  $AWS lambda add-permission --function-name "$NAME" --statement-id public-url-invoke \
    --action lambda:InvokeFunction --principal "*" >/dev/null 2>&1 || true
fi
URL=$($AWS lambda get-function-url-config --function-name "$NAME" --query FunctionUrl --output text)
$AWS logs put-retention-policy --log-group-name "/aws/lambda/$NAME" --retention-in-days 14 2>/dev/null || true

echo "== deployed: $URL"
echo "   health check: curl ${URL}health"
