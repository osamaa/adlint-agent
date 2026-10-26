#!/usr/bin/env bash
# Remove everything aws/deploy.sh created. Deletes all stored runs.
set -euo pipefail
AWS="${AWS_CLI:-aws}"
export AWS_PAGER=""
REGION="${AWS_DEFAULT_REGION:?set AWS_DEFAULT_REGION}"
NAME="${ADLINT_NAME:-adlint}"
ACCOUNT=$($AWS sts get-caller-identity --query Account --output text)
BUCKET="${ADLINT_BUCKET:-$NAME-$ACCOUNT-$REGION}"

$AWS lambda delete-function-url-config --function-name "$NAME" 2>/dev/null || true
$AWS lambda delete-function --function-name "$NAME" 2>/dev/null || true
$AWS logs delete-log-group --log-group-name "/aws/lambda/$NAME" 2>/dev/null || true
$AWS iam delete-role-policy --role-name "$NAME-lambda-role" --policy-name "$NAME-s3-runs" 2>/dev/null || true
$AWS iam detach-role-policy --role-name "$NAME-lambda-role" --policy-arn arn:aws:iam::aws:policy/service-role/AWSLambdaBasicExecutionRole 2>/dev/null || true
$AWS iam delete-role --role-name "$NAME-lambda-role" 2>/dev/null || true
$AWS s3 rm "s3://$BUCKET" --recursive --only-show-errors 2>/dev/null || true
$AWS s3api delete-bucket --bucket "$BUCKET" 2>/dev/null || true
echo "removed AdLint resources in $REGION"
