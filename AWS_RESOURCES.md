# AWS resources created for AdLint (OpenCV AI Competition 2026)

Account `483863386902`, region **eu-north-1 (Stockholm)**, created 2026-10-26 by IAM user `opencv-deploy` with `aws/deploy.sh`.
Every resource that supports tags carries `Project=opencv-contest`.

| resource | id / name | notes |
|---|---|---|
| Lambda function | `arn:aws:lambda:eu-north-1:483863386902:function:adlint` | python3.12, **arm64 (Graviton)**, 3008 MB, 300 s timeout, 1 GB /tmp, reserved concurrency 3 (cost guard) |
| Lambda Function URL | https://hkuqxujx6iftfyt6gcq2q64ewm0kndgt.lambda-url.eu-north-1.on.aws/ | auth NONE (public demo), resource-policy statements `public-url`, `public-url-invoke` |
| IAM role | `arn:aws:iam::483863386902:role/adlint-lambda-role` | managed `AWSLambdaBasicExecutionRole` + inline `adlint-s3-runs` (s3:GetObject/PutObject on `runs/*` of the bucket only) |
| S3 bucket | `adlint-483863386902-eu-north-1` | private (all public access blocked), lifecycle `expire-runs`: `runs/` deleted after 7 days; `artifacts/` holds the live code zip |
| CloudWatch log group | `/aws/lambda/adlint` | 14-day retention |

Not tagged by AWS design: the Function URL, the role's inline policy, the log group (created by Lambda).

## Cost
Lambda is billed per request: a 4-placement image check measured 3.8 s warm / 6.8 s cold at ~2.9 GB = 11-20 GB-s
(the free tier's 400,000 GB-s per month covers roughly 20,000+ checks); an 8 s video for 4 placements took 60 s = ~175 GB-s.
S3 storage is a few MB per run and expires after 7 days. Idle cost is zero.

## Tear down
```
set -a; . <env file with AWS creds>; set +a
AWS_DEFAULT_REGION=eu-north-1 aws/teardown.sh
```
