#!/usr/bin/env bash
# Build dist/lambda.zip for the python3.12 arm64 (Graviton) Lambda runtime (Amazon Linux 2023, glibc 2.34). No Docker needed.
set -euo pipefail
cd "$(dirname "$0")/.."
rm -rf dist/lambda dist/lambda.zip
mkdir -p dist/lambda
python3 -m pip install --quiet --platform manylinux2014_aarch64 --platform manylinux_2_28_aarch64 --implementation cp --python-version 3.12 \
  --only-binary=:all: --target dist/lambda -r aws/requirements-lambda.txt
cp -r adlint webapp models lambda_handler.py dist/lambda/
find dist/lambda -name "__pycache__" -type d -prune -exec rm -rf {} +
rm -rf dist/lambda/*.dist-info/RECORD
(cd dist/lambda && zip -qr9 ../lambda.zip .)
echo "unzipped: $(du -sh dist/lambda | cut -f1)  zipped: $(du -h dist/lambda.zip | cut -f1)"
