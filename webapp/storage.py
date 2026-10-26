"""Where run files live: local disk by default, S3 when ADLINT_S3_BUCKET is set."""

import mimetypes
import os

BUCKET = os.environ.get("ADLINT_S3_BUCKET")
PREFIX = os.environ.get("ADLINT_S3_PREFIX", "runs")

_client = None


def _s3():
    global _client
    if _client is None:
        import boto3
        from botocore.config import Config

        region = os.environ.get("AWS_REGION") or os.environ.get("AWS_DEFAULT_REGION")
        endpoint = f"https://s3.{region}.amazonaws.com" if region else None  # presigned URLs must name the region
        _client = boto3.client("s3", region_name=region, endpoint_url=endpoint, config=Config(connect_timeout=5, read_timeout=30, retries={"max_attempts": 3},
                                                    signature_version="s3v4"))
    return _client


def uses_s3():
    return bool(BUCKET)


def publish(run_id, folder):
    """Copy a finished run folder to S3 (no-op locally: files are served from disk)."""
    if not BUCKET:
        return
    for root, _, files in os.walk(folder):
        for name in files:
            path = os.path.join(root, name)
            key = f"{PREFIX}/{run_id}/{os.path.relpath(path, folder)}"
            content_type = mimetypes.guess_type(name)[0] or "application/octet-stream"
            _s3().upload_file(path, BUCKET, key, ExtraArgs={"ContentType": content_type})


def presigned_url(run_id, name, seconds=3600):
    return _s3().generate_presigned_url("get_object", Params={"Bucket": BUCKET, "Key": f"{PREFIX}/{run_id}/{name}"},
                                        ExpiresIn=seconds)


def read(run_id, name):
    """Small text files (report.html, *.json) are proxied so their relative links keep working."""
    return _s3().get_object(Bucket=BUCKET, Key=f"{PREFIX}/{run_id}/{name}")["Body"].read()
