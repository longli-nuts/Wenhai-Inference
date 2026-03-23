# S3 upload utilities using boto3.
import os
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

SINGLE_UPLOAD_LIMIT = 5 * 1024 * 1024 * 1024


def get_s3_client():
    # Create boto3 S3 client from environment variables.
    endpoint = os.environ.get("AWS_S3_ENDPOINT", "")
    if not endpoint.startswith("http"):
        endpoint = f"https://{endpoint}"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
        config=Config(
            retries={"max_attempts": 10, "mode": "adaptive"},
            connect_timeout=60,
            read_timeout=300,
            max_pool_connections=2,
        ),
    )


def save_bytes_to_s3(bucket_name, object_bytes, object_key):
    # Upload raw bytes to S3.
    get_s3_client().put_object(Bucket=bucket_name, Body=object_bytes, Key=object_key)
    print(f"[OK] s3://{bucket_name}/{object_key}")


def download_from_s3(bucket_name, object_key, local_file_path):
    # Download a single file from S3 to a local path.
    local_path = Path(local_file_path)
    local_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"Downloading s3://{bucket_name}/{object_key} -> {local_path}...")
    get_s3_client().download_file(bucket_name, object_key, str(local_path))
    print(f"[OK] {local_path}")
    return str(local_path)


def save_file_to_s3(bucket_name, local_file_path, object_key):
    # Upload a local file to S3.
    print(f"Uploading {local_file_path} -> s3://{bucket_name}/{object_key}...")
    local_path = Path(local_file_path)
    s3_client = get_s3_client()

    if local_path.stat().st_size < SINGLE_UPLOAD_LIMIT:
        with local_path.open("rb") as file_handle:
            s3_client.put_object(Bucket=bucket_name, Key=object_key, Body=file_handle)
        print(f"[OK] s3://{bucket_name}/{object_key}")
        return

    transfer_config = TransferConfig(
        multipart_threshold=1024 * 1024 * 512,
        multipart_chunksize=1024 * 1024 * 1024,
        max_concurrency=1,
        use_threads=False,
    )
    s3_client.upload_file(
        str(local_path),
        bucket_name,
        object_key,
        Config=transfer_config,
    )
    print(f"[OK] s3://{bucket_name}/{object_key}")
