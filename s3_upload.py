# S3 upload utilities using boto3.
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import boto3
from boto3.s3.transfer import TransferConfig
from botocore.config import Config

SINGLE_UPLOAD_LIMIT = 5 * 1024 * 1024 * 1024
MAX_UPLOAD_WORKERS = int(os.environ.get("S3_UPLOAD_WORKERS", "32"))
SUCCESS_MARKER = "_SUCCESS"


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


def upload_bytes_to_s3(bucket_name, data_bytes, object_key, content_type="application/octet-stream"):
    # Upload raw bytes directly to S3.
    get_s3_client().put_object(
        Bucket=bucket_name,
        Body=data_bytes,
        Key=object_key,
        ContentType=content_type,
    )
    s3_url = f"s3://{bucket_name}/{object_key}"
    print(f"[OK] {s3_url}")
    return s3_url


def delete_s3_prefix(bucket_name, object_prefix):
    # Delete all objects under an S3 prefix in parallel.
    s3_client = get_s3_client()
    object_prefix = str(object_prefix).strip("/")
    if not object_prefix:
        raise ValueError("Refusing to delete an empty S3 prefix")

    paginator = s3_client.get_paginator("list_objects_v2")
    keys = []
    for page in paginator.paginate(Bucket=bucket_name, Prefix=object_prefix):
        keys.extend(
            obj["Key"]
            for obj in page.get("Contents", [])
            if obj["Key"] == object_prefix or obj["Key"].startswith(f"{object_prefix}/")
        )

    if not keys:
        return

    print(
        f"Deleting existing S3 prefix: s3://{bucket_name}/{object_prefix}/ "
        f"({len(keys)} objects, {MAX_UPLOAD_WORKERS} workers)"
    )

    deleted_count = 0
    with ThreadPoolExecutor(max_workers=MAX_UPLOAD_WORKERS) as executor:
        futures = [
            executor.submit(s3_client.delete_object, Bucket=bucket_name, Key=key)
            for key in keys
        ]
        for future in as_completed(futures):
            future.result()
            deleted_count += 1
            if deleted_count % 1000 == 0 or deleted_count == len(keys):
                print(f"  Deleted {deleted_count}/{len(keys)} objects")


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


def save_directory_to_s3(bucket_name, local_dir_path, object_prefix):
    # Upload a local directory tree to S3 under object_prefix.
    s3_client = get_s3_client()
    local_dir = Path(local_dir_path)

    if not local_dir.is_dir():
        raise FileNotFoundError(f"Local directory not found: {local_dir_path}")

    transfer_config = TransferConfig(
        multipart_threshold=64 * 1024 * 1024,
        multipart_chunksize=64 * 1024 * 1024,
        max_concurrency=1,
        use_threads=False,
    )

    files = [path for path in sorted(local_dir.rglob("*")) if path.is_file()]
    if not files:
        raise RuntimeError(f"No files found in local directory: {local_dir_path}")

    total_size_mb = sum(path.stat().st_size for path in files) / 1e6
    print(
        f"Uploading directory: {local_dir.name} "
        f"({len(files)} files, {total_size_mb:.2f} MB, {MAX_UPLOAD_WORKERS} workers)"
    )

    object_prefix = str(object_prefix).strip("/")
    if object_prefix:
        delete_s3_prefix(bucket_name, object_prefix)
    success_key = f"{object_prefix}/{SUCCESS_MARKER}" if object_prefix else SUCCESS_MARKER

    def upload_one(path):
        relative_key = path.relative_to(local_dir).as_posix()
        object_key = f"{object_prefix}/{relative_key}" if object_prefix else relative_key
        s3_client.upload_file(str(path), bucket_name, object_key, Config=transfer_config)

    uploaded = 0
    with ThreadPoolExecutor(max_workers=MAX_UPLOAD_WORKERS) as executor:
        futures = [executor.submit(upload_one, path) for path in files]
        for future in as_completed(futures):
            future.result()
            uploaded += 1
            if uploaded % 1000 == 0 or uploaded == len(files):
                print(f"  Uploaded {uploaded}/{len(files)} files")

    s3_client.put_object(
        Bucket=bucket_name,
        Key=success_key,
        Body=b"",
        ContentType="text/plain",
    )
    print(f"[OK] Ready marker: s3://{bucket_name}/{success_key}")

    s3_url = f"s3://{bucket_name}/{object_prefix}"
    print(f"[OK] {s3_url}")
    return s3_url
