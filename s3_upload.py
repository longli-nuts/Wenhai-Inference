# S3 upload utilities using boto3.
import os
import boto3


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
    )


def save_bytes_to_s3(bucket_name, object_bytes, object_key):
    # Upload raw bytes to S3.
    get_s3_client().put_object(Bucket=bucket_name, Body=object_bytes, Key=object_key)
    print(f"[OK] s3://{bucket_name}/{object_key}")


def save_file_to_s3(bucket_name, local_file_path, object_key):
    # Upload a local file to S3.
    print(f"Uploading {local_file_path} -> s3://{bucket_name}/{object_key}...")
    get_s3_client().upload_file(local_file_path, bucket_name, object_key)
    print(f"[OK] s3://{bucket_name}/{object_key}")