# Download WenHai ONNX model and statistics files from S3.
import os
from pathlib import Path
import boto3
from botocore.exceptions import ClientError

MODELS_S3_BUCKET = os.environ.get("WENHAI_S3_BUCKET", "project-moi-ai")
MODELS_S3_PREFIX = "WENHAI_EDITO/WenHai"

MODEL_FILES = ["WenHai.onnx", "min_GLORYS.npy", "max_GLORYS.npy", "min_flux.npy", "max_flux.npy", "mask_GLORYS.nc"]


def get_s3_client():
    # Create boto3 S3 client from environment variables.
    endpoint = os.environ.get("AWS_S3_ENDPOINT", "minio.dive.edito.eu")
    if not endpoint.startswith("http"):
        endpoint = f"https://{endpoint}"
    return boto3.client(
        "s3",
        endpoint_url=endpoint,
        aws_access_key_id=os.environ["AWS_ACCESS_KEY_ID"],
        aws_secret_access_key=os.environ["AWS_SECRET_ACCESS_KEY"],
        aws_session_token=os.environ.get("AWS_SESSION_TOKEN"),
    )


def download_wenhai_model(output_dir):
    # Download WenHai model files from S3, or use local dir if WENHAI_LOCAL_DIR is set.
    local_dir = os.environ.get("WENHAI_LOCAL_DIR")
    if local_dir:
        local_path = Path(local_dir)
        missing = [f for f in MODEL_FILES if not (local_path / f).exists()]
        if missing:
            raise FileNotFoundError(f"Missing WenHai files in {local_dir}: {missing}")
        print(f"[OK] Using local WenHai files from {local_dir}")
        return {f: str(local_path / f) for f in MODEL_FILES}

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    s3 = get_s3_client()
    bucket = MODELS_S3_BUCKET

    paths = {}
    for filename in MODEL_FILES:
        local_file = output_path / filename
        if local_file.exists():
            print(f"   ✓ {filename} (cached)")
        else:
            s3_key = f"{MODELS_S3_PREFIX}/{filename}"
            try:
                print(f"   ⬇ {filename}...", end=" ", flush=True)
                s3.download_file(bucket, s3_key, str(local_file))
                print(f"✓ ({local_file.stat().st_size / 1e6:.1f} MB)")
            except ClientError as e:
                raise RuntimeError(f"Failed to download {filename}: {e}")
        paths[filename] = str(local_file)

    print(f"[OK] All WenHai files ready in {output_dir}")
    return paths