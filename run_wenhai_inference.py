# Main orchestrator: download model, fetch data, run inference, upload zarr to S3.
import os
import shutil
import sys
from datetime import datetime, timedelta
from pathlib import Path

import xarray as xr

from model_manager import download_wenhai_model
from fetch_copernicus_marine import fetch_marine_data
from fetch_ifs import fetch_ifs_data
from wenhai_inference import run_inference
from s3_upload import download_from_s3, save_directory_to_s3
from generate_thumbnails import generate_thumbnails
from add_metadata import add_metadata_to_zarr

LOCAL_WORK_DIR = os.environ.get("LOCAL_WORK_DIR", "/tmp/wenhai")
MARINE_INIT_FILE_NAME = "marine_init.nc"
ATMOS_INIT_FILE_NAME = "era5_init.nc"


def validate_environment(use_custom_init: bool, require_aws_bucket_name: bool):
    required = [
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_S3_ENDPOINT",
    ]
    if require_aws_bucket_name:
        required.insert(0, "AWS_BUCKET_NAME")
    if not use_custom_init:
        required.extend(
            [
                "COPERNICUSMARINE_SERVICE_USERNAME",
                "COPERNICUSMARINE_SERVICE_PASSWORD",
                "ECMWF_API_KEY",
                "ECMWF_API_EMAIL",
            ]
        )
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print("[ERROR] Missing required environment variables:")
        for v in missing:
            print(f"   - {v}")
        sys.exit(1)


def download_init_file(init_url: str, local_dir: Path) -> str:
    # Download a custom init file from S3 into the local work directory.
    if not init_url.startswith("s3://"):
        raise ValueError(f"Only s3:// URLs are supported for custom init files, got: {init_url}")

    bucket_and_key = init_url[len("s3://"):]
    bucket_name, object_key = bucket_and_key.split("/", 1)
    local_path = local_dir / Path(object_key).name
    return download_from_s3(bucket_name, object_key, str(local_path))


def build_s3_file_url(folder_url: str, file_name: str) -> str:
    # Build a file URL from a custom init folder URL and a fixed file name.
    return folder_url.rstrip("/") + "/" + file_name


def extract_forecast_date_from_marine_file(marine_file: str):
    # Read the forecast date from the last time value in a marine init NetCDF.
    with xr.open_dataset(marine_file) as dataset:
        return (
            dataset.time.values[-1]
            .astype("datetime64[D]")
            .astype("datetime64[ms]")
            .astype(datetime)
            .date()
        )


def s3_output_is_file_path(s3_output_folder: str):
    output_value = s3_output_folder.strip()
    if output_value.startswith("s3://"):
        output_value = output_value[len("s3://"):]
    return output_value.endswith(".zarr")


def normalize_s3_key(*parts):
    return "/".join(
        segment
        for part in parts
        for segment in str(part).strip("/").split("/")
        if segment
    )


def resolve_s3_output(aws_bucket_name: str, s3_output_folder: str, forecast_date, zarr_name: str):
    # Resolve the final S3 upload target and the shared parent prefix for thumbnails.
    output_value = s3_output_folder.strip()

    if output_value.startswith("s3://"):
        output_value = output_value[len("s3://"):]
    output_value = output_value.strip("/")

    if output_value.endswith(".zarr"):
        if "/" not in output_value:
            raise ValueError(
                "S3_OUTPUT_FOLDER full file path must include bucket and key, "
                "for example: bucket/path/forecast_date/output.zarr"
            )
        bucket_name, file_key = output_value.split("/", 1)
        file_key = normalize_s3_key(file_key)
        if "/" not in file_key:
            raise ValueError(
                "S3_OUTPUT_FOLDER full file path must include a parent folder, "
                "for example: bucket/path/forecast_date/output.zarr"
            )
        output_prefix = file_key.rsplit("/", 1)[0]
        return bucket_name, file_key, output_prefix

    output_prefix = normalize_s3_key(output_value, forecast_date)
    file_key = normalize_s3_key(output_prefix, zarr_name)
    return aws_bucket_name, file_key, output_prefix


def main():
    print("=" * 60)
    print("WenHai Inference - EDITO Process")
    print("=" * 60)

    init_folder_url = os.environ.get("INIT_FILES_FOLDER_URL")
    use_custom_init = bool(init_folder_url)
    s3_output_folder = os.environ.get("S3_OUTPUT_FOLDER", "wenhai-inference")

    validate_environment(
        use_custom_init,
        require_aws_bucket_name=not s3_output_is_file_path(s3_output_folder),
    )

    print(f"Mode: {'CUSTOM' if use_custom_init else 'AUTO'}")

    print(f"S3 output: {s3_output_folder}")

    bucket_name = os.environ.get("AWS_BUCKET_NAME")

    print(f"\n{'=' * 60}")
    print("Step 1/7 - Downloading WenHai model files")
    print("=" * 60)
    model_dir = Path(LOCAL_WORK_DIR) / "model"
    download_wenhai_model(str(model_dir))

    if use_custom_init:
        print(f"\n{'=' * 60}")
        print("Step 2/7 - Using custom init files")
        print("=" * 60)
        custom_init_dir = Path(LOCAL_WORK_DIR) / "custom_init"
        custom_init_dir.mkdir(parents=True, exist_ok=True)
        marine_init_url = build_s3_file_url(init_folder_url, MARINE_INIT_FILE_NAME)
        atmos_init_url = build_s3_file_url(init_folder_url, ATMOS_INIT_FILE_NAME)
        nowcast_file = download_init_file(marine_init_url, custom_init_dir)
        atmos_file = download_init_file(atmos_init_url, custom_init_dir)
        FORECAST_DATE = extract_forecast_date_from_marine_file(nowcast_file)
        print(f"[OK] Marine init: {nowcast_file}")
        print(f"[OK] Atmos init:  {atmos_file}")
    else:
        forecast_date_str = os.environ.get("FORECAST_DATE")
        if not forecast_date_str:
            print("[ERROR] FORECAST_DATE is required in AUTO mode")
            sys.exit(1)
        try:
            FORECAST_DATE = datetime.strptime(forecast_date_str, "%Y-%m-%d").date()
        except ValueError:
            print(f"[ERROR] FORECAST_DATE '{forecast_date_str}' must be YYYY-MM-DD")
            sys.exit(1)

        print(f"\n{'=' * 60}")
        print("Step 2/7 - Fetching Copernicus Marine nowcast")
        print("=" * 60)
        nowcast_file = fetch_marine_data(FORECAST_DATE, str(Path(LOCAL_WORK_DIR) / "marine"))

        print(f"\n{'=' * 60}")
        print("Step 3/7 - Fetching IFS atmospheric forcing")
        print("=" * 60)
        atmos_file = fetch_ifs_data(FORECAST_DATE, str(Path(LOCAL_WORK_DIR) / "ifs"))

    forecast_start = FORECAST_DATE + timedelta(days=1)
    forecast_end   = FORECAST_DATE + timedelta(days=10)
    zarr_name      = f"WenHai_MOI_{forecast_start}_{forecast_end}.zarr"
    zarr_path      = Path(LOCAL_WORK_DIR) / zarr_name
    output_bucket, file_key, output_prefix = resolve_s3_output(
        aws_bucket_name=bucket_name,
        s3_output_folder=s3_output_folder,
        forecast_date=FORECAST_DATE,
        zarr_name=zarr_name,
    )
    print(f"Forecast date: {FORECAST_DATE}")
    print(f"Output: s3://{output_bucket}/{file_key}")

    print(f"\n{'=' * 60}")
    print("Step 4/7 - Running WenHai inference")
    print("=" * 60)
    forecast_ds = run_inference(
        nowcast_file=nowcast_file,
        atmos_file=atmos_file,
        model_dir=str(model_dir),
        output_path=str(Path(LOCAL_WORK_DIR) / "output"),
    )

    print(f"\n{'=' * 60}")
    print("Step 5/7 - Saving zarr")
    print("=" * 60)

    if zarr_path.exists():
        if zarr_path.is_dir():
            shutil.rmtree(zarr_path)
        else:
            zarr_path.unlink()
    forecast_ds.to_zarr(str(zarr_path), mode="w", consolidated=True)
    forecast_ds.close()
    print(f"[OK] Zarr saved: {zarr_path}")

    print(f"\n{'=' * 60}")
    print("Step 6/7 - Adding metadata")
    print("=" * 60)
    add_metadata_to_zarr(zarr_path)

    print(f"\n{'=' * 60}")
    print("Step 7/7 - Uploading zarr and thumbnails to S3")
    print("=" * 60)
    result_url = save_directory_to_s3(
        bucket_name=output_bucket,
        local_dir_path=zarr_path,
        object_prefix=file_key,
    )

    try:
        thumbnail_urls = generate_thumbnails(
            zarr_path=zarr_path,
            bucket_name=output_bucket,
            s3_prefix=output_prefix,
        )
        print(f"[OK] {len(thumbnail_urls)} thumbnails")
    except Exception as e:
        print(f"[WARNING] Thumbnail generation failed (non-fatal): {e}")

    shutil.rmtree(Path(LOCAL_WORK_DIR), ignore_errors=True)
    print(f"[OK] Cleaned up: {LOCAL_WORK_DIR}")

    print(f"\n{'=' * 60}")
    print("WenHai Inference - DONE")
    print(f"  Output: {result_url}")
    print(f"  Forecast: {forecast_start} to {forecast_end} (10 days)")
    print("=" * 60)


if __name__ == "__main__":
    main()
