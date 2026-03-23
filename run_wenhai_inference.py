# Main orchestrator: download model, fetch data, run inference, upload zarr to S3.
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

import xarray as xr

from model_manager import download_wenhai_model
from fetch_copernicus_marine import fetch_marine_data
from fetch_era5 import fetch_era5_data
from wenhai_inference import run_inference
from s3_upload import download_from_s3, save_file_to_s3
from generate_thumbnails import generate_thumbnails

LOCAL_WORK_DIR = os.environ.get("LOCAL_WORK_DIR", "/tmp/wenhai")
MARINE_INIT_FILE_NAME = "marine_init.nc"
ERA5_INIT_FILE_NAME = "era5_init.nc"


def validate_environment(use_custom_init: bool):
    required = [
        "AWS_BUCKET_NAME",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_S3_ENDPOINT",
    ]
    if not use_custom_init:
        required.extend(
            [
                "COPERNICUSMARINE_SERVICE_USERNAME",
                "COPERNICUSMARINE_SERVICE_PASSWORD",
                "CDS_API_KEY",
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


def upload_with_retry(bucket, local_path, s3_key, retries=5, wait=30):
    # Upload file to S3 with exponential backoff retry on SlowDown errors.
    for attempt in range(retries):
        try:
            save_file_to_s3(bucket, local_path, s3_key)
            return
        except Exception as e:
            if attempt < retries - 1:
                print(f"[WARNING] Upload failed (attempt {attempt+1}/{retries}): {e}")
                time.sleep(wait * (2 ** attempt))
            else:
                raise


def main():
    print("=" * 60)
    print("WenHai Inference - EDITO Process")
    print("=" * 60)

    init_folder_url = os.environ.get("INIT_FILES_FOLDER_URL")
    use_custom_init = bool(init_folder_url)

    validate_environment(use_custom_init)

    print(f"Mode: {'CUSTOM' if use_custom_init else 'AUTO'}")

    s3_output_folder = os.environ.get("S3_OUTPUT_FOLDER")
    if not s3_output_folder:
        bucket = os.environ.get("AWS_BUCKET_NAME")
        s3_output_folder = f"{bucket}/wenhai-inference"
    print(f"S3 output: {s3_output_folder}")

    bucket_name = s3_output_folder.split("/", 1)[0]
    s3_prefix = s3_output_folder.split("/", 1)[1] if "/" in s3_output_folder else ""

    print(f"\n{'=' * 60}")
    print("Step 1/5 - Downloading WenHai model files")
    print("=" * 60)
    model_dir = Path(LOCAL_WORK_DIR) / "model"
    download_wenhai_model(str(model_dir))

    if use_custom_init:
        print(f"\n{'=' * 60}")
        print("Step 2/5 - Using custom init files")
        print("=" * 60)
        custom_init_dir = Path(LOCAL_WORK_DIR) / "custom_init"
        custom_init_dir.mkdir(parents=True, exist_ok=True)
        marine_init_url = build_s3_file_url(init_folder_url, MARINE_INIT_FILE_NAME)
        era5_init_url = build_s3_file_url(init_folder_url, ERA5_INIT_FILE_NAME)
        nowcast_file = download_init_file(marine_init_url, custom_init_dir)
        era5_file = download_init_file(era5_init_url, custom_init_dir)
        FORECAST_DATE = extract_forecast_date_from_marine_file(nowcast_file)
        print(f"[OK] Marine init: {nowcast_file}")
        print(f"[OK] ERA5 init:   {era5_file}")
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
        print("Step 2/5 - Fetching Copernicus Marine nowcast")
        print("=" * 60)
        nowcast_file = fetch_marine_data(FORECAST_DATE, str(Path(LOCAL_WORK_DIR) / "marine"))

        print(f"\n{'=' * 60}")
        print("Step 3/5 - Fetching ERA5 atmospheric forcing")
        print("=" * 60)
        era5_file = fetch_era5_data(FORECAST_DATE, str(Path(LOCAL_WORK_DIR) / "era5"))

    forecast_start = FORECAST_DATE + timedelta(days=1)
    forecast_end   = FORECAST_DATE + timedelta(days=10)
    zarr_filename  = f"WenHai_MOI_{forecast_start}_{forecast_end}.zarr.zip"
    s3_zarr_key    = f"{s3_prefix}/{FORECAST_DATE}/{zarr_filename}"
    print(f"Forecast date: {FORECAST_DATE}")

    print(f"\n{'=' * 60}")
    print("Step 4/5 - Running WenHai inference")
    print("=" * 60)
    forecast_ds = run_inference(
        nowcast_file=nowcast_file,
        era5_file=era5_file,
        model_dir=str(model_dir),
        output_path=str(Path(LOCAL_WORK_DIR) / "output"),
    )

    print("Generating thumbnails...")
    thumb_prefix = f"{s3_prefix}/{FORECAST_DATE}/thumbnails"
    thumbnail_urls = {
        var: f"s3://{bucket_name}/{thumb_prefix}/{var}.png"
        for var in ["zos", "thetao", "so", "uo", "vo"]
    }
    try:
        generate_thumbnails(
            bucket_name=bucket_name,
            forecast_netcdf_file_url=f"s3://{bucket_name}/{s3_zarr_key}",
            thumbnail_urls=thumbnail_urls,
            forecast_dataset=forecast_ds,
        )
        print("[OK] Thumbnails uploaded")
    except Exception as e:
        print(f"[WARNING] Thumbnail generation failed (non-fatal): {e}")

    print(f"\n{'=' * 60}")
    print("Step 5/5 - Saving zarr.zip and uploading to S3")
    print("=" * 60)

    final_zarr = Path(LOCAL_WORK_DIR) / zarr_filename
    import zarr
    store = zarr.ZipStore(str(final_zarr), mode="w")
    forecast_ds.to_zarr(store, consolidated=True)
    store.close()
    size_mb = final_zarr.stat().st_size / 1e6
    print(f"[OK] Zarr zip saved: {final_zarr} ({size_mb:.1f} MB)")

    upload_with_retry(bucket_name, str(final_zarr), s3_zarr_key)

    forecast_ds.close()

    print(f"\n{'=' * 60}")
    print("WenHai Inference - DONE")
    print(f"  Output: s3://{bucket_name}/{s3_zarr_key}")
    print(f"  Forecast: {forecast_start} to {forecast_end} (10 days)")
    print("=" * 60)


if __name__ == "__main__":
    main()
