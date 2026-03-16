# Main orchestrator: download model, fetch data, run inference, upload zarr to S3.
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path

from model_manager import download_wenhai_model
from fetch_copernicus_marine import fetch_marine_data
from fetch_era5 import fetch_era5_data
from wenhai_inference import run_inference
from s3_upload import save_file_to_s3
from generate_thumbnails import generate_thumbnails

LOCAL_WORK_DIR = os.environ.get("LOCAL_WORK_DIR", "/tmp/wenhai")


def validate_environment():
    required = [
        "COPERNICUSMARINE_SERVICE_USERNAME",
        "COPERNICUSMARINE_SERVICE_PASSWORD",
        "CDS_API_KEY",
        "AWS_BUCKET_NAME",
        "AWS_ACCESS_KEY_ID",
        "AWS_SECRET_ACCESS_KEY",
        "AWS_SESSION_TOKEN",
        "AWS_S3_ENDPOINT",
    ]
    missing = [v for v in required if not os.environ.get(v)]
    if missing:
        print("[ERROR] Missing required environment variables:")
        for v in missing:
            print(f"   - {v}")
        sys.exit(1)


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

    validate_environment()

    default_date_str = (datetime.now() - timedelta(days=1)).strftime("%Y-%m-%d")
    forecast_date_str = os.environ.get("FORECAST_DATE", default_date_str) or default_date_str
    try:
        FORECAST_DATE = datetime.strptime(forecast_date_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"[ERROR] FORECAST_DATE '{forecast_date_str}' must be YYYY-MM-DD")
        sys.exit(1)

    print(f"Forecast date: {FORECAST_DATE}")

    s3_output_folder = os.environ.get("S3_OUTPUT_FOLDER")
    if not s3_output_folder:
        bucket = os.environ.get("AWS_BUCKET_NAME")
        s3_output_folder = f"{bucket}/wenhai-inference"
    print(f"S3 output: {s3_output_folder}")

    bucket_name = s3_output_folder.split("/", 1)[0]
    s3_prefix = s3_output_folder.split("/", 1)[1] if "/" in s3_output_folder else ""

    forecast_start = FORECAST_DATE + timedelta(days=1)
    forecast_end   = FORECAST_DATE + timedelta(days=10)
    zarr_filename  = f"WenHai_MOI_{forecast_start}_{forecast_end}.zarr.zip"
    s3_zarr_key    = f"{s3_prefix}/{FORECAST_DATE}/{zarr_filename}"

    print(f"\n{'=' * 60}")
    print("Step 1/5 - Downloading WenHai model files")
    print("=" * 60)
    model_dir = Path(LOCAL_WORK_DIR) / "model"
    download_wenhai_model(str(model_dir))

    print(f"\n{'=' * 60}")
    print("Step 2/5 - Fetching Copernicus Marine nowcast")
    print("=" * 60)
    nowcast_file = fetch_marine_data(FORECAST_DATE, str(Path(LOCAL_WORK_DIR) / "marine"))

    print(f"\n{'=' * 60}")
    print("Step 3/5 - Fetching ERA5 atmospheric forcing")
    print("=" * 60)
    era5_file = fetch_era5_data(FORECAST_DATE, str(Path(LOCAL_WORK_DIR) / "era5"))

    print(f"\n{'=' * 60}")
    print("Step 4/5 - Running WenHai inference")
    print("=" * 60)
    forecast_ds = run_inference(
        nowcast_file=nowcast_file,
        era5_file=era5_file,
        model_dir=str(model_dir),
        output_path=str(Path(LOCAL_WORK_DIR) / "output"),
    )

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

    forecast_ds.close()

    print(f"\n{'=' * 60}")
    print("WenHai Inference - DONE")
    print(f"  Output: s3://{bucket_name}/{s3_zarr_key}")
    print(f"  Forecast: {forecast_start} to {forecast_end} (10 days)")
    print("=" * 60)


if __name__ == "__main__":
    main()