#!/usr/bin/env python3
import os
import sys
from datetime import datetime
from pathlib import Path

from fetch_copernicus_marine import fetch_marine_data
from fetch_era5 import fetch_era5_data
from s3_upload import save_file_to_s3


LOCAL_WORK_DIR = os.environ.get("LOCAL_WORK_DIR", "/tmp/wenhai_test_init")
MARINE_INIT_FILE_NAME = "marine_init.nc"
ERA5_INIT_FILE_NAME = "era5_init.nc"


def main():
    bucket_name = os.environ.get("AWS_BUCKET_NAME")
    s3_prefix = os.environ.get("TEST_INIT_S3_PREFIX", "wenhai-test-init")
    test_date_str = os.environ.get("TEST_INIT_DATE")

    if not bucket_name:
        print("[ERROR] AWS_BUCKET_NAME is required")
        sys.exit(1)
    if not test_date_str:
        print("[ERROR] TEST_INIT_DATE is required")
        sys.exit(1)

    try:
        test_date = datetime.strptime(test_date_str, "%Y-%m-%d").date()
    except ValueError:
        print(f"[ERROR] TEST_INIT_DATE '{test_date_str}' must be YYYY-MM-DD")
        sys.exit(1)

    work_dir = Path(LOCAL_WORK_DIR)
    marine_dir = work_dir / "marine"
    era5_dir = work_dir / "era5"
    marine_dir.mkdir(parents=True, exist_ok=True)
    era5_dir.mkdir(parents=True, exist_ok=True)

    print(f"Generating test init for {test_date}...")
    marine_file = fetch_marine_data(
        forecast_date=test_date,
        output_dir=str(marine_dir),
    )
    era5_file = fetch_era5_data(
        forecast_date=test_date,
        output_dir=str(era5_dir),
    )

    custom_init_prefix = f"{s3_prefix}/{test_date}"
    fixed_marine_file = work_dir / MARINE_INIT_FILE_NAME
    fixed_era5_file = work_dir / ERA5_INIT_FILE_NAME
    Path(marine_file).replace(fixed_marine_file)
    Path(era5_file).replace(fixed_era5_file)

    marine_key = f"{custom_init_prefix}/{MARINE_INIT_FILE_NAME}"
    era5_key = f"{custom_init_prefix}/{ERA5_INIT_FILE_NAME}"

    save_file_to_s3(
        bucket_name=bucket_name,
        local_file_path=str(fixed_marine_file),
        object_key=marine_key,
    )
    save_file_to_s3(
        bucket_name=bucket_name,
        local_file_path=str(fixed_era5_file),
        object_key=era5_key,
    )

    print("\nUse this for CUSTOM mode:")
    print(f"  export INIT_FILES_FOLDER_URL=s3://{bucket_name}/{custom_init_prefix}")


if __name__ == "__main__":
    main()
