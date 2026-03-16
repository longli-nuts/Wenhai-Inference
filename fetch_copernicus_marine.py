"""
Fetch ocean nowcast data from Copernicus Marine Data Store for WenHai.

WenHai uses 23 specific depth levels from GLO12 (GLORYS12).
Variables: thetao, so, uo, vo, zos.
"""
import os
from datetime import datetime, timedelta
from pathlib import Path
import copernicusmarine
import xarray as xr
import numpy as np

DATASETS = {
    "cmems_mod_glo_phy-thetao_anfc_0.083deg_P1D-m": ["thetao"],
    "cmems_mod_glo_phy-so_anfc_0.083deg_P1D-m": ["so"],
    "cmems_mod_glo_phy-cur_anfc_0.083deg_P1D-m": ["uo", "vo"],
    "cmems_mod_glo_phy_anfc_0.083deg_P1D-m": ["zos"],
}

# WenHai's 23 specific depth levels (metres)
WENHAI_DEPTHS = [
    0.494025, 2.645669, 5.078224, 7.92956, 11.405, 15.81007, 21.59882,
    29.44473, 40.34405, 55.76429, 77.85385, 92.32607, 109.7293, 130.666,
    155.8507, 186.1256, 222.4752, 266.0403, 318.1274, 380.213, 453.9377,
    541.0889, 643.5668,
]

DEPTH_MIN = min(WENHAI_DEPTHS)
DEPTH_MAX = max(WENHAI_DEPTHS)
LON_MIN = -180.0
LON_MAX = 180.0
LAT_MIN = -80.0
LAT_MAX = 90.0


def fetch_marine_data(forecast_date, output_dir):
    """
    Download Copernicus Marine nowcast data for WenHai initial conditions.

    Args:
        forecast_date: datetime.date - Analysis date (init day of forecast)
        output_dir: str - Output directory

    Returns:
        str - Path to merged NetCDF file
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    date_str = forecast_date.strftime("%Y%m%d")
    output_file = output_path / f"wenhai_nowcast_{date_str}.nc"

    if output_file.exists():
        print(f"[OK] Already exists: {output_file}")
        return str(output_file)

    start_date = forecast_date.strftime("%Y-%m-%d")
    end_date = forecast_date.strftime("%Y-%m-%d")

    print(f"Fetching Copernicus Marine nowcast data:")
    print(f"   Date: {start_date}")
    print(f"   Depth levels: {len(WENHAI_DEPTHS)} levels ({DEPTH_MIN}-{DEPTH_MAX} m)")

    tmp_files = []

    for dataset_id, variables in DATASETS.items():
        tmp_file = output_path / f"tmp_{dataset_id}_{date_str}.nc"
        print(f"   Downloading {variables} from {dataset_id}...")

        is_zos = variables == ["zos"]
        depth_min = None if is_zos else DEPTH_MIN
        depth_max = None if is_zos else DEPTH_MAX

        try:
            copernicusmarine.subset(
                dataset_id=dataset_id,
                variables=variables,
                minimum_longitude=LON_MIN,
                maximum_longitude=LON_MAX,
                minimum_latitude=LAT_MIN,
                maximum_latitude=LAT_MAX,
                start_datetime=start_date,
                end_datetime=end_date,
                minimum_depth=depth_min,
                maximum_depth=depth_max,
                output_filename=str(tmp_file.name),
                output_directory=str(output_path),
                force_download=True,
                username=os.environ.get("COPERNICUSMARINE_SERVICE_USERNAME"),
                password=os.environ.get("COPERNICUSMARINE_SERVICE_PASSWORD"),
            )
            tmp_files.append(str(tmp_file))
            print(f"   [OK] {variables}")
        except Exception as e:
            print(f"   [ERROR] {dataset_id}: {e}")
            raise

    print(f"Merging {len(tmp_files)} files...")
    datasets = [xr.open_dataset(f) for f in tmp_files]
    merged = xr.merge(datasets)

    # Select exact WenHai depth levels (nearest)
    if "depth" in merged.dims:
        merged = merged.sel(depth=WENHAI_DEPTHS, method="nearest")

    merged.to_netcdf(str(output_file))
    for ds in datasets:
        ds.close()

    for f in tmp_files:
        Path(f).unlink(missing_ok=True)

    size_mb = output_file.stat().st_size / 1e6
    print(f"[OK] Nowcast file: {output_file} ({size_mb:.1f} MB)")

    return str(output_file)


if __name__ == "__main__":
    import sys
    test_date = datetime.strptime(sys.argv[1], "%Y-%m-%d").date() if len(sys.argv) > 1 \
        else datetime.now().date() - timedelta(days=2)
    print(f"Testing CMEMS download for {test_date}")
    out = fetch_copernicus_marine(test_date, "/tmp/wenhai_test/marine")
    print(f"Success: {out}")