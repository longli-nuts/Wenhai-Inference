"""
Fetch ERA5 reanalysis data from Copernicus Climate Data Store for WenHai.

WenHai needs 8 atmospheric variables for aerobulk bulk flux computation:
  t2m, d2m, u10, v10, msl (instantaneous)
  tp, ssrd, strd (accumulated)

CDS returns a ZIP with two separate NetCDF files:
  - data_stream-oper_stepType-instant.nc  (instantaneous vars)
  - data_stream-oper_stepType-accum.nc    (accumulated vars)

Downloads at 0.25° for the 10 forecast days, then upsamples to 0.083°.
"""
import os
import calendar
import shutil
import zipfile
from datetime import datetime, timedelta
from pathlib import Path
import cdsapi
import xarray as xr
import numpy as np


DATASET = "reanalysis-era5-single-levels"

ERA5_VARIABLES = [
    "2m_temperature",
    "2m_dewpoint_temperature",
    "10m_u_component_of_wind",
    "10m_v_component_of_wind",
    "mean_sea_level_pressure",
    "total_precipitation",
    "surface_solar_radiation_downwards",
    "surface_thermal_radiation_downwards",
]

TARGET_LON = np.arange(-180, 180, 1/12)
TARGET_LAT = np.arange(-80, 90.083, 1/12)

INSTANTANEOUS = ["t2m", "d2m", "u10", "v10", "msl"]
ACCUMULATED   = ["tp", "ssrd", "strd"]
ERA5_ALIASES = {
    "ssrd": ["ssrd", "ssr"],
    "strd": ["strd", "str"],
}


def _write_cdsapirc():
    cds_key = os.environ.get("CDS_API_KEY")
    if not cds_key:
        raise ValueError("CDS_API_KEY environment variable not set")
    cdsapi_rc = Path.home() / ".cdsapirc"
    cdsapi_rc.write_text(f"url: https://cds.climate.copernicus.eu/api\nkey: {cds_key}\n")


def _upsample_to_wenhai_grid(ds):
    lat_target = TARGET_LAT[TARGET_LAT <= 90.0]
    return ds.interp(latitude=lat_target, longitude=TARGET_LON, method="linear")


def _sanitize_era5_dataset(ds):
    if "expver" in ds.variables:
        ds = ds.drop_vars("expver")
    if "number" in ds.dims and ds.sizes["number"] == 1:
        ds = ds.isel(number=0, drop=True)
    return ds


def _open_era5_zip(zip_path, extract_dir):
    """Unzip CDS archive and merge instant + accum NetCDF files."""
    with zipfile.ZipFile(zip_path, "r") as zf:
        zf.extractall(extract_dir)

    instant_file = extract_dir / "data_stream-oper_stepType-instant.nc"
    accum_file   = extract_dir / "data_stream-oper_stepType-accum.nc"

    datasets = []
    if instant_file.exists():
        datasets.append(_sanitize_era5_dataset(xr.open_dataset(instant_file)))
    if accum_file.exists():
        datasets.append(_sanitize_era5_dataset(xr.open_dataset(accum_file)))

    if not datasets:
        raise FileNotFoundError(f"No NetCDF files found after unzipping {zip_path}")

    return xr.merge(datasets) if len(datasets) > 1 else datasets[0]


def _get_era5_var(ds, name):
    for candidate in ERA5_ALIASES.get(name, [name]):
        if candidate in ds:
            data = ds[candidate]
            if candidate != name:
                data = data.rename(name)
            return data
    raise KeyError(f"Missing ERA5 variable '{name}'. Available variables: {list(ds.data_vars)}")


def _validate_era5_dataset(ds, context):
    expected_vars = INSTANTANEOUS + ACCUMULATED
    missing_vars = [var for var in expected_vars if var not in ds]
    if missing_vars:
        raise ValueError(
            f"{context} is missing required variables {missing_vars}. "
            f"Available variables: {list(ds.data_vars)}"
        )


def fetch_era5_data(forecast_date, output_dir):
    """
    Download ERA5 forcing data for WenHai (10 forecast days).

    Args:
        forecast_date: datetime.date - Init date of the forecast
        output_dir: str - Output directory

    Returns:
        str - Path to upsampled NetCDF file at 0.083°
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    forecast_start = forecast_date + timedelta(days=1)
    forecast_end   = forecast_date + timedelta(days=10)

    date_str = forecast_date.strftime("%Y%m%d")
    upsampled_file = output_path / f"era5_wenhai_{date_str}.nc"

    if upsampled_file.exists():
        with xr.open_dataset(upsampled_file) as cached_ds:
            try:
                _validate_era5_dataset(cached_ds, f"Cached ERA5 file {upsampled_file}")
                print(f"[OK] ERA5 file already exists: {upsampled_file}")
                return str(upsampled_file)
            except ValueError as exc:
                print(f"[WARNING] {exc}")
                print(f"[WARNING] Removing invalid cached ERA5 file: {upsampled_file}")
        upsampled_file.unlink(missing_ok=True)

    print(f"Fetching ERA5 forcing data:")
    print(f"   Variables: t2m, d2m, u10, v10, msl, tp, ssrd, strd (8 vars)")
    print(f"   Forecast window: {forecast_start} to {forecast_end} (10 days)")

    _write_cdsapirc()
    c = cdsapi.Client()

    months_needed = set()
    for i in range(10):
        d = forecast_start + timedelta(days=i)
        months_needed.add((d.year, d.month))

    monthly_datasets = []

    for year, month in sorted(months_needed):
        ym = f"{year}{month:02d}"
        zip_file    = output_path / f"era5_raw_{ym}.zip"
        extract_dir = output_path / f"era5_raw_{ym}"
        extract_dir.mkdir(exist_ok=True)

        if not zip_file.exists():
            last_day = calendar.monthrange(year, month)[1]
            all_days = [f"{d:02d}" for d in range(1, last_day + 1)]

            request = {
                "product_type":    "reanalysis",
                "variable":        ERA5_VARIABLES,
                "year":            str(year),
                "month":           f"{month:02d}",
                "day":             all_days,
                "time":            ["00:00", "06:00", "12:00", "18:00"],
                "area":            [90, -180, -80, 180],
                "data_format":     "netcdf",
                "download_format": "zip",
            }

            print(f"   Downloading ERA5 {ym} (0.25°)...")
            c.retrieve(DATASET, request, str(zip_file))
            size_mb = zip_file.stat().st_size / 1e6
            print(f"   [OK] {zip_file} ({size_mb:.1f} MB)")
        else:
            print(f"   [OK] Already cached: {zip_file}")

        ds = _open_era5_zip(zip_file, extract_dir)
        monthly_datasets.append(ds)

    ds = xr.merge(monthly_datasets) if len(monthly_datasets) > 1 else monthly_datasets[0]
    ds = ds.sel(valid_time=slice(str(forecast_start), str(forecast_end)))

    # Daily aggregation
    daily_parts = {}
    for var in INSTANTANEOUS:
        daily_parts[var] = _get_era5_var(ds, var).resample(valid_time="1D").mean()
    for var in ACCUMULATED:
        daily_parts[var] = _get_era5_var(ds, var).resample(valid_time="1D").sum()

    ds_daily = xr.Dataset(daily_parts)
    ds_daily = ds_daily.rename({"valid_time": "time_counter"})

    _validate_era5_dataset(ds_daily, "ERA5 daily dataset")

    print("Upsampling ERA5 from 0.25° to 0.083°...")
    ds_upsampled = _upsample_to_wenhai_grid(ds_daily)
    ds_upsampled.to_netcdf(str(upsampled_file))

    # Cleanup
    for ds_ in monthly_datasets:
        ds_.close()
    for year, month in months_needed:
        ym = f"{year}{month:02d}"
        (output_path / f"era5_raw_{ym}.zip").unlink(missing_ok=True)
        shutil.rmtree(str(output_path / f"era5_raw_{ym}"), ignore_errors=True)

    size_mb = upsampled_file.stat().st_size / 1e6
    print(f"[OK] ERA5 forcing file: {upsampled_file} ({size_mb:.1f} MB)")

    return str(upsampled_file)


if __name__ == "__main__":
    import sys
    test_date = datetime.strptime(sys.argv[1], "%Y-%m-%d").date() if len(sys.argv) > 1 \
        else datetime.now().date() - timedelta(days=2)
    print(f"Testing ERA5 download for {test_date}")
    out = fetch_era5_data(test_date, "/tmp/wenhai_test/era5")
    print(f"Success: {out}")
