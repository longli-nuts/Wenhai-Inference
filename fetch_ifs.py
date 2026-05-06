"""
Fetch ECMWF IFS operational forecast forcing data for WenHai.

WenHai needs 8 atmospheric variables for aerobulk bulk flux computation:
  t2m, d2m, u10, v10, msl (instantaneous)
  tp, ssrd, strd (accumulated from forecast start)

The output NetCDF matches the existing ERA5 forcing contract:
  dims: time_counter, latitude, longitude
  instantaneous variables: daily mean
  accumulated variables: daily total, later divided by 86400 in wenhai_inference.py
"""
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import xarray as xr
from ecmwfapi import ECMWFService


TARGET_LON = np.arange(-180, 180, 1 / 12)
TARGET_LAT = np.arange(-80, 90.083, 1 / 12)

INSTANTANEOUS = ["t2m", "d2m", "u10", "v10", "msl"]
ACCUMULATED = ["tp", "ssrd", "strd"]

IFS_PARAMS = {
    "msl": "151.128",
    "u10": "165.128",
    "v10": "166.128",
    "t2m": "167.128",
    "d2m": "168.128",
    "ssrd": "169.128",
    "strd": "175.128",
    "tp": "228.128",
}


def _get_mars_client():
    api_key = os.environ.get("ECMWF_API_KEY")
    api_email = os.environ.get("ECMWF_API_EMAIL")
    if not api_key or not api_email:
        missing = [
            name
            for name, value in {
                "ECMWF_API_KEY": api_key,
                "ECMWF_API_EMAIL": api_email,
            }.items()
            if not value
        ]
        raise ValueError(f"Missing environment variables: {', '.join(missing)}")
    os.environ["ECMWF_API_URL"] = "https://api.ecmwf.int/v1"
    return ECMWFService("mars")


def _validate_forcing_dataset(ds, context):
    expected_vars = INSTANTANEOUS + ACCUMULATED
    missing_vars = [var for var in expected_vars if var not in ds]
    if missing_vars:
        raise ValueError(
            f"{context} is missing required variables {missing_vars}. "
            f"Available variables: {list(ds.data_vars)}"
        )


def _validate_daily_forcing_dataset(ds, context):
    _validate_forcing_dataset(ds, context)
    if "time_counter" not in ds.dims:
        raise ValueError(f"{context} is missing time_counter dimension")


def _fetch_ifs_grib(forecast_date, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    date_str = forecast_date.strftime("%Y%m%d")
    raw_file = output_path / f"ifs_wenhai_raw_{date_str}.grib"
    if raw_file.exists():
        print(f"[OK] IFS GRIB already exists: {raw_file}")
        return raw_file

    print("Fetching IFS atmospheric forcing via MARS:")
    print(f"   Date:      {forecast_date}")
    print("   Variables: t2m, d2m, u10, v10, msl, tp, ssrd, strd")
    print("   Steps:     0 to 240h (10 days, every 6h)")

    _get_mars_client().execute(
        {
            "class": "od",
            "date": forecast_date.strftime("%Y-%m-%d"),
            "expver": "1",
            "levtype": "sfc",
            "param": "/".join(IFS_PARAMS.values()),
            "step": "0/to/240/by/6",
            "stream": "oper",
            "time": "00",
            "type": "fc",
            "grid": "0.25/0.25",
            "area": "90/-180/-80/180",
        },
        str(raw_file),
    )

    print(f"[OK] IFS GRIB downloaded: {raw_file} ({raw_file.stat().st_size / 1e6:.1f} MB)")
    return raw_file


def _normalize_coordinates(ds):
    if "valid_time" in ds.coords:
        ds = ds.drop_vars("valid_time")
    if "time" in ds.coords:
        ds = ds.drop_vars("time")

    if "longitude" in ds.coords and float(ds.longitude.max()) > 180.0:
        longitude = ((ds.longitude + 180) % 360) - 180
        ds = ds.assign_coords(longitude=longitude).sortby("longitude")

    if "longitude" in ds.coords:
        _, unique_indices = np.unique(ds.longitude.values, return_index=True)
        if len(unique_indices) != ds.sizes["longitude"]:
            ds = ds.isel(longitude=np.sort(unique_indices)).sortby("longitude")

    if "latitude" in ds.coords:
        ds = ds.sortby("latitude")

    return ds


def _step_hours(step_coord):
    values = step_coord.values
    if np.issubdtype(values.dtype, np.timedelta64):
        return (values / np.timedelta64(1, "h")).astype(int)
    return values.astype(int)


def _daily_instantaneous(data_array, step_hours, days=10):
    daily = []
    for day in range(1, days + 1):
        start = (day - 1) * 24
        end = day * 24
        indices = np.where((step_hours > start) & (step_hours <= end))[0]
        if len(indices) == 0:
            raise ValueError(f"No IFS steps found for day {day} in {data_array.name}")
        daily.append(data_array.isel(step=indices).mean(dim="step"))
    return xr.concat(daily, dim="time_counter")


def _daily_accumulated_total(data_array, step_hours, days=10):
    daily = []
    for day in range(1, days + 1):
        start = (day - 1) * 24
        end = day * 24
        start_indices = np.where(step_hours == start)[0]
        end_indices = np.where(step_hours == end)[0]
        if len(start_indices) == 0 or len(end_indices) == 0:
            raise ValueError(
                f"Missing cumulative endpoints {start}h/{end}h for {data_array.name}"
            )
        total = data_array.isel(step=end_indices[0]) - data_array.isel(step=start_indices[0])
        daily.append(total.clip(min=0))
    return xr.concat(daily, dim="time_counter")


def _make_daily_dataset(raw_file, forecast_date):
    ds = xr.open_dataset(raw_file, engine="cfgrib", backend_kwargs={"indexpath": ""})
    try:
        ds = _normalize_coordinates(ds)
        _validate_forcing_dataset(ds, f"IFS GRIB {raw_file}")

        step_hours = _step_hours(ds.step)
        forecast_start = forecast_date + timedelta(days=1)
        time_counter = np.array(
            [np.datetime64(forecast_start + timedelta(days=i)) for i in range(10)]
        )

        daily_parts = {}
        for var in INSTANTANEOUS:
            daily_parts[var] = _daily_instantaneous(ds[var], step_hours)
        for var in ACCUMULATED:
            daily_parts[var] = _daily_accumulated_total(ds[var], step_hours)

        ds_daily = xr.Dataset(daily_parts)
        ds_daily = ds_daily.assign_coords(time_counter=time_counter)
        return ds_daily
    finally:
        ds.close()


def _upsample_and_write_ifs(ds_daily, output_file):
    lat_target = TARGET_LAT[TARGET_LAT <= 90.0]
    output_path = Path(output_file)
    if output_path.exists():
        output_path.unlink()

    all_vars = INSTANTANEOUS + ACCUMULATED
    for idx, var in enumerate(all_vars):
        print(f"   Upsampling {var} ({idx + 1}/{len(all_vars)})...")
        ds_var = (
            ds_daily[[var]]
            .astype(np.float32)
            .interp(latitude=lat_target, longitude=TARGET_LON, method="linear")
        )
        ds_var.to_netcdf(
            str(output_path),
            mode="w" if idx == 0 else "a",
            engine="netcdf4",
            encoding={var: {"dtype": "float32", "zlib": True, "complevel": 1}},
        )
        ds_var.close()


def fetch_ifs_data(forecast_date, output_dir):
    """
    Download IFS forcing data for WenHai and write a daily 1/12 degree NetCDF.

    Args:
        forecast_date: datetime.date - Init date of the forecast
        output_dir: str - Output directory

    Returns:
        str - Path to upsampled NetCDF file at 1/12 degree
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    date_str = forecast_date.strftime("%Y%m%d")
    upsampled_file = output_path / f"ifs_wenhai_{date_str}.nc"

    if upsampled_file.exists():
        with xr.open_dataset(upsampled_file) as cached_ds:
            try:
                _validate_daily_forcing_dataset(cached_ds, f"Cached IFS file {upsampled_file}")
                print(f"[OK] IFS forcing file already exists: {upsampled_file}")
                return str(upsampled_file)
            except ValueError as exc:
                print(f"[WARNING] {exc}")
                print(f"[WARNING] Removing invalid cached IFS file: {upsampled_file}")
        upsampled_file.unlink(missing_ok=True)

    forecast_start = forecast_date + timedelta(days=1)
    forecast_end = forecast_date + timedelta(days=10)
    print("Fetching IFS forcing data:")
    print("   Variables: t2m, d2m, u10, v10, msl, tp, ssrd, strd (8 vars)")
    print(f"   Forecast window: {forecast_start} to {forecast_end} (10 days)")

    raw_file = _fetch_ifs_grib(forecast_date, output_path)
    ds_daily = _make_daily_dataset(raw_file, forecast_date)
    _validate_daily_forcing_dataset(ds_daily, "IFS daily dataset")

    print("Upsampling IFS from 0.25° to 0.083°...")
    _upsample_and_write_ifs(ds_daily, upsampled_file)
    ds_daily.close()

    raw_file.unlink(missing_ok=True)
    size_mb = upsampled_file.stat().st_size / 1e6
    print(f"[OK] IFS forcing file: {upsampled_file} ({size_mb:.1f} MB)")
    return str(upsampled_file)


if __name__ == "__main__":
    import sys

    test_date = (
        datetime.strptime(sys.argv[1], "%Y-%m-%d").date()
        if len(sys.argv) > 1
        else datetime.now().date() - timedelta(days=2)
    )
    print(f"Testing IFS download for {test_date}")
    out = fetch_ifs_data(test_date, "/tmp/wenhai_test/ifs")
    print(f"Success: {out}")
