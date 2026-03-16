# WenHai 10-day autoregressive ocean forecast engine using ONNX and aerobulk bulk fluxes.
import multiprocessing
import os
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
import onnxruntime as ort
from metpy.calc import specific_humidity_from_dewpoint
from metpy.units import units
from aerobulk.flux import noskin_np


REQUIRED_ERA5_VARS = ["t2m", "d2m", "u10", "v10", "msl", "tp", "ssrd", "strd"]


def _noskin_worker(args):
    t0, t2m, h2m, u, v, msl = args
    return noskin_np(t0, t2m, h2m, u, v, msl, "ncar", 2, 10, 4, False)


def _compute_bulk_flux(ocean_state, era5_ds, day_idx, min_GLORYS, max_GLORYS, min_flux, max_flux, mask):
    # Compute aerobulk bulk air-sea fluxes for one forecast day.
    sst = ((max_GLORYS[0, 46] - min_GLORYS[0, 46]) * ocean_state[0, 46].astype(np.float32)
           + min_GLORYS[0, 46]) * mask[0, 46] + 273.15
    u0 = ((max_GLORYS[0, 0] - min_GLORYS[0, 0]) * ocean_state[0, 0].astype(np.float32)
          + min_GLORYS[0, 0]) * mask[0, 0]
    v0 = ((max_GLORYS[0, 23] - min_GLORYS[0, 23]) * ocean_state[0, 23].astype(np.float32)
          + min_GLORYS[0, 23]) * mask[0, 23]

    era5_day = era5_ds.isel(time_counter=day_idx)
    t2m  = era5_day["t2m"].values.astype(np.float32)
    d2m  = era5_day["d2m"].values.astype(np.float32)
    u10  = era5_day["u10"].values.astype(np.float32)
    v10  = era5_day["v10"].values.astype(np.float32)
    msl  = era5_day["msl"].values.astype(np.float32)
    tp   = era5_day["tp"].values.astype(np.float32) / 86400.0
    ssrd = era5_day["ssrd"].values.astype(np.float32) / 86400.0
    strd = era5_day["strd"].values.astype(np.float32) / 86400.0

    h2m = specific_humidity_from_dewpoint(msl * units.Pa, d2m * units.K).to("kg/kg").magnitude
    h2m = np.nan_to_num(h2m)

    W = sst.shape[1]
    pool = multiprocessing.Pool(processes=min(12, os.cpu_count() or 4))
    results = pool.map(
        _noskin_worker,
        [[sst[:, i], t2m[:, i], h2m[:, i], u10[:, i] - u0[:, i], v10[:, i] - v0[:, i], msl[:, i]]
         for i in range(W)],
    )
    pool.close()
    pool.join()

    qe, qh, taux, tauy, evap = zip(*results)
    qe   = np.array(qe).T.reshape(sst.shape)
    qh   = np.array(qh).T.reshape(sst.shape)
    taux = np.array(taux).T.reshape(sst.shape)
    tauy = np.array(tauy).T.reshape(sst.shape)
    evap = np.array(evap).T.reshape(sst.shape) / 1000.0

    sigma = 5.67e-8
    ql = strd - sigma * (sst ** 4)

    bulk_flux = np.stack([ql, ssrd, qh, qe, taux, tauy, evap, tp], axis=0)
    bulk_flux = np.nan_to_num(bulk_flux)
    bulk_flux = (bulk_flux - min_flux) / (max_flux - min_flux)
    bulk_flux *= mask[:, 0]
    return bulk_flux.astype(np.float16).clip(0, 1)[None]


def _make_dataset(output, max_GLORYS, min_GLORYS, mask, longitude, latitude, depth, fcst_date):
    # Denormalize model output and build an xarray Dataset for one forecast day.
    out = output * (max_GLORYS - min_GLORYS) + min_GLORYS
    out[mask == 0] = np.nan
    out = out[0].astype(np.float32)

    u, v, t, s, ssh = out[:23], out[23:46], out[46:69], out[69:92], out[92]
    time_coord = [pd.to_datetime(fcst_date.strftime("%Y%m%d"))]
    c3d = dict(depth=(["depth"], depth), latitude=(["latitude"], latitude), longitude=(["longitude"], longitude))
    c2d = dict(latitude=(["latitude"], latitude), longitude=(["longitude"], longitude))

    return xr.Dataset({
        "thetao": xr.DataArray(t, dims=["depth","latitude","longitude"], coords=c3d,
                               attrs={"long_name":"Potential temperature","units":"degrees_C"}).expand_dims(time=time_coord),
        "so":     xr.DataArray(s, dims=["depth","latitude","longitude"], coords=c3d,
                               attrs={"long_name":"Salinity","units":"1e-3"}).expand_dims(time=time_coord),
        "uo":     xr.DataArray(u, dims=["depth","latitude","longitude"], coords=c3d,
                               attrs={"long_name":"Eastward velocity","units":"m s-1"}).expand_dims(time=time_coord),
        "vo":     xr.DataArray(v, dims=["depth","latitude","longitude"], coords=c3d,
                               attrs={"long_name":"Northward velocity","units":"m s-1"}).expand_dims(time=time_coord),
        "zos":    xr.DataArray(ssh, dims=["latitude","longitude"], coords=c2d,
                               attrs={"long_name":"Sea surface height","units":"m"}).expand_dims(time=time_coord),
    }, attrs={
        "Conventions": "CF-1.8",
        "title": "WenHai Ocean Forecast - 1/12 degree resolution",
        "institution": "Mercator Ocean International",
        "source": "WenHai Deep Learning Ocean Forecast Model",
        "history": f"Created on {datetime.now().isoformat()}",
        "forecast_reference_time": fcst_date.strftime("%Y-%m-%d"),
    })


def run_inference(nowcast_file, era5_file, model_dir, output_path):
    # Load model, run 10-day autoregressive forecast and return concatenated xarray Dataset.
    model_dir = Path(model_dir)
    #Path(output_path).mkdir(parents=True, exist_ok=True)

    min_GLORYS = np.load(model_dir / "min_GLORYS.npy").reshape(1, -1, 1, 1)
    max_GLORYS = np.load(model_dir / "max_GLORYS.npy").reshape(1, -1, 1, 1)
    min_flux   = np.load(model_dir / "min_flux.npy").reshape(-1, 1, 1)
    max_flux   = np.load(model_dir / "max_flux.npy").reshape(-1, 1, 1)
    mask = xr.open_dataset(model_dir / "mask_GLORYS.nc").mask.values[None]

    providers = ["CUDAExecutionProvider"] if ort.get_device() == "GPU" else ["CPUExecutionProvider"]
    session = ort.InferenceSession(str(model_dir / "WenHai.onnx"), providers=providers)
    name_ocean = session.get_inputs()[0].name
    name_flux  = session.get_inputs()[1].name
    print(f"[OK] WenHai ONNX loaded on {'GPU' if ort.get_device() == 'GPU' else 'CPU'}")

    ds_init   = xr.open_dataset(nowcast_file)
    longitude = ds_init.longitude.values
    latitude  = ds_init.latitude.values
    depth     = ds_init.depth.values
    init_date = datetime.fromtimestamp(
        (ds_init.time[0].values - np.datetime64(0, "s")) / np.timedelta64(1, "s")
    )
    init = np.concatenate([ds_init.uo.values, ds_init.vo.values, ds_init.thetao.values,
                           ds_init.so.values, ds_init.zos.values[None]], axis=1)
    ds_init.close()

    init = np.nan_to_num((init - min_GLORYS) / (max_GLORYS - min_GLORYS))

    era5_ds = xr.open_dataset(era5_file)
    missing_era5_vars = [var for var in REQUIRED_ERA5_VARS if var not in era5_ds]
    if missing_era5_vars:
        era5_ds.close()
        raise ValueError(
            f"ERA5 forcing file {era5_file} is missing required variables {missing_era5_vars}. "
            f"Available variables: {list(era5_ds.data_vars)}"
        )
    nday = len(era5_ds.time_counter)
    print(f"Starting {nday}-day WenHai forecast from {init_date.strftime('%Y-%m-%d')}...")

    all_datasets = []
    current_state = init

    for i in range(nday):
        fcst_date = init_date + timedelta(days=1 + i)
        print(f"   Day {i+1}/{nday}: {fcst_date.strftime('%Y-%m-%d')} ...", end=" ", flush=True)

        bulk_flux = _compute_bulk_flux(current_state, era5_ds, i, min_GLORYS, max_GLORYS, min_flux, max_flux, mask)

        inputs = {name_ocean: current_state.astype(np.float16).clip(0, 1), name_flux: bulk_flux}
        output = session.run(None, inputs)[0]
        output = (output + current_state).clip(0, 1) * mask
        current_state = output.copy()

        ds_day = _make_dataset(output, max_GLORYS, min_GLORYS, mask, longitude, latitude, depth, fcst_date)
        all_datasets.append(ds_day)
        print("✓")

    era5_ds.close()
    print(f"[OK] Forecast complete: {nday} days")

    final_ds = xr.concat(all_datasets, dim="time")
    return final_ds
