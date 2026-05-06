from pathlib import Path

import numpy as np
import xarray as xr
import zarr


DEFAULT_TITLE = "daily mean fields from WenHai 1/12 degree resolution Forecast updated Daily"
DEFAULT_SOURCE = "Laoshan Laboratory"
DEFAULT_INSTITUTION = "Laoshan Laboratory"
DEFAULT_CONTACT = ""
DEFAULT_REFERENCES = "https://github.com/Cuiyingzhe/WenHai"

VAR_METADATA = {
    "zos": {
        "cell_methods": "area: mean",
        "long_name": "Sea surface height",
        "standard_name": "sea_surface_height_above_geoid",
        "units": "m",
        "valid_min": -5.0,
        "valid_max": 5.0,
    },
    "thetao": {
        "cell_methods": "area: mean",
        "long_name": "Temperature",
        "standard_name": "sea_water_potential_temperature",
        "units": "degrees_C",
        "valid_min": -10.0,
        "valid_max": 40.0,
    },
    "so": {
        "cell_methods": "area: mean",
        "long_name": "Salinity",
        "standard_name": "sea_water_salinity",
        "units": "1e-3",
        "valid_min": 0.0,
        "valid_max": 50.0,
    },
    "uo": {
        "cell_methods": "area: mean",
        "long_name": "Eastward velocity",
        "standard_name": "eastward_sea_water_velocity",
        "units": "m s-1",
        "valid_min": -5.0,
        "valid_max": 5.0,
    },
    "vo": {
        "cell_methods": "area: mean",
        "long_name": "Northward velocity",
        "standard_name": "northward_sea_water_velocity",
        "units": "m s-1",
        "valid_min": -5.0,
        "valid_max": 5.0,
    },
}


def _coord_step(values: xr.DataArray) -> float | None:
    if values.size < 2:
        return None
    diffs = np.diff(values.values.astype(np.float64))
    if diffs.size == 0:
        return None
    return float(np.median(diffs))


def _merge_attrs(existing: dict, updates: dict) -> dict:
    merged = dict(existing)
    merged.update({key: value for key, value in updates.items() if value is not None})
    return merged


def _normalize_attr_value(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return value.item()
        return value.tolist()
    return value


def _build_metadata_dataset(
    ds: xr.Dataset,
    title: str,
    source: str,
    institution: str,
    contact: str,
    references: str,
) -> xr.Dataset:
    ds = ds.copy(deep=False)
    ds.attrs = _merge_attrs(
        ds.attrs,
        {
            "Conventions": "CF-1.8",
            "area": "Global",
            "contact": contact,
            "institution": institution,
            "source": source,
            "title": title,
            "references": references,
        },
    )
    ds.attrs.pop("regrid_method", None)

    for var_name, attrs in VAR_METADATA.items():
        if var_name in ds:
            ds[var_name].attrs = _merge_attrs(ds[var_name].attrs, attrs)

    if "latitude" in ds.coords:
        lat = ds["latitude"]
        ds["latitude"].attrs = _merge_attrs(
            lat.attrs,
            {
                "axis": "Y",
                "long_name": "Latitude",
                "standard_name": "latitude",
                "step": _coord_step(lat),
                "units": "degrees_north",
                "valid_min": float(np.nanmin(lat.values)),
                "valid_max": float(np.nanmax(lat.values)),
            },
        )

    if "longitude" in ds.coords:
        lon = ds["longitude"]
        ds["longitude"].attrs = _merge_attrs(
            lon.attrs,
            {
                "axis": "X",
                "long_name": "Longitude",
                "standard_name": "longitude",
                "step": _coord_step(lon),
                "units": "degrees_east",
                "valid_min": float(np.nanmin(lon.values)),
                "valid_max": float(np.nanmax(lon.values)),
            },
        )

    if "time" in ds.coords and ds["time"].size > 0:
        first_time = str(np.asarray(ds["time"].values[0], dtype="datetime64[ns]"))
        last_time = str(np.asarray(ds["time"].values[-1], dtype="datetime64[ns]"))
        ds["time"].attrs = _merge_attrs(
            ds["time"].attrs,
            {
                "valid_min": first_time,
                "valid_max": last_time,
            },
        )

    if "depth" in ds.coords and ds["depth"].size > 0:
        depth = ds["depth"]
        ds["depth"].attrs = _merge_attrs(
            depth.attrs,
            {
                "axis": "Z",
                "long_name": "Elevation",
                "positive": "down",
                "standard_name": "elevation",
                "unit_long": "Meters",
                "units": "m",
                "valid_min": float(np.nanmin(depth.values)),
                "valid_max": float(np.nanmax(depth.values)),
            },
        )

    return ds


def _apply_zarr_metadata(store_path: Path, ds: xr.Dataset) -> None:
    group = zarr.open_group(str(store_path), mode="a")
    for key, value in ds.attrs.items():
        group.attrs[key] = _normalize_attr_value(value)

    for name in ds.coords:
        if name in group:
            for key, value in ds[name].attrs.items():
                group[name].attrs[key] = _normalize_attr_value(value)

    for name in ds.data_vars:
        if name in group:
            for key, value in ds[name].attrs.items():
                group[name].attrs[key] = _normalize_attr_value(value)


def add_metadata_to_zarr(
    zarr_path,
    title: str | None = None,
    source: str | None = None,
    institution: str | None = None,
    contact: str | None = None,
    references: str | None = None,
):
    zarr_path = Path(zarr_path)
    if not zarr_path.is_dir() or not zarr_path.name.endswith(".zarr"):
        raise ValueError(f"Expected a local .zarr directory, got: {zarr_path}")

    print(f"Adding metadata to Zarr: {zarr_path}")
    ds = xr.open_zarr(str(zarr_path))
    try:
        ds_with_metadata = _build_metadata_dataset(
            ds=ds,
            title=DEFAULT_TITLE if title is None else title,
            source=DEFAULT_SOURCE if source is None else source,
            institution=DEFAULT_INSTITUTION if institution is None else institution,
            contact=DEFAULT_CONTACT if contact is None else contact,
            references=DEFAULT_REFERENCES if references is None else references,
        )
        _apply_zarr_metadata(zarr_path, ds_with_metadata)
        zarr.consolidate_metadata(str(zarr_path))
    finally:
        ds.close()

    print(f"[OK] Metadata added: {zarr_path}")
    return str(zarr_path)
