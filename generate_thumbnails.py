import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
import rioxarray  # noqa: F401 - registers the .rio accessor on xarray objects
from PIL import Image

from s3_upload import upload_bytes_to_s3


THUMBNAIL_SETTINGS = {
    "zos": {"lead": 9, "cmap": "seismic"},
    "thetao": {"lead": 9, "depth": 0, "cmap": "viridis"},
    "so": {"lead": 9, "depth": 0, "cmap": "jet"},
    "uo": {"lead": 2, "depth": 0, "cmap": "coolwarm"},
    "vo": {"lead": 2, "depth": 0, "cmap": "coolwarm"},
}


def _isel_existing(data_array, **indexers):
    valid_indexers = {}
    for dim, index in indexers.items():
        if dim in data_array.dims:
            dim_size = data_array.sizes[dim]
            valid_indexers[dim] = min(index, dim_size - 1)
    return data_array.isel(**valid_indexers)


def _render_png(data_array, cmap_name):
    data_array = (
        data_array.rio.set_spatial_dims(x_dim="longitude", y_dim="latitude")
        .rio.write_crs("EPSG:4326")
        .rio.reproject("EPSG:4326")
    )
    values = data_array.values
    vmin = np.nanmin(values)
    vmax = np.nanmax(values)

    if vmax == vmin:
        normalized = np.zeros_like(values, dtype=np.uint8)
    else:
        normalized = ((values - vmin) / (vmax - vmin) * 255)
        normalized = np.nan_to_num(normalized, nan=0).astype(np.uint8)

    cmap = plt.get_cmap(cmap_name)
    colored = cmap(normalized)
    colored[..., 3] = np.where(np.isnan(values), 0, 255).astype(np.uint8) / 255.0
    rgba = (colored * 255).astype(np.uint8)

    buffer = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buffer, format="PNG")
    buffer.seek(0)
    return buffer.getvalue()


def generate_thumbnails(zarr_path, bucket_name, s3_prefix):
    # Generate PNG thumbnails for all ocean variables and upload them to S3.
    print(f"Generating thumbnails from: {Path(zarr_path).name}")

    ds = xr.open_zarr(zarr_path)
    thumbnail_urls = {}

    try:
        for var_name, config in THUMBNAIL_SETTINGS.items():
            if var_name not in ds:
                print(f"  [WARNING] {var_name} not found, skipping")
                continue

            try:
                data = _isel_existing(
                    ds[var_name],
                    time=config["lead"],
                    depth=config.get("depth", 0),
                )
                png_bytes = _render_png(data, config["cmap"])

                object_prefix = "/".join(
                    part for part in s3_prefix.strip("/").split("/") if part
                )
                object_key = (
                    f"{object_prefix}/{var_name}.png"
                    if object_prefix
                    else f"{var_name}.png"
                )
                s3_url = upload_bytes_to_s3(
                    bucket_name=bucket_name,
                    data_bytes=png_bytes,
                    object_key=object_key,
                    content_type="image/png",
                )
                thumbnail_urls[var_name] = s3_url
                print(f"  [OK] {var_name}.png -> {s3_url}")

            except Exception as e:
                print(f"  [WARNING] Failed for {var_name}: {e}")
    finally:
        ds.close()

    print(f"[OK] {len(thumbnail_urls)} thumbnails generated")
    return thumbnail_urls
