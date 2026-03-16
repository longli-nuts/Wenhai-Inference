# Generate PNG thumbnail previews from WenHai forecast dataset and upload to S3.
import io
import numpy as np
from PIL import Image
from matplotlib import pyplot as plt
from s3_upload import save_bytes_to_s3


def _make_png(data_2d, cmap_name):
    # Render a 2D array as a PNG bytes using the given colormap.
    arr = np.array(data_2d, dtype=np.float32)
    vmin, vmax = np.nanmin(arr), np.nanmax(arr)
    norm = ((arr - vmin) / (vmax - vmin) * 255).astype(np.uint8) if vmax > vmin else np.zeros_like(arr, dtype=np.uint8)
    cmap = plt.get_cmap(cmap_name)
    rgba = (cmap(norm) * 255).astype(np.uint8)
    rgba[..., 3] = np.where(np.isnan(arr), 0, 255).astype(np.uint8)
    buf = io.BytesIO()
    Image.fromarray(rgba, mode="RGBA").save(buf, format="PNG")
    return buf.getvalue()


def generate_thumbnails(bucket_name, forecast_netcdf_file_url, thumbnail_urls, forecast_dataset):
    # Generate and upload thumbnail PNGs for zos, thetao, so, uo, vo.
    ds = forecast_dataset
    configs = [
        ("zos",    ds["zos"].isel(time=-1).values,            "seismic"),
        ("thetao", ds["thetao"].isel(time=-1, depth=0).values, "viridis"),
        ("so",     ds["so"].isel(time=-1, depth=0).values,     "jet"),
        ("uo",     ds["uo"].isel(time=2, depth=0).values,      "coolwarm"),
        ("vo",     ds["vo"].isel(time=2, depth=0).values,      "coolwarm"),
    ]
    for var, data, cmap in configs:
        png = _make_png(data, cmap)
        key = thumbnail_urls[var].partition(bucket_name + "/")[2]
        save_bytes_to_s3(bucket_name, png, key)
        print(f"[OK] Thumbnail {var}")