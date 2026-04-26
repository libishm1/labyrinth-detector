"""Generate terrain visualizations for archaeological search using RVT_py.

Produces selected derivatives such as SLRM, openness, and SVF.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio

try:
    from rvt.default import DefaultValues
except ImportError as exc:
    raise SystemExit("rvt-py is required: pip install rvt-py") from exc


def write_like(reference_path: str, out_path: str, arr: np.ndarray):
    arr = arr.astype(np.float32)
    if arr.ndim == 2:
        arr = arr[None, ...]
    else:
        arr = np.transpose(arr, (2, 0, 1))
    with rasterio.open(reference_path) as src:
        profile = src.profile.copy()
    profile.update(dtype="float32", count=arr.shape[0], compress="lzw")
    with rasterio.open(out_path, "w", **profile) as dst:
        dst.write(arr)


def preprocess_dem(dem_path: str, out_dir: str, resolution: float = 0.5):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
        nodata = src.nodata
        if nodata is not None:
            dem = np.where(dem == nodata, np.nan, dem)

    dv = DefaultValues()
    dv.slrm_rad_cell = 20
    dv.svf_r_max = 10
    dv.opns_r_max = 10

    slrm = dv.get_slrm(dem=dem)
    pos_opn = dv.get_opns(dem=dem, resolution=resolution, positive=True)
    neg_opn = dv.get_opns(dem=dem, resolution=resolution, positive=False)
    svf = dv.get_sky_view_factor(dem=dem, resolution=resolution)["svf"]

    write_like(dem_path, out_dir / "slrm.tif", slrm)
    write_like(dem_path, out_dir / "openness_positive.tif", pos_opn)
    write_like(dem_path, out_dir / "openness_negative.tif", neg_opn)
    write_like(dem_path, out_dir / "svf.tif", svf)

    terrain_stack = np.dstack([slrm, pos_opn, neg_opn]).astype(np.float32)
    write_like(dem_path, out_dir / "terrain_stack.tif", terrain_stack)
    print(f"saved derivatives to {out_dir}")


if __name__ == "__main__":
    preprocess_dem("data/dem/example_dem.tif", "data/dem/derived", resolution=0.5)
