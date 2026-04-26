from __future__ import annotations

from pathlib import Path
from typing import Tuple
import numpy as np
import cv2
import rasterio
from rasterio.windows import Window
from rasterio.warp import transform as rio_transform


def ensure_rgb(arr: np.ndarray) -> np.ndarray:
    """Convert raster array to 8-bit RGB for patch export and scoring."""
    if arr.ndim == 2:
        arr = arr[:, :, None]
    if arr.ndim == 3 and arr.shape[0] <= 16 and arr.shape[0] < arr.shape[-1]:
        arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] >= 3:
        arr = arr[:, :, :3]

    arr = arr.astype(np.float32)
    vals = arr[np.isfinite(arr)]
    if vals.size:
        lo, hi = np.percentile(vals, [2, 98])
        arr = np.clip((arr - lo) / max(hi - lo, 1e-6), 0, 1)
    else:
        arr = np.zeros_like(arr)
    return (arr * 255).astype(np.uint8)


def meters_to_pixels(src: rasterio.io.DatasetReader, meters: float) -> int:
    xres = abs(src.transform.a)
    yres = abs(src.transform.e)
    gsd = (xres + yres) / 2.0
    return max(1, int(round(meters / max(gsd, 1e-9))))


def latlon_to_pixel(src: rasterio.io.DatasetReader, lat: float, lon: float) -> Tuple[int, int]:
    if src.crs is None:
        raise ValueError("Raster has no CRS.")
    xs, ys = rio_transform("EPSG:4326", src.crs, [lon], [lat])
    return src.index(xs[0], ys[0])


def pixel_to_latlon(src: rasterio.io.DatasetReader, row: int, col: int):
    x, y = src.xy(row, col)
    if src.crs is None:
        return None, None, x, y
    lon, lat = rio_transform(src.crs, "EPSG:4326", [x], [y])
    return lat[0], lon[0], x, y


def read_window_rgb(src: rasterio.io.DatasetReader, row0: int, col0: int, row1: int, col1: int):
    row0, col0 = max(0, row0), max(0, col0)
    row1, col1 = min(src.height, row1), min(src.width, col1)
    arr = src.read(window=Window(col0, row0, max(1, col1 - col0), max(1, row1 - row0)))
    return ensure_rgb(arr), row0, col0, row1, col1


def save_rgb(path: str | Path, rgb: np.ndarray) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(path), cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR))
