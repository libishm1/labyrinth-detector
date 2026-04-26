"""Extract positive seed patches around known labyrinth coordinates.

This script reads point seeds from a CSV such as ``data/known_sites.csv``,
projects them into the raster CRS, crops centered windows, and writes them as
PNG patches plus a metadata CSV.

It is designed for the Nazca-style path where you may have seed coordinates
before you have polygon masks.

Expected seed CSV columns
-------------------------
Required:
- site_group
- site_id
- label
- lat
- lon

Optional:
- crs          default: EPSG:4326
- raster_path  optional override per row
- notes

Examples
--------
Use a single raster for all seeds:
    python extract_seed_patches.py \
        --seed-csv data/known_sites.csv \
        --raster data/test/example.tif \
        --out-dir data/patch_dataset \
        --patch-sizes 128 256 384 \
        --jitter-count 4 \
        --jitter-px 16

Use per-row raster paths from the CSV:
    python extract_seed_patches.py \
        --seed-csv data/known_sites.csv \
        --out-dir data/patch_dataset \
        --patch-sizes 256

Notes
-----
- For classification, these crops are positive seed patches.
- For segmentation, you still need polygon masks later.
- Nearby seeds should share the same ``site_group`` so that train/val splitting
  can keep them in the same fold.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
from rasterio.warp import transform as rio_transform

try:
    from config import PATHS
except Exception:
    PATHS = None


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}
DEFAULT_SEED_CSV = "data/known_sites.csv"
DEFAULT_OUT_DIR = "data/patch_dataset"
DEFAULT_OUT_CSV = "seed_patches.csv"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract positive seed patches from georeferenced rasters.")
    parser.add_argument("--seed-csv", type=str, default=getattr(PATHS, "data_dir", Path("data")) / "known_sites.csv" if PATHS else DEFAULT_SEED_CSV,
                        help="CSV of known site coordinates.")
    parser.add_argument("--raster", type=str, default=None,
                        help="Optional raster used for every seed. If omitted, use per-row raster_path.")
    parser.add_argument("--out-dir", type=str, default=getattr(PATHS, "patch_dataset_dir", Path(DEFAULT_OUT_DIR)) if PATHS else DEFAULT_OUT_DIR,
                        help="Output patch dataset directory.")
    parser.add_argument("--out-csv", type=str, default=DEFAULT_OUT_CSV,
                        help="Metadata CSV filename written inside out-dir.")
    parser.add_argument("--patch-sizes", nargs="+", type=int, default=[128, 256, 384],
                        help="Patch sizes in pixels.")
    parser.add_argument("--jitter-count", type=int, default=0,
                        help="Number of extra offset crops per patch size for each seed.")
    parser.add_argument("--jitter-px", type=int, default=12,
                        help="Maximum absolute offset in pixels for jittered crops.")
    parser.add_argument("--min-valid-ratio", type=float, default=0.80,
                        help="Minimum ratio of valid in-raster pixels before skipping the crop.")
    parser.add_argument("--save-geotiff", action="store_true",
                        help="Also save georeferenced GeoTIFF crops next to PNGs.")
    parser.add_argument("--prefix", type=str, default="seed",
                        help="Prefix for patch filenames.")
    return parser.parse_args()


def ensure_required_columns(df: pd.DataFrame) -> None:
    required = {"site_group", "site_id", "label", "lat", "lon"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns in seed CSV: {sorted(missing)}")


def normalize_to_uint8(arr: np.ndarray) -> np.ndarray:
    arr = arr.astype(np.float32)
    out_channels: List[np.ndarray] = []
    for ch in range(arr.shape[2]):
        band = arr[:, :, ch]
        finite = np.isfinite(band)
        if not finite.any():
            out_channels.append(np.zeros_like(band, dtype=np.uint8))
            continue
        mn = np.nanmin(band)
        mx = np.nanmax(band)
        if mx - mn < 1e-8:
            out_channels.append(np.zeros_like(band, dtype=np.uint8))
        else:
            norm = (band - mn) / (mx - mn)
            norm[~finite] = 0
            out_channels.append((norm * 255).clip(0, 255).astype(np.uint8))
    return np.stack(out_channels, axis=2)


def read_rgb_window(src: rasterio.io.DatasetReader, window: rasterio.windows.Window) -> np.ndarray:
    band_count = min(src.count, 3)
    arr = src.read(indexes=list(range(1, band_count + 1)), window=window, boundless=True, fill_value=0)
    arr = np.transpose(arr, (1, 2, 0))
    if arr.shape[2] == 1:
        arr = np.repeat(arr, 3, axis=2)
    elif arr.shape[2] == 2:
        arr = np.concatenate([arr, arr[:, :, :1]], axis=2)
    return normalize_to_uint8(arr)


def pixel_window(center_row: int, center_col: int, patch_px: int) -> rasterio.windows.Window:
    half = patch_px // 2
    row_off = center_row - half
    col_off = center_col - half
    return rasterio.windows.Window(col_off=col_off, row_off=row_off, width=patch_px, height=patch_px)


def valid_ratio(src: rasterio.io.DatasetReader, center_row: int, center_col: int, patch_px: int) -> float:
    half = patch_px // 2
    r0 = center_row - half
    c0 = center_col - half
    r1 = r0 + patch_px
    c1 = c0 + patch_px
    rr0 = max(0, r0)
    cc0 = max(0, c0)
    rr1 = min(src.height, r1)
    cc1 = min(src.width, c1)
    inside = max(0, rr1 - rr0) * max(0, cc1 - cc0)
    total = patch_px * patch_px
    return inside / max(total, 1)


def window_transform(transform: Affine, window: rasterio.windows.Window) -> Affine:
    return rasterio.windows.transform(window, transform)


def sanitize_name(value: str) -> str:
    return "".join(ch if ch.isalnum() or ch in {"-", "_"} else "_" for ch in str(value))


def project_point(lon: float, lat: float, src_crs: str, dst_crs) -> Tuple[float, float]:
    xs, ys = rio_transform(src_crs, dst_crs, [lon], [lat])
    return xs[0], ys[0]


def make_offsets(jitter_count: int, jitter_px: int) -> List[Tuple[int, int]]:
    offsets = [(0, 0)]
    if jitter_count <= 0:
        return offsets
    if jitter_px <= 0:
        return offsets
    angles = np.linspace(0, 2 * math.pi, num=jitter_count, endpoint=False)
    radius = max(1, jitter_px)
    for a in angles:
        dr = int(round(math.sin(a) * radius))
        dc = int(round(math.cos(a) * radius))
        offsets.append((dr, dc))
    return offsets


def save_png(img_rgb: np.ndarray, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR))


def save_geotiff_crop(src: rasterio.io.DatasetReader, window: rasterio.windows.Window, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = src.read(window=window, boundless=True, fill_value=0)
    meta = src.meta.copy()
    meta.update({
        "height": int(window.height),
        "width": int(window.width),
        "transform": window_transform(src.transform, window),
    })
    with rasterio.open(out_path, "w", **meta) as dst:
        dst.write(arr)


def extract_seed_patches(
    seed_csv: str | Path,
    raster_path: str | Path | None,
    out_dir: str | Path,
    out_csv_name: str,
    patch_sizes: Sequence[int],
    jitter_count: int,
    jitter_px: int,
    min_valid_ratio: float,
    save_geotiff: bool,
    prefix: str,
) -> Path:
    seed_csv = Path(seed_csv)
    out_dir = Path(out_dir)
    df = pd.read_csv(seed_csv)
    ensure_required_columns(df)
    if "crs" not in df.columns:
        df["crs"] = "EPSG:4326"

    rows = []
    grouped = df.groupby(df["raster_path"] if (raster_path is None and "raster_path" in df.columns) else pd.Series([str(raster_path)] * len(df), index=df.index))

    for raster_key, sub_df in grouped:
        if raster_key in {None, "None", "nan"}:
            raise ValueError("No raster supplied. Pass --raster or include raster_path in the CSV.")
        raster_key = str(raster_key)
        with rasterio.open(raster_key) as src:
            for _, row in sub_df.iterrows():
                site_group = sanitize_name(row["site_group"])
                site_id = sanitize_name(row["site_id"])
                label = int(row["label"])
                lat = float(row["lat"])
                lon = float(row["lon"])
                point_crs = row.get("crs", "EPSG:4326")
                notes = row.get("notes", "")

                x_map, y_map = project_point(lon=lon, lat=lat, src_crs=point_crs, dst_crs=src.crs)
                center_row, center_col = src.index(x_map, y_map)

                for patch_px in patch_sizes:
                    for aug_idx, (dr, dc) in enumerate(make_offsets(jitter_count, jitter_px)):
                        rr = center_row + dr
                        cc = center_col + dc
                        vr = valid_ratio(src, rr, cc, patch_px)
                        if vr < min_valid_ratio:
                            continue

                        window = pixel_window(rr, cc, patch_px)
                        rgb = read_rgb_window(src, window)

                        rel_dir = Path("seed_patches") / site_group / f"{site_id}_px{patch_px}"
                        stem = f"{prefix}_{site_group}_{site_id}_px{patch_px}_aug{aug_idx:02d}"
                        png_path = out_dir / rel_dir / f"{stem}.png"
                        save_png(rgb, png_path)

                        tif_rel = None
                        if save_geotiff:
                            tif_path = out_dir / rel_dir / f"{stem}.tif"
                            save_geotiff_crop(src, window, tif_path)
                            tif_rel = str(tif_path)

                        # Derive crop bounds in raster CRS and original point location.
                        crop_transform = window_transform(src.transform, window)
                        x0, y0 = crop_transform * (0, 0)
                        x1, y1 = crop_transform * (patch_px, patch_px)

                        rows.append({
                            "image_path": str(png_path),
                            "geotiff_path": tif_rel,
                            "label": label,
                            "site_group": site_group,
                            "site_id": site_id,
                            "source_seed_csv": str(seed_csv),
                            "source_raster": raster_key,
                            "seed_lat": lat,
                            "seed_lon": lon,
                            "seed_crs": point_crs,
                            "raster_crs": str(src.crs),
                            "seed_x_map": x_map,
                            "seed_y_map": y_map,
                            "center_row": rr,
                            "center_col": cc,
                            "patch_px": patch_px,
                            "augment_index": aug_idx,
                            "offset_row": dr,
                            "offset_col": dc,
                            "valid_ratio": vr,
                            "crop_x_min": min(x0, x1),
                            "crop_y_min": min(y0, y1),
                            "crop_x_max": max(x0, x1),
                            "crop_y_max": max(y0, y1),
                            "notes": notes,
                        })

    if not rows:
        raise RuntimeError("No seed patches were extracted. Check CRS, raster coverage, or patch size.")

    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / out_csv_name
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    return out_csv


if __name__ == "__main__":
    args = parse_args()
    out_csv = extract_seed_patches(
        seed_csv=args.seed_csv,
        raster_path=args.raster,
        out_dir=args.out_dir,
        out_csv_name=args.out_csv,
        patch_sizes=args.patch_sizes,
        jitter_count=args.jitter_count,
        jitter_px=args.jitter_px,
        min_valid_ratio=args.min_valid_ratio,
        save_geotiff=args.save_geotiff,
        prefix=args.prefix,
    )
    print(f"Saved seed patch metadata to {out_csv}")
