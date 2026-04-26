"""
visualize_top_candidates.py

Visualize known sites or ranked candidates from a georeferenced raster.
Run from repository root.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys
import cv2
import numpy as np
import pandas as pd
import rasterio

sys.path.append(str(Path(__file__).resolve().parent))
from raster_utils import read_window_rgb, latlon_to_pixel, save_rgb


def add_header(img, title, subtitle=""):
    h, w = img.shape[:2]
    hh = 42 if subtitle else 26
    canvas = np.zeros((h + hh, w, 3), dtype=np.uint8)
    canvas[:hh] = (20, 20, 20)
    canvas[hh:] = img
    cv2.putText(canvas, title[:90], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle[:120], (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 190, 190), 1, cv2.LINE_AA)
    return canvas


def contact_sheet(images, out_path, cols=4):
    if not images:
        return
    mh, mw = max(i.shape[0] for i in images), max(i.shape[1] for i in images)
    canvas = np.full((math.ceil(len(images) / cols) * mh, cols * mw, 3), 245, dtype=np.uint8)
    for i, img in enumerate(images):
        y, x = (i // cols) * mh, (i % cols) * mw
        h, w = img.shape[:2]
        canvas[y:y + h, x:x + w] = img
    save_rgb(out_path, canvas)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raster-path", required=True)
    p.add_argument("--csv-path", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--mode", choices=["auto", "latlon", "bbox_pixels", "center_pixels"], default="auto")
    p.add_argument("--top-k", type=int, default=20)
    p.add_argument("--chip-size-px", type=int, default=512)
    p.add_argument("--sort-by", default="candidate_score")
    return p.parse_args()


def infer_mode(df):
    if {"lat", "lon"}.issubset(df.columns):
        return "latlon"
    if {"row0", "row1", "col0", "col1"}.issubset(df.columns):
        return "bbox_pixels"
    if {"center_row", "center_col"}.issubset(df.columns):
        return "center_pixels"
    raise ValueError("Cannot infer CSV mode.")


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    chip_dir = out_dir / "chips"
    chip_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.csv_path)
    mode = infer_mode(df) if args.mode == "auto" else args.mode
    if args.sort_by in df.columns:
        df = df.sort_values(args.sort_by, ascending=False)
    df = df.head(args.top_k)

    records, imgs = [], []
    with rasterio.open(args.raster_path) as src:
        for _, row in df.iterrows():
            title = str(row.get("site_id", row.get("tile_id", f"item_{len(records) + 1}")))
            subtitle = f"score={row[args.sort_by]:.3f}" if args.sort_by in row and pd.notna(row[args.sort_by]) else ""
            if mode == "latlon":
                r, c = latlon_to_pixel(src, float(row["lat"]), float(row["lon"]))
                half = args.chip_size_px // 2
                img, *_ = read_window_rgb(src, r - half, c - half, r + half, c + half)
                cv2.drawMarker(img, (img.shape[1] // 2, img.shape[0] // 2), (255, 0, 0), markerType=cv2.MARKER_CROSS, markerSize=24, thickness=2)
            elif mode == "center_pixels":
                r, c = int(row["center_row"]), int(row["center_col"])
                half = args.chip_size_px // 2
                img, *_ = read_window_rgb(src, r - half, c - half, r + half, c + half)
            else:
                img, *_ = read_window_rgb(src, int(row["row0"]), int(row["col0"]), int(row["row1"]), int(row["col1"]))

            labeled = add_header(img, title, subtitle)
            out_path = chip_dir / f"{len(records) + 1:03d}_{title}.png".replace("/", "_").replace(" ", "_")
            save_rgb(out_path, labeled)
            imgs.append(labeled)
            records.append({"title": title, "chip_path": str(out_path)})

    pd.DataFrame(records).to_csv(out_dir / "visualized_candidates.csv", index=False)
    contact_sheet(imgs, out_dir / "contact_sheet.png")
    print(f"Saved: {out_dir}")


if __name__ == "__main__":
    main()
