"""Export raw multiscale candidate dictionaries to a clean georeferenced CSV.

Reads the .npy object array written by infer_multiscale_heatmap.py and expands
cluster bounds from grid coordinates into pixel and map coordinates.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import xy

from config import PATHS


def load_candidates(candidates_path: str):
    arr = np.load(candidates_path, allow_pickle=True)
    if isinstance(arr, np.ndarray):
        return [dict(x) for x in arr.tolist()]
    return [dict(x) for x in arr]


def export_candidates_csv(raster_path: str, candidates_path: str, out_csv: str):
    candidates = load_candidates(candidates_path)
    if not candidates:
        raise ValueError(f"No candidates found in {candidates_path}")

    rows = []
    with rasterio.open(raster_path) as src:
        transform = src.transform
        crs = str(src.crs) if src.crs else ""
        for cand in candidates:
            patch_px = int(cand["patch_px"])
            stride_px = int(cand["stride_px"])

            row0 = int(cand["grid_y0"] * stride_px)
            row1 = int(cand["grid_y1"] * stride_px + patch_px)
            col0 = int(cand["grid_x0"] * stride_px)
            col1 = int(cand["grid_x1"] * stride_px + patch_px)

            x0, y0 = xy(transform, row0, col0, offset="ul")
            x1, y1 = xy(transform, row1, col1, offset="ul")
            xc, yc = xy(transform, (row0 + row1) // 2, (col0 + col1) // 2, offset="center")

            rows.append({
                "patch_px": patch_px,
                "stride_px": stride_px,
                "score": float(cand.get("score", 0.0)),
                "row0": row0,
                "row1": row1,
                "col0": col0,
                "col1": col1,
                "height_px": row1 - row0,
                "width_px": col1 - col0,
                "center_row": (row0 + row1) // 2,
                "center_col": (col0 + col1) // 2,
                "x0": float(x0),
                "y0": float(y0),
                "x1": float(x1),
                "y1": float(y1),
                "center_x": float(xc),
                "center_y": float(yc),
                "crs": crs,
                "raw_candidate_json": json.dumps(cand),
            })

    df = pd.DataFrame(rows).sort_values(["score", "patch_px"], ascending=[False, True])
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"saved {len(df)} candidates to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raster_path", default=str(PATHS.test_raster))
    parser.add_argument("--candidates_path", default=str(PATHS.raw_candidates_npy))
    parser.add_argument("--out_csv", default=str(PATHS.candidates_csv))
    args = parser.parse_args()
    export_candidates_csv(args.raster_path, args.candidates_path, args.out_csv)
