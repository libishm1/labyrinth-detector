"""
make_search_tiles.py

Create overlapping search tiles from a georeferenced raster.
Run from repository root.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import sys
import pandas as pd
import rasterio
from rasterio.windows import Window

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from raster_utils import ensure_rgb, meters_to_pixels, pixel_to_latlon, save_rgb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raster-path", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--tile-size-m", type=float, default=32.0)
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--min-valid-ratio", type=float, default=0.9)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    tile_dir = out_dir / "tiles"
    tile_dir.mkdir(parents=True, exist_ok=True)
    records = []

    with rasterio.open(args.raster_path) as src:
        tile_px = max(64, meters_to_pixels(src, args.tile_size_m))
        stride_px = max(1, int(round(tile_px * (1.0 - args.overlap))))

        row_starts = list(range(0, max(1, src.height - tile_px + 1), stride_px)) or [0]
        col_starts = list(range(0, max(1, src.width - tile_px + 1), stride_px)) or [0]
        if row_starts[-1] != max(0, src.height - tile_px):
            row_starts.append(max(0, src.height - tile_px))
        if col_starts[-1] != max(0, src.width - tile_px):
            col_starts.append(max(0, src.width - tile_px))

        tile_id = 0
        for row0 in row_starts:
            for col0 in col_starts:
                row1 = min(src.height, row0 + tile_px)
                col1 = min(src.width, col0 + tile_px)
                arr = src.read(window=Window(col0, row0, col1 - col0, row1 - row0))
                valid_ratio = float((arr == arr).mean())
                if valid_ratio < args.min_valid_ratio:
                    continue

                rgb = ensure_rgb(arr)
                tile_name = f"tile_{tile_id:06d}"
                tile_path = tile_dir / f"{tile_name}.png"
                save_rgb(tile_path, rgb)

                center_row = row0 + (row1 - row0) // 2
                center_col = col0 + (col1 - col0) // 2
                center_lat, center_lon, center_x, center_y = pixel_to_latlon(src, center_row, center_col)
                ul_lat, ul_lon, _, _ = pixel_to_latlon(src, row0, col0)
                lr_lat, lr_lon, _, _ = pixel_to_latlon(src, row1 - 1, col1 - 1)

                records.append({
                    "tile_id": tile_name,
                    "image_path": str(tile_path),
                    "row0": row0, "row1": row1, "col0": col0, "col1": col1,
                    "center_row": center_row, "center_col": center_col,
                    "center_x": center_x, "center_y": center_y,
                    "center_lat": center_lat, "center_lon": center_lon,
                    "ul_lat": ul_lat, "ul_lon": ul_lon, "lr_lat": lr_lat, "lr_lon": lr_lon,
                    "tile_size_px": row1 - row0,
                    "tile_size_m": args.tile_size_m,
                    "source_raster": args.raster_path,
                    "valid_ratio": valid_ratio,
                })
                tile_id += 1

    pd.DataFrame(records).to_csv(out_dir / "search_tiles.csv", index=False)
    print(f"Saved {len(records)} tiles to {tile_dir}")
    print(f"Saved metadata: {out_dir / 'search_tiles.csv'}")


if __name__ == "__main__":
    main()
