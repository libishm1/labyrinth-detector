"""
extract_sites_from_multiple_rasters.py

Extract positive chips around documented sites from one or more georeferenced rasters.
Run from repository root.
"""
from __future__ import annotations

import argparse
import random
from pathlib import Path
import sys
import pandas as pd
import rasterio

sys.path.append(str(Path(__file__).resolve().parents[1] / "utils"))
from raster_utils import latlon_to_pixel, meters_to_pixels, read_window_rgb, save_rgb


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--sites-csv", required=True)
    p.add_argument("--rasters", nargs="+", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--chip-sizes-m", nargs="+", type=float, default=[16, 32, 64])
    p.add_argument("--jitter-count", type=int, default=12)
    p.add_argument("--jitter-frac", type=float, default=0.20)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def point_inside(src, row, col, margin=1):
    return margin <= row < src.height - margin and margin <= col < src.width - margin


def main():
    args = parse_args()
    random.seed(args.seed)
    sites = pd.read_csv(args.sites_csv)
    out_dir = Path(args.out_dir)
    chip_dir = out_dir / "positive_chips"
    chip_dir.mkdir(parents=True, exist_ok=True)
    records = []

    for raster_path in args.rasters:
        with rasterio.open(raster_path) as src:
            for _, site in sites.iterrows():
                try:
                    row, col = latlon_to_pixel(src, float(site["lat"]), float(site["lon"]))
                except Exception:
                    continue
                if not point_inside(src, row, col, margin=10):
                    continue

                for chip_m in args.chip_sizes_m:
                    chip_px = max(32, meters_to_pixels(src, chip_m))
                    half = chip_px // 2
                    jitter_px = max(1, int(chip_px * args.jitter_frac))

                    offsets = [(0, 0)]
                    for _j in range(args.jitter_count):
                        offsets.append((random.randint(-jitter_px, jitter_px), random.randint(-jitter_px, jitter_px)))

                    for j, (dr, dc) in enumerate(offsets):
                        rr, cc = row + dr, col + dc
                        rgb, r0, c0, r1, c1 = read_window_rgb(src, rr - half, cc - half, rr + half, cc + half)
                        if rgb.shape[0] < 16 or rgb.shape[1] < 16:
                            continue
                        name = f"{site['site_group']}_{site['site_id']}_{int(chip_m)}m_j{j:02d}_{Path(raster_path).stem}.png".replace(" ", "_")
                        out_path = chip_dir / name
                        save_rgb(out_path, rgb)
                        records.append({
                            "image_path": str(out_path),
                            "label": 1,
                            "site_id": site["site_id"],
                            "site_group": site["site_group"],
                            "domain": site.get("domain", "india_documented"),
                            "label_strength": site.get("label_strength", 1.0),
                            "chip_size_m": chip_m,
                            "source_raster": raster_path,
                            "row0": r0, "row1": r1, "col0": c0, "col1": c1,
                        })

    out_csv = out_dir / "positive_chips.csv"
    pd.DataFrame(records).to_csv(out_csv, index=False)
    print(f"Saved {len(records)} positive chips: {out_csv}")


if __name__ == "__main__":
    main()
