"""
run_salem_geometry_only.py

One-command geometry-only scan for the Salem tile.
Place your raster at: data/test/salem_cartosat3_pan.tif
Run from the repository root.
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


GEOM_DIR = Path("scripts/geometry")


def run(cmd):
    print("\n" + " ".join(cmd))
    subprocess.check_call(cmd)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--raster-path", default="data/test/salem_cartosat3_pan.tif")
    p.add_argument("--known-sites-csv", default="data/sites/known_sites_salem.csv")
    p.add_argument("--out-dir", default="outputs/geometry_salem")
    p.add_argument("--tile-sizes-m", nargs="+", type=int, default=[16, 32, 64])
    p.add_argument("--overlap", type=float, default=0.5)
    p.add_argument("--top-k", type=int, default=200)
    return p.parse_args()


def main():
    args = parse_args()
    out_root = Path(args.out_dir)
    out_root.mkdir(parents=True, exist_ok=True)
    score_csvs = []

    for size in args.tile_sizes_m:
        scale_dir = out_root / f"tiles_{size}m"

        run([sys.executable, str(GEOM_DIR / "make_search_tiles.py"), "--raster-path", args.raster_path, "--out-dir", str(scale_dir), "--tile-size-m", str(size), "--overlap", str(args.overlap)])

        score_csv = scale_dir / "tile_scores.csv"
        run([sys.executable, str(GEOM_DIR / "score_tiles_geometry.py"), "--tiles-csv", str(scale_dir / "search_tiles.csv"), "--out-csv", str(score_csv)])

        run([sys.executable, str(GEOM_DIR / "evaluate_known_sites.py"), "--known-sites-csv", args.known_sites_csv, "--tile-scores-csv", str(score_csv), "--out-dir", str(scale_dir / "evaluation")])

        run([sys.executable, str(GEOM_DIR / "review_queue.py"), "--tile-scores-csv", str(score_csv), "--out-dir", str(scale_dir / "review"), "--top-k", str(args.top_k)])

        score_csvs.append(str(score_csv))

    merged = out_root / "merged_multiscale_scores.csv"
    run([sys.executable, str(GEOM_DIR / "merge_multiscale_scores.py"), "--score-csvs", *score_csvs, "--out-csv", str(merged), "--top-k-per-scale", str(args.top_k)])

    run([sys.executable, str(GEOM_DIR / "review_queue.py"), "--tile-scores-csv", str(merged), "--out-dir", str(out_root / "merged_review"), "--top-k", str(args.top_k)])

    print(f"\nDone. Review: {out_root / 'merged_review'}")


if __name__ == "__main__":
    main()
