"""
evaluate_known_sites.py

Check whether known sites fall inside generated tiles and rank highly.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import numpy as np
import pandas as pd


def containing_tiles(tile_df, lat, lon):
    lat_max = np.maximum(tile_df["ul_lat"].astype(float), tile_df["lr_lat"].astype(float))
    lat_min = np.minimum(tile_df["ul_lat"].astype(float), tile_df["lr_lat"].astype(float))
    lon_max = np.maximum(tile_df["ul_lon"].astype(float), tile_df["lr_lon"].astype(float))
    lon_min = np.minimum(tile_df["ul_lon"].astype(float), tile_df["lr_lon"].astype(float))
    return tile_df[(lat >= lat_min) & (lat <= lat_max) & (lon >= lon_min) & (lon <= lon_max)].copy()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--known-sites-csv", required=True)
    p.add_argument("--tile-scores-csv", required=True)
    p.add_argument("--out-dir", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    known = pd.read_csv(args.known_sites_csv)
    tiles = pd.read_csv(args.tile_scores_csv).sort_values("candidate_score", ascending=False).reset_index(drop=True)

    records = []
    for _, site in known.iterrows():
        matches = containing_tiles(tiles, float(site["lat"]), float(site["lon"]))
        if len(matches) == 0:
            records.append({"site_id": site.get("site_id", ""), "found_containing_tile": 0})
            continue
        best = matches.sort_values("candidate_score", ascending=False).iloc[0]
        rank = int((tiles["candidate_score"] > best["candidate_score"]).sum()) + 1
        percentile = 100 * rank / len(tiles)
        records.append({
            "site_id": site.get("site_id", ""),
            "site_group": site.get("site_group", ""),
            "lat": site["lat"], "lon": site["lon"],
            "found_containing_tile": 1,
            "best_tile_id": best["tile_id"],
            "best_tile_score": best["candidate_score"],
            "best_tile_rank": rank,
            "best_tile_percentile": percentile,
            "in_top_1_percent": int(percentile <= 1),
            "in_top_5_percent": int(percentile <= 5),
            "in_top_10_percent": int(percentile <= 10),
        })

    out = pd.DataFrame(records)
    out.to_csv(out_dir / "known_site_evaluation.csv", index=False)
    summary = (
        f"Known sites evaluated: {len(out)}\n"
        f"Found inside generated tiles: {int(out.get('found_containing_tile', pd.Series(dtype=int)).sum()) if len(out) else 0}\n"
        f"Top 1 percent hits: {int(out.get('in_top_1_percent', pd.Series(dtype=int)).sum()) if len(out) else 0}\n"
        f"Top 5 percent hits: {int(out.get('in_top_5_percent', pd.Series(dtype=int)).sum()) if len(out) else 0}\n"
        f"Top 10 percent hits: {int(out.get('in_top_10_percent', pd.Series(dtype=int)).sum()) if len(out) else 0}\n"
    )
    (out_dir / "known_site_evaluation_summary.txt").write_text(summary)
    print(summary)


if __name__ == "__main__":
    main()
