"""Rank multiscale candidates with archaeology-specific concentric ring heuristics."""
from __future__ import annotations

import argparse
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import rasterio

from concentric_ring_scorer import score_patch
from config import PATHS, SEARCH


def rank_candidates(raster_path: str, candidate_csv: str, out_csv: str):
    df = pd.read_csv(candidate_csv)
    rows = []
    with rasterio.open(raster_path) as src:
        img = np.transpose(src.read()[:3], (1, 2, 0)).astype(np.float32)
        img -= img.min()
        if img.max() > 0:
            img /= img.max()
        img = (img * 255).astype(np.uint8)

        for _, row in df.iterrows():
            r0, r1 = int(row["row0"]), int(row["row1"])
            c0, c1 = int(row["col0"]), int(row["col1"])
            patch = img[r0:r1, c0:c1]
            if patch.size == 0:
                continue

            gray = cv2.cvtColor(patch, cv2.COLOR_RGB2GRAY)
            ring_scores = score_patch(gray)
            final = (
                SEARCH.score_weight * float(row["score"])
                + SEARCH.circularity_weight * ring_scores["circularity"]
                + SEARCH.concentricity_weight * ring_scores["concentricity"]
                + SEARCH.radial_symmetry_weight * ring_scores["radial_symmetry"]
                + SEARCH.ring_count_weight * ring_scores["ring_density"]
            )

            rows.append({
                **row.to_dict(),
                **ring_scores,
                "final_score": float(final),
            })

    out = pd.DataFrame(rows).sort_values("final_score", ascending=False)
    Path(out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(out_csv, index=False)
    print(f"saved ranked candidates to {out_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--raster_path", default=str(PATHS.test_raster))
    parser.add_argument("--candidate_csv", default=str(PATHS.candidates_csv))
    parser.add_argument("--out_csv", default=str(PATHS.ranked_candidates_csv))
    args = parser.parse_args()
    rank_candidates(args.raster_path, args.candidate_csv, args.out_csv)
