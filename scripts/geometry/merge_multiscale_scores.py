from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--score-csvs", nargs="+", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--top-k-per-scale", type=int, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    frames = []
    for csv in args.score_csvs:
        df = pd.read_csv(csv).sort_values("candidate_score", ascending=False).reset_index(drop=True)
        df["rank_within_scale"] = np.arange(1, len(df) + 1)
        df["source_score_csv"] = csv
        if args.top_k_per_scale:
            df = df.head(args.top_k_per_scale)
        frames.append(df)

    out = pd.concat(frames, ignore_index=True).sort_values("candidate_score", ascending=False).reset_index(drop=True)
    out["global_rank"] = np.arange(1, len(out) + 1)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
