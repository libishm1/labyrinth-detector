from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd
import numpy as np

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input-csv", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--geometry-weight", type=float, default=0.50)
    p.add_argument("--classifier-weight", type=float, default=0.35)
    p.add_argument("--context-weight", type=float, default=0.15)
    return p.parse_args()

def getcol(df, names, default=0.0):
    for n in names:
        if n in df.columns:
            return df[n].astype(float).fillna(default)
    return pd.Series([default] * len(df))

def main():
    args = parse_args()
    df = pd.read_csv(args.input_csv)
    geometry = getcol(df, ["geometry_score", "candidate_score"])
    classifier = getcol(df, ["classifier_score"], default=0.0)
    context = getcol(df, ["context_score"], default=0.0)
    if "classifier_score" not in df.columns:
        final = geometry
    else:
        final = args.geometry_weight * geometry + args.classifier_weight * classifier + args.context_weight * context
    df["hybrid_score"] = np.clip(final, 0, 1)
    df = df.sort_values("hybrid_score", ascending=False).reset_index(drop=True)
    df["hybrid_rank"] = np.arange(1, len(df) + 1)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved hybrid ranking: {args.out_csv}")

if __name__ == "__main__":
    main()
