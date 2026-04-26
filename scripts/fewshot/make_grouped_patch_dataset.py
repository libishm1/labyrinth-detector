"""
make_grouped_patch_dataset.py

Combine positive chips and hard negatives into a grouped patch dataset.
Preserves site_group for grouped validation.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--positive-csv", required=True)
    p.add_argument("--negative-csvs", nargs="*", default=[])
    p.add_argument("--negative-dir", default=None)
    p.add_argument("--out-csv", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    frames = []

    pos = pd.read_csv(args.positive_csv)
    pos["label"] = 1
    frames.append(pos)

    for neg_csv in args.negative_csvs:
        neg = pd.read_csv(neg_csv)
        neg["label"] = 0
        if "site_group" not in neg.columns:
            neg["site_group"] = "negative"
        if "site_id" not in neg.columns:
            neg["site_id"] = ["neg_%06d" % i for i in range(len(neg))]
        frames.append(neg)

    if args.negative_dir:
        paths = sorted(Path(args.negative_dir).glob("*.png"))
        neg = pd.DataFrame({
            "image_path": [str(p) for p in paths],
            "label": [0] * len(paths),
            "site_group": ["negative"] * len(paths),
            "site_id": [p.stem for p in paths],
            "domain": ["hard_negative"] * len(paths),
            "label_strength": [1.0] * len(paths),
        })
        frames.append(neg)

    out = pd.concat(frames, ignore_index=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved dataset: {args.out_csv}")
    print(out["label"].value_counts())


if __name__ == "__main__":
    main()
