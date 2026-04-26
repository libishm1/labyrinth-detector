from __future__ import annotations
import argparse
from pathlib import Path
import pandas as pd

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--positive-only-groups", action="store_true")
    return p.parse_args()

def main():
    args = parse_args()
    df = pd.read_csv(args.dataset_csv)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    groups = sorted(df["site_group"].dropna().unique().tolist())
    if args.positive_only_groups:
        groups = [g for g, gdf in df.groupby("site_group") if (gdf["label"] == 1).any()]
    manifest = []
    for g in groups:
        val = df[df["site_group"] == g].copy()
        train = df[df["site_group"] != g].copy()
        fold_dir = out_dir / f"holdout_{g}"
        fold_dir.mkdir(parents=True, exist_ok=True)
        train.to_csv(fold_dir / "train.csv", index=False)
        val.to_csv(fold_dir / "val.csv", index=False)
        manifest.append({"holdout_group": g, "train_csv": str(fold_dir / "train.csv"), "val_csv": str(fold_dir / "val.csv"), "n_train": len(train), "n_val": len(val)})
    pd.DataFrame(manifest).to_csv(out_dir / "fold_manifest.csv", index=False)
    print(f"Saved leave-one-site-out folds: {out_dir / 'fold_manifest.csv'}")

if __name__ == "__main__":
    main()
