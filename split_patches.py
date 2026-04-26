"""Split patch CSV into train and validation CSVs.

Stratified by label when possible.
"""
from __future__ import annotations

from pathlib import Path
import argparse

import pandas as pd
from sklearn.model_selection import train_test_split

from config import PATHS


def split_patches(input_csv: str, train_csv: str, val_csv: str, val_fraction: float = 0.2, seed: int = 42):
    df = pd.read_csv(input_csv)
    if df.empty:
        raise ValueError(f"No rows found in {input_csv}")
    if "label" not in df.columns:
        raise ValueError("Input CSV must contain a 'label' column")

    stratify = df["label"] if df["label"].nunique() > 1 else None
    train_df, val_df = train_test_split(
        df,
        test_size=val_fraction,
        random_state=seed,
        stratify=stratify,
    )

    Path(train_csv).parent.mkdir(parents=True, exist_ok=True)
    train_df.to_csv(train_csv, index=False)
    val_df.to_csv(val_csv, index=False)

    summary = {
        "train_rows": len(train_df),
        "val_rows": len(val_df),
        "train_pos": int(train_df["label"].sum()),
        "val_pos": int(val_df["label"].sum()),
    }
    print(summary)
    print(f"saved train split to {train_csv}")
    print(f"saved val split to {val_csv}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_csv", default=str(PATHS.patch_csv))
    parser.add_argument("--train_csv", default=str(PATHS.train_csv))
    parser.add_argument("--val_csv", default=str(PATHS.val_csv))
    parser.add_argument("--val_fraction", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    split_patches(args.input_csv, args.train_csv, args.val_csv, args.val_fraction, args.seed)
