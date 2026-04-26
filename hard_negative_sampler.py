"""Mine hard negatives from false-positive candidate regions."""
from __future__ import annotations

from pathlib import Path

import cv2
import pandas as pd
import rasterio


def mine_hard_negatives(raster_path: str, candidate_csv: str, out_dir: str, max_count: int = 100):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(candidate_csv).sort_values("score", ascending=False)

    with rasterio.open(raster_path) as src:
        img = src.read()[:3]
        img = img.transpose(1, 2, 0).astype("float32")
        img -= img.min()
        if img.max() > 0:
            img /= img.max()
        img = (img * 255).astype("uint8")

    saved = 0
    for i, row in df.iterrows():
        r0, r1 = int(row["row0"]), int(row["row1"])
        c0, c1 = int(row["col0"]), int(row["col1"])
        patch = img[r0:r1, c0:c1]
        if patch.size == 0:
            continue
        path = out_dir / f"hard_negative_{saved:04d}.png"
        cv2.imwrite(str(path), cv2.cvtColor(patch, cv2.COLOR_RGB2BGR))
        saved += 1
        if saved >= max_count:
            break
    print(f"saved {saved} hard negatives to {out_dir}")


if __name__ == "__main__":
    mine_hard_negatives("data/test/example.tif", "outputs/false_positives.csv", "data/hard_negatives")
