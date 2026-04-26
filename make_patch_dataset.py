"""Create classification patches from rasters and masks.

Positive patches are sampled around labyrinth masks.
Negative patches are sampled from mask-free regions.
"""
from __future__ import annotations

from pathlib import Path
import random

import cv2
import numpy as np
import pandas as pd
import rasterio

from dataset import read_mask, read_rgb


random.seed(42)
np.random.seed(42)


def crop_with_pad(img: np.ndarray, center_r: int, center_c: int, size: int) -> np.ndarray:
    half = size // 2
    r0, r1 = center_r - half, center_r + half
    c0, c1 = center_c - half, center_c + half
    pad_top = max(0, -r0)
    pad_left = max(0, -c0)
    pad_bottom = max(0, r1 - img.shape[0])
    pad_right = max(0, c1 - img.shape[1])
    if any([pad_top, pad_bottom, pad_left, pad_right]):
        img = cv2.copyMakeBorder(img, pad_top, pad_bottom, pad_left, pad_right, cv2.BORDER_REFLECT_101)
        r0 += pad_top
        r1 += pad_top
        c0 += pad_left
        c1 += pad_left
    return img[r0:r1, c0:c1]


def save_patch(img: np.ndarray, out_path: Path):
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), cv2.cvtColor(img, cv2.COLOR_RGB2BGR))


def build_patch_dataset(rgb_dir: str, mask_dir: str, out_dir: str, patch_px: int = 256, negatives_per_image: int = 20):
    rgb_dir = Path(rgb_dir)
    mask_dir = Path(mask_dir)
    out_dir = Path(out_dir)
    rows = []

    for rgb_path in sorted(rgb_dir.iterdir()):
        if rgb_path.suffix.lower() not in {".png", ".jpg", ".jpeg", ".tif", ".tiff"}:
            continue
        stem = rgb_path.stem
        mask_path = None
        for ext in [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"]:
            p = mask_dir / f"{stem}{ext}"
            if p.exists():
                mask_path = p
                break
        if mask_path is None:
            continue

        rgb = read_rgb(rgb_path)
        mask = read_mask(mask_path)
        ys, xs = np.where(mask > 0)
        if len(xs):
            center_r, center_c = int(np.mean(ys)), int(np.mean(xs))
            pos_patch = crop_with_pad(rgb, center_r, center_c, patch_px)
            pos_path = out_dir / "patches" / "positive" / f"{stem}_pos.png"
            save_patch(pos_patch, pos_path)
            rows.append({"image_path": str(pos_path), "label": 1})

        for i in range(negatives_per_image):
            for _ in range(100):
                r = random.randint(patch_px // 2, rgb.shape[0] - patch_px // 2 - 1)
                c = random.randint(patch_px // 2, rgb.shape[1] - patch_px // 2 - 1)
                patch_mask = crop_with_pad(mask[..., None], r, c, patch_px)[:, :, 0]
                if patch_mask.mean() == 0:
                    neg_patch = crop_with_pad(rgb, r, c, patch_px)
                    neg_path = out_dir / "patches" / "negative" / f"{stem}_neg_{i:03d}.png"
                    save_patch(neg_patch, neg_path)
                    rows.append({"image_path": str(neg_path), "label": 0})
                    break

    df = pd.DataFrame(rows)
    out_csv = out_dir / "patches.csv"
    df.to_csv(out_csv, index=False)
    print(f"saved {len(df)} rows to {out_csv}")


if __name__ == "__main__":
    build_patch_dataset("data/train/rgb", "data/train/masks", "data/patch_dataset", patch_px=256)
