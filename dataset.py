"""Shared dataset and raster utilities for labyrinth detection.

Supports:
- patch classification datasets
- multimodal segmentation datasets
- raster stack loading for RGB + terrain channels
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Iterable, Optional

import cv2
import numpy as np
import rasterio
import torch
from torch.utils.data import Dataset

IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".npy"}


def normalize_minmax(x: np.ndarray) -> np.ndarray:
    x = x.astype(np.float32)
    if np.all(~np.isfinite(x)):
        return np.zeros_like(x, dtype=np.float32)
    mn = np.nanmin(x)
    mx = np.nanmax(x)
    if mx - mn < 1e-8:
        return np.zeros_like(x, dtype=np.float32)
    return (x - mn) / (mx - mn)


def read_raster(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path)
        if arr.ndim == 2:
            arr = arr[..., None]
        return arr.astype(np.float32)
    with rasterio.open(path) as src:
        arr = src.read()  # C,H,W
    arr = np.transpose(arr, (1, 2, 0))
    return arr.astype(np.float32)


def read_rgb(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff", ".npy"}:
        arr = read_raster(path)
        if arr.shape[2] > 3:
            arr = arr[:, :, :3]
        arr = normalize_minmax(arr)
        return (arr * 255).astype(np.uint8)
    img = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(path)
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)


def read_mask(path: str | Path) -> np.ndarray:
    path = Path(path)
    if path.suffix.lower() in {".tif", ".tiff", ".npy"}:
        arr = read_raster(path)
        if arr.ndim == 3:
            arr = arr[:, :, 0]
        return (arr > 0).astype(np.float32)
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return (img > 127).astype(np.float32)


def find_matching_file(folder: str | Path, stem: str) -> Path:
    folder = Path(folder)
    for ext in IMAGE_EXTS:
        p = folder / f"{stem}{ext}"
        if p.exists():
            return p
    raise FileNotFoundError(f"No matching file for stem={stem} in {folder}")


@dataclass
class SegmentationSample:
    x: torch.Tensor
    y: torch.Tensor
    sample_id: str


class LabyrinthSegmentationDataset(Dataset):
    def __init__(
        self,
        rgb_dir: str | Path,
        terrain_dir: str | Path,
        mask_dir: str | Path,
        transform: Optional[Callable] = None,
    ) -> None:
        self.rgb_dir = Path(rgb_dir)
        self.terrain_dir = Path(terrain_dir)
        self.mask_dir = Path(mask_dir)
        self.transform = transform
        self.ids = sorted([p.stem for p in self.rgb_dir.iterdir() if p.suffix.lower() in IMAGE_EXTS])
        if not self.ids:
            raise ValueError(f"No samples found in {self.rgb_dir}")

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, idx: int) -> SegmentationSample:
        sample_id = self.ids[idx]
        rgb = read_rgb(find_matching_file(self.rgb_dir, sample_id)).astype(np.float32) / 255.0
        terrain = normalize_minmax(read_raster(find_matching_file(self.terrain_dir, sample_id)))
        if terrain.ndim == 2:
            terrain = terrain[..., None]
        mask = read_mask(find_matching_file(self.mask_dir, sample_id))
        x = np.concatenate([rgb, terrain], axis=2)

        if self.transform is not None:
            out = self.transform(image=x, mask=mask)
            x_t = out["image"]
            y_t = out["mask"].unsqueeze(0).float()
        else:
            x_t = torch.tensor(np.transpose(x, (2, 0, 1)), dtype=torch.float32)
            y_t = torch.tensor(mask, dtype=torch.float32).unsqueeze(0)
        return SegmentationSample(x=x_t, y=y_t, sample_id=sample_id)


class PatchClassificationDataset(Dataset):
    def __init__(self, csv_path: str | Path, transform: Optional[Callable] = None) -> None:
        import pandas as pd

        self.df = pd.read_csv(csv_path)
        required = {"image_path", "label"}
        missing = required - set(self.df.columns)
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {missing}")
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        img = read_rgb(row["image_path"])
        if self.transform is not None:
            out = self.transform(image=img)
            x = out["image"]
        else:
            x = torch.tensor(np.transpose(img.astype(np.float32) / 255.0, (2, 0, 1)), dtype=torch.float32)
        y = torch.tensor([float(row["label"])], dtype=torch.float32)
        return x, y, str(row["image_path"])


class RasterPatchDataset(Dataset):
    """In-memory patch dataset for sliding-window inference batches."""

    def __init__(self, patches: Iterable[np.ndarray]) -> None:
        self.patches = list(patches)

    def __len__(self) -> int:
        return len(self.patches)

    def __getitem__(self, idx: int):
        patch = self.patches[idx].astype(np.float32) / 255.0
        patch = np.transpose(patch, (2, 0, 1))
        return torch.tensor(patch, dtype=torch.float32)
