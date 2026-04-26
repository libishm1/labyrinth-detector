"""Nazca-style multiscale sliding-window inference for labyrinth search."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
from scipy.ndimage import label
import torch
from torchvision import models
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


@dataclass
class ScaleConfig:
    patch_px: int
    stride_px: int
    prob_threshold: float
    min_neighbors: int


def build_model(checkpoint: str):
    model = models.resnet50(weights=None)
    model.fc = torch.nn.Sequential(
        torch.nn.Linear(model.fc.in_features, 256),
        torch.nn.ReLU(inplace=True),
        torch.nn.Dropout(0.3),
        torch.nn.Linear(256, 1),
    )
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.to(DEVICE).eval()
    return model


@torch.no_grad()
def infer_one_scale(src, model, cfg: ScaleConfig, batch_size: int = 64):
    h, w = src.height, src.width
    patch_px = cfg.patch_px
    stride_px = cfg.stride_px
    heat = np.zeros((h, w), dtype=np.float32)
    hits = np.zeros((h, w), dtype=np.float32)
    coords = []
    patches = []

    def flush():
        nonlocal coords, patches, heat, hits
        if not patches:
            return
        x = torch.tensor(np.stack(patches).transpose(0, 3, 1, 2), dtype=torch.float32, device=DEVICE) / 255.0
        probs = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
        for (r, c), p in zip(coords, probs):
            heat[r:r+patch_px, c:c+patch_px] += p
            hits[r:r+patch_px, c:c+patch_px] += 1.0
        coords, patches = [], []

    for r in tqdm(range(0, h - patch_px + 1, stride_px), desc=f"scale {patch_px}"):
        for c in range(0, w - patch_px + 1, stride_px):
            arr = src.read(window=Window(c, r, patch_px, patch_px))
            arr = np.transpose(arr, (1, 2, 0))[:, :, :3].astype(np.float32)
            arr -= arr.min()
            if arr.max() > 0:
                arr /= arr.max()
            patches.append((arr * 255).astype(np.uint8))
            coords.append((r, c))
            if len(patches) >= batch_size:
                flush()
    flush()

    avg = heat / np.maximum(hits, 1.0)
    grid_rows = list(range(0, h - patch_px + 1, stride_px))
    grid_cols = list(range(0, w - patch_px + 1, stride_px))
    score_grid = np.zeros((len(grid_rows), len(grid_cols)), dtype=np.float32)
    for i, r in enumerate(grid_rows):
        for j, c in enumerate(grid_cols):
            score_grid[i, j] = avg[r:r+patch_px, c:c+patch_px].mean()

    binary = (score_grid >= cfg.prob_threshold).astype(np.uint8)
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(binary, -1, kernel)
    binary = np.where(neighbor_count >= cfg.min_neighbors, 1, 0).astype(np.uint8)
    labeled, num = label(binary)

    candidates = []
    for k in range(1, num + 1):
        ys, xs = np.where(labeled == k)
        if len(xs) == 0:
            continue
        score = float(score_grid[ys, xs].max())
        candidates.append({
            "patch_px": patch_px,
            "stride_px": stride_px,
            "score": score,
            "grid_y0": int(ys.min()),
            "grid_y1": int(ys.max()),
            "grid_x0": int(xs.min()),
            "grid_x1": int(xs.max()),
        })
    return avg, score_grid, candidates


def infer_multiscale(raster_path: str, checkpoint: str, scales: list[ScaleConfig]):
    model = build_model(checkpoint)
    all_candidates = []
    outputs = {}
    with rasterio.open(raster_path) as src:
        for cfg in scales:
            heat, score_grid, candidates = infer_one_scale(src, model, cfg)
            outputs[cfg.patch_px] = {"heat": heat, "score_grid": score_grid}
            all_candidates.extend(candidates)
    return outputs, sorted(all_candidates, key=lambda d: d["score"], reverse=True)


if __name__ == "__main__":
    scales = [
        ScaleConfig(128, 10, 0.50, 3),
        ScaleConfig(256, 20, 0.50, 4),
        ScaleConfig(384, 30, 0.52, 5),
    ]
    outputs, candidates = infer_multiscale("data/test/example.tif", "nazca_labyrinth_classifier.pt", scales)
    Path("outputs").mkdir(exist_ok=True)
    np.save("outputs/candidates.npy", np.array(candidates, dtype=object))
    print(f"found {len(candidates)} clustered candidates")
