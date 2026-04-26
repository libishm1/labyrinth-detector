"""Single-scale sliding window heatmap inference for trained classifiers."""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import rasterio
from rasterio.windows import Window
import torch
from torchvision import models
from tqdm import tqdm


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(checkpoint: str):
    model = models.resnet34(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, 1)
    state = torch.load(checkpoint, map_location=DEVICE)
    model.load_state_dict(state)
    model.to(DEVICE).eval()
    return model


@torch.no_grad()
def infer_heatmap(raster_path: str, checkpoint: str, patch_px: int = 256, stride_px: int = 64, batch_size: int = 64):
    model = build_model(checkpoint)
    with rasterio.open(raster_path) as src:
        h, w = src.height, src.width
        heat = np.zeros((h, w), dtype=np.float32)
        hits = np.zeros((h, w), dtype=np.float32)
        batch = []
        coords = []

        def flush():
            nonlocal batch, coords, heat, hits
            if not batch:
                return
            x = torch.tensor(np.stack(batch).transpose(0, 3, 1, 2), dtype=torch.float32, device=DEVICE) / 255.0
            probs = torch.sigmoid(model(x)).squeeze(1).cpu().numpy()
            for (r, c), p in zip(coords, probs):
                heat[r:r+patch_px, c:c+patch_px] += p
                hits[r:r+patch_px, c:c+patch_px] += 1
            batch, coords = [], []

        total = max(1, ((h - patch_px) // stride_px + 1) * ((w - patch_px) // stride_px + 1))
        for r in tqdm(range(0, h - patch_px + 1, stride_px), total=((h - patch_px) // stride_px + 1)):
            for c in range(0, w - patch_px + 1, stride_px):
                arr = src.read(window=Window(c, r, patch_px, patch_px))
                arr = np.transpose(arr, (1, 2, 0))
                arr = arr[:, :, :3].astype(np.float32)
                arr -= arr.min()
                if arr.max() > 0:
                    arr /= arr.max()
                batch.append((arr * 255).astype(np.uint8))
                coords.append((r, c))
                if len(batch) >= batch_size:
                    flush()
        flush()

        avg = heat / np.maximum(hits, 1)
        return avg, src.transform, src.crs


def save_heatmap(heatmap: np.ndarray, out_png: str):
    vis = np.clip(heatmap * 255, 0, 255).astype(np.uint8)
    vis = cv2.applyColorMap(vis, cv2.COLORMAP_VIRIDIS)
    cv2.imwrite(out_png, vis)


if __name__ == "__main__":
    raster_path = "data/test/example.tif"
    checkpoint = "classifier_baseline.pt"
    heat, _, _ = infer_heatmap(raster_path, checkpoint)
    Path("outputs").mkdir(exist_ok=True)
    save_heatmap(heat, "outputs/heatmap.png")
