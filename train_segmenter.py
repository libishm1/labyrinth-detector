"""Train a multimodal labyrinth segmenter on RGB + terrain channels."""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import LabyrinthSegmentationDataset


@dataclass
class CFG:
    train_rgb_dir: str = "data/train/rgb"
    train_terrain_dir: str = "data/train/terrain"
    train_mask_dir: str = "data/train/masks"
    val_rgb_dir: str = "data/val/rgb"
    val_terrain_dir: str = "data/val/terrain"
    val_mask_dir: str = "data/val/masks"
    image_size: int = 256
    batch_size: int = 8
    epochs: int = 40
    lr: float = 1e-4
    num_workers: int = 2
    encoder_name: str = "resnet34"
    checkpoint: str = "segmenter_best.pt"
    pretrained: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class DiceBCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.bce = nn.BCEWithLogitsLoss()

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        intersection = (probs * targets).sum(dim=(2, 3))
        union = probs.sum(dim=(2, 3)) + targets.sum(dim=(2, 3))
        dice = (2 * intersection + 1e-6) / (union + 1e-6)
        return bce + (1 - dice.mean())


def compute_iou(logits, targets, threshold=0.5) -> float:
    probs = torch.sigmoid(logits)
    preds = (probs > threshold).float()
    intersection = (preds * targets).sum(dim=(2, 3))
    union = ((preds + targets) > 0).float().sum(dim=(2, 3))
    return float(((intersection + 1e-6) / (union + 1e-6)).mean().item())


def train_tf(size):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.10, rotate_limit=20, p=0.5),
        ToTensorV2(),
    ])


def val_tf(size):
    return A.Compose([A.Resize(size, size), ToTensorV2()])


def build_model(in_channels: int, encoder_name: str, pretrained: bool = True) -> nn.Module:
    encoder_weights = "imagenet" if pretrained else None
    try:
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            in_channels=in_channels,
            classes=1,
        )
    except Exception as exc:
        warnings.warn(f"Falling back to randomly initialized encoder because pretrained weights failed: {exc}")
        return smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=None,
            in_channels=in_channels,
            classes=1,
        )


def run_epoch(model, loader, criterion, device, optimizer=None):
    train = optimizer is not None
    model.train() if train else model.eval()
    losses = []
    ious = []
    for batch in tqdm(loader, leave=False):
        x, y = batch.x.to(device), batch.y.to(device)
        logits = model(x)
        loss = criterion(logits, y)
        if train:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        losses.append(loss.item())
        ious.append(compute_iou(logits, y))
    return float(np.mean(losses)), float(np.mean(ious))


def main() -> None:
    cfg = CFG()
    train_ds = LabyrinthSegmentationDataset(
        cfg.train_rgb_dir, cfg.train_terrain_dir, cfg.train_mask_dir, transform=train_tf(cfg.image_size)
    )
    val_ds = LabyrinthSegmentationDataset(
        cfg.val_rgb_dir, cfg.val_terrain_dir, cfg.val_mask_dir, transform=val_tf(cfg.image_size)
    )
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    sample = train_ds[0]
    model = build_model(sample.x.shape[0], cfg.encoder_name, pretrained=cfg.pretrained).to(cfg.device)
    criterion = DiceBCELoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr)

    best_iou = -1.0
    for epoch in range(cfg.epochs):
        tr_loss, tr_iou = run_epoch(model, train_dl, criterion, cfg.device, optimizer)
        va_loss, va_iou = run_epoch(model, val_dl, criterion, cfg.device)
        print(f"epoch={epoch+1} train_loss={tr_loss:.4f} train_iou={tr_iou:.4f} val_loss={va_loss:.4f} val_iou={va_iou:.4f}")
        if va_iou > best_iou:
            best_iou = va_iou
            torch.save(model.state_dict(), cfg.checkpoint)
            print(f"saved {cfg.checkpoint}")


if __name__ == "__main__":
    main()
