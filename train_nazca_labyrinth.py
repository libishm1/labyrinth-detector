"""Nazca-style labyrinth patch classifier with focal loss.

Use this after make_patch_dataset.py and before infer_multiscale_heatmap.py.
"""
from __future__ import annotations

from dataclasses import dataclass
import warnings

import albumentations as A
from albumentations.pytorch import ToTensorV2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import models
from tqdm import tqdm

from dataset import PatchClassificationDataset


@dataclass
class CFG:
    train_csv: str = "data/patch_dataset/train.csv"
    val_csv: str = "data/patch_dataset/val.csv"
    image_size: int = 256
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-4
    num_workers: int = 2
    checkpoint: str = "nazca_labyrinth_classifier.pt"
    pretrained: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


class FocalLoss(nn.Module):
    def __init__(self, alpha: float = 0.25, gamma: float = 2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, logits, targets):
        bce = self.bce(logits, targets)
        probs = torch.sigmoid(logits)
        pt = torch.where(targets == 1, probs, 1 - probs)
        return (self.alpha * (1 - pt) ** self.gamma * bce).mean()


def build_model(pretrained: bool = True):
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    try:
        model = models.resnet50(weights=weights)
    except Exception as exc:
        warnings.warn(f"Falling back to randomly initialized ResNet50 because pretrained weights failed: {exc}")
        model = models.resnet50(weights=None)
    model.fc = nn.Sequential(
        nn.Linear(model.fc.in_features, 256),
        nn.ReLU(inplace=True),
        nn.Dropout(0.3),
        nn.Linear(256, 1),
    )
    return model


def train_tf(size):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=25, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(),
        ToTensorV2(),
    ])


def val_tf(size):
    return A.Compose([A.Resize(size, size), A.Normalize(), ToTensorV2()])


def validate(model, loader, criterion, device):
    model.eval()
    losses = []
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses.append(criterion(logits, y).item())
    return float(np.mean(losses))


def main():
    cfg = CFG()
    train_ds = PatchClassificationDataset(cfg.train_csv, transform=train_tf(cfg.image_size))
    val_ds = PatchClassificationDataset(cfg.val_csv, transform=val_tf(cfg.image_size))
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = build_model(pretrained=cfg.pretrained).to(cfg.device)
    criterion = FocalLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    best_val = 1e9
    for epoch in range(cfg.epochs):
        model.train()
        epoch_losses = []
        for x, y, _ in tqdm(train_dl, desc=f"epoch {epoch+1}/{cfg.epochs}"):
            x, y = x.to(cfg.device), y.to(cfg.device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_losses.append(loss.item())
        val_loss = validate(model, val_dl, criterion, cfg.device)
        print(f"train_loss={np.mean(epoch_losses):.4f} val_loss={val_loss:.4f}")
        if val_loss < best_val:
            best_val = val_loss
            torch.save(model.state_dict(), cfg.checkpoint)
            print(f"saved {cfg.checkpoint}")


if __name__ == "__main__":
    main()
