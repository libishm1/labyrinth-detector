"""Generic binary classifier training for patch datasets.

Use this for quick baselines before Nazca-style multiscale scanning.
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
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-4
    num_workers: int = 2
    checkpoint: str = "classifier_baseline.pt"
    pretrained: bool = True
    device: str = "cuda" if torch.cuda.is_available() else "cpu"


def build_model(pretrained: bool = True) -> nn.Module:
    weights = models.ResNet34_Weights.IMAGENET1K_V1 if pretrained else None
    try:
        model = models.resnet34(weights=weights)
    except Exception as exc:
        warnings.warn(f"Falling back to randomly initialized ResNet34 because pretrained weights failed: {exc}")
        model = models.resnet34(weights=None)
    model.fc = nn.Linear(model.fc.in_features, 1)
    return model


def train_tf(size: int):
    return A.Compose([
        A.Resize(size, size),
        A.HorizontalFlip(p=0.5),
        A.VerticalFlip(p=0.5),
        A.RandomRotate90(p=0.5),
        A.ShiftScaleRotate(shift_limit=0.05, scale_limit=0.1, rotate_limit=20, p=0.5),
        A.RandomBrightnessContrast(p=0.2),
        A.Normalize(),
        ToTensorV2(),
    ])


def val_tf(size: int):
    return A.Compose([A.Resize(size, size), A.Normalize(), ToTensorV2()])


def evaluate(model, loader, criterion, device: str):
    model.eval()
    losses = []
    probs_all = []
    ys_all = []
    with torch.no_grad():
        for x, y, _ in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            loss = criterion(logits, y)
            losses.append(loss.item())
            probs_all.append(torch.sigmoid(logits).cpu().numpy())
            ys_all.append(y.cpu().numpy())
    probs = np.concatenate(probs_all).ravel()
    ys = np.concatenate(ys_all).ravel()
    preds = (probs >= 0.5).astype(np.float32)
    acc = float((preds == ys).mean()) if len(ys) else 0.0
    return float(np.mean(losses)), acc


def main() -> None:
    cfg = CFG()
    train_ds = PatchClassificationDataset(cfg.train_csv, transform=train_tf(cfg.image_size))
    val_ds = PatchClassificationDataset(cfg.val_csv, transform=val_tf(cfg.image_size))
    train_dl = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers)
    val_dl = DataLoader(val_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers)

    model = build_model(pretrained=cfg.pretrained).to(cfg.device)
    criterion = nn.BCEWithLogitsLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-4)

    best_acc = -1.0
    for epoch in range(cfg.epochs):
        model.train()
        losses = []
        for x, y, _ in tqdm(train_dl, desc=f"epoch {epoch+1}/{cfg.epochs}"):
            x, y = x.to(cfg.device), y.to(cfg.device)
            logits = model(x)
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            losses.append(loss.item())

        val_loss, val_acc = evaluate(model, val_dl, criterion, cfg.device)
        print(f"train_loss={np.mean(losses):.4f} val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(model.state_dict(), cfg.checkpoint)
            print(f"saved {cfg.checkpoint}")


if __name__ == "__main__":
    main()
