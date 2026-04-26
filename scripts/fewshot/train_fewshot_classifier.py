"""
Small grouped few-shot classifier. Run from repository root.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd
import numpy as np
import cv2
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import roc_auc_score, average_precision_score


class PatchDataset(Dataset):
    def __init__(self, df, image_size=224, train=False):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
        self.train = train

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        if img is None:
            raise FileNotFoundError(row["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        if self.train:
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=1).copy()
            if np.random.rand() < 0.5:
                img = np.flip(img, axis=0).copy()
            img = np.rot90(img, np.random.randint(0, 4)).copy()
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485, 0.456, 0.406], dtype=np.float32)) / np.array([0.229, 0.224, 0.225], dtype=np.float32)
        return torch.tensor(img.transpose(2, 0, 1), dtype=torch.float32), torch.tensor([float(row["label"])], dtype=torch.float32), torch.tensor([float(row.get("label_strength", 1.0))], dtype=torch.float32)


def build_model(name="resnet18"):
    if name == "resnet18":
        model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=models.MobileNet_V3_Small_Weights.IMAGENET1K_V1)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
        return model
    raise ValueError(name)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--model-name", default="resnet18", choices=["resnet18", "mobilenet_v3_small"])
    p.add_argument("--image-size", type=int, default=224)
    p.add_argument("--epochs", type=int, default=20)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--val-size", type=float, default=0.25)
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.dataset_csv)
    if "site_group" not in df.columns:
        raise ValueError("dataset CSV must include site_group for grouped splitting")
    splitter = GroupShuffleSplit(n_splits=1, test_size=args.val_size, random_state=args.seed)
    train_idx, val_idx = next(splitter.split(df, groups=df["site_group"]))
    train_df = df.iloc[train_idx].copy()
    val_df = df.iloc[val_idx].copy()
    train_df.to_csv(out_dir / "train_split.csv", index=False)
    val_df.to_csv(out_dir / "val_split.csv", index=False)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = build_model(args.model_name).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    bce = nn.BCEWithLogitsLoss(reduction="none")
    train_dl = DataLoader(PatchDataset(train_df, args.image_size, True), batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = DataLoader(PatchDataset(val_df, args.image_size, False), batch_size=args.batch_size, shuffle=False, num_workers=0)
    best_ap = -1
    for epoch in range(1, args.epochs + 1):
        model.train()
        losses = []
        for x, y, w in train_dl:
            x, y, w = x.to(device), y.to(device), w.to(device)
            loss = (bce(model(x), y) * w).mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            losses.append(loss.item())
        model.eval()
        probs, ys = [], []
        with torch.no_grad():
            for x, y, _ in val_dl:
                p = torch.sigmoid(model(x.to(device))).cpu().numpy().ravel()
                probs.extend(p.tolist())
                ys.extend(y.numpy().ravel().tolist())
        try: auc = roc_auc_score(ys, probs)
        except Exception: auc = float("nan")
        try: ap = average_precision_score(ys, probs)
        except Exception: ap = float("nan")
        print(f"epoch={epoch} loss={np.mean(losses):.4f} val_auc={auc:.4f} val_ap={ap:.4f}")
        metric = ap if not np.isnan(ap) else -1
        if metric > best_ap:
            best_ap = metric
            torch.save({"model": model.state_dict(), "model_name": args.model_name, "image_size": args.image_size}, out_dir / "fewshot_classifier_best.pt")
    print(f"Saved best checkpoint: {out_dir / 'fewshot_classifier_best.pt'}")

if __name__ == "__main__":
    main()
