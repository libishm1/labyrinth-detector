from __future__ import annotations
import argparse
from pathlib import Path
import cv2, torch, numpy as np, pandas as pd
from torch.utils.data import Dataset, DataLoader
from torchvision import models
import torch.nn as nn

class TileDataset(Dataset):
    def __init__(self, df, image_size):
        self.df = df.reset_index(drop=True)
        self.image_size = image_size
    def __len__(self): return len(self.df)
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        if img is None: raise FileNotFoundError(row["image_path"])
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img = cv2.resize(img, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
        img = img.astype(np.float32) / 255.0
        img = (img - np.array([0.485,0.456,0.406], np.float32)) / np.array([0.229,0.224,0.225], np.float32)
        return torch.tensor(img.transpose(2,0,1), dtype=torch.float32), idx

def build_model(name):
    if name == "resnet18":
        model = models.resnet18(weights=None)
        model.fc = nn.Linear(model.fc.in_features, 1)
        return model
    if name == "mobilenet_v3_small":
        model = models.mobilenet_v3_small(weights=None)
        model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 1)
        return model
    raise ValueError(name)

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tiles-csv", required=True)
    p.add_argument("--out-csv", required=True)
    p.add_argument("--batch-size", type=int, default=64)
    return p.parse_args()

def main():
    args = parse_args()
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model = build_model(ckpt.get("model_name", "resnet18"))
    model.load_state_dict(ckpt["model"])
    image_size = int(ckpt.get("image_size", 224))
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device).eval()
    df = pd.read_csv(args.tiles_csv)
    dl = DataLoader(TileDataset(df, image_size), batch_size=args.batch_size, shuffle=False, num_workers=0)
    scores = np.zeros(len(df), dtype=np.float32)
    with torch.no_grad():
        for x, idx in dl:
            scores[idx.numpy()] = torch.sigmoid(model(x.to(device))).cpu().numpy().ravel()
    df["classifier_score"] = scores
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(args.out_csv, index=False)
    print(f"Saved classifier-scored tiles: {args.out_csv}")

if __name__ == "__main__":
    main()
