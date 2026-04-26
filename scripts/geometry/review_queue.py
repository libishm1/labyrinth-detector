"""
review_queue.py

Create top candidate image chips and a review CSV.
"""
from __future__ import annotations

import argparse
import math
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def add_header(img, title, subtitle=""):
    h, w = img.shape[:2]
    hh = 42 if subtitle else 26
    canvas = np.zeros((h + hh, w, 3), dtype=np.uint8)
    canvas[:hh] = (20, 20, 20)
    canvas[hh:] = img
    cv2.putText(canvas, title[:90], (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 1, cv2.LINE_AA)
    if subtitle:
        cv2.putText(canvas, subtitle[:120], (8, 36), cv2.FONT_HERSHEY_SIMPLEX, 0.43, (190, 190, 190), 1, cv2.LINE_AA)
    return canvas


def contact_sheet(images, out_path, cols=4):
    if not images:
        return
    mh = max(i.shape[0] for i in images)
    mw = max(i.shape[1] for i in images)
    canvas = np.full((math.ceil(len(images) / cols) * mh, cols * mw, 3), 245, dtype=np.uint8)
    for idx, img in enumerate(images):
        y = (idx // cols) * mh
        x = (idx % cols) * mw
        h, w = img.shape[:2]
        canvas[y:y + h, x:x + w] = img
    cv2.imwrite(str(out_path), cv2.cvtColor(canvas, cv2.COLOR_RGB2BGR))


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tile-scores-csv", required=True)
    p.add_argument("--out-dir", required=True)
    p.add_argument("--top-k", type=int, default=200)
    p.add_argument("--score-threshold", type=float, default=None)
    return p.parse_args()


def main():
    args = parse_args()
    out_dir = Path(args.out_dir)
    chip_dir = out_dir / "review_tiles"
    chip_dir.mkdir(parents=True, exist_ok=True)
    df = pd.read_csv(args.tile_scores_csv).sort_values("candidate_score", ascending=False)
    if args.score_threshold is not None:
        df = df[df["candidate_score"] >= args.score_threshold]
    df = df.head(args.top_k)

    rows, imgs = [], []
    for _, row in df.iterrows():
        img = cv2.imread(str(row["image_path"]), cv2.IMREAD_COLOR)
        if img is None:
            continue
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        title = f"{len(rows) + 1:03d} | {row['tile_id']}"
        subtitle = f"score={row['candidate_score']:.3f} circ={row.get('circularity_score', 0):.2f} conc={row.get('concentric_edge_score', 0):.2f}"
        labeled = add_header(img, title, subtitle)
        out_path = chip_dir / f"{len(rows) + 1:03d}_{row['tile_id']}.png"
        cv2.imwrite(str(out_path), cv2.cvtColor(labeled, cv2.COLOR_RGB2BGR))
        imgs.append(labeled)
        rec = row.to_dict()
        rec.update({"review_rank": len(rows) + 1, "review_image_path": str(out_path), "label": "", "review_note": "", "false_positive_category": ""})
        rows.append(rec)

    pd.DataFrame(rows).to_csv(out_dir / "review_queue.csv", index=False)
    contact_sheet(imgs, out_dir / "review_contact_sheet.png")
    print(f"Saved review queue: {out_dir / 'review_queue.csv'}")


if __name__ == "__main__":
    main()
