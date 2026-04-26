"""
score_tiles_geometry.py

Geometry-only candidate scoring for labyrinth-like forms.
This is a ranking aid, not proof of identification.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import cv2
import numpy as np
import pandas as pd


def read_gray(path):
    img = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(path)
    return img


def norm01(x, lo, hi):
    return float(np.clip((x - lo) / max(hi - lo, 1e-6), 0, 1))


def local_peak_count(values, min_prominence=0.015):
    values = np.asarray(values, dtype=np.float32)
    peaks = 0
    for i in range(1, len(values) - 1):
        if values[i] > values[i - 1] and values[i] > values[i + 1]:
            if min(values[i] - values[i - 1], values[i] - values[i + 1]) >= min_prominence:
                peaks += 1
    return peaks


def concentric_profile_score(gray):
    h, w = gray.shape
    cx, cy = w // 2, h // 2
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 60, 140)

    yy, xx = np.indices(edges.shape)
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    max_r = max(5, int(min(h, w) * 0.45))
    radial_sum = np.zeros(max_r + 1, dtype=np.float32)
    radial_cnt = np.zeros(max_r + 1, dtype=np.float32)
    r_int = np.clip(rr.astype(np.int32), 0, max_r)
    np.add.at(radial_sum, r_int.ravel(), edges.ravel() / 255.0)
    np.add.at(radial_cnt, r_int.ravel(), 1)
    profile = radial_sum / np.maximum(radial_cnt, 1)
    peaks = local_peak_count(profile)
    return float(np.clip(peaks / 6.0, 0, 1)), int(peaks)


def radial_symmetry_score(gray):
    s = min(gray.shape)
    crop = gray[:s, :s]
    rot = cv2.rotate(crop, cv2.ROTATE_180)
    corr = np.corrcoef(crop.astype(np.float32).ravel(), rot.astype(np.float32).ravel())[0, 1]
    if np.isnan(corr):
        corr = 0
    return float(np.clip((corr + 1) / 2, 0, 1))


def circularity_score(gray):
    blur = cv2.GaussianBlur(gray, (7, 7), 1.2)
    circles = cv2.HoughCircles(
        blur, cv2.HOUGH_GRADIENT, dp=1.2,
        minDist=min(gray.shape) * 0.15,
        param1=80, param2=20,
        minRadius=max(4, int(min(gray.shape) * 0.06)),
        maxRadius=max(8, int(min(gray.shape) * 0.45)),
    )
    if circles is None:
        return 0.0
    return float(np.clip(circles.shape[1] / 3.0, 0, 1))


def square_structure_score(gray):
    edges = cv2.Canny(gray, 50, 140)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=25, minLineLength=20, maxLineGap=6)
    if lines is None:
        return 0.0
    angles = []
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) % 180
        angles.append(angle)
    angles = np.array(angles)
    return float(((angles < 15) | (np.abs(angles - 90) < 15) | (np.abs(angles - 180) < 15)).mean())


def line_density_score(gray):
    edges = cv2.Canny(gray, 50, 150)
    return float(np.clip((edges > 0).mean() * 4.0, 0, 1))


def straight_line_penalty(gray):
    edges = cv2.Canny(gray, 50, 150)
    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, threshold=30, minLineLength=max(15, gray.shape[1] // 6), maxLineGap=6)
    if lines is None:
        return 0.0
    total = 0
    for line in lines[:, 0]:
        x1, y1, x2, y2 = line
        total += np.hypot(x2 - x1, y2 - y1)
    return float(np.clip((total / max(gray.size, 1)) * 8, 0, 1))


def contrast_score(gray):
    return norm01(float(np.std(gray.astype(np.float32))), 12, 55)


def score_one(path):
    gray = read_gray(path)
    circular = circularity_score(gray)
    radial = radial_symmetry_score(gray)
    concentric, peaks = concentric_profile_score(gray)
    square = square_structure_score(gray)
    line_density = line_density_score(gray)
    contrast = contrast_score(gray)
    straight_penalty = straight_line_penalty(gray)

    circular_core = 0.45 * circular + 0.25 * radial + 0.30 * concentric
    square_core = 0.35 * square + 0.25 * radial + 0.20 * concentric + 0.20 * contrast
    geometry_core = max(circular_core, square_core)

    candidate_score = 0.55 * geometry_core + 0.18 * line_density + 0.12 * contrast + 0.10 * square - 0.20 * straight_penalty
    candidate_score = float(np.clip(candidate_score, 0, 1))

    return {
        "circularity_score": circular,
        "radial_symmetry_score": radial,
        "concentric_edge_score": concentric,
        "concentric_peak_count": peaks,
        "square_structure_score": square,
        "line_density_score": line_density,
        "contrast_score": contrast,
        "straight_line_penalty": straight_penalty,
        "geometry_score": candidate_score,
        "candidate_score": candidate_score,
    }


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--tiles-csv", required=True)
    p.add_argument("--out-csv", required=True)
    return p.parse_args()


def main():
    args = parse_args()
    df = pd.read_csv(args.tiles_csv)
    rows = []
    for i, row in df.iterrows():
        rec = row.to_dict()
        rec.update(score_one(row["image_path"]))
        rows.append(rec)
        if (i + 1) % 100 == 0 or (i + 1) == len(df):
            print(f"Scored {i + 1}/{len(df)}")

    out = pd.DataFrame(rows).sort_values("candidate_score", ascending=False).reset_index(drop=True)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out_csv, index=False)
    print(f"Saved: {args.out_csv}")


if __name__ == "__main__":
    main()
