"""Archaeology-oriented concentric ring scoring utilities.

This module favors low-relief circular candidates with repeated radial edge peaks,
which is more specific to labyrinth-like concentric structures than generic circularity.
"""
from __future__ import annotations

import math
from typing import Dict

import cv2
import numpy as np


def _normalize_gray(gray: np.ndarray) -> np.ndarray:
    gray = gray.astype(np.float32)
    gray -= gray.min()
    if gray.max() > 0:
        gray /= gray.max()
    return gray


def circularity_score(binary_mask: np.ndarray) -> float:
    contours, _ = cv2.findContours(binary_mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours:
        return 0.0
    cnt = max(contours, key=cv2.contourArea)
    area = cv2.contourArea(cnt)
    perimeter = cv2.arcLength(cnt, True)
    if perimeter < 1e-6:
        return 0.0
    score = (4 * np.pi * area) / (perimeter ** 2 + 1e-6)
    return float(np.clip(score, 0.0, 1.0))


def estimate_center_from_edges(gray: np.ndarray) -> tuple[float, float]:
    edges = cv2.Canny((_normalize_gray(gray) * 255).astype(np.uint8), 60, 140)
    ys, xs = np.where(edges > 0)
    if len(xs) < 10:
        h, w = gray.shape[:2]
        return w / 2.0, h / 2.0
    return float(xs.mean()), float(ys.mean())


def radial_profile(gray: np.ndarray, center: tuple[float, float], n_bins: int | None = None) -> np.ndarray:
    h, w = gray.shape[:2]
    cx, cy = center
    yy, xx = np.indices((h, w))
    rr = np.sqrt((xx - cx) ** 2 + (yy - cy) ** 2)
    gray_n = _normalize_gray(gray)
    edges = cv2.Canny((gray_n * 255).astype(np.uint8), 60, 140).astype(np.float32) / 255.0
    mag = cv2.GaussianBlur(edges, (0, 0), 1.0)

    max_r = rr.max()
    if n_bins is None:
        n_bins = max(32, int(max_r))
    bins = np.linspace(0, max_r + 1e-6, n_bins + 1)
    profile = np.zeros(n_bins, dtype=np.float32)

    for i in range(n_bins):
        m = (rr >= bins[i]) & (rr < bins[i + 1])
        if np.any(m):
            profile[i] = mag[m].mean()
    return profile


def smooth_profile(profile: np.ndarray, k: int = 5) -> np.ndarray:
    if k <= 1:
        return profile
    kernel = np.ones(k, dtype=np.float32) / float(k)
    return np.convolve(profile, kernel, mode="same")


def count_ring_peaks(profile: np.ndarray, min_peak_height: float = 0.08, min_peak_distance: int = 3) -> int:
    peaks = []
    for i in range(1, len(profile) - 1):
        if profile[i] > min_peak_height and profile[i] >= profile[i - 1] and profile[i] >= profile[i + 1]:
            if not peaks or i - peaks[-1] >= min_peak_distance:
                peaks.append(i)
            elif profile[i] > profile[peaks[-1]]:
                peaks[-1] = i
    return len(peaks)


def concentricity_score(gray: np.ndarray, center: tuple[float, float]) -> float:
    profile = smooth_profile(radial_profile(gray, center), k=7)
    if profile.max() <= 1e-6:
        return 0.0
    p = profile / (profile.max() + 1e-6)
    peak_count = count_ring_peaks(p, min_peak_height=0.12, min_peak_distance=3)
    # softly reward 3-10 visible rings in small noisy crops
    score = min(peak_count / 6.0, 1.0)
    return float(np.clip(score, 0.0, 1.0))


def radial_symmetry_score(gray: np.ndarray, center: tuple[float, float], n_angles: int = 24) -> float:
    gray_n = _normalize_gray(gray)
    h, w = gray.shape[:2]
    cx, cy = center
    max_r = int(min(cx, cy, w - cx - 1, h - cy - 1))
    if max_r < 8:
        return 0.0

    samples = []
    for theta in np.linspace(0, 2 * np.pi, n_angles, endpoint=False):
        vals = []
        for r in range(1, max_r):
            x = int(round(cx + r * math.cos(theta)))
            y = int(round(cy + r * math.sin(theta)))
            if 0 <= x < w and 0 <= y < h:
                vals.append(gray_n[y, x])
        if len(vals) >= 8:
            samples.append(np.array(vals, dtype=np.float32))
    if len(samples) < 4:
        return 0.0

    # compare all profiles to the mean truncated profile length
    min_len = min(len(v) for v in samples)
    arr = np.stack([v[:min_len] for v in samples], axis=0)
    mean_profile = arr.mean(axis=0, keepdims=True)
    dev = np.mean(np.abs(arr - mean_profile))
    return float(np.clip(1.0 - 2.0 * dev, 0.0, 1.0))


def ring_density_score(gray: np.ndarray, center: tuple[float, float]) -> float:
    profile = smooth_profile(radial_profile(gray, center), k=7)
    if profile.max() <= 1e-6:
        return 0.0
    p = profile / (profile.max() + 1e-6)
    peak_count = count_ring_peaks(p, min_peak_height=0.12, min_peak_distance=3)
    return float(np.clip(peak_count / 10.0, 0.0, 1.0))


def score_patch(gray: np.ndarray) -> Dict[str, float]:
    gray_u8 = (_normalize_gray(gray) * 255).astype(np.uint8)
    center = estimate_center_from_edges(gray_u8)
    _, th = cv2.threshold(gray_u8, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    binary = (gray_u8 >= th).astype(np.uint8)

    scores = {
        "circularity": circularity_score(binary),
        "concentricity": concentricity_score(gray_u8, center),
        "radial_symmetry": radial_symmetry_score(gray_u8, center),
        "ring_density": ring_density_score(gray_u8, center),
        "center_x": float(center[0]),
        "center_y": float(center[1]),
    }
    return scores
