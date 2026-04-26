"""Central project configuration for labyrinth detection.

Edit these defaults once, then override from scripts when needed.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import List


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "outputs"
MODEL_DIR = PROJECT_ROOT / "models"


@dataclass
class PathsConfig:
    project_root: Path = PROJECT_ROOT
    data_dir: Path = DATA_DIR
    output_dir: Path = OUTPUT_DIR
    model_dir: Path = MODEL_DIR

    patch_dataset_dir: Path = DATA_DIR / "patch_dataset"
    patch_csv: Path = DATA_DIR / "patch_dataset" / "patches.csv"
    train_csv: Path = DATA_DIR / "patch_dataset" / "train.csv"
    val_csv: Path = DATA_DIR / "patch_dataset" / "val.csv"

    train_rgb_dir: Path = DATA_DIR / "train" / "rgb"
    train_terrain_dir: Path = DATA_DIR / "train" / "terrain"
    train_mask_dir: Path = DATA_DIR / "train" / "masks"
    val_rgb_dir: Path = DATA_DIR / "val" / "rgb"
    val_terrain_dir: Path = DATA_DIR / "val" / "terrain"
    val_mask_dir: Path = DATA_DIR / "val" / "masks"

    dem_input: Path = DATA_DIR / "dem" / "example_dem.tif"
    dem_derived_dir: Path = DATA_DIR / "dem" / "derived"

    test_raster: Path = DATA_DIR / "test" / "example.tif"
    raw_candidates_npy: Path = OUTPUT_DIR / "candidates.npy"
    candidates_csv: Path = OUTPUT_DIR / "candidates.csv"
    ranked_candidates_csv: Path = OUTPUT_DIR / "candidates_ranked.csv"
    false_positive_csv: Path = OUTPUT_DIR / "false_positives.csv"
    hard_negative_dir: Path = DATA_DIR / "hard_negatives"


@dataclass
class ClassifierConfig:
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 20
    lr: float = 1e-4
    num_workers: int = 2
    checkpoint: str = "classifier_baseline.pt"


@dataclass
class NazcaClassifierConfig:
    image_size: int = 256
    batch_size: int = 64
    epochs: int = 30
    lr: float = 1e-4
    num_workers: int = 2
    checkpoint: str = "nazca_labyrinth_classifier.pt"


@dataclass
class SegmenterConfig:
    image_size: int = 256
    batch_size: int = 8
    epochs: int = 40
    lr: float = 1e-4
    num_workers: int = 2
    encoder_name: str = "resnet34"
    checkpoint: str = "segmenter_best.pt"


@dataclass
class ScaleConfig:
    patch_px: int
    stride_px: int
    prob_threshold: float
    min_neighbors: int


@dataclass
class SearchConfig:
    scales: List[ScaleConfig] = field(default_factory=lambda: [
        ScaleConfig(128, 10, 0.50, 3),
        ScaleConfig(256, 20, 0.50, 4),
        ScaleConfig(384, 30, 0.52, 5),
    ])

    # geology / archaeology-aware ranking weights
    score_weight: float = 0.40
    circularity_weight: float = 0.10
    concentricity_weight: float = 0.25
    radial_symmetry_weight: float = 0.15
    ring_count_weight: float = 0.10


PATHS = PathsConfig()
CLASSIFIER = ClassifierConfig()
NAZCA = NazcaClassifierConfig()
SEGMENTER = SegmenterConfig()
SEARCH = SearchConfig()
