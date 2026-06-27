from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DATA_DIR = PROJECT_ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = PROJECT_ROOT / "models"
REPORTS_DIR = PROJECT_ROOT / "reports"
DOCS_DIR = PROJECT_ROOT / "docs"

BRACOL_DATASET_ID = "yy2k5y8mxg"
BRACOL_VERSION = 1
BRACOL_DOI = "10.17632/yy2k5y8mxg.1"
BRACOL_PAGE = "https://data.mendeley.com/datasets/yy2k5y8mxg/1"
BRACOL_ZIP_URL = (
    f"https://data.mendeley.com/public-api/zip/{BRACOL_DATASET_ID}/download/{BRACOL_VERSION}"
)

CLASSES = [
    "healthy",
    "rust",
    "leaf_miner",
    "brown_leaf_spot",
    "cercospora_leaf_spot",
]

DISPLAY_NAMES = {
    "healthy": "Healthy",
    "rust": "Rust",
    "leaf_miner": "Leaf Miner",
    "brown_leaf_spot": "Brown Leaf Spot",
    "cercospora_leaf_spot": "Cercospora Leaf Spot",
}

ADVISORY_MESSAGES = {
    "healthy": "No visible disease pattern was detected in this image.",
    "rust": "Rust-like symptoms were detected. Expert confirmation and early management are recommended.",
    "leaf_miner": "Leaf miner-like damage was detected. Inspect nearby plants and monitor spread.",
    "brown_leaf_spot": "Brown leaf spot symptoms were detected. Monitor spread and seek agronomy guidance.",
    "cercospora_leaf_spot": "Cercospora-like symptoms were detected. Seek agronomy confirmation before treatment.",
}


@dataclass(frozen=True)
class TrainConfig:
    seed: int = 42
    image_size: int = 224
    batch_size: int = 32
    epochs: int = 20
    learning_rate: float = 3e-4
    weight_decay: float = 1e-4
    confidence_threshold: float = 0.60
    train_ratio: float = 0.70
    val_ratio: float = 0.15
    test_ratio: float = 0.15


TRAIN_CONFIG = TrainConfig()


def ensure_dirs() -> None:
    for path in [RAW_DIR, INTERIM_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, DOCS_DIR]:
        path.mkdir(parents=True, exist_ok=True)
