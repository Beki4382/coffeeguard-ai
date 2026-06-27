from __future__ import annotations

import csv
import random
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import pandas as pd
from PIL import Image

from .config import CLASSES, DISPLAY_NAMES, PROCESSED_DIR, REPORTS_DIR, TRAIN_CONFIG

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def is_image(path: Path) -> bool:
    return path.suffix.lower() in IMAGE_EXTENSIONS


def verify_image(path: Path) -> bool:
    try:
        with Image.open(path) as img:
            img.verify()
        with Image.open(path) as img:
            img.convert("RGB").load()
        return True
    except Exception:
        return False


def label_from_path(path: Path) -> str | None:
    normalized = " ".join(part.lower().replace("_", " ").replace("-", " ") for part in path.parts)
    filename = path.name.lower().replace("_", " ").replace("-", " ")

    if "healthy" in normalized or "sadias" in normalized or "sadia" in normalized:
        return "healthy"
    if "cercospora" in normalized:
        return "cercospora_leaf_spot"
    if "brown" in normalized or "phoma" in normalized:
        return "brown_leaf_spot"
    if "miner" in normalized or "mineiro" in normalized:
        return "leaf_miner"
    if "rust" in normalized or "ferrugem" in normalized:
        return "rust"

    if filename.startswith("healthy"):
        return "healthy"
    return None


def collect_labeled_images(root: Path) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file() or not is_image(path):
            continue
        label = label_from_path(path)
        if label not in CLASSES:
            continue
        if not verify_image(path):
            continue
        rows.append({"image_path": str(path), "label": label})
    return rows


BRACOL_PREDOMINANT_LABELS = {
    0: "healthy",
    1: "leaf_miner",
    2: "rust",
    3: "brown_leaf_spot",
    4: "cercospora_leaf_spot",
}


def collect_bracol_leaf_csv(leaf_dir: Path) -> list[dict[str, str]]:
    csv_path = leaf_dir / "dataset.csv"
    images_dir = leaf_dir / "images"
    if not csv_path.exists() or not images_dir.exists():
        return []

    df = pd.read_csv(csv_path)
    rows: list[dict[str, str]] = []
    for _, item in df.iterrows():
        label = BRACOL_PREDOMINANT_LABELS.get(int(item["predominant_stress"]))
        if label is None:
            continue
        image_path = images_dir / f"{int(item['id'])}.jpg"
        if not image_path.exists() or not verify_image(image_path):
            continue
        rows.append(
            {
                "image_path": str(image_path),
                "label": label,
                "source_id": str(int(item["id"])),
                "severity": str(int(item["severity"])),
            }
        )
    return rows


def stratified_split(rows: Iterable[dict[str, str]], seed: int = TRAIN_CONFIG.seed) -> list[dict[str, str]]:
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        grouped[row["label"]].append(row)

    rng = random.Random(seed)
    split_rows: list[dict[str, str]] = []
    for label in CLASSES:
        items = grouped[label]
        rng.shuffle(items)
        n = len(items)
        n_train = round(n * TRAIN_CONFIG.train_ratio)
        n_val = round(n * TRAIN_CONFIG.val_ratio)
        for idx, item in enumerate(items):
            if idx < n_train:
                split = "train"
            elif idx < n_train + n_val:
                split = "val"
            else:
                split = "test"
            split_rows.append({**item, "split": split})
    return sorted(split_rows, key=lambda r: (r["split"], r["label"], r["image_path"]))


def write_manifest(rows: list[dict[str, str]], path: Path = PROCESSED_DIR / "bracol_splits.csv") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["image_path", "label", "split", "source_id", "severity"],
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)


def write_audit(rows: list[dict[str, str]], path: Path = REPORTS_DIR / "dataset_audit.csv") -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    total_counts = Counter(row["label"] for row in rows)
    split_counts = Counter((row["label"], row["split"]) for row in rows)
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["label", "display_name", "total", "train", "val", "test"],
        )
        writer.writeheader()
        for label in CLASSES:
            writer.writerow(
                {
                    "label": label,
                    "display_name": DISPLAY_NAMES[label],
                    "total": total_counts[label],
                    "train": split_counts[(label, "train")],
                    "val": split_counts[(label, "val")],
                    "test": split_counts[(label, "test")],
                }
            )
