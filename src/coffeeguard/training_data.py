from __future__ import annotations

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset

from .config import CLASSES, PROCESSED_DIR


class CoffeeLeafDataset(Dataset):
    def __init__(self, manifest_path=PROCESSED_DIR / "bracol_splits.csv", split="train", transform=None):
        self.df = pd.read_csv(manifest_path)
        self.df = self.df[self.df["split"] == split].reset_index(drop=True)
        self.transform = transform
        self.label_to_idx = {label: idx for idx, label in enumerate(CLASSES)}

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row["image_path"]).convert("RGB")
        if self.transform:
            image = self.transform(image)
        label = torch.tensor(self.label_to_idx[row["label"]], dtype=torch.long)
        return image, label, row["image_path"]


def class_weights(manifest_path=PROCESSED_DIR / "bracol_splits.csv") -> torch.Tensor:
    df = pd.read_csv(manifest_path)
    train = df[df["split"] == "train"]
    counts = train["label"].value_counts().to_dict()
    weights = []
    total = len(train)
    for label in CLASSES:
        weights.append(total / (len(CLASSES) * counts[label]))
    return torch.tensor(weights, dtype=torch.float32)
