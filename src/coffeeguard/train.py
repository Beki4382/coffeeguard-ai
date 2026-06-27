from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CLASSES, MODELS_DIR, REPORTS_DIR, TRAIN_CONFIG, ensure_dirs
from .model import create_model, get_device
from .training_data import CoffeeLeafDataset, class_weights
from .transforms import eval_transform, train_transform


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def run_epoch(model, loader, criterion, optimizer, device, training: bool):
    model.train(training)
    losses = []
    y_true = []
    y_pred = []
    for images, labels, _ in tqdm(loader, leave=False):
        images = images.to(device)
        labels = labels.to(device)
        optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            logits = model(images)
            loss = criterion(logits, labels)
            if training:
                loss.backward()
                optimizer.step()
        losses.append(loss.detach().cpu().item())
        preds = torch.argmax(logits.detach(), dim=1)
        y_true.extend(labels.detach().cpu().tolist())
        y_pred.extend(preds.cpu().tolist())
    accuracy = float(np.mean(np.array(y_true) == np.array(y_pred)))
    macro_f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
    return {"loss": float(np.mean(losses)), "accuracy": accuracy, "macro_f1": macro_f1}


def plot_history(history: list[dict], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(history)
    for metric in ["loss", "accuracy", "macro_f1"]:
        plt.figure(figsize=(7, 4))
        plt.plot(df["epoch"], df[f"train_{metric}"], label=f"train {metric}")
        plt.plot(df["epoch"], df[f"val_{metric}"], label=f"val {metric}")
        plt.xlabel("Epoch")
        plt.ylabel(metric)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_dir / f"training_{metric}.png", dpi=180)
        plt.close()
    df.to_csv(out_dir / "training_history.csv", index=False)


def main() -> None:
    ensure_dirs()
    set_seed(TRAIN_CONFIG.seed)
    device = get_device()
    print(f"Using device: {device}")

    train_ds = CoffeeLeafDataset(split="train", transform=train_transform())
    val_ds = CoffeeLeafDataset(split="val", transform=eval_transform())
    train_loader = DataLoader(train_ds, batch_size=TRAIN_CONFIG.batch_size, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=TRAIN_CONFIG.batch_size, shuffle=False, num_workers=0)

    model = create_model(num_classes=len(CLASSES), pretrained=True).to(device)
    weights = class_weights().to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=TRAIN_CONFIG.learning_rate,
        weight_decay=TRAIN_CONFIG.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=TRAIN_CONFIG.epochs)

    best_f1 = -1.0
    history = []
    label_map = {str(idx): label for idx, label in enumerate(CLASSES)}
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)

    config_payload = {
        "model": "EfficientNet-B0",
        "classes": CLASSES,
        "train_config": TRAIN_CONFIG.__dict__,
        "criterion": "weighted_cross_entropy",
        "optimizer": "AdamW",
        "scheduler": "CosineAnnealingLR",
        "checkpoint_rule": "best validation macro-F1",
    }
    (MODELS_DIR / "label_map.json").write_text(json.dumps(label_map, indent=2))
    (MODELS_DIR / "training_config.json").write_text(json.dumps(config_payload, indent=2))

    for epoch in range(1, TRAIN_CONFIG.epochs + 1):
        print(f"Epoch {epoch}/{TRAIN_CONFIG.epochs}")
        train_metrics = run_epoch(model, train_loader, criterion, optimizer, device, training=True)
        val_metrics = run_epoch(model, val_loader, criterion, optimizer, device, training=False)
        scheduler.step()
        row = {
            "epoch": epoch,
            **{f"train_{k}": v for k, v in train_metrics.items()},
            **{f"val_{k}": v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(row)
        if val_metrics["macro_f1"] > best_f1:
            best_f1 = val_metrics["macro_f1"]
            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "classes": CLASSES,
                    "epoch": epoch,
                    "best_val_macro_f1": best_f1,
                    "train_config": TRAIN_CONFIG.__dict__,
                },
                MODELS_DIR / "best_model.pt",
            )

    plot_history(history, REPORTS_DIR / "training")
    print(f"Best validation macro-F1: {best_f1:.4f}")


if __name__ == "__main__":
    main()
