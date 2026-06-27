from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
from PIL import Image
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from torch.utils.data import DataLoader
from tqdm import tqdm

from .config import CLASSES, DISPLAY_NAMES, MODELS_DIR, REPORTS_DIR, TRAIN_CONFIG, ensure_dirs
from .gradcam import GradCAM, overlay_gradcam, tensor_from_image
from .model import get_device, load_checkpoint
from .training_data import CoffeeLeafDataset
from .transforms import eval_transform


def save_confusion_matrix(y_true, y_pred, out_path: Path) -> None:
    labels = list(range(len(CLASSES)))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Greens",
        xticklabels=[DISPLAY_NAMES[c] for c in CLASSES],
        yticklabels=[DISPLAY_NAMES[c] for c in CLASSES],
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.xticks(rotation=35, ha="right")
    plt.yticks(rotation=0)
    plt.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, dpi=180)
    plt.close()


def main() -> None:
    ensure_dirs()
    out_dir = REPORTS_DIR / "evaluation"
    out_dir.mkdir(parents=True, exist_ok=True)
    device = get_device()
    model, checkpoint = load_checkpoint(MODELS_DIR / "best_model.pt", device, len(CLASSES))

    test_ds = CoffeeLeafDataset(split="test", transform=eval_transform())
    loader = DataLoader(test_ds, batch_size=TRAIN_CONFIG.batch_size, shuffle=False, num_workers=0)
    y_true, y_pred, confidences, paths = [], [], [], []
    all_probs = []
    with torch.no_grad():
        for images, labels, image_paths in tqdm(loader, leave=False):
            images = images.to(device)
            logits = model(images)
            probs = torch.softmax(logits, dim=1).cpu()
            preds = probs.argmax(dim=1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())
            confidences.extend(probs.max(dim=1).values.tolist())
            paths.extend(image_paths)
            all_probs.extend(probs.tolist())

    report = classification_report(
        y_true,
        y_pred,
        target_names=[DISPLAY_NAMES[c] for c in CLASSES],
        output_dict=True,
        zero_division=0,
    )
    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "weighted_f1": f1_score(y_true, y_pred, average="weighted", zero_division=0),
        "classification_report": report,
        "checkpoint": {
            "epoch": checkpoint.get("epoch"),
            "best_val_macro_f1": checkpoint.get("best_val_macro_f1"),
        },
    }
    (out_dir / "metrics.json").write_text(json.dumps(metrics, indent=2))
    pd.DataFrame(report).transpose().to_csv(out_dir / "classification_report.csv")
    save_confusion_matrix(y_true, y_pred, out_dir / "confusion_matrix.png")

    predictions = []
    for path, true_idx, pred_idx, conf, probs in zip(paths, y_true, y_pred, confidences, all_probs):
        predictions.append(
            {
                "image_path": path,
                "true_label": CLASSES[true_idx],
                "predicted_label": CLASSES[pred_idx],
                "confidence": conf,
                **{f"prob_{label}": probs[i] for i, label in enumerate(CLASSES)},
            }
        )
    pd.DataFrame(predictions).to_csv(out_dir / "test_predictions.csv", index=False)

    failures = [p for p in predictions if p["true_label"] != p["predicted_label"]]
    selected_failures = failures[:5]
    with (out_dir / "failure_cases.csv").open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "image_path",
                "true_label",
                "predicted_label",
                "confidence",
                "likely_reason",
            ],
        )
        writer.writeheader()
        for row in selected_failures:
            writer.writerow(
                {
                    "image_path": row["image_path"],
                    "true_label": row["true_label"],
                    "predicted_label": row["predicted_label"],
                    "confidence": row["confidence"],
                    "likely_reason": "Likely visual similarity, mild symptoms, class imbalance, or image quality/background variation.",
                }
            )

    gradcam_dir = out_dir / "gradcam"
    gradcam_dir.mkdir(exist_ok=True)
    cam = GradCAM(model, model.features[-1])
    try:
        examples = []
        for cls_idx, cls in enumerate(CLASSES):
            for row, true_idx, pred_idx in zip(predictions, y_true, y_pred):
                if true_idx == cls_idx and pred_idx == cls_idx:
                    examples.append((row, pred_idx, f"correct_{cls}"))
                    break
        for row in selected_failures[:2]:
            examples.append((row, CLASSES.index(row["predicted_label"]), f"failure_{Path(row['image_path']).stem}"))
        for row, class_idx, name in examples:
            image = Image.open(row["image_path"]).convert("RGB")
            tensor = tensor_from_image(image, device)
            heatmap = cam(tensor, class_idx)
            overlay = overlay_gradcam(image, heatmap)
            overlay.save(gradcam_dir / f"{name}.png")
    finally:
        cam.remove()

    print(json.dumps({k: metrics[k] for k in ["accuracy", "macro_f1", "weighted_f1"]}, indent=2))


if __name__ == "__main__":
    main()
