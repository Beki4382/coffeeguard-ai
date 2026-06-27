from __future__ import annotations

import json
from pathlib import Path

import torch
from PIL import Image

from .config import ADVISORY_MESSAGES, DISPLAY_NAMES, MODELS_DIR, TRAIN_CONFIG
from .gradcam import GradCAM, overlay_gradcam, tensor_from_image
from .model import get_device, load_checkpoint


class CoffeeGuardPredictor:
    def __init__(
        self,
        model_path: Path = MODELS_DIR / "best_model.pt",
        label_map_path: Path = MODELS_DIR / "label_map.json",
    ):
        self.device = get_device()
        self.label_map = json.loads(Path(label_map_path).read_text())
        self.classes = [self.label_map[str(i)] for i in range(len(self.label_map))]
        self.model, self.checkpoint = load_checkpoint(model_path, self.device, len(self.classes))

    def predict(self, image: Image.Image, with_gradcam: bool = True) -> dict:
        image = image.convert("RGB")
        tensor = tensor_from_image(image, self.device)
        with torch.no_grad():
            logits = self.model(tensor)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu()
        top_idx = int(torch.argmax(probs).item())
        confidence = float(probs[top_idx].item())
        label = self.classes[top_idx]
        display_name = DISPLAY_NAMES[label]
        is_uncertain = confidence < TRAIN_CONFIG.confidence_threshold
        top3 = sorted(
            [
                {"label": DISPLAY_NAMES[self.classes[i]], "probability": float(probs[i].item())}
                for i in range(len(self.classes))
            ],
            key=lambda item: item["probability"],
            reverse=True,
        )[:3]

        heatmap_image = None
        if with_gradcam:
            cam = GradCAM(self.model, self.model.features[-1])
            try:
                heatmap = cam(tensor, top_idx)
                heatmap_image = overlay_gradcam(image, heatmap)
            finally:
                cam.remove()

        return {
            "label": label,
            "display_name": "Uncertain prediction" if is_uncertain else display_name,
            "raw_display_name": display_name,
            "confidence": confidence,
            "top3": top3,
            "is_uncertain": is_uncertain,
            "advisory": "Prediction uncertain. Please upload a clearer coffee leaf image."
            if is_uncertain
            else ADVISORY_MESSAGES[label],
            "heatmap": heatmap_image,
        }
