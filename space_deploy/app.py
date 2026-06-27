from __future__ import annotations

import sys
from pathlib import Path

import gradio as gr
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from coffeeguard.config import TRAIN_CONFIG
from coffeeguard.inference import CoffeeGuardPredictor

predictor = None


def get_predictor() -> CoffeeGuardPredictor:
    global predictor
    if predictor is None:
        predictor = CoffeeGuardPredictor()
    return predictor


def predict(image):
    if image is None:
        return "No image uploaded", "", None
    try:
        pil_image = Image.fromarray(image).convert("RGB")
        result = get_predictor().predict(pil_image, with_gradcam=True)
        top3 = "\n".join(f"{item['label']}: {item['probability']:.2%}" for item in result["top3"])
        summary = (
            f"Prediction: {result['display_name']}\n"
            f"Model class: {result['raw_display_name']}\n"
            f"Confidence: {result['confidence']:.2%}\n"
            f"Threshold: {TRAIN_CONFIG.confidence_threshold:.0%}\n\n"
            f"{result['advisory']}"
        )
        return summary, top3, result["heatmap"]
    except Exception as exc:
        return f"Could not process this image: {exc}", "", None


with gr.Blocks(title="CoffeeGuard AI") as demo:
    gr.Markdown(
        """
        # CoffeeGuard AI
        Upload one coffee leaf image to classify visible disease symptoms. This is a decision-support tool, not a definitive agricultural diagnosis.
        """
    )
    with gr.Row():
        image_input = gr.Image(label="Coffee leaf image", type="numpy")
        with gr.Column():
            prediction_output = gr.Textbox(label="Prediction", lines=8)
            top3_output = gr.Textbox(label="Top-3 probabilities", lines=4)
    gradcam_output = gr.Image(label="Grad-CAM explanation", type="pil")
    predict_btn = gr.Button("Analyze leaf")
    predict_btn.click(predict, inputs=image_input, outputs=[prediction_output, top3_output, gradcam_output])


if __name__ == "__main__":
    demo.launch()
