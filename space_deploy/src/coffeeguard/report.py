from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
from reportlab.lib import colors
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.units import inch
from reportlab.platypus import Image, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from .config import BRACOL_DOI, BRACOL_PAGE, DISPLAY_NAMES, REPORTS_DIR


def paragraph(text: str, style):
    return Paragraph(text.replace("\n", "<br/>"), style)


def add_table(story, rows):
    table = Table(rows, hAlign="LEFT")
    table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9ead3")),
                ("GRID", (0, 0), (-1, -1), 0.3, colors.grey),
                ("FONTNAME", (0, 0), (-1, 0), "Helvetica-Bold"),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 5),
                ("RIGHTPADDING", (0, 0), (-1, -1), 5),
            ]
        )
    )
    story.append(table)
    story.append(Spacer(1, 0.15 * inch))


def main() -> None:
    REPORTS_DIR.mkdir(parents=True, exist_ok=True)
    out_path = REPORTS_DIR / "final_report.pdf"
    doc = SimpleDocTemplate(str(out_path), pagesize=A4, rightMargin=44, leftMargin=44, topMargin=44, bottomMargin=44)
    styles = getSampleStyleSheet()
    story = []

    story.append(Paragraph("CoffeeGuard AI: Coffee Leaf Disease Classification", styles["Title"]))
    story.append(Paragraph("Computer Vision Capstone Project Documentation", styles["Heading2"]))
    story.append(Spacer(1, 0.2 * inch))

    sections = [
        (
            "Introduction",
            "Coffee leaf diseases reduce productivity and quality in coffee-growing regions. This project builds a practical computer vision decision-support application that classifies Arabica coffee leaf images into healthy and disease categories using transfer learning.",
        ),
        (
            "Problem Statement",
            "Given a coffee leaf image, the system will classify the likely coffee leaf disease so that farmers, students, or extension workers can identify symptoms earlier and decide whether expert follow-up is needed.",
        ),
        (
            "Methodology",
            "The project uses the BRACOL Leaf Dataset only. Images are split into train, validation, and test sets with seed 42. EfficientNet-B0 pretrained on ImageNet is fine-tuned with a five-class output layer. The best checkpoint is selected by validation macro-F1 and evaluated once on the untouched test set.",
        ),
        (
            "Proposed Solution",
            "The solution is a Gradio web application. A user uploads one coffee leaf image. The app preprocesses the image, runs EfficientNet-B0 inference, applies a confidence threshold, and returns the predicted disease, confidence score, top-3 probabilities, Grad-CAM explanation, and an advisory message.",
        ),
        (
            "System Architecture",
            "User upload -> Gradio frontend -> preprocessing pipeline -> EfficientNet-B0 model -> softmax probabilities -> confidence threshold -> prediction, advisory message, and Grad-CAM heatmap.",
        ),
        (
            "Implementation Details",
            "The implementation uses PyTorch, torchvision, scikit-learn, Gradio, and ReportLab. Training uses 224x224 images, AdamW, weighted cross-entropy, data augmentation, and a single EfficientNet-B0 training run.",
        ),
    ]
    for title, body in sections:
        story.append(Paragraph(title, styles["Heading1"]))
        story.append(paragraph(body, styles["BodyText"]))
        story.append(Spacer(1, 0.15 * inch))

    audit_path = REPORTS_DIR / "dataset_audit.csv"
    if audit_path.exists():
        audit = pd.read_csv(audit_path)
        story.append(Paragraph("Dataset Audit", styles["Heading1"]))
        rows = [["Class", "Total", "Train", "Validation", "Test"]]
        for _, row in audit.iterrows():
            rows.append([row["display_name"], int(row["total"]), int(row["train"]), int(row["val"]), int(row["test"])])
        add_table(story, rows)

    metrics_path = REPORTS_DIR / "evaluation" / "metrics.json"
    if metrics_path.exists():
        metrics = json.loads(metrics_path.read_text())
        story.append(Paragraph("Results", styles["Heading1"]))
        rows = [
            ["Metric", "Value"],
            ["Accuracy", f"{metrics['accuracy']:.4f}"],
            ["Macro-F1", f"{metrics['macro_f1']:.4f}"],
            ["Weighted-F1", f"{metrics['weighted_f1']:.4f}"],
            ["Best Validation Macro-F1", f"{metrics['checkpoint'].get('best_val_macro_f1', 0):.4f}"],
            ["Best Epoch", str(metrics["checkpoint"].get("epoch"))],
        ]
        add_table(story, rows)
        cm_path = REPORTS_DIR / "evaluation" / "confusion_matrix.png"
        if cm_path.exists():
            story.append(Paragraph("Confusion Matrix", styles["Heading2"]))
            story.append(Image(str(cm_path), width=6.8 * inch, height=5.0 * inch))
            story.append(Spacer(1, 0.15 * inch))
    else:
        story.append(Paragraph("Results", styles["Heading1"]))
        story.append(paragraph("Results will be finalized after the single training run and test-set evaluation.", styles["BodyText"]))

    failure_path = REPORTS_DIR / "evaluation" / "failure_cases.csv"
    if failure_path.exists():
        failures = pd.read_csv(failure_path)
        story.append(Paragraph("Failure Case Analysis", styles["Heading1"]))
        rows = [["True Label", "Predicted Label", "Confidence", "Likely Reason"]]
        for _, row in failures.head(5).iterrows():
            rows.append(
                [
                    DISPLAY_NAMES.get(row["true_label"], row["true_label"]),
                    DISPLAY_NAMES.get(row["predicted_label"], row["predicted_label"]),
                    f"{row['confidence']:.2%}",
                    row["likely_reason"],
                ]
            )
        add_table(story, rows)

    story.append(Paragraph("Deployment", styles["Heading1"]))
    story.append(
        paragraph(
            "The application is designed for Gradio on Hugging Face Spaces. Final submission must include the working app link, GitHub repository link, and a demo video under 1 minute 30 seconds.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("Conclusion", styles["Heading1"]))
    story.append(
        paragraph(
            "CoffeeGuard AI demonstrates an end-to-end computer vision workflow: dataset preparation, transfer learning, evaluation, explainability, application building, and deployment packaging. It is a decision-support tool and not a replacement for expert agronomy diagnosis.",
            styles["BodyText"],
        )
    )

    story.append(Paragraph("References", styles["Heading1"]))
    refs = [
        f"Krohling, R. A., Esgario, G. J. M., and Ventura, J. A. BRACOL dataset. Mendeley Data. DOI: {BRACOL_DOI}. {BRACOL_PAGE}",
        "Tan, M. and Le, Q. EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks.",
        "PyTorch and torchvision documentation.",
        "Gradio documentation for model deployment.",
    ]
    for ref in refs:
        story.append(paragraph(ref, styles["BodyText"]))
        story.append(Spacer(1, 0.08 * inch))

    doc.build(story)
    print(out_path)


if __name__ == "__main__":
    main()
