# CoffeeGuard AI

CoffeeGuard AI is a computer vision capstone project for coffee leaf disease decision support. It uses one public dataset, one transfer-learning training run, and a deployed Gradio app.

## Fixed Project Choices

- Dataset: BRACOL Leaf Dataset, DOI `10.17632/yy2k5y8mxg.1`
- Model: EfficientNet-B0 pretrained on ImageNet
- Task: five-class coffee leaf classification
- Deployment target: Gradio on Hugging Face Spaces

## Classes

- Healthy
- Rust
- Leaf Miner
- Brown Leaf Spot
- Cercospora Leaf Spot

## Setup

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install -e .
```

## Pipeline

Download BRACOL:

```bash
.venv/bin/python scripts/download_bracol.py
```

Prepare the fixed train/validation/test split:

```bash
.venv/bin/python scripts/prepare_bracol.py
```

Train exactly one model:

```bash
.venv/bin/python -m coffeeguard.train
```

Evaluate on the untouched test split:

```bash
.venv/bin/python -m coffeeguard.evaluate
```

Run the app locally:

```bash
.venv/bin/python app.py
```

Build the report PDF after training/evaluation:

```bash
.venv/bin/python -m coffeeguard.report
```

## Outputs

- `data/processed/bracol_splits.csv`: fixed split manifest
- `reports/dataset_audit.csv`: class counts and split counts
- `models/best_model.pt`: best EfficientNet-B0 checkpoint by validation macro-F1
- `models/label_map.json`: class-index mapping
- `reports/evaluation/`: metrics, confusion matrix, Grad-CAM, failure cases
- `reports/final_report.pdf`: assignment-aligned documentation draft

## Deployment

For Hugging Face Spaces, upload:

- `app.py`
- `src/`
- `requirements.txt`
- `models/best_model.pt`
- `models/label_map.json`

The app is a decision-support tool and does not replace expert agronomy diagnosis.
