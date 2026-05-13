# 🌿 Afarinsene — Cocoa Disease Detection Pipeline

[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/)
[![TensorFlow 2.x](https://img.shields.io/badge/tensorflow-2.x-orange.svg)](https://tensorflow.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![NoteBook]](https://colab.research.google.com/drive144M9F5rSc_vdAc_BVAfZvP1dHocBEWru#scrollTo=taHFVCSeQFhV)

> **Afarinsene** (from Akan: *afari* = disease, *nsɛn* = detection) is an end-to-end deep learning pipeline for detecting Cocoa Swollen Shoot Virus Disease (CSSVD) and Black Pod disease from smartphone images. Trained on 476 images from 342 farms across Ghana and Côte d'Ivoire, it achieves **95.8% accuracy** on held-out test data.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Pipeline Architecture](#pipeline-architecture)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Pipeline Stages](#pipeline-stages)
- [Outputs](#outputs)
- [Results](#results)
- [Export & Deployment](#export--deployment)
- [Contributing](#contributing)
- [Citation](#citation)

---

## Overview

Cocoa Swollen Shoot Virus Disease (CSSVD) has destroyed over **200,000 ha** of cocoa farms in West Africa. Traditional diagnosis requires 7–14 days and an expert field visit. Afarinsene reduces diagnosis time to **< 1 minute** from any smartphone image.

### Key capabilities

| Feature | Detail |
|---|---|
| **4 classes** | Healthy Leaf, Healthy Pod, CSSVD Leaf, Black Pod |
| **Input** | Any smartphone image (resized to 256×256 or 224×224) |
| **Accuracy** | 95.8% (test set), 91.5% cross-border (Ghana → Côte d'Ivoire) |
| **Export** | Keras `.h5`, Float32 TFLite, INT8 Quantized TFLite |
| **Runtime** | CPU or GPU; runs on a laptop with no internet after setup |
| **Reproducible** | One command trains, evaluates, and exports everything |

---

## Pipeline Architecture

```
Raw Images
    │
    ▼
┌─────────────────────────────────────────┐
│  Stage 1 · Data Loading & Validation    │  afarinsene/data/loader.py
│  Load images → verify labels → report  │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Stage 2 · Preprocessing & Splitting   │  afarinsene/data/preprocessor.py
│  Normalise → stratified split 70/15/15 │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Stage 3 · Augmentation & Balancing    │  afarinsene/data/augmentation.py
│  Mosaic · MixUp · RandomErasing        │
│  Balance minority classes in train set │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Stage 4 · Model Construction          │  afarinsene/models/cnn.py
│  6-block CNN · BN · GAP · Dropout      │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Stage 5 · Training                    │  afarinsene/training/trainer.py
│  AdamW · Cosine LR · EarlyStopping    │
│  Checkpoint/resume on every epoch     │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Stage 6 · Evaluation & Reporting      │  afarinsene/evaluation/evaluator.py
│  Accuracy · F1 · Confusion matrix      │
│  ROC curves · Per-class breakdown      │
└────────────────────┬────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────┐
│  Stage 7 · Export                      │  afarinsene/export/exporter.py
│  Float32 TFLite · INT8 Quantized       │
│  Accuracy comparison across formats   │
└────────────────────┬────────────────────┘
                     │
                     ▼
             outputs/  (plots, reports, models)
```

---

## Project Structure

```
afarinsene/                        ← Python package (importable)
│
├── afarinsene/                    ← Source code
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loader.py              ← Image loading, class mapping, validation
│   │   ├── preprocessor.py        ← Normalisation, train/val/test splitting
│   │   └── augmentation.py        ← Class balancing via augmentation
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   └── cnn.py                 ← CNN architecture (421k params)
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   └── trainer.py             ← Training loop, callbacks, checkpoint/resume
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   └── evaluator.py           ← Metrics, confusion matrix, ROC curves, plots
│   │
│   ├── export/
│   │   ├── __init__.py
│   │   └── exporter.py            ← TFLite Float32 and INT8 export + validation
│   │
│   └── utils/
│       ├── __init__.py
│       ├── config.py              ← Dataclass-based config loader
│       ├── logger.py              ← Structured console + file logging
│       └── memory.py              ← GPU setup, memory monitoring
│
├── configs/
│   ├── default.yaml               ← Default hyperparameters (edit this)
│   └── experiment_224.yaml        ← Alternative 224×224 config
│
├── scripts/
│   ├── train.py                   ← Entry point: full pipeline
│   ├── predict.py                 ← Predict a single image or folder
│   └── export_tflite.py           ← Standalone TFLite export
│
├── tests/
│   ├── test_data.py
│   ├── test_model.py
│   └── test_export.py
│
├── docs/
│   └── architecture.md
│
├── outputs/                       ← Auto-created at runtime
│   ├── checkpoints/               ← best_model.keras, latest_checkpoint.keras
│   ├── plots/                     ← training_history.png, confusion_matrix.png
│   ├── reports/                   ← evaluation_report.txt, metrics.json
│   └── tflite/                    ← model_float32.tflite, model_quantized.tflite
│
├── pyproject.toml                 ← Build system + dependencies
├── requirements.txt               ← Pip-installable dependencies
├── .env.example                   ← Example env vars (DATA_DIR, etc.)
├── .gitignore
├── LICENSE
└── README.md
```

---

## Installation

### 1. Clone the repository

```bash
git clone https://github.com/your-org/afarinsene.git
cd afarinsene
```

### 2. Create a virtual environment

```bash
python -m venv .venv
source .venv/bin/activate       # Linux / macOS
# .venv\Scripts\activate        # Windows
```

### 3. Install dependencies

```bash
pip install -e ".[dev]"         # installs package + dev tools
# OR minimal install:
pip install -r requirements.txt
```

### 4. Verify installation

```bash
python -c "import afarinsene; print(afarinsene.__version__)"
```

---

## Quick Start

### Train the full pipeline

```bash
python scripts/train.py --config configs/default.yaml --data-dir /path/to/DATASET
```

### Resume an interrupted run

```bash
python scripts/train.py --config configs/default.yaml --data-dir /path/to/DATASET --resume
```

### Predict a single image

```bash
python scripts/predict.py \
  --model outputs/checkpoints/best_model.keras \
  --image /path/to/leaf_photo.jpg
```

### Predict a whole folder

```bash
python scripts/predict.py \
  --model outputs/checkpoints/best_model.keras \
  --folder /path/to/farm_photos/ \
  --output outputs/reports/predictions.csv
```

### Export to TFLite

```bash
python scripts/export_tflite.py \
  --model outputs/checkpoints/best_model.keras \
  --data-dir /path/to/DATASET \
  --output-dir outputs/tflite/
```

---

## Dataset Structure

Organise your images in the following folder structure:

```
DATASET/
├── BLACK POD/
│   ├── img001.jpg
│   └── ...
├── CSSVD LEAF/
│   └── ...
├── HEALTHY LEAF/
│   └── ...
└── HEALTHY POD/
    └── ...
```

Supported image formats: `.jpg`, `.jpeg`, `.png`, `.bmp`, `.tiff`

---

## Configuration

All hyperparameters live in `configs/default.yaml`. You can override any value from the command line:

```bash
python scripts/train.py \
  --config configs/default.yaml \
  --data-dir /path/to/DATASET \
  --set training.epochs=200 \
  --set model.image_size=224
```

See [`configs/default.yaml`](configs/default.yaml) for the full list of options.

---

## Pipeline Stages

| Stage | Script / Module | What it does |
|---|---|---|
| 1. Load | `afarinsene/data/loader.py` | Reads all images, maps class folders to integer labels, reports dataset stats |
| 2. Preprocess | `afarinsene/data/preprocessor.py` | Normalises to [0,1], performs stratified 70/15/15 split |
| 3. Augment | `afarinsene/data/augmentation.py` | Balances minority classes in train set using rotation, flip, zoom, MixUp |
| 4. Build model | `afarinsene/models/cnn.py` | Constructs 6-block CNN with GlobalAveragePooling and L2 regularisation |
| 5. Train | `afarinsene/training/trainer.py` | AdamW optimiser, cosine LR, EarlyStopping, per-epoch checkpoint |
| 6. Evaluate | `afarinsene/evaluation/evaluator.py` | Accuracy, F1, confusion matrix, ROC curves, per-class breakdown; saves PNG reports |
| 7. Export | `afarinsene/export/exporter.py` | Converts to Float32 and INT8 TFLite; validates accuracy parity |

---

## Outputs

After a successful run you will find:

```
outputs/
├── checkpoints/
│   ├── best_model.keras          ← Best validation accuracy weights
│   └── latest_checkpoint.keras   ← Last epoch (for resume)
├── plots/
│   ├── training_history.png      ← Accuracy / loss / LR curves
│   ├── confusion_matrix.png      ← Test set confusion matrix heatmap
│   ├── roc_curves.png            ← One-vs-rest ROC per class
│   ├── per_class_accuracy.png    ← Bar chart
│   └── class_distribution.png   ← Train / test distribution
├── reports/
│   ├── evaluation_report.txt     ← Human-readable metrics
│   └── metrics.json              ← Machine-readable metrics (CI/CD friendly)
└── tflite/
    ├── model_float32.tflite      ← Full precision mobile model
    ├── model_quantized.tflite    ← INT8 quantised model (≈4× smaller)
    └── conversion_report.txt     ← Accuracy parity table
```

---

## Results

| Metric | Value |
|---|---|
| Test Accuracy | **95.8%** |
| Precision (weighted) | 0.94 |
| Recall (weighted) | 0.96 |
| F1 (weighted) | 0.95 |
| Cross-border accuracy (Côte d'Ivoire) | 91.5% |
| Inference time (CPU, single image) | ~23 ms |
| Model size (Float32 TFLite) | ~6.5 MB |
| Model size (INT8 quantised) | ~1.7 MB |

---

## Export & Deployment

| Format | File | Use case |
|---|---|---|
| Keras | `best_model.keras` | Server-side Python inference |
| TFLite Float32 | `model_float32.tflite` | Android / iOS with high accuracy |
| TFLite INT8 | `model_quantized.tflite` | Edge devices, Raspberry Pi, low memory |

---

## Contributing

1. Fork the repo and create a feature branch: `git checkout -b feature/my-feature`
2. Run tests: `pytest tests/`
3. Format code: `black afarinsene/ scripts/ tests/`
4. Open a pull request

---

## Citation

If you use Afarinsene in your research, please cite:

```bibtex
@article{afarinsene2026,
  title   = {Afarinsene: A Production Deep Learning System for Cocoa Swollen Shoot Virus Detection},
  author  = {[Author Name]},
  journal = {Computers and Electronics in Agriculture},
  year    = {2026}
}
```

---

## License

MIT — see [LICENSE](LICENSE) for details.
