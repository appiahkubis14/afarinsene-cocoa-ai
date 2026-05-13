#!/usr/bin/env python3
"""
scripts/export_tflite.py
========================
Standalone TFLite export script — use this when you have a trained model
and want to (re-)export without rerunning the full pipeline.

Usage
-----
::

    python scripts/export_tflite.py \\
        --model outputs/checkpoints/best_model.keras \\
        --data-dir /path/to/DATASET \\
        --output-dir outputs/tflite/
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args():
    p = argparse.ArgumentParser(description="Afarinsene — Standalone TFLite Export")
    p.add_argument("--model", required=True, help="Path to trained .keras model")
    p.add_argument("--data-dir", required=True, help="Dataset root (for calibration)")
    p.add_argument("--output-dir", default="outputs/tflite/")
    p.add_argument("--config", default="configs/default.yaml")
    p.add_argument("--cal-samples", type=int, default=200,
                   help="Number of calibration samples for INT8 quantisation")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    from afarinsene.utils.config import load_config
    from afarinsene.utils.logger import get_logger, configure_from_config
    from afarinsene.data.loader import DatasetLoader
    from afarinsene.data.preprocessor import Preprocessor
    from afarinsene.export.exporter import Exporter

    cfg = load_config(args.config, data_dir=args.data_dir)
    cfg["export"]["output_dir"] = args.output_dir
    cfg["export"]["quantized"]["calibration_samples"] = args.cal_samples
    configure_from_config(cfg)
    log = get_logger("afarinsene.export_standalone")

    log.info("Loading model from: %s", args.model)
    import tensorflow as tf
    model = tf.keras.models.load_model(args.model)
    model.summary()

    log.info("Loading dataset for calibration: %s", args.data_dir)
    loader = DatasetLoader(cfg)
    X_raw, y, _ = loader.load()

    pre = Preprocessor(cfg)
    splits = pre.split(X_raw, y)
    del X_raw

    exp = Exporter(cfg)
    results = exp.export(model, splits)

    log.info("Export complete.")
    for fmt, info in results.items():
        log.info("  %s → %s", fmt, info.get("path"))


if __name__ == "__main__":
    main()
