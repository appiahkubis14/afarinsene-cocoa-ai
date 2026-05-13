#!/usr/bin/env python3
"""
scripts/train.py
================
Full Afarinsene pipeline entry point.

Runs all 7 stages in sequence:
  1. Load dataset
  2. Preprocess & split
  3. Augment & balance training set
  4. Build CNN model
  5. Train with checkpoint/resume
  6. Evaluate and produce reports
  7. Export to TFLite

Usage
-----
Basic run::

    python scripts/train.py --config configs/default.yaml --data-dir /path/to/DATASET

Resume an interrupted run::

    python scripts/train.py --config configs/default.yaml --data-dir /path/to/DATASET --resume

Override any config value::

    python scripts/train.py \\
        --config configs/default.yaml \\
        --data-dir /path/to/DATASET \\
        --set training.epochs=200 \\
        --set model.image_size=224 \\
        --set training.batch_size=16

Skip export::

    python scripts/train.py --config configs/default.yaml --data-dir /path/to/DATASET --no-export
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Make sure the project root is on sys.path so `afarinsene` package imports.
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Afarinsene — Cocoa Disease Detection Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument(
        "--config", "-c",
        default="configs/default.yaml",
        help="Path to YAML config file (default: configs/default.yaml)",
    )
    p.add_argument(
        "--data-dir", "-d",
        default=None,
        help="Root data directory (overrides config and DATA_DIR env var)",
    )
    p.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from the latest checkpoint",
    )
    p.add_argument(
        "--no-export",
        action="store_true",
        help="Skip TFLite export (Stage 7)",
    )
    p.add_argument(
        "--set",
        metavar="KEY=VALUE",
        action="append",
        default=[],
        help="Override a config value, e.g. --set training.epochs=200",
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()

    # ---- Config ----------------------------------------------------------
    from afarinsene.utils.config import load_config, pretty_print
    from afarinsene.utils.logger import get_logger, configure_from_config
    from afarinsene.utils.memory import setup_gpu, show_memory

    cfg = load_config(args.config, overrides=args.set, data_dir=args.data_dir)
    configure_from_config(cfg)
    log = get_logger("afarinsene.pipeline")

    log.info("=" * 55)
    log.info("  🌿  Afarinsene — Cocoa Disease Detection Pipeline")
    log.info("=" * 55)
    log.info("Config : %s", Path(args.config).resolve())
    log.info("Data   : %s", cfg["data"]["dir"])
    log.info("Resume : %s", args.resume)
    log.info("")

    t_start = time.perf_counter()

    # ---- GPU / memory setup ----------------------------------------------
    setup_gpu()
    show_memory("startup")

    # ---- Stage 1: Load ---------------------------------------------------
    from afarinsene.data.loader import DatasetLoader

    loader = DatasetLoader(cfg)
    X_raw, y, class_names = loader.load()
    show_memory("after load")

    # ---- Stage 2: Preprocess & Split ------------------------------------
    from afarinsene.data.preprocessor import Preprocessor

    pre = Preprocessor(cfg)
    splits = pre.split(X_raw, y)
    del X_raw  # free raw uint8 array
    show_memory("after split")

    # ---- Stage 3: Augment -----------------------------------------------
    from afarinsene.data.augmentation import Augmenter

    aug = Augmenter(cfg)
    X_train_bal, y_train_bal = aug.balance(splits.X_train, splits.y_train)
    show_memory("after augmentation")

    # Patch splits with balanced training data
    from tensorflow.keras.utils import to_categorical
    import numpy as np

    splits.X_train = X_train_bal
    splits.y_train = y_train_bal
    splits.y_train_cat = to_categorical(y_train_bal, splits.num_classes)

    # ---- Stage 4: Build model -------------------------------------------
    from afarinsene.models.cnn import build_model

    model = build_model(cfg, num_classes=splits.num_classes)
    model.summary(print_fn=lambda s: log.info("  %s", s))

    # ---- Stage 5: Train -------------------------------------------------
    from afarinsene.training.trainer import Trainer

    trainer = Trainer(cfg)
    history = trainer.train(model, splits, resume=args.resume)
    show_memory("after training")

    # ---- Stage 6: Evaluate ----------------------------------------------
    from afarinsene.evaluation.evaluator import Evaluator

    ev = Evaluator(cfg)
    metrics = ev.evaluate(model, splits, history)

    # ---- Stage 7: Export (optional) ------------------------------------
    if not args.no_export:
        from afarinsene.export.exporter import Exporter

        exp = Exporter(cfg)
        exp.export(model, splits)

    # ---- Summary --------------------------------------------------------
    elapsed = time.perf_counter() - t_start
    log.info("")
    log.info("=" * 55)
    log.info("  Pipeline complete — %.1f minutes", elapsed / 60)
    log.info("  Test accuracy : %.4f (%.2f%%)",
             metrics["test_accuracy"], metrics["test_accuracy"] * 100)
    log.info("  F1 score      : %.4f", metrics["f1"])
    log.info("  Outputs       : %s", (ROOT / "outputs").resolve())
    log.info("=" * 55)


if __name__ == "__main__":
    main()
