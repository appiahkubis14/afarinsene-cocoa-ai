"""
tests/test_export.py
====================
Smoke tests for TFLite export.
Run: pytest tests/test_export.py -v
"""

import sys
import tempfile
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from afarinsene.utils.config import load_config
from afarinsene.models.cnn import build_model
from afarinsene.export.exporter import Exporter
from afarinsene.data.preprocessor import DataSplits


@pytest.fixture
def cfg():
    c = load_config(ROOT / "configs" / "default.yaml")
    c["data"]["image_size"] = 32
    return c


@pytest.fixture
def tiny_model(cfg):
    return build_model(cfg, num_classes=4)


@pytest.fixture
def tiny_splits(cfg):
    """Minimal DataSplits for export testing."""
    n = 20
    img_size = cfg["data"]["image_size"]
    X = np.random.rand(n, img_size, img_size, 3).astype("float32")
    y = np.repeat(np.arange(4), 5).astype(np.int32)
    from tensorflow.keras.utils import to_categorical
    return DataSplits(
        X_train=X, X_val=X[:4], X_test=X[:4],
        y_train=y, y_val=y[:4], y_test=y[:4],
        y_train_cat=to_categorical(y, 4),
        y_val_cat=to_categorical(y[:4], 4),
        y_test_cat=to_categorical(y[:4], 4),
        num_classes=4,
        class_names=["Black Pod", "CSSVD Leaf", "Healthy Leaf", "Healthy Pod"],
    )


class TestExporter:
    def test_float32_tflite_produced(self, cfg, tiny_model, tiny_splits):
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg["export"]["output_dir"] = tmpdir
            cfg["export"]["quantized"]["enabled"] = False  # skip quant for speed
            exp = Exporter(cfg)
            results = exp.export(tiny_model, tiny_splits)

            out = Path(results["float32"]["path"])
            assert out.exists(), "TFLite file not created"
            assert out.stat().st_size > 0, "TFLite file is empty"

    def test_float32_accuracy_reasonable(self, cfg, tiny_model, tiny_splits):
        """Untrained model accuracy should be > 0 and <= 1."""
        with tempfile.TemporaryDirectory() as tmpdir:
            cfg["export"]["output_dir"] = tmpdir
            cfg["export"]["quantized"]["enabled"] = False
            exp = Exporter(cfg)
            results = exp.export(tiny_model, tiny_splits)

            acc = results["float32"]["accuracy"]
            assert 0.0 <= acc <= 1.0
