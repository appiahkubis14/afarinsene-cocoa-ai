"""
tests/test_model.py
===================
Unit tests for model construction.
Run: pytest tests/test_model.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from afarinsene.utils.config import load_config
from afarinsene.models.cnn import build_model


@pytest.fixture
def cfg():
    c = load_config(ROOT / "configs" / "default.yaml")
    c["data"]["image_size"] = 32  # tiny images for fast test
    return c


class TestCNN:
    def test_build_compiles(self, cfg):
        model = build_model(cfg, num_classes=4)
        assert model is not None

    def test_output_shape(self, cfg):
        model = build_model(cfg, num_classes=4)
        img_size = cfg["data"]["image_size"]
        dummy = np.zeros((2, img_size, img_size, 3), dtype=np.float32)
        out = model.predict(dummy, verbose=0)
        assert out.shape == (2, 4), f"Expected (2, 4), got {out.shape}"

    def test_output_sums_to_one(self, cfg):
        """Softmax outputs should sum to 1 per sample."""
        model = build_model(cfg, num_classes=4)
        img_size = cfg["data"]["image_size"]
        dummy = np.random.rand(5, img_size, img_size, 3).astype("float32")
        out = model.predict(dummy, verbose=0)
        sums = out.sum(axis=1)
        np.testing.assert_allclose(sums, np.ones(5), atol=1e-5)

    def test_parameter_count_reasonable(self, cfg):
        """Should have more than 10k but fewer than 50M params."""
        model = build_model(cfg, num_classes=4)
        n = model.count_params()
        assert n > 10_000, f"Too few parameters: {n}"
        assert n < 50_000_000, f"Too many parameters: {n}"

    def test_different_num_classes(self, cfg):
        for num_classes in [2, 4, 8]:
            model = build_model(cfg, num_classes=num_classes)
            img_size = cfg["data"]["image_size"]
            dummy = np.zeros((1, img_size, img_size, 3), dtype=np.float32)
            out = model.predict(dummy, verbose=0)
            assert out.shape == (1, num_classes)
