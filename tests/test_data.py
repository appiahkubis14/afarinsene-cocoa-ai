"""
tests/test_data.py
==================
Unit tests for data loading, preprocessing, and augmentation.
Run: pytest tests/test_data.py -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from afarinsene.utils.config import load_config
from afarinsene.data.preprocessor import Preprocessor
from afarinsene.data.augmentation import Augmenter


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def cfg():
    return load_config(ROOT / "configs" / "default.yaml")


def make_fake_dataset(n=120, h=32, w=32, c=3, num_classes=4):
    """Create tiny synthetic images for fast unit tests."""
    X = np.random.randint(0, 256, (n, h, w, c), dtype=np.uint8)
    y = np.repeat(np.arange(num_classes), n // num_classes).astype(np.int32)
    return X, y


# ---------------------------------------------------------------------------
# Preprocessor tests
# ---------------------------------------------------------------------------

class TestPreprocessor:
    def test_normalisation_range(self, cfg):
        X, y = make_fake_dataset()
        pre = Preprocessor(cfg)
        splits = pre.split(X, y)
        assert splits.X_train.min() >= 0.0
        assert splits.X_train.max() <= 1.0
        assert splits.X_val.min() >= 0.0
        assert splits.X_test.max() <= 1.0

    def test_split_sizes(self, cfg):
        n = 120
        X, y = make_fake_dataset(n=n)
        pre = Preprocessor(cfg)
        splits = pre.split(X, y)

        total = len(splits.X_train) + len(splits.X_val) + len(splits.X_test)
        assert total == n, f"Expected {n} total samples, got {total}"

    def test_no_train_val_overlap(self, cfg):
        X, y = make_fake_dataset()
        pre = Preprocessor(cfg)
        splits = pre.split(X, y)

        train_set = set(map(tuple, splits.X_train.reshape(len(splits.X_train), -1).tolist()))
        val_set = set(map(tuple, splits.X_val.reshape(len(splits.X_val), -1).tolist()))
        # Very unlikely to overlap on random data
        assert len(train_set & val_set) == 0

    def test_one_hot_shape(self, cfg):
        X, y = make_fake_dataset()
        pre = Preprocessor(cfg)
        splits = pre.split(X, y)

        assert splits.y_train_cat.shape[1] == splits.num_classes
        assert splits.y_val_cat.shape[1] == splits.num_classes
        assert splits.y_test_cat.shape[1] == splits.num_classes

    def test_num_classes(self, cfg):
        X, y = make_fake_dataset(num_classes=4)
        pre = Preprocessor(cfg)
        splits = pre.split(X, y)
        assert splits.num_classes == 4


# ---------------------------------------------------------------------------
# Augmenter tests
# ---------------------------------------------------------------------------

class TestAugmenter:
    def test_balance_output_shape(self, cfg):
        """After balancing, all classes should have the same count."""
        # Create imbalanced data: class 0 has 10 samples, class 1 has 30
        X = np.random.rand(40, 32, 32, 3).astype("float32")
        y = np.array([0] * 10 + [1] * 30, dtype=np.int32)

        aug = Augmenter(cfg)
        aug.mixup_alpha = 0.0  # disable MixUp to keep label interpretation simple
        X_bal, y_bal = aug.balance(X, y)

        unique, counts = np.unique(y_bal, return_counts=True)
        assert all(c == counts[0] for c in counts), "Classes not balanced"

    def test_balance_preserves_dtype(self, cfg):
        X = np.random.rand(80, 32, 32, 3).astype("float32")
        y = np.repeat(np.arange(4), 20).astype(np.int32)

        aug = Augmenter(cfg)
        X_bal, y_bal = aug.balance(X, y)

        assert X_bal.dtype == np.float32
        assert X_bal.min() >= 0.0
        assert X_bal.max() <= 1.0

    def test_disabled_augmentation(self, cfg):
        cfg_copy = dict(cfg)
        cfg_copy["augmentation"] = {"enabled": False}
        X = np.random.rand(40, 32, 32, 3).astype("float32")
        y = np.repeat(np.arange(4), 10).astype(np.int32)

        aug = Augmenter(cfg_copy)
        X_out, y_out = aug.balance(X, y)

        assert X_out.shape == X.shape
        np.testing.assert_array_equal(y_out, y)
