"""
afarinsene.data.preprocessor
=============================
Stage 2 of the pipeline: normalise pixel values to [0, 1] and create
stratified train / val / test splits.

Usage
-----
>>> from afarinsene.data.preprocessor import Preprocessor
>>> pre = Preprocessor(cfg)
>>> splits = pre.split(X, y)  # returns DataSplits namedtuple
>>> print(splits.X_train.shape)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from afarinsene.utils.logger import get_logger

log = get_logger(__name__)


@dataclass
class DataSplits:
    """Container for train / val / test arrays returned by :class:`Preprocessor`."""

    X_train: np.ndarray   # float32, [0, 1], shape (N_train, H, W, C)
    X_val: np.ndarray     # float32, [0, 1], shape (N_val, H, W, C)
    X_test: np.ndarray    # float32, [0, 1], shape (N_test, H, W, C)

    y_train: np.ndarray   # integer labels, shape (N_train,)
    y_val: np.ndarray
    y_test: np.ndarray

    y_train_cat: np.ndarray  # one-hot, shape (N_train, num_classes)
    y_val_cat: np.ndarray
    y_test_cat: np.ndarray

    num_classes: int
    class_names: list[str]


class Preprocessor:
    """
    Normalise images and produce stratified train/val/test splits.

    Parameters
    ----------
    cfg:
        Top-level config dict (reads ``cfg["data"]``).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        dcfg = cfg["data"]
        self.train_ratio = dcfg.get("train_ratio", 0.70)
        self.val_ratio = dcfg.get("val_ratio", 0.15)
        self.test_ratio = dcfg.get("test_ratio", 0.15)
        self.random_seed = dcfg.get("random_seed", 42)
        self.class_names: list[str] = dcfg["class_names"]

    def split(self, X: np.ndarray, y: np.ndarray) -> DataSplits:
        """
        Normalise *X* and split into train / val / test.

        Parameters
        ----------
        X:
            Raw uint8 images, shape (N, H, W, C).
        y:
            Integer labels, shape (N,).

        Returns
        -------
        DataSplits
        """
        log.info("=" * 55)
        log.info("STAGE 2 — Preprocessing & Splitting")
        log.info(
            "  Ratios: train=%.0f%%  val=%.0f%%  test=%.0f%%",
            self.train_ratio * 100,
            self.val_ratio * 100,
            self.test_ratio * 100,
        )
        log.info("=" * 55)

        num_classes = len(np.unique(y))

        # ---- Normalise ---------------------------------------------------
        X_f = X.astype("float32") / 255.0

        # ---- First split: separate test set ------------------------------
        test_size = self.test_ratio
        X_tv, X_test, y_tv, y_test = train_test_split(
            X_f, y,
            test_size=test_size,
            random_state=self.random_seed,
            stratify=y,
        )

        # ---- Second split: separate val from train -----------------------
        # val ratio relative to the remaining (train + val) data
        relative_val = self.val_ratio / (self.train_ratio + self.val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_tv, y_tv,
            test_size=relative_val,
            random_state=self.random_seed,
            stratify=y_tv,
        )

        # ---- One-hot encode labels ---------------------------------------
        y_train_cat = to_categorical(y_train, num_classes)
        y_val_cat = to_categorical(y_val, num_classes)
        y_test_cat = to_categorical(y_test, num_classes)

        self._report(X_train, X_val, X_test, y_train, y_val, y_test)
        self._validate_ranges(X_train, X_val, X_test)

        return DataSplits(
            X_train=X_train, X_val=X_val, X_test=X_test,
            y_train=y_train, y_val=y_val, y_test=y_test,
            y_train_cat=y_train_cat, y_val_cat=y_val_cat, y_test_cat=y_test_cat,
            num_classes=num_classes,
            class_names=self.class_names,
        )

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _report(X_tr, X_v, X_te, y_tr, y_v, y_te) -> None:
        log.info("")
        log.info("  Split sizes (normalised float32):")
        log.info("    Train : %s  range [%.3f, %.3f]", X_tr.shape, X_tr.min(), X_tr.max())
        log.info("    Val   : %s  range [%.3f, %.3f]", X_v.shape, X_v.min(), X_v.max())
        log.info("    Test  : %s  range [%.3f, %.3f]", X_te.shape, X_te.min(), X_te.max())

    @staticmethod
    def _validate_ranges(*arrays: np.ndarray) -> None:
        for arr in arrays:
            assert arr.min() >= 0.0 and arr.max() <= 1.0, (
                f"Normalisation failed: range [{arr.min():.4f}, {arr.max():.4f}]"
            )
        log.info("  All arrays verified in [0, 1] ✓")
