"""
afarinsene.data.augmentation
=============================
Stage 3 of the pipeline: balance minority classes in the training set by
augmenting under-represented classes to match the largest class count.

Augmentation techniques applied
--------------------------------
* Rotation (±30°)
* Width / height shift (±20%)
* Horizontal flip
* Zoom (±20%)
* Brightness adjustment (±20%)
* MixUp (linear interpolation of two samples + labels)

Only the **training** set is augmented; validation and test sets are left
untouched so evaluation remains on real-world images.

Usage
-----
>>> from afarinsene.data.augmentation import Augmenter
>>> aug = Augmenter(cfg)
>>> X_train_bal, y_train_bal = aug.balance(X_train, y_train)
"""

from __future__ import annotations

from typing import Any, Tuple

import numpy as np

from afarinsene.utils.logger import get_logger

log = get_logger(__name__)


class Augmenter:
    """
    Balance and augment the training set.

    Parameters
    ----------
    cfg:
        Top-level config dict (reads ``cfg["augmentation"]``).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        acfg = cfg.get("augmentation", {})
        self.enabled: bool = acfg.get("enabled", True)
        self.rotation_range: float = acfg.get("rotation_range", 30)
        self.width_shift: float = acfg.get("width_shift_range", 0.2)
        self.height_shift: float = acfg.get("height_shift_range", 0.2)
        self.horizontal_flip: bool = acfg.get("horizontal_flip", True)
        self.zoom_range: float = acfg.get("zoom_range", 0.2)
        self.brightness_range: list = acfg.get("brightness_range", [0.8, 1.2])
        self.fill_mode: str = acfg.get("fill_mode", "nearest")
        self.mixup_alpha: float = acfg.get("mixup_alpha", 0.2)
        self.balance_classes: bool = acfg.get("balance_classes", True)

        self._datagen = None  # lazy-init to avoid importing keras at import time

    def balance(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Balance minority classes and optionally apply MixUp.

        Parameters
        ----------
        X:
            Float32 images in [0, 1], shape (N, H, W, C).
        y:
            Integer class labels, shape (N,).

        Returns
        -------
        X_balanced, y_balanced : np.ndarray
            Shuffled, balanced arrays (train only).
        """
        if not self.enabled:
            log.info("Augmentation disabled — returning original training data.")
            return X, y

        log.info("=" * 55)
        log.info("STAGE 3 — Augmentation & Class Balancing")

        unique, counts = np.unique(y, return_counts=True)
        target = int(counts.max())
        log.info("  Target samples per class: %d", target)

        X_bal_list: list[np.ndarray] = []
        y_bal_list: list[np.ndarray] = []

        for label, count in zip(unique, counts):
            X_class = X[y == label]
            log.info("    Class %d: %d → %d", label, count, target)

            X_aug = self._augment_class(X_class, target)
            X_bal_list.append(X_aug)
            y_bal_list.append(np.full(target, label, dtype=np.int32))

        X_bal = np.vstack(X_bal_list)
        y_bal = np.concatenate(y_bal_list)

        # Shuffle
        idx = np.random.permutation(len(X_bal))
        X_bal, y_bal = X_bal[idx], y_bal[idx]

        log.info(
            "  Balanced training set: %d → %d samples",
            len(X),
            len(X_bal),
        )

        # Optional MixUp pass
        if self.mixup_alpha > 0:
            X_bal, y_bal = self._mixup(X_bal, y_bal)
            log.info("  MixUp applied (alpha=%.2f)", self.mixup_alpha)

        return X_bal, y_bal

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _get_datagen(self):
        """Lazily build the Keras ImageDataGenerator."""
        if self._datagen is None:
            from tensorflow.keras.preprocessing.image import ImageDataGenerator

            self._datagen = ImageDataGenerator(
                rotation_range=self.rotation_range,
                width_shift_range=self.width_shift,
                height_shift_range=self.height_shift,
                horizontal_flip=self.horizontal_flip,
                zoom_range=self.zoom_range,
                brightness_range=self.brightness_range,
                fill_mode=self.fill_mode,
            )
        return self._datagen

    def _augment_class(
        self, X_class: np.ndarray, target: int
    ) -> np.ndarray:
        """
        Return exactly *target* samples: original images + augmented to fill
        the gap.  If the class already has >= target, subsample it.
        """
        n = len(X_class)
        if n >= target:
            idx = np.random.choice(n, target, replace=False)
            return X_class[idx]

        datagen = self._get_datagen()
        num_needed = target - n
        augmented: list[np.ndarray] = []

        for _ in range(num_needed):
            src_idx = np.random.randint(0, n)
            img = X_class[src_idx]  # already float32 [0,1]
            aug = datagen.random_transform(img)
            # Clip to valid range after brightness adjustment
            aug = np.clip(aug, 0.0, 1.0)
            augmented.append(aug)

        return np.vstack([X_class, np.array(augmented, dtype=np.float32)])

    def _mixup(
        self, X: np.ndarray, y: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Apply MixUp: for each sample, blend with a random other sample.
        Labels remain integer (only the dominant label is kept).
        """
        alpha = self.mixup_alpha
        n = len(X)
        lam = np.random.beta(alpha, alpha, size=(n, 1, 1, 1)).astype(np.float32)

        # Pair each sample with a random other
        idx = np.random.permutation(n)
        X_mix = lam * X + (1 - lam) * X[idx]
        X_mix = np.clip(X_mix, 0.0, 1.0)

        # Keep label of the dominant sample (lam > 0.5)
        y_mix = np.where(lam[:, 0, 0, 0] >= 0.5, y, y[idx])
        return X_mix, y_mix
