"""
afarinsene.data.loader
======================
Stage 1 of the pipeline: load images from a class-folder dataset and
return validated numpy arrays.

Expected dataset structure
--------------------------
::

    DATASET/
    ├── BLACK POD/          → label 0
    ├── CSSVD LEAF/         → label 1
    ├── HEALTHY LEAF/       → label 2
    └── HEALTHY POD/        → label 3

Usage
-----
>>> from afarinsene.utils.config import load_config
>>> from afarinsene.data.loader import DatasetLoader
>>> cfg = load_config("configs/default.yaml")
>>> loader = DatasetLoader(cfg)
>>> X, y, class_names = loader.load()
>>> print(X.shape, y.shape)  # e.g. (476, 256, 256, 3) (476,)
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Tuple

import numpy as np

from afarinsene.utils.logger import get_logger

log = get_logger(__name__)

# Supported image extensions (lower-case)
_IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


class DatasetLoader:
    """
    Load a class-folder image dataset into memory as uint8 numpy arrays.

    Parameters
    ----------
    cfg:
        Top-level config dict (reads ``cfg["data"]``).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        self.cfg = cfg
        data_cfg = cfg["data"]

        self.data_dir = Path(data_cfg["dir"])
        self.class_map: dict[str, int] = data_cfg["class_map"]
        self.class_names: list[str] = data_cfg["class_names"]
        self.image_size: int = data_cfg["image_size"]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load(self) -> Tuple[np.ndarray, np.ndarray, list[str]]:
        """
        Walk the dataset directory, load every image, and return arrays.

        Returns
        -------
        X : np.ndarray, shape (N, H, W, 3), dtype uint8
            Raw pixel values in [0, 255].
        y : np.ndarray, shape (N,), dtype int32
            Integer class labels.
        class_names : list[str]
            Human-readable class name for each label index.
        """
        if not self.data_dir.exists():
            raise FileNotFoundError(
                f"Dataset directory not found: {self.data_dir.resolve()}\n"
                "Set --data-dir or the DATA_DIR environment variable."
            )

        log.info("=" * 55)
        log.info("STAGE 1 — Dataset Loading")
        log.info("  Directory : %s", self.data_dir)
        log.info("  Image size: %d × %d", self.image_size, self.image_size)
        log.info("=" * 55)

        all_images: list[np.ndarray] = []
        all_labels: list[int] = []
        missing_dirs: list[str] = []

        for folder_name, label in self.class_map.items():
            folder_path = self.data_dir / folder_name

            if not folder_path.exists():
                log.warning("Class folder not found — skipping: %s", folder_path)
                missing_dirs.append(folder_name)
                continue

            images, errors = self._load_class(folder_path, label)
            all_images.extend(images)
            all_labels.extend([label] * len(images))

            if errors:
                log.warning(
                    "  [%s] Loaded %d images, %d failed",
                    folder_name,
                    len(images),
                    errors,
                )

        if not all_images:
            raise RuntimeError(
                "No images were loaded. Check that --data-dir points to a "
                "folder containing the expected class sub-directories."
            )

        X = np.array(all_images, dtype=np.uint8)
        y = np.array(all_labels, dtype=np.int32)

        self._report(X, y, missing_dirs)
        return X, y, self.class_names

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _load_class(
        self, folder: Path, label: int
    ) -> Tuple[list[np.ndarray], int]:
        """Load all images in *folder*, resize to target size."""
        import tensorflow as tf

        image_files = [
            f for f in folder.iterdir() if f.suffix.lower() in _IMG_EXTS
        ]

        if not image_files:
            log.warning("No images found in %s", folder)
            return [], 0

        log.info(
            "  Loading %-15s (label %d) — %d images ...",
            folder.name,
            label,
            len(image_files),
        )

        images: list[np.ndarray] = []
        errors = 0

        for i, fpath in enumerate(image_files):
            try:
                img = tf.keras.preprocessing.image.load_img(
                    str(fpath),
                    target_size=(self.image_size, self.image_size),
                )
                arr = tf.keras.preprocessing.image.img_to_array(img).astype(
                    np.uint8
                )
                images.append(arr)
            except Exception as exc:
                log.debug("    Could not load %s: %s", fpath.name, exc)
                errors += 1

            if (i + 1) % 100 == 0:
                log.debug("    ... %d / %d", i + 1, len(image_files))

        return images, errors

    def _report(
        self, X: np.ndarray, y: np.ndarray, missing: list[str]
    ) -> None:
        log.info("")
        log.info("Dataset loading complete")
        log.info("  Total images : %d", len(X))
        log.info("  Array shape  : %s  dtype=%s", X.shape, X.dtype)
        log.info("  Value range  : [%d, %d]", int(X.min()), int(X.max()))

        log.info("  Class breakdown:")
        label_values, counts = np.unique(y, return_counts=True)
        for lv, cnt in zip(label_values, counts):
            name = self.class_names[lv] if lv < len(self.class_names) else f"Class_{lv}"
            pct = 100.0 * cnt / len(y)
            log.info("    [%d] %-15s — %4d samples (%.1f%%)", lv, name, cnt, pct)

        if missing:
            log.warning("  Missing class folders: %s", missing)
