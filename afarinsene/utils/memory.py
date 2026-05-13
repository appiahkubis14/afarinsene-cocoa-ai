"""
afarinsene.utils.memory
=======================
GPU / CPU configuration and memory monitoring utilities.

Usage
-----
>>> from afarinsene.utils.memory import setup_gpu, show_memory
>>> setup_gpu()
>>> show_memory()
"""

from __future__ import annotations

import gc
import os
from typing import Optional

from afarinsene.utils.logger import get_logger

log = get_logger(__name__)


def setup_gpu(memory_growth: bool = True, visible_device: int = 0) -> bool:
    """
    Configure TensorFlow GPU settings.

    Parameters
    ----------
    memory_growth:
        If True, TensorFlow allocates GPU memory on demand rather than
        pre-allocating everything.  Recommended for shared machines.
    visible_device:
        Index of the GPU to use (default: first GPU).

    Returns
    -------
    bool
        True if at least one GPU was found and configured.
    """
    import tensorflow as tf  # lazy import so module loads without TF

    log.info("TensorFlow version: %s", tf.__version__)
    gpus = tf.config.list_physical_devices("GPU")

    if not gpus:
        log.warning("No GPU detected — running on CPU only.")
        return False

    try:
        if memory_growth:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)

        tf.config.set_visible_devices(gpus[visible_device], "GPU")
        tf.config.set_soft_device_placement(True)

        logical_gpus = tf.config.list_logical_devices("GPU")
        log.info(
            "GPU configured: %d physical → %d logical GPU(s)",
            len(gpus),
            len(logical_gpus),
        )
        return True

    except RuntimeError as exc:
        log.error("GPU configuration error: %s", exc)
        return False


def show_memory(label: str = "") -> None:
    """
    Log current process RAM usage.

    Parameters
    ----------
    label:
        Optional label prepended to the log message.
    """
    try:
        import psutil  # optional dependency

        process = __import__("psutil").Process(os.getpid())
        mem_gb = process.memory_info().rss / 1024**3
        avail_gb = psutil.virtual_memory().available / 1024**3
        total_gb = psutil.virtual_memory().total / 1024**3
        tag = f"[{label}] " if label else ""
        log.info(
            "%sRAM: %.1f GB used | %.1f GB available / %.1f GB total",
            tag,
            mem_gb,
            avail_gb,
            total_gb,
        )
    except ImportError:
        log.debug("psutil not installed — memory stats unavailable.")


def free_memory(large_arrays: Optional[list] = None) -> None:
    """
    Release TensorFlow session and run garbage collection.

    Parameters
    ----------
    large_arrays:
        Optional list of numpy arrays to delete before collecting.
    """
    import tensorflow as tf

    if large_arrays:
        for arr in large_arrays:
            del arr

    try:
        tf.keras.backend.clear_session()
    except Exception:
        pass

    for _ in range(3):
        gc.collect()

    log.debug("Memory freed.")
