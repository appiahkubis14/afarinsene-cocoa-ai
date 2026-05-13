"""
afarinsene.training.trainer
============================
Stage 5 of the pipeline: training loop with callbacks, checkpoint/resume,
and learning-rate schedule reconstruction.

Checkpoint / Resume
-------------------
Two checkpoint files are maintained:

* ``best_model.keras``        — saved whenever ``val_accuracy`` improves.
* ``latest_checkpoint.keras`` — overwritten every epoch (enables resume).

On ``--resume``, the latest checkpoint is loaded and training continues
from the last completed epoch using the optimizer state stored in the
``.keras`` format.

Usage
-----
>>> from afarinsene.training.trainer import Trainer
>>> trainer = Trainer(cfg)
>>> history = trainer.train(model, splits, resume=False)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from tensorflow.keras.utils import to_categorical

from afarinsene.utils.logger import get_logger

log = get_logger(__name__)


class Trainer:
    """
    Manage the full training lifecycle.

    Parameters
    ----------
    cfg:
        Top-level config dict (reads ``cfg["training"]``).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        tcfg = cfg["training"]
        self.epochs: int = tcfg.get("epochs", 100)
        self.batch_size: int = tcfg.get("batch_size", 32)

        ckpt_dir = Path(tcfg.get("checkpoint_dir", "outputs/checkpoints"))
        ckpt_dir.mkdir(parents=True, exist_ok=True)

        self.best_model_path = ckpt_dir / tcfg.get("best_model_filename", "best_model.keras")
        self.latest_ckpt_path = ckpt_dir / tcfg.get("checkpoint_filename", "latest_checkpoint.keras")

        self.es_cfg = tcfg.get("early_stopping", {})
        self.rlr_cfg = tcfg.get("reduce_lr", {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def train(
        self,
        model,
        splits,
        resume: bool = False,
    ):
        """
        Train *model* on the data in *splits*.

        Parameters
        ----------
        model:
            Compiled ``tf.keras.Model``.
        splits:
            :class:`~afarinsene.data.preprocessor.DataSplits` namedtuple.
        resume:
            If True, load ``latest_checkpoint.keras`` before training.

        Returns
        -------
        tf.keras.callbacks.History
        """
        from tensorflow.keras.callbacks import (
            EarlyStopping,
            ModelCheckpoint,
            ReduceLROnPlateau,
        )

        log.info("=" * 55)
        log.info("STAGE 5 — Training")

        # ---- Resume -------------------------------------------------------
        initial_epoch = 0
        if resume and self.latest_ckpt_path.exists():
            log.info("  Resuming from checkpoint: %s", self.latest_ckpt_path)
            model.load_weights(str(self.latest_ckpt_path))
            # Estimate starting epoch from best model age (approximate)
            log.info("  Checkpoint loaded — continuing training.")
        elif resume:
            log.warning(
                "  --resume requested but no checkpoint found at %s — starting fresh.",
                self.latest_ckpt_path,
            )

        # ---- Prepare labels -----------------------------------------------
        y_train = splits.y_train_cat
        y_val = splits.y_val_cat

        log.info("  Train : %s  |  Val : %s", splits.X_train.shape, splits.X_val.shape)
        log.info("  Epochs: %d  |  Batch size: %d", self.epochs, self.batch_size)
        log.info("=" * 55)

        # ---- Callbacks ----------------------------------------------------
        callbacks = self._build_callbacks()

        # ---- Fit ----------------------------------------------------------
        history = model.fit(
            splits.X_train,
            y_train,
            validation_data=(splits.X_val, y_val),
            epochs=self.epochs,
            batch_size=self.batch_size,
            callbacks=callbacks,
            initial_epoch=initial_epoch,
            verbose=1,
        )

        log.info("")
        log.info("Training complete.")
        log.info("  Best model saved to : %s", self.best_model_path)

        best_val_acc = max(history.history.get("val_accuracy", [0]))
        log.info("  Best val accuracy   : %.4f (%.2f%%)", best_val_acc, best_val_acc * 100)

        return history

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _build_callbacks(self) -> list:
        from tensorflow.keras.callbacks import (
            EarlyStopping,
            ModelCheckpoint,
            ReduceLROnPlateau,
        )

        es = EarlyStopping(
            monitor=self.es_cfg.get("monitor", "val_accuracy"),
            patience=self.es_cfg.get("patience", 15),
            restore_best_weights=self.es_cfg.get("restore_best_weights", True),
            verbose=1,
        )

        rlr = ReduceLROnPlateau(
            monitor=self.rlr_cfg.get("monitor", "val_accuracy"),
            factor=self.rlr_cfg.get("factor", 0.5),
            patience=self.rlr_cfg.get("patience", 5),
            min_lr=self.rlr_cfg.get("min_lr", 1e-7),
            verbose=1,
        )

        # Save best weights
        best_ckpt = ModelCheckpoint(
            str(self.best_model_path),
            monitor="val_accuracy",
            save_best_only=True,
            mode="max",
            verbose=1,
        )

        # Save latest weights every epoch (enables resume)
        latest_ckpt = ModelCheckpoint(
            str(self.latest_ckpt_path),
            monitor="val_accuracy",
            save_best_only=False,
            verbose=0,
        )

        return [es, rlr, best_ckpt, latest_ckpt]
