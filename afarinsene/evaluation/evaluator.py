"""
afarinsene.evaluation.evaluator
=================================
Stage 6 of the pipeline: compute and save all evaluation metrics and plots.

Outputs
-------
* ``outputs/plots/training_history.png``   — accuracy, loss, LR curves
* ``outputs/plots/confusion_matrix.png``   — test-set heatmap
* ``outputs/plots/roc_curves.png``         — one-vs-rest ROC per class
* ``outputs/plots/per_class_accuracy.png`` — bar chart
* ``outputs/reports/evaluation_report.txt``— human-readable summary
* ``outputs/reports/metrics.json``         — machine-readable metrics (CI friendly)

Usage
-----
>>> from afarinsene.evaluation.evaluator import Evaluator
>>> ev = Evaluator(cfg)
>>> metrics = ev.evaluate(model, splits, history)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

from afarinsene.utils.logger import get_logger

log = get_logger(__name__)


class Evaluator:
    """
    Compute and persist evaluation metrics and plots.

    Parameters
    ----------
    cfg:
        Top-level config dict (reads ``cfg["evaluation"]``).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        ecfg = cfg.get("evaluation", {})
        self.plots_dir = Path(ecfg.get("plots_dir", "outputs/plots"))
        self.reports_dir = Path(ecfg.get("reports_dir", "outputs/reports"))
        self.dpi: int = ecfg.get("figure_dpi", 150)

        self.plots_dir.mkdir(parents=True, exist_ok=True)
        self.reports_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def evaluate(self, model, splits, history) -> dict[str, Any]:
        """
        Run full evaluation on the test split.

        Parameters
        ----------
        model:
            Trained ``tf.keras.Model`` (best weights already restored by EarlyStopping).
        splits:
            :class:`~afarinsene.data.preprocessor.DataSplits`.
        history:
            ``tf.keras.callbacks.History`` from training.

        Returns
        -------
        dict
            All scalar metrics (accuracy, F1, precision, recall, per-class).
        """
        import matplotlib
        matplotlib.use("Agg")  # non-interactive backend
        import matplotlib.pyplot as plt
        from sklearn.metrics import (
            classification_report,
            confusion_matrix,
            roc_curve,
            auc,
            precision_score,
            recall_score,
            f1_score,
        )

        log.info("=" * 55)
        log.info("STAGE 6 — Evaluation")

        class_names = splits.class_names
        num_classes = splits.num_classes

        # ---- Evaluate on test set ----------------------------------------
        test_loss, test_acc = model.evaluate(
            splits.X_test, splits.y_test_cat, verbose=0
        )
        log.info("  Test accuracy : %.4f (%.2f%%)", test_acc, test_acc * 100)
        log.info("  Test loss     : %.4f", test_loss)

        # ---- Predictions -------------------------------------------------
        y_pred_prob = model.predict(splits.X_test, verbose=0)
        y_pred = np.argmax(y_pred_prob, axis=1)
        y_true = splits.y_test

        precision = precision_score(y_true, y_pred, average="weighted", zero_division=0)
        recall    = recall_score(y_true, y_pred, average="weighted", zero_division=0)
        f1        = f1_score(y_true, y_pred, average="weighted", zero_division=0)

        log.info("  Precision     : %.4f", precision)
        log.info("  Recall        : %.4f", recall)
        log.info("  F1            : %.4f", f1)

        # ---- Per-class accuracy ------------------------------------------
        cm = confusion_matrix(y_true, y_pred)
        per_class_acc = cm.diagonal() / cm.sum(axis=1)

        log.info("")
        log.info("  Per-class accuracy:")
        for i, (name, acc_i) in enumerate(zip(class_names, per_class_acc)):
            n = int(cm.sum(axis=1)[i])
            log.info("    [%d] %-15s  %.2f%%  (%d samples)", i, name, acc_i * 100, n)

        log.info("")
        log.info("  Classification report:")
        report_str = classification_report(y_true, y_pred, target_names=class_names)
        for line in report_str.splitlines():
            log.info("    %s", line)

        # ---- Build metrics dict ------------------------------------------
        metrics: dict[str, Any] = {
            "test_accuracy":  round(float(test_acc), 6),
            "test_loss":      round(float(test_loss), 6),
            "precision":      round(float(precision), 6),
            "recall":         round(float(recall), 6),
            "f1":             round(float(f1), 6),
            "per_class_accuracy": {
                class_names[i]: round(float(per_class_acc[i]), 6)
                for i in range(num_classes)
            },
        }

        # ---- Save plots --------------------------------------------------
        self._plot_training_history(history, test_acc, test_loss)
        self._plot_confusion_matrix(cm, class_names, test_acc)
        self._plot_roc_curves(splits.y_test_cat, y_pred_prob, class_names, num_classes)
        self._plot_per_class_accuracy(per_class_acc, class_names, test_acc)

        # ---- Save reports ------------------------------------------------
        self._save_text_report(metrics, report_str)
        self._save_json_metrics(metrics)

        log.info("  Plots saved to  : %s", self.plots_dir)
        log.info("  Reports saved to: %s", self.reports_dir)
        log.info("=" * 55)

        return metrics

    # ------------------------------------------------------------------
    # Plots
    # ------------------------------------------------------------------

    def _plot_training_history(self, history, test_acc: float, test_loss: float) -> None:
        import matplotlib.pyplot as plt

        h = history.history
        epochs = range(1, len(h["accuracy"]) + 1)

        fig, axes = plt.subplots(1, 3, figsize=(18, 5))
        fig.suptitle("Afarinsene — Training History", fontsize=14, fontweight="bold")

        # Accuracy
        ax = axes[0]
        ax.plot(epochs, h["accuracy"], label="Train", lw=2)
        ax.plot(epochs, h["val_accuracy"], label="Val", lw=2)
        ax.axhline(test_acc, color="red", ls="--", lw=1.5, label=f"Test {test_acc:.3f}")
        ax.set(title="Accuracy", xlabel="Epoch", ylabel="Accuracy")
        ax.legend(); ax.grid(alpha=0.3); ax.set_ylim(0, 1)

        # Loss
        ax = axes[1]
        ax.plot(epochs, h["loss"], label="Train", lw=2)
        ax.plot(epochs, h["val_loss"], label="Val", lw=2)
        ax.axhline(test_loss, color="red", ls="--", lw=1.5, label=f"Test {test_loss:.3f}")
        ax.set(title="Loss", xlabel="Epoch", ylabel="Loss")
        ax.legend(); ax.grid(alpha=0.3)

        # Learning rate (reconstructed if not logged)
        ax = axes[2]
        lr_hist = h.get("lr", self._reconstruct_lr(h))
        ax.semilogy(epochs, lr_hist, color="green", lw=2)
        ax.set(title="Learning Rate", xlabel="Epoch", ylabel="LR (log scale)")
        ax.grid(alpha=0.3)

        plt.tight_layout()
        out = self.plots_dir / "training_history.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)
        log.debug("  Saved: %s", out)

    def _plot_confusion_matrix(self, cm, class_names, test_acc: float) -> None:
        import matplotlib.pyplot as plt
        import seaborn as sns

        fig, ax = plt.subplots(figsize=(8, 7))
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_names, yticklabels=class_names,
            ax=ax, linewidths=0.5,
        )
        ax.set_title(f"Confusion Matrix  (Test Acc: {test_acc:.2%})", fontsize=13, pad=12)
        ax.set_xlabel("Predicted", fontsize=11)
        ax.set_ylabel("True", fontsize=11)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=30, ha="right")

        plt.tight_layout()
        out = self.plots_dir / "confusion_matrix.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_roc_curves(self, y_true_cat, y_pred_prob, class_names, num_classes) -> None:
        import matplotlib.pyplot as plt
        from sklearn.metrics import roc_curve, auc

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728",
                  "#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]

        fig, ax = plt.subplots(figsize=(8, 7))
        for i in range(num_classes):
            fpr, tpr, _ = roc_curve(y_true_cat[:, i], y_pred_prob[:, i])
            roc_auc = auc(fpr, tpr)
            ax.plot(fpr, tpr, lw=2, color=colors[i % len(colors)],
                    label=f"{class_names[i]}  (AUC={roc_auc:.3f})")

        ax.plot([0, 1], [0, 1], "k--", lw=1.5, alpha=0.5)
        ax.set(title="ROC Curves (One-vs-Rest)", xlabel="FPR", ylabel="TPR",
               xlim=[0, 1], ylim=[0, 1])
        ax.legend(loc="lower right", fontsize=9)
        ax.grid(alpha=0.3)

        plt.tight_layout()
        out = self.plots_dir / "roc_curves.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    def _plot_per_class_accuracy(self, per_class_acc, class_names, overall_acc) -> None:
        import matplotlib.pyplot as plt

        colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728"]
        fig, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(class_names, per_class_acc,
                      color=colors[: len(class_names)], edgecolor="white")
        ax.axhline(overall_acc, color="red", ls="--", lw=1.5,
                   label=f"Overall: {overall_acc:.1%}")

        for bar, acc_i in zip(bars, per_class_acc):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.01,
                    f"{acc_i:.1%}", ha="center", va="bottom", fontsize=10)

        ax.set(title="Per-Class Accuracy", xlabel="Class", ylabel="Accuracy",
               ylim=[0, 1.1])
        ax.set_xticklabels(class_names, rotation=20, ha="right")
        ax.legend(); ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        out = self.plots_dir / "per_class_accuracy.png"
        fig.savefig(out, dpi=self.dpi, bbox_inches="tight")
        plt.close(fig)

    # ------------------------------------------------------------------
    # Reports
    # ------------------------------------------------------------------

    def _save_text_report(self, metrics: dict, report_str: str) -> None:
        out = self.reports_dir / "evaluation_report.txt"
        lines = [
            "=" * 60,
            "AFARINSENE — E-VALUATION REPORT",
            "=" * 60,
            "",
            f"Test Accuracy  : {metrics['test_accuracy']:.4f}  ({metrics['test_accuracy']*100:.2f}%)",
            f"Test Loss      : {metrics['test_loss']:.4f}",
            f"Precision (W)  : {metrics['precision']:.4f}",
            f"Recall    (W)  : {metrics['recall']:.4f}",
            f"F1        (W)  : {metrics['f1']:.4f}",
            "",
            "Per-class accuracy:",
        ]
        for cls, acc_i in metrics["per_class_accuracy"].items():
            lines.append(f"  {cls:<20s}: {acc_i*100:.2f}%")

        lines += ["", "Classification Report:", "─" * 50, report_str]

        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))

    def _save_json_metrics(self, metrics: dict) -> None:
        out = self.reports_dir / "metrics.json"
        with open(out, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)

    # ------------------------------------------------------------------
    # LR reconstruction (when ReduceLROnPlateau doesn't log 'lr')
    # ------------------------------------------------------------------

    @staticmethod
    def _reconstruct_lr(history: dict) -> list[float]:
        """Approximate LR history from val_loss trajectory."""
        lr = 0.001
        factor = 0.5
        patience = 5
        min_lr = 1e-7

        val_loss = history.get("val_loss", [])
        lr_hist = []
        best = float("inf")
        counter = 0

        for loss in val_loss:
            lr_hist.append(lr)
            if loss < best:
                best = loss
                counter = 0
            else:
                counter += 1
            if counter >= patience:
                lr = max(lr * factor, min_lr)
                counter = 0

        return lr_hist
