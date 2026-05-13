"""
afarinsene.export.exporter
===========================
Stage 7 of the pipeline: convert the trained Keras model to TFLite formats
and validate accuracy parity between formats.

Formats produced
----------------
* ``model_float32.tflite``  — full-precision mobile model (~6 MB)
* ``model_quantized.tflite``— INT8 quantised for edge devices (~1.7 MB)

Usage
-----
>>> from afarinsene.export.exporter import Exporter
>>> exp = Exporter(cfg)
>>> exp.export(model, splits)
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from afarinsene.utils.logger import get_logger

log = get_logger(__name__)


class Exporter:
    """
    Export a trained Keras model to TFLite formats.

    Parameters
    ----------
    cfg:
        Top-level config dict (reads ``cfg["export"]``).
    """

    def __init__(self, cfg: dict[str, Any]) -> None:
        ecfg = cfg.get("export", {})
        self.output_dir = Path(ecfg.get("output_dir", "outputs/tflite"))
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.float32_cfg = ecfg.get("float32", {})
        self.quant_cfg = ecfg.get("quantized", {})

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def export(self, model, splits) -> dict[str, Any]:
        """
        Convert *model* to TFLite formats and compare accuracy.

        Parameters
        ----------
        model:
            Best-weights Keras model.
        splits:
            :class:`~afarinsene.data.preprocessor.DataSplits`.

        Returns
        -------
        dict
            Size (bytes) and accuracy for each format.
        """
        import tensorflow as tf

        log.info("=" * 55)
        log.info("STAGE 7 — TFLite Export")
        log.info("  Output dir: %s", self.output_dir)

        results: dict[str, Any] = {}

        # ---- Float32 TFLite ---------------------------------------------
        if self.float32_cfg.get("enabled", True):
            fname = self.float32_cfg.get("filename", "model_float32.tflite")
            tflite_bytes = self._convert_float32(model, tf)
            out_path = self.output_dir / fname
            out_path.write_bytes(tflite_bytes)

            size_mb = len(tflite_bytes) / 1024 / 1024
            acc = self._evaluate_tflite(str(out_path), splits, "Float32")
            results["float32"] = {"path": str(out_path), "size_mb": round(size_mb, 3), "accuracy": round(acc, 6)}
            log.info("  Float32 TFLite — %.2f MB | acc=%.4f", size_mb, acc)

        # ---- INT8 Quantised TFLite --------------------------------------
        if self.quant_cfg.get("enabled", True):
            fname = self.quant_cfg.get("filename", "model_quantized.tflite")
            cal_n = self.quant_cfg.get("calibration_samples", 200)
            tflite_bytes = self._convert_quantized(model, splits.X_train, tf, cal_n)
            out_path = self.output_dir / fname
            out_path.write_bytes(tflite_bytes)

            size_kb = len(tflite_bytes) / 1024
            acc = self._evaluate_tflite(str(out_path), splits, "INT8")
            results["quantized"] = {"path": str(out_path), "size_kb": round(size_kb, 1), "accuracy": round(acc, 6)}
            log.info("  INT8 Quantised  — %.1f KB | acc=%.4f", size_kb, acc)

        self._write_report(results)
        log.info("=" * 55)
        return results

    # ------------------------------------------------------------------
    # Conversion helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _convert_float32(model, tf) -> bytes:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        return converter.convert()

    @staticmethod
    def _convert_quantized(model, X_train: np.ndarray, tf, cal_samples: int) -> bytes:
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]

        n_cal = min(cal_samples, len(X_train))

        def representative_dataset():
            for i in range(n_cal):
                img = X_train[i].astype("float32")
                if img.max() > 1.0:
                    img /= 255.0
                yield [np.expand_dims(img, axis=0)]

        converter.representative_dataset = representative_dataset
        converter.target_spec.supported_ops = [
            tf.lite.OpsSet.TFLITE_BUILTINS_INT8,
            tf.lite.OpsSet.TFLITE_BUILTINS,
        ]
        converter.inference_input_type = tf.float32
        converter.inference_output_type = tf.float32

        return converter.convert()

    # ------------------------------------------------------------------
    # Evaluation helper
    # ------------------------------------------------------------------

    @staticmethod
    def _evaluate_tflite(
        tflite_path: str, splits, label: str
    ) -> float:
        """Run inference using TFLite interpreter; return accuracy."""
        import tensorflow as tf

        interpreter = tf.lite.Interpreter(model_path=tflite_path)
        interpreter.allocate_tensors()

        in_details = interpreter.get_input_details()
        out_details = interpreter.get_output_details()

        X_test = splits.X_test
        y_true = splits.y_test

        correct = 0
        for i in range(len(X_test)):
            img = X_test[i].astype("float32")
            if img.max() > 1.0:
                img /= 255.0
            img = np.expand_dims(img, axis=0)

            interpreter.set_tensor(in_details[0]["index"], img)
            interpreter.invoke()
            output = interpreter.get_tensor(out_details[0]["index"])

            if np.argmax(output[0]) == y_true[i]:
                correct += 1

        return correct / len(X_test)

    # ------------------------------------------------------------------
    # Report
    # ------------------------------------------------------------------

    def _write_report(self, results: dict) -> None:
        lines = [
            "=" * 55,
            "AFARINSENE — TFLITE EXPORT REPORT",
            "=" * 55,
        ]
        for fmt, info in results.items():
            lines.append(f"\n  Format  : {fmt}")
            for k, v in info.items():
                lines.append(f"  {k:<10}: {v}")

        report_path = self.output_dir / "conversion_report.txt"
        with open(report_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        log.info("  Report saved: %s", report_path)
