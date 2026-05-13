#!/usr/bin/env python3
"""
scripts/predict.py
==================
Run inference on a single image or a folder of images using a trained
Afarinsene model.

Usage
-----
Single image::

    python scripts/predict.py \\
        --model outputs/checkpoints/best_model.keras \\
        --image /path/to/leaf.jpg

Folder (saves a CSV)::

    python scripts/predict.py \\
        --model outputs/checkpoints/best_model.keras \\
        --folder /path/to/farm_photos/ \\
        --output outputs/reports/predictions.csv

TFLite model::

    python scripts/predict.py \\
        --model outputs/tflite/model_float32.tflite \\
        --image /path/to/leaf.jpg
"""

from __future__ import annotations

import argparse
import csv
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CLASS_NAMES = ["Black Pod", "CSSVD Leaf", "Healthy Leaf", "Healthy Pod"]
IMAGE_SIZE = 256
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".tif"}


def parse_args():
    p = argparse.ArgumentParser(description="Afarinsene — Prediction")
    p.add_argument("--model", "-m", required=True,
                   help="Path to .keras or .tflite model")
    p.add_argument("--image", "-i", default=None,
                   help="Single image path")
    p.add_argument("--folder", "-f", default=None,
                   help="Folder of images to predict")
    p.add_argument("--output", "-o", default="outputs/reports/predictions.csv",
                   help="CSV output path (folder mode only)")
    p.add_argument("--image-size", type=int, default=IMAGE_SIZE,
                   help=f"Image resize dimension (default: {IMAGE_SIZE})")
    p.add_argument("--classes", nargs="+", default=CLASS_NAMES,
                   help="Class names (in label order)")
    return p.parse_args()


def load_model(model_path: str):
    """Load a Keras or TFLite model."""
    path = Path(model_path)
    if not path.exists():
        raise FileNotFoundError(f"Model not found: {path.resolve()}")

    if path.suffix == ".tflite":
        import tensorflow as tf
        interpreter = tf.lite.Interpreter(model_path=str(path))
        interpreter.allocate_tensors()
        return ("tflite", interpreter)
    else:
        import tensorflow as tf
        model = tf.keras.models.load_model(str(path))
        return ("keras", model)


def preprocess_image(image_path: str, size: int):
    """Load and normalise a single image."""
    import numpy as np
    import tensorflow as tf

    img = tf.keras.preprocessing.image.load_img(image_path, target_size=(size, size))
    arr = tf.keras.preprocessing.image.img_to_array(img).astype("float32") / 255.0
    return arr


def predict_one(model_bundle, img_arr, class_names: list[str]) -> dict:
    """Run inference on a single pre-processed image array."""
    import numpy as np

    kind, model = model_bundle
    inp = img_arr[np.newaxis]  # add batch dim

    if kind == "keras":
        probs = model.predict(inp, verbose=0)[0]
    else:
        # TFLite
        in_details = model.get_input_details()
        out_details = model.get_output_details()
        model.set_tensor(in_details[0]["index"], inp)
        model.invoke()
        probs = model.get_tensor(out_details[0]["index"])[0]

    label_idx = int(probs.argmax())
    return {
        "class_id": label_idx,
        "class_name": class_names[label_idx],
        "confidence": float(probs[label_idx]),
        "all_probs": {class_names[i]: float(p) for i, p in enumerate(probs)},
    }


def main() -> None:
    args = parse_args()
    model_bundle = load_model(args.model)
    class_names = args.classes

    if args.image:
        # ---- Single image mode ------------------------------------------
        arr = preprocess_image(args.image, args.image_size)
        result = predict_one(model_bundle, arr, class_names)

        print("\n" + "=" * 50)
        print(f"  Image     : {args.image}")
        print(f"  Prediction: {result['class_name']}")
        print(f"  Confidence: {result['confidence']:.2%}")
        print("\n  All class probabilities:")
        for cls, prob in result["all_probs"].items():
            bar = "█" * int(prob * 30)
            print(f"    {cls:<15} {prob:6.2%}  {bar}")
        print("=" * 50)

    elif args.folder:
        # ---- Folder mode ------------------------------------------------
        folder = Path(args.folder)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder}")

        image_files = sorted(
            p for p in folder.rglob("*") if p.suffix.lower() in IMG_EXTS
        )
        if not image_files:
            print("No images found in", folder)
            return

        print(f"\nPredicting {len(image_files)} images from {folder}...")
        out_path = Path(args.output)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        rows = []
        for i, img_path in enumerate(image_files):
            try:
                arr = preprocess_image(str(img_path), args.image_size)
                result = predict_one(model_bundle, arr, class_names)
                row = {
                    "file": img_path.name,
                    "path": str(img_path),
                    "predicted_class": result["class_name"],
                    "confidence": f"{result['confidence']:.4f}",
                    **{f"prob_{cls.lower().replace(' ', '_')}": f"{p:.4f}"
                       for cls, p in result["all_probs"].items()},
                }
                rows.append(row)
                if (i + 1) % 20 == 0 or (i + 1) == len(image_files):
                    print(f"  {i+1}/{len(image_files)} processed...")
            except Exception as exc:
                print(f"  WARN: could not process {img_path.name}: {exc}")

        # Write CSV
        if rows:
            with open(out_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)
            print(f"\nResults saved to: {out_path.resolve()}")

        # Quick summary
        from collections import Counter
        summary = Counter(r["predicted_class"] for r in rows)
        print("\nPrediction summary:")
        for cls, cnt in sorted(summary.items()):
            print(f"  {cls:<18} {cnt:4d}  ({cnt/len(rows)*100:.1f}%)")

    else:
        print("Provide --image or --folder. Use --help for usage.")
        sys.exit(1)


if __name__ == "__main__":
    main()
