"""
afarinsene.models.cnn
======================
Stage 4 of the pipeline: construct the Afarinsene CNN.

Architecture summary
--------------------
4 convolutional blocks, each with:
  Conv2D → BatchNorm → ReLU → Conv2D → BatchNorm → ReLU → MaxPool → Dropout

Followed by:
  GlobalAveragePooling2D
  Dense(256, L2) → BatchNorm → Dropout(0.5)
  Dense(128, L2) → BatchNorm → Dropout(0.3)
  Dense(num_classes, softmax)

Total trainable parameters: ≈ 3.5 M (256×256) or ≈ 3.5 M (224×224)
The bottleneck is GlobalAveragePooling, not Flatten, so parameter count
is independent of spatial resolution.

Usage
-----
>>> from afarinsene.models.cnn import build_model
>>> model = build_model(cfg, num_classes=4)
>>> model.summary()
"""

from __future__ import annotations

from typing import Any

from afarinsene.utils.logger import get_logger

log = get_logger(__name__)


def build_model(cfg: dict[str, Any], num_classes: int):
    """
    Build and compile the Afarinsene CNN.

    Parameters
    ----------
    cfg:
        Top-level config dict (reads ``cfg["model"]`` and ``cfg["training"]``).
    num_classes:
        Number of output classes (derived from the dataset).

    Returns
    -------
    tf.keras.Model
        Compiled model ready for training.
    """
    import tensorflow as tf
    from tensorflow.keras import regularizers
    from tensorflow.keras.layers import (
        BatchNormalization,
        Conv2D,
        Dense,
        Dropout,
        GlobalAveragePooling2D,
        Input,
        MaxPooling2D,
    )
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.optimizers import Adam

    mcfg = cfg.get("model", {})
    tcfg = cfg.get("training", {})
    dcfg = cfg.get("data", {})

    image_size: int = dcfg.get("image_size", 256)
    channels: int = dcfg.get("channels", 3)

    blocks: list[dict] = mcfg.get("blocks", [
        {"filters": 32},
        {"filters": 64},
        {"filters": 128},
        {"filters": 256},
    ])
    dense_units: list[int] = mcfg.get("dense_units", [256, 128])
    dropout_rates: list[float] = mcfg.get("dropout_rates", [0.5, 0.3])
    l2: float = mcfg.get("l2_regularization", 1e-4)
    clip_norm = mcfg.get("gradient_clip_norm", 1.0)

    lr: float = tcfg.get("optimizer", {}).get("learning_rate", 0.001)
    wd: float = tcfg.get("optimizer", {}).get("weight_decay", 1e-4)

    log.info("=" * 55)
    log.info("STAGE 4 — Model Construction")
    log.info("  Input  : %d × %d × %d", image_size, image_size, channels)
    log.info("  Blocks : %s", [b["filters"] for b in blocks])
    log.info("  Dense  : %s", dense_units)
    log.info("  Classes: %d", num_classes)
    log.info("=" * 55)

    reg = regularizers.l2(l2)
    input_shape = (image_size, image_size, channels)

    layers = []

    for i, block in enumerate(blocks):
        f = block["filters"]
        if i == 0:
            # First conv needs explicit input_shape
            layers += [
                Conv2D(f, (3, 3), activation="relu", padding="same",
                       input_shape=input_shape),
            ]
        else:
            layers += [Conv2D(f, (3, 3), activation="relu", padding="same")]

        layers += [
            BatchNormalization(),
            Conv2D(f, (3, 3), activation="relu", padding="same"),
            BatchNormalization(),
            MaxPooling2D((2, 2)),
            Dropout(0.25),
        ]

    layers.append(GlobalAveragePooling2D())

    for units, drop_rate in zip(dense_units, dropout_rates):
        layers += [
            Dense(units, activation="relu", kernel_regularizer=reg),
            BatchNormalization(),
            Dropout(drop_rate),
        ]

    layers.append(Dense(num_classes, activation="softmax"))

    model = Sequential(layers, name="AfarinsenneCNN")

    # ---- Optimizer -------------------------------------------------------
    optimizer_cfg = tcfg.get("optimizer", {})
    opt_name = optimizer_cfg.get("name", "adamw").lower()

    if opt_name in ("adamw", "adam"):
        optimizer = Adam(
            learning_rate=lr,
            clipnorm=clip_norm if clip_norm else None,
        )
    else:
        raise ValueError(f"Unsupported optimizer: {opt_name!r}")

    model.compile(
        optimizer=optimizer,
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )

    param_count = model.count_params()
    log.info("  Model built — %s trainable parameters", f"{param_count:,}")

    return model
