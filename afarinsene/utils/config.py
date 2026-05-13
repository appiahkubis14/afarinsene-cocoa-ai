"""
afarinsene.utils.config
=======================
YAML-based configuration loader with dot-notation access, environment
variable overrides, and CLI ``--set key=value`` patching.

Usage
-----
>>> cfg = load_config("configs/default.yaml")
>>> cfg["training"]["epochs"]
100
>>> # Override from CLI:  --set training.epochs=200
>>> cfg = load_config("configs/default.yaml", overrides=["training.epochs=200"])
"""

from __future__ import annotations

import copy
import os
from pathlib import Path
from typing import Any

import yaml


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_config(
    config_path: str | Path,
    overrides: list[str] | None = None,
    data_dir: str | None = None,
) -> dict[str, Any]:
    """
    Load a YAML config file and apply optional overrides.

    Parameters
    ----------
    config_path:
        Path to the YAML file (e.g. ``"configs/default.yaml"``).
    overrides:
        List of ``"key.subkey=value"`` strings (from ``--set``).
        Values are cast to int, float, or bool when possible.
    data_dir:
        If provided, sets ``cfg["data"]["dir"]``, overriding the YAML
        value and the ``DATA_DIR`` env var.

    Returns
    -------
    dict
        The merged configuration.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path.resolve()}")

    with open(path, "r", encoding="utf-8") as f:
        cfg: dict[str, Any] = yaml.safe_load(f) or {}

    # ---- Environment variable overrides -----------------------------------
    env_data_dir = os.environ.get("DATA_DIR")
    if env_data_dir:
        cfg.setdefault("data", {})["dir"] = env_data_dir

    # ---- CLI --data-dir override ------------------------------------------
    if data_dir:
        cfg.setdefault("data", {})["dir"] = data_dir

    # ---- --set key=value overrides ----------------------------------------
    for override in overrides or []:
        if "=" not in override:
            raise ValueError(f"Invalid override (expected key=value): {override!r}")
        key_path, raw_value = override.split("=", 1)
        keys = key_path.strip().split(".")
        value = _cast(raw_value.strip())
        _set_nested(cfg, keys, value)

    return cfg


def merge_configs(base: dict, override: dict) -> dict:
    """
    Deep-merge *override* into *base* (non-destructive; returns new dict).
    """
    result = copy.deepcopy(base)
    for k, v in override.items():
        if k in result and isinstance(result[k], dict) and isinstance(v, dict):
            result[k] = merge_configs(result[k], v)
        else:
            result[k] = copy.deepcopy(v)
    return result


def pretty_print(cfg: dict, indent: int = 0) -> None:
    """Print the config in a readable tree format."""
    for key, value in cfg.items():
        if isinstance(value, dict):
            print(" " * indent + f"{key}:")
            pretty_print(value, indent + 2)
        else:
            print(" " * indent + f"{key}: {value!r}")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _cast(value: str) -> Any:
    """Try to cast a string to int, float, bool, None, or keep as str."""
    if value.lower() in ("true", "yes"):
        return True
    if value.lower() in ("false", "no"):
        return False
    if value.lower() in ("null", "none", "~"):
        return None
    try:
        return int(value)
    except ValueError:
        pass
    try:
        return float(value)
    except ValueError:
        pass
    return value


def _set_nested(d: dict, keys: list[str], value: Any) -> None:
    """Set a nested dict value given a list of keys."""
    for key in keys[:-1]:
        d = d.setdefault(key, {})
    d[keys[-1]] = value
