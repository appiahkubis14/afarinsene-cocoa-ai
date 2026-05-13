"""
afarinsene.utils.logger
=======================
Structured logging with optional Rich console output and a file sink.

Usage
-----
>>> from afarinsene.utils.logger import get_logger
>>> log = get_logger(__name__)
>>> log.info("Training started")
"""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path
from typing import Optional

_LOGGERS: dict[str, logging.Logger] = {}


def get_logger(
    name: str,
    level: str = "INFO",
    log_file: Optional[str] = None,
    rich_console: bool = True,
) -> logging.Logger:
    """
    Return (or create) a named logger.

    Parameters
    ----------
    name:
        Typically ``__name__`` of the calling module.
    level:
        One of ``DEBUG``, ``INFO``, ``WARNING``, ``ERROR``.
    log_file:
        If provided, also write logs to this file path (appended).
    rich_console:
        Use Rich's pretty handler if the ``rich`` package is available.

    Returns
    -------
    logging.Logger
    """
    if name in _LOGGERS:
        return _LOGGERS[name]

    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False  # don't bubble up to root

    fmt = logging.Formatter(
        "%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # ---- Console handler ------------------------------------------------
    if rich_console:
        try:
            from rich.logging import RichHandler  # type: ignore

            console_handler = RichHandler(
                level=getattr(logging, level.upper(), logging.INFO),
                rich_tracebacks=True,
                markup=True,
                show_path=False,
            )
        except ImportError:
            console_handler = logging.StreamHandler(sys.stdout)
            console_handler.setFormatter(fmt)
    else:
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setFormatter(fmt)

    logger.addHandler(console_handler)

    # ---- File handler ---------------------------------------------------
    if log_file:
        Path(log_file).parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(fmt)
        logger.addHandler(file_handler)

    _LOGGERS[name] = logger
    return logger


def configure_from_config(cfg: dict) -> None:
    """
    Call once at startup to apply logging settings from the YAML config.

    Parameters
    ----------
    cfg:
        The top-level config dict (``cfg["logging"]`` is read).
    """
    logging_cfg = cfg.get("logging", {})
    level = logging_cfg.get("level", "INFO")
    log_file = logging_cfg.get("log_file", None)
    rich_console = logging_cfg.get("rich_console", True)

    if log_file:
        os.makedirs(Path(log_file).parent, exist_ok=True)

    root = get_logger("afarinsene", level=level, log_file=log_file, rich_console=rich_console)
    root.info("Logging initialised — level=%s", level)
