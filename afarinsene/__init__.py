"""
Afarinsene — Cocoa Disease Detection Pipeline
=============================================
End-to-end deep learning pipeline for detecting CSSVD and Black Pod disease
from smartphone cocoa leaf / pod images.

Typical usage
-------------
>>> from afarinsene.utils.config import load_config
>>> from afarinsene.data.loader import DatasetLoader
>>> cfg = load_config("configs/default.yaml")
>>> loader = DatasetLoader(cfg)
>>> X, y = loader.load()
"""

__version__ = "1.0.0"
__author__ = "KNUST Geomatic Engineering"
__license__ = "MIT"

# Surface the most commonly used symbols at package level
from afarinsene.utils.config import load_config  # noqa: F401
from afarinsene.utils.logger import get_logger    # noqa: F401
