# src/__init__.py
from __future__ import annotations

"""
Top-level package for tri-objective-robust-xai-medimg.

We keep this light to avoid heavy side effects on `import src`, but
we expose a few core dataset abstractions and convenience aliases.
"""

from importlib import metadata as _metadata

# ---------------------------------------------------------------------
# Package version (used when the project is installed as a package)
# ---------------------------------------------------------------------
try:  # pragma: no cover - only used in installed environments
    __version__ = _metadata.version("tri_objective_robust_xai_medimg")
except _metadata.PackageNotFoundError:  # pragma: no cover
    __version__ = "0.0.0"

# ---------------------------------------------------------------------
# Dataset abstractions + concrete dataset classes
# ---------------------------------------------------------------------
# Base abstract dataset + split enum
from .datasets.base_dataset import BaseMedicalDataset, Split  # noqa: E402,F401
from .datasets.chest_xray import ChestXRayDataset  # noqa: E402,F401
from .datasets.derm7pt import Derm7ptDataset  # noqa: E402,F401

# Concrete datasets
from .datasets.isic import ISICDataset  # noqa: E402,F401

__all__ = [
    "BaseMedicalDataset",
    "Split",
    "ISICDataset",
    "Derm7ptDataset",
    "ChestXRayDataset",
    "__version__",
]
