# src/datasets/__init__.py
"""
Medical imaging datasets for tri-objective robust XAI research.

This package provides PyTorch Dataset implementations for:
- ISIC 2018, 2019, 2020 (skin lesion classification)
- Derm7pt (dermatology 7-point checklist)
- NIH Chest X-Ray (CXR-14)
- PadChest (chest X-ray)

All datasets inherit from BaseMedicalDataset and support:
- Train/val/test splits
- Data augmentation via torchvision transforms
- Class imbalance handling
- Unified interface for medical imaging tasks
"""

from src.datasets.base_dataset import BaseMedicalDataset, Sample, Split
from src.datasets.chest_xray import ChestXRayDataset
from src.datasets.derm7pt import Derm7ptDataset
from src.datasets.isic import ISICDataset

# Backwards compatibility alias
BaseDataset = BaseMedicalDataset

__all__ = [
    "BaseMedicalDataset",
    "BaseDataset",  # Alias for backwards compatibility
    "Sample",
    "Split",
    "ISICDataset",
    "Derm7ptDataset",
    "ChestXRayDataset",
]
