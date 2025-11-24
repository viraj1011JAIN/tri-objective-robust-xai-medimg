# src/datasets/base_dataset.py
from __future__ import annotations

"""
Shared abstractions for all medical-image datasets in this project.

This module provides:
- Split: enum for train/val/test splits
- BaseMedicalDataset: abstract base class with:
    * common attributes (root, split, transform, class_names, etc.)
    * __len__/__getitem__ wired to image loading + transforms
    * utilities for computing class weights for imbalance handling
    * basic class statistics for tests / EDA
"""

from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


class Split(str, Enum):
    """Standard dataset split identifiers."""

    TRAIN = "train"
    VAL = "val"
    TEST = "test"

    @classmethod
    def from_str(cls, value: str) -> "Split":
        value = value.lower()
        if value in {"train", "training"}:
            return cls.TRAIN
        if value in {"val", "valid", "validation"}:
            return cls.VAL
        if value in {"test", "testing"}:
            return cls.TEST
        raise ValueError(f"Unknown split: {value}")


@dataclass
class Sample:
    """
    Lightweight container for a single dataset sample.

    Attributes
    ----------
    image_path:
        Path to the image file on disk.
    label:
        Class label. For single-label tasks this is an int tensor of
        shape [] or [1]. For multi-label tasks this is a float/bool
        tensor of shape [num_classes] with 0/1 entries.
    meta:
        Optional additional metadata (e.g. patient id, study id).
    """

    image_path: Path
    label: torch.Tensor
    meta: Optional[Dict[str, Any]] = None


class BaseMedicalDataset(Dataset):
    """
    Abstract base class for all medical-image datasets in this project.

    Subclasses **must** implement `_load_metadata`, which is responsible
    for populating:

        - `self.samples`: a list of `Sample` objects
        - `self.class_names`: list of human-readable class names

    After `_load_metadata` has run, this base class will:

        - build `class_to_idx`
        - set `num_classes`
        - expose `__len__` and `__getitem__`
        - provide `compute_class_weights` and `compute_class_statistics`
    """

    # Type alias for transforms: anything callable taking an image and
    # returning a tensor or transformed image.
    Transform = Callable[..., Any]

    def __init__(
        self,
        root: Union[str, Path],
        split: Union[Split, str] = Split.TRAIN,
        *,
        transform: Optional[Transform] = None,
        transforms: Optional[Transform] = None,
        **kwargs: Any,
    ) -> None:
        """
        Initialize the dataset.

        Parameters
        ----------
        root:
            Root directory for this dataset.
        split:
            Split identifier (e.g. "train", "val", "test").
        transform:
            Optional transform/augmentation pipeline. Accepts either
            `transform=` or `transforms=` for compatibility.
        transforms:
            Alias for `transform`. Only one should be provided.
        **kwargs:
            Additional keyword arguments (ignored, for forward compatibility).
        """
        super().__init__()

        # Handle both transform= and transforms= for compatibility
        if transform is not None and transforms is not None:
            raise ValueError("Pass only one of 'transform' or 'transforms', not both.")

        # Prefer transforms if provided, else use transform
        if transforms is not None:
            transform = transforms

        # Normalize root and split
        self.root: Path = Path(root)
        self.split: Split = (
            split if isinstance(split, Split) else Split.from_str(str(split))
        )

        # Store under both attribute names for backward compatibility
        self.transform: Optional[BaseMedicalDataset.Transform] = transform
        self.transforms: Optional[BaseMedicalDataset.Transform] = transform

        # Populated by subclasses in _load_metadata()
        self.samples: List[Sample] = []
        self.class_names: List[str] = []

        # Filled by _finalize_metadata()
        self.class_to_idx: Dict[str, int] = {}
        self.num_classes: int = 0

        # Lazily cached statistics
        self._class_weights: Optional[torch.Tensor] = None

        # Let subclasses read metadata / build sample list.
        self._load_metadata()
        self._finalize_metadata()

    # ------------------------------------------------------------------
    # Methods subclasses must implement
    # ------------------------------------------------------------------
    def _load_metadata(self) -> None:  # pragma: no cover - abstract template
        """
        Populate `self.samples` and `self.class_names`.

        Typical implementation workflow:
            1. Parse CSV / Excel metadata for ALL data.
            2. Build complete class vocabulary from ALL data.
            3. Filter samples by the requested split.
            4. For each row in the split, create a `Sample(image_path, label_tensor, meta)`.
            5. Append to `self.samples`.

        This method should **not** perform any heavy preprocessing (no
        resizing, caching tensors, etc.); that belongs in a separate
        preprocessing pipeline, not in the dataset constructor.
        """
        raise NotImplementedError(
            f"{self.__class__.__name__} must implement _load_metadata()"
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------
    def _finalize_metadata(self) -> None:
        """Run consistency checks and derive basic label statistics."""
        if not self.samples:
            raise ValueError(
                f"{self.__class__.__name__}: no samples found for split={self.split}"
            )

        if self.class_names:
            self.class_to_idx = {name: idx for idx, name in enumerate(self.class_names)}
            self.num_classes = len(self.class_names)
        else:
            # Infer num_classes from labels if not provided explicitly.
            labels = self._stack_labels()
            if labels.ndim == 1:
                self.num_classes = int(labels.max().item()) + 1
            else:
                self.num_classes = int(labels.size(1))

    def _stack_labels(self) -> torch.Tensor:
        """Stack all labels into a single tensor [N] or [N, C]."""
        if not self.samples:
            raise ValueError("Cannot stack labels: no samples present.")
        labels = [s.label for s in self.samples]
        return torch.stack(labels, dim=0)

    # ------------------------------------------------------------------
    # Image path resolution / loading
    # ------------------------------------------------------------------
    def _resolve_image_path(self, raw: Union[str, Path]) -> Path:
        """
        Best-effort resolver for image paths coming from metadata.

        Handles cases where metadata stores:
        - "img.png" (with extension)
        - "img" (without extension - will search for common extensions)
        - "images/img.png"
        - "subdir/img.png"
        - an absolute path
        and where images may live directly under `root`, `root / 'images'`,
        or deeper subdirectories (fixtures sometimes move things around).
        """
        p = Path(raw)

        # Absolute path case
        if p.is_absolute() and p.is_file():
            return p

        candidates: List[Path] = []

        # Relative to root
        candidates.append(self.root / p)

        # Just the basename under root
        if p.name != str(p):
            candidates.append(self.root / p.name)

        # basename under root / "images"
        candidates.append(self.root / "images" / p.name)

        # full relative under root / "images"
        if p.parent != Path("."):
            candidates.append(self.root / "images" / p)

        # Check all candidates
        for cand in candidates:
            if cand.is_file():
                return cand

        # If no extension provided, try common image extensions
        if not p.suffix:
            common_extensions = [
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".gif",
                ".tiff",
                ".webp",
            ]

            # Try adding extensions to all candidate paths
            for cand in candidates:
                for ext in common_extensions:
                    cand_with_ext = cand.parent / f"{cand.name}{ext}"
                    if cand_with_ext.is_file():
                        return cand_with_ext

        # LAST-RESORT FALLBACK:
        # Search recursively under root for a file with the same basename
        # (with or without extension).
        # This is mainly to keep tests robust to small path mismatches
        # between fixtures and metadata.

        # First try exact match
        matches = list(self.root.rglob(p.name))
        if matches:
            return matches[0]

        # Then try with common extensions if no extension provided
        if not p.suffix:
            for ext in common_extensions:
                matches = list(self.root.rglob(f"{p.name}{ext}"))
                if matches:
                    return matches[0]

        # If still nothing, return the first candidate (will likely error
        # later, but at least it's deterministic).
        return candidates[0]

    def _load_image(self, path: Path) -> np.ndarray:
        """
        Load an image from disk as a numpy array (RGB, uint8).

        Returns
        -------
        np.ndarray
            Image as numpy array of shape (H, W, 3) with dtype uint8.
            This format is compatible with Albumentations 2.x.
        """
        if not path.is_file():
            path = self._resolve_image_path(str(path))

        # Load as PIL, convert to RGB, then to numpy array
        pil_img = Image.open(path).convert("RGB")
        return np.array(pil_img, dtype=np.uint8)

    # ------------------------------------------------------------------
    # Dataset interface
    # ------------------------------------------------------------------
    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, Dict[str, Any]]:
        """
        Return (image, label, meta) tuple.

        Returns
        -------
        tuple:
            - image: transformed image (torch.Tensor)
            - label: torch.Tensor
            - meta: dict with at least {"split": ..., "path": ...}
        """
        sample = self.samples[index]
        image = self._load_image(sample.image_path)

        if self.transform is not None:
            # Albumentations 2.x expects keyword arguments
            try:
                out = self.transform(image=image)
                # Handle Albumentations-style dict output
                if isinstance(out, dict):
                    image = out.get("image", out)
                else:
                    image = out
            except (TypeError, KeyError):
                # Fallback for non-Albumentations transforms
                image = self.transform(image)

        # Merge sample metadata with path
        meta = (sample.meta or {}).copy()
        meta["path"] = str(sample.image_path)
        meta["split"] = self.split.value

        return image, sample.label, meta

    # ------------------------------------------------------------------
    # Imbalance handling / statistics
    # ------------------------------------------------------------------
    def compute_class_weights(self) -> torch.Tensor:
        """
        Compute inverse-frequency class weights.

        For single-label classification, assumes labels are integer class
        ids [N]. For multi-label classification, assumes labels are
        {0,1} vectors of shape [N, C] and uses per-class positive counts.

        Returns
        -------
        torch.Tensor
            1D tensor of shape [num_classes] with normalized weights.
        """
        labels = self._stack_labels().to(torch.float32)

        if labels.ndim == 1:
            # Single-label case: integer class ids
            num_classes = int(labels.max().item()) + 1
            counts = torch.bincount(labels.to(torch.long), minlength=num_classes)
        else:
            # Multi-label: sum positive examples per class
            counts = labels.sum(dim=0)
            num_classes = counts.numel()

        counts = counts.clamp(min=1.0)  # avoid division by zero
        weights = 1.0 / counts
        # Normalize so that mean weight ~= 1
        weights = weights * (num_classes / weights.sum())
        self._class_weights = weights
        return weights

    @property
    def class_weights(self) -> torch.Tensor:
        """Cached view of `compute_class_weights`."""
        if self._class_weights is None:
            self._class_weights = self.compute_class_weights()
        return self._class_weights

    def compute_class_statistics(self) -> Dict[str, Any]:
        """
        Compute simple per-class statistics for tests and EDA.

        Returns
        -------
        dict with keys:
            - "dataset": dataset name (DATASET_NAME or class name)
            - "split": split name ("train"/"val"/"test")
            - "num_samples": int
            - "num_classes": int
            - "class_names": list[str]
            - "class_counts": list[float]
            - "class_weights": list[float]
            - "positive_rates": list[float]
        """
        labels = self._stack_labels().to(torch.float32)
        num_samples = int(labels.shape[0])

        if labels.ndim == 1:
            # Single-label case
            num_classes = int(labels.max().item()) + 1 if num_samples > 0 else 0
            counts = torch.bincount(labels.to(torch.long), minlength=num_classes).to(
                torch.float32
            )
            positive_rates = counts / max(num_samples, 1)
        else:
            # Multi-label: labels shape [N, C] with 0/1 entries
            num_classes = int(labels.size(1))
            counts = labels.sum(dim=0)
            positive_rates = counts / max(num_samples, 1)

        # Compute class weights
        weights = self.compute_class_weights()

        dataset_name = getattr(self, "DATASET_NAME", self.__class__.__name__)

        stats: Dict[str, Any] = {
            "dataset": dataset_name,
            "split": self.split.value,
            "num_samples": num_samples,
            "num_classes": int(num_classes),
            "class_names": (
                list(self.class_names)
                if self.class_names
                else [str(i) for i in range(int(num_classes))]
            ),
            "class_counts": counts.tolist(),
            "class_weights": weights.tolist(),
            "positive_rates": positive_rates.tolist(),
        }
        return stats

    def validate(self, strict: bool = True) -> Dict[str, Any]:
        """
        Validate dataset integrity.

        Parameters
        ----------
        strict:
            If True, raise errors on validation failures.
            If False, just report issues in the summary.

        Returns
        -------
        dict:
            Summary with keys:
                - "num_samples": int
                - "num_missing_files": int
                - "missing_files": list[str]
                - "is_valid": bool
        """
        missing_files: List[str] = []

        for sample in self.samples:
            if not sample.image_path.is_file():
                missing_files.append(str(sample.image_path))

        is_valid = len(missing_files) == 0

        summary = {
            "num_samples": len(self.samples),
            "num_missing_files": len(missing_files),
            "missing_files": missing_files,
            "is_valid": is_valid,
        }

        if strict and not is_valid:
            raise FileNotFoundError(
                f"Validation failed: {len(missing_files)} missing files. "
                f"First missing: {missing_files[0]}"
            )

        return summary


# Backwards compatibility alias for tests
BaseDataset = BaseMedicalDataset


__all__ = [
    "BaseMedicalDataset",
    "BaseDataset",  # Alias
    "Sample",
    "Split",
]
