# src/datasets/isic.py
from __future__ import annotations

"""
ISIC-style dermoscopy dataset for skin lesion classification.

Typical (fixture) layout:

    root/
      metadata.csv
      images/
        img1.png
        img2.png
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import pandas as pd
import torch

from .base_dataset import BaseMedicalDataset, Sample, Split


class ISICDataset(BaseMedicalDataset):
    """
    ISIC dermoscopy dataset wrapper (single-label classification).

    Parameters
    ----------
    root:
        Root directory for the ISIC data.
    split:
        "train", "val", or "test".
    transforms:
        Albumentations Compose or any callable transform.
    csv_path:
        Path to metadata CSV. Defaults to root/metadata.csv.
    split_column:
        Column name for split identifier (default: "split").
    image_column:
        Column name for image paths. Auto-detected if None.
    label_column:
        Column name for labels/diagnosis. Auto-detected if None.
    """

    DATASET_NAME = "ISIC"

    def __init__(
        self,
        root: Union[str, Path],
        split: Union[Split, str] = Split.TRAIN,
        transforms: Optional[BaseMedicalDataset.Transform] = None,
        *,
        csv_path: Optional[Union[str, Path]] = None,
        split_column: str = "split",
        image_column: Optional[str] = None,
        label_column: Optional[str] = None,
    ) -> None:
        self.root = Path(root)
        self.csv_path = (
            Path(csv_path) if csv_path is not None else self.root / "metadata.csv"
        )
        self.split_column = split_column
        self.image_column = image_column
        self.label_column = label_column

        super().__init__(root=root, split=split, transforms=transforms)

    def _load_metadata(self) -> None:
        """Load metadata and create sample list."""
        if not self.csv_path.is_file():
            raise FileNotFoundError(f"ISIC metadata CSV not found at {self.csv_path}")

        df = pd.read_csv(self.csv_path)

        # Validate split column exists
        if self.split_column not in df.columns:
            raise KeyError(
                f"ISIC metadata at {self.csv_path} is missing split column "
                f"'{self.split_column}'"
            )

        # Auto-detect image column if not provided
        img_col = self.image_column
        if img_col is None:
            for cand in ["image_path", "image", "filename", "file", "image_id"]:
                if cand in df.columns:
                    img_col = cand
                    break
        if img_col is None:
            raise KeyError(
                f"ISIC metadata at {self.csv_path} lacks an image column "
                "(tried: image_path, image, filename, file, image_id)"
            )

        # Auto-detect label column if not provided
        label_col = self.label_column
        if label_col is None:
            for cand in ["label", "labels", "diagnosis", "target", "class", "y"]:
                if cand in df.columns:
                    label_col = cand
                    break
        if label_col is None:
            raise KeyError(
                f"ISIC metadata at {self.csv_path} lacks a label column "
                "(tried: label, labels, diagnosis, target, class, y)"
            )

        # BUILD CLASS VOCABULARY FROM ALL DATA (not just current split)
        # This ensures consistent class_names across splits
        all_label_values = df[label_col].astype(str).tolist()
        label_names: List[str] = sorted(set(all_label_values))
        self.class_names = label_names
        name_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(label_names)
        }

        # NOW filter by split
        split_series = df[self.split_column].astype(str).str.lower()
        df_split = df.loc[split_series == self.split.value.lower()].reset_index(
            drop=True
        )

        if df_split.empty:
            # Still keep class_names even if split is empty
            self.samples = []
            # But we need at least one sample, so raise error
            raise ValueError(
                f"{self.__class__.__name__}: no samples found for split={self.split}"
            )

        # Create samples
        samples: List[Sample] = []
        for _, row in df_split.iterrows():
            raw_path = str(row[img_col])
            image_path = self._resolve_image_path(raw_path)

            label_name = str(row[label_col])
            y = torch.tensor(name_to_idx[label_name], dtype=torch.long)

            meta: Dict[str, Any] = {
                "split": self.split.value,
                "raw_label": label_name,
            }

            samples.append(Sample(image_path=image_path, label=y, meta=meta))

        self.samples = samples
