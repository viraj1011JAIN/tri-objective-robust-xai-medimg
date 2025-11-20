# src/datasets/derm7pt.py
from __future__ import annotations

"""
Derm7pt dermoscopy dataset with single-label diagnosis and optional concepts.

Typical (fixture) layout:

    root/
      metadata.csv
      images/
        d1.png
        d2.png
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Union

import numpy as np
import pandas as pd
import torch

from .base_dataset import BaseMedicalDataset, Sample, Split


class Derm7ptDataset(BaseMedicalDataset):
    """
    Derm7pt dermoscopy dataset wrapper.

    Parameters
    ----------
    root:
        Root directory for the Derm7pt data.
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
        Column name for diagnosis labels. Auto-detected if None.
    concept_columns:
        Optional list of concept column names (e.g. ["pigment_network"]).
        Auto-detected if None (searches for numerical columns excluding
        the label column).
    """

    DATASET_NAME = "Derm7pt"

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
        concept_columns: Optional[Sequence[str]] = None,
    ) -> None:
        self.root = Path(root)
        self.csv_path = (
            Path(csv_path) if csv_path is not None else self.root / "metadata.csv"
        )
        self.split_column = split_column
        self.image_column = image_column
        self.label_column = label_column
        self._concept_columns = (
            list(concept_columns) if concept_columns is not None else None
        )

        # Will be filled in _load_metadata
        self.concept_names: List[str] = []
        self.metadata: pd.DataFrame = pd.DataFrame()  # Store the filtered metadata
        self._concept_matrix: Optional[np.ndarray] = None

        super().__init__(root=root, split=split, transforms=transforms)

    def _load_metadata(self) -> None:
        """Load metadata and create sample list."""
        if not self.csv_path.is_file():
            raise FileNotFoundError(
                f"Derm7pt metadata CSV not found at {self.csv_path}"
            )

        df = pd.read_csv(self.csv_path)

        # Validate split column exists
        if self.split_column not in df.columns:
            raise KeyError(
                f"Derm7pt metadata at {self.csv_path} is missing split column "
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
                f"Derm7pt metadata at {self.csv_path} lacks an image column "
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
                f"Derm7pt metadata at {self.csv_path} lacks a label column "
                "(tried: label, labels, diagnosis, target, class, y)"
            )

        # Auto-detect concept columns if not provided
        if self._concept_columns is None:
            # Look for numeric columns that are not the label or split columns
            exclude_cols = {self.split_column, label_col, img_col}
            numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
            self._concept_columns = [c for c in numeric_cols if c not in exclude_cols]

        self._concept_columns = self._concept_columns or []
        self.concept_names = self._concept_columns.copy()

        # BUILD CLASS VOCABULARY FROM ALL DATA (not just current split)
        all_label_values = df[label_col].astype(str).tolist()
        label_names = sorted(set(all_label_values))
        self.class_names = label_names
        name_to_idx = {name: idx for idx, name in enumerate(label_names)}

        # NOW filter by split
        split_series = df[self.split_column].astype(str).str.lower()
        df_split = df.loc[split_series == self.split.value.lower()].reset_index(
            drop=True
        )

        if df_split.empty:
            self.samples = []
            self.metadata = pd.DataFrame()
            self._concept_matrix = np.array([]).reshape(0, len(self.concept_names))
            raise ValueError(
                f"{self.__class__.__name__}: no samples found for split={self.split}"
            )

        # Store the filtered metadata
        self.metadata = df_split.copy()

        # Build concept matrix for this split
        if self._concept_columns:
            concept_data = df_split[self._concept_columns].copy()
            # Replace NaN with 0, and map -1 to 0 (negative/absent concepts)
            concept_data = concept_data.fillna(0.0)
            concept_data = concept_data.replace(-1, 0)
            self._concept_matrix = concept_data.to_numpy(dtype=np.float32)
        else:
            self._concept_matrix = np.array([]).reshape(len(df_split), 0)

        # Create samples
        samples: List[Sample] = []
        for idx, row in df_split.iterrows():
            raw_path = str(row[img_col])
            image_path = self._resolve_image_path(raw_path)

            label_name = str(row[label_col])
            y = torch.tensor(name_to_idx[label_name], dtype=torch.long)

            meta: Dict[str, Any] = {"split": self.split.value, "raw_label": label_name}

            # Add concept annotations if available
            # Add concept annotations if available
            if self._concept_columns:
                concepts = {}
                for col in self._concept_columns:
                    if col in row.index:
                        val = row[col]
                        # Handle NaN and -1
                        if pd.isna(val) or val == -1:
                            val = 0
                        concepts[col] = val
                meta["concepts"] = concepts

            samples.append(Sample(image_path=image_path, label=y, meta=meta))

        self.samples = samples

    @property
    def concept_matrix(self) -> np.ndarray:
        """
        Return concept annotations as a matrix [N, C_concepts].

        Returns
        -------
        np.ndarray
            Float array of shape [num_samples, num_concepts] with values
            typically in {0, 1}. NaN and -1 values are mapped to 0.
        """
        if self._concept_matrix is None:
            return np.array([]).reshape(0, len(self.concept_names))
        return self._concept_matrix

    def compute_class_statistics(self) -> Dict[str, Any]:
        """
        Extend base class statistics with concept information.
        """
        stats = super().compute_class_statistics()
        stats["num_concepts"] = len(self.concept_names)
        if self.concept_names:
            stats["concept_names"] = self.concept_names
        return stats
