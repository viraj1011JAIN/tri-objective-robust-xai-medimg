# src/datasets/chest_xray.py
from __future__ import annotations

"""
Multi-label chest X-ray dataset (NIH + PadChest).

Test fixture layout (simplified):

    root/
      metadata.csv   # image_path, labels, split, dataset
      images/
        cxr1.png
        cxr2.png
"""

from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Union

import pandas as pd
import torch

from .base_dataset import BaseMedicalDataset, Sample, Split


class ChestXRayDataset(BaseMedicalDataset):
    """
    Multi-label chest X-ray dataset supporting NIH + PadChest style metadata.

    Parameters
    ----------
    root:
        Root directory for the chest X-ray data.
    split:
        "train", "val", or "test".
    transforms:
        Albumentations Compose or any callable transform.
    csv_path:
        Path to a metadata CSV with at least:
            image_path, labels, split
    allowed_datasets:
        Optional list of dataset names to include (e.g. ["NIH", "PadChest"]).
    image_path_column:
        Column name for image paths (default: "image_path").
    labels_column:
        Column name for labels (default: "labels"). Expects pipe-separated
        strings like "Pneumonia|Effusion" or single labels.
    dataset_column:
        Optional column name for dataset identifier (e.g. "NIH", "PadChest").
    label_separator:
        Character separating multiple labels (default: "|").
    label_harmonization:
        Optional dict mapping label names to standardized versions
        (e.g. {"pneumothorax": "Pneumothorax"}).
    split_column:
        Column name for split identifier (default: "split").
    """

    DATASET_NAME = "ChestXRay"

    def __init__(
        self,
        root: Union[str, Path],
        split: Union[Split, str] = Split.TRAIN,
        transforms: Optional[BaseMedicalDataset.Transform] = None,
        *,
        csv_path: Union[str, Path],
        image_path_column: str = "image_path",
        labels_column: str = "labels",
        dataset_column: Optional[str] = "dataset",
        allowed_datasets: Optional[Sequence[str]] = None,
        label_separator: str = "|",
        label_harmonization: Optional[Mapping[str, str]] = None,
        split_column: str = "split",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.image_path_column = image_path_column
        self.labels_column = labels_column
        self.dataset_column = dataset_column
        self.allowed_datasets = (
            {d.lower() for d in allowed_datasets}
            if allowed_datasets is not None
            else None
        )
        self.label_separator = label_separator
        self.label_harmonization = (
            {k.lower(): v for k, v in label_harmonization.items()}
            if label_harmonization
            else None
        )
        self.split_column = split_column

        super().__init__(root=root, split=split, transforms=transforms)

    # ------------------------------------------------------------------
    # Metadata loading
    # ------------------------------------------------------------------
    def _load_metadata(self) -> None:
        """Load metadata and create sample list."""
        df = pd.read_csv(self.csv_path)

        # Optional filtering by dataset (e.g. NIH vs PadChest)
        if (
            self.dataset_column is not None
            and self.dataset_column in df.columns
            and self.allowed_datasets is not None
        ):
            ds_series = df[self.dataset_column].astype(str).str.lower()
            df = df.loc[ds_series.isin(self.allowed_datasets)].reset_index(drop=True)

        # Helper to parse label strings
        def _parse_label_string(raw: str) -> List[str]:
            """Parse pipe-separated label string into list of labels."""
            raw = str(raw)
            if not raw or raw.lower() in {"nan", "none"}:
                return []
            parts = [p.strip() for p in raw.split(self.label_separator)]
            parts = [p for p in parts if p]
            if self.label_harmonization:
                parts = [self.label_harmonization.get(p.lower(), p) for p in parts]
            return parts

        # BUILD COMPLETE CLASS VOCABULARY FROM ALL DATA (not just current split)
        all_labels_set: set[str] = set()
        for _, row in df.iterrows():
            labels_raw = row[self.labels_column]
            labels = _parse_label_string(labels_raw)
            all_labels_set.update(labels)

        class_names = sorted(all_labels_set)
        self.class_names = class_names
        name_to_idx: Dict[str, int] = {
            name: idx for idx, name in enumerate(class_names)
        }

        # NOW filter by split
        split_str = self.split.value.lower()
        split_series = df[self.split_column].astype(str).str.lower()
        df_split = df.loc[split_series == split_str].reset_index(drop=True)

        if df_split.empty:
            self.samples = []
            raise ValueError(
                f"{self.__class__.__name__}: no samples found for split={self.split}"
            )

        # ------------------------------------------------------------------
        # Create Sample objects with multi-hot label vectors
        # ------------------------------------------------------------------
        samples: List[Sample] = []

        for _, row in df_split.iterrows():
            raw_path = str(row[self.image_path_column]).strip()
            image_path = self._resolve_image_path(raw_path)

            labels_raw = row[self.labels_column]
            labels = _parse_label_string(labels_raw)

            # Create multi-hot vector
            y = torch.zeros(len(class_names), dtype=torch.float32)
            for name in labels:
                idx = name_to_idx[name]
                y[idx] = 1.0

            meta = {
                "split": self.split.value,
                "dataset": (
                    row.get(self.dataset_column) if self.dataset_column else None
                ),
                "raw_labels": labels,
            }

            samples.append(
                Sample(
                    image_path=image_path,
                    label=y,
                    meta=meta,
                )
            )

        self.samples = samples

    # ------------------------------------------------------------------
    # Simple statistics helpers (handy for tests & EDA)
    # ------------------------------------------------------------------
    def positive_rates(self) -> torch.Tensor:
        """
        Fraction of images in this split that are positive for each label.

        Returns
        -------
        torch.Tensor
            1D tensor of shape [num_classes] with values in [0, 1].
        """
        labels = self._stack_labels().to(torch.float32)
        if labels.ndim != 2:
            raise ValueError("positive_rates expects multi-label [N, C] targets.")
        return labels.mean(dim=0)

    def get_positive_rates(self) -> Dict[str, float]:
        """
        Alias for positive_rates that returns a dict {class_name: rate}.

        Returns
        -------
        dict
            Dictionary mapping class names to their positive rates.
        """
        rates_tensor = self.positive_rates()
        return {
            name: float(rates_tensor[i].item())
            for i, name in enumerate(self.class_names)
        }
