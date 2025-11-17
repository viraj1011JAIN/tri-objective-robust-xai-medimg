# tests/test_datasets_derm7pt.py
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.datasets.derm7pt import Derm7ptDataset
from src.datasets.transforms import get_derm7pt_transforms


def _create_dummy_derm7pt_root(tmp_path: Path) -> Path:
    root = tmp_path / "derm7pt"
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Two tiny images
    for name in ["d1.jpg", "d2.jpg"]:
        img = np.full((16, 16, 3), 200, dtype=np.uint8)
        cv2.imwrite(str(images_dir / name), img)

    # Metadata with diagnosis + 7-point concepts, including missing/negative values
    df = pd.DataFrame(
        {
            "image_id": ["d1", "d2"],
            "diagnosis": ["melanoma", "nevus"],
            "split": ["train", "val"],
            "pigment_network": [1, -1],
            "negative_network": [0, 1],
            "streaks": [np.nan, 1],
            "dots_globules": [1, 0],
            "regression_structures": [0, 0],
            "blue_whitish_veil": [1, 0],
            "vascular_structures": [0, 1],
        }
    )
    df.to_csv(root / "metadata.csv", index=False)
    return root


def test_derm7pt_dataset_and_concepts(tmp_path: Path) -> None:
    root = _create_dummy_derm7pt_root(tmp_path)
    t_train = get_derm7pt_transforms(split="train", image_size=32)

    ds_train = Derm7ptDataset(root=root, split="train", transforms=t_train)

    assert len(ds_train) == 1
    assert ds_train.num_classes >= 2

    x, y, meta = ds_train[0]
    assert isinstance(x, torch.Tensor)
    assert isinstance(y, torch.Tensor)
    assert meta["split"] == "train"

    # Concept matrix: N x C_concepts
    concepts = ds_train.concept_matrix
    assert concepts.shape[0] == len(ds_train.metadata)
    assert concepts.shape[1] == len(ds_train.concept_names)

    # No NaNs, -1 should have been mapped to 0
    assert not np.isnan(concepts).any()
    assert (concepts >= 0).all()

    stats = ds_train.compute_class_statistics()
    assert stats["dataset"] == "Derm7pt"
    assert "num_concepts" in stats
    assert stats["num_concepts"] == concepts.shape[1]
