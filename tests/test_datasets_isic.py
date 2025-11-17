# tests/test_datasets_isic.py
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.datasets.isic import ISICDataset
from src.datasets.transforms import get_isic_transforms


def _create_dummy_isic_root(tmp_path: Path) -> Path:
    root = tmp_path / "isic_2018"
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Create three tiny RGB images
    for name in ["img1.jpg", "img2.jpg", "img3.jpg"]:
        img = np.full((16, 16, 3), 127, dtype=np.uint8)
        cv2.imwrite(str(images_dir / name), img)

    # Minimal metadata with split + label
    df = pd.DataFrame(
        {
            "image_id": ["img1", "img2", "img3"],
            "label": ["nevus", "melanoma", "nevus"],
            "split": ["train", "val", "test"],
        }
    )
    df.to_csv(root / "metadata.csv", index=False)
    return root


def test_isic_dataset_splits_and_labels(tmp_path: Path) -> None:
    root = _create_dummy_isic_root(tmp_path)
    t_train = get_isic_transforms(split="train", image_size=32)
    t_val = get_isic_transforms(split="val", image_size=32)

    ds_train = ISICDataset(root=root, split="train", transforms=t_train)
    ds_val = ISICDataset(root=root, split="val", transforms=t_val)

    assert len(ds_train) == 1
    assert len(ds_val) == 1
    assert set(ds_train.class_names) == {"melanoma", "nevus"}

    x, y, meta = ds_train[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape[0] == 3  # C
    assert isinstance(y, torch.Tensor)
    assert y.dtype == torch.long
    assert meta["split"] == "train"
    assert meta["path"].endswith(".jpg")

    # Validate should not report missing files
    summary = ds_train.validate(strict=False)
    assert summary["num_missing_files"] == 0


def test_isic_class_statistics(tmp_path: Path) -> None:
    root = _create_dummy_isic_root(tmp_path)
    ds_train = ISICDataset(root=root, split="train", transforms=None)

    stats = ds_train.compute_class_statistics()
    assert stats["dataset"] == "ISIC"
    assert stats["split"] == "train"
    assert stats["num_samples"] == 1
    assert len(stats["class_names"]) == ds_train.num_classes
    assert len(stats["class_counts"]) == ds_train.num_classes

    weights = torch.tensor(stats["class_weights"], dtype=torch.float32)
    assert (weights > 0).all()
