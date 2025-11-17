# tests/test_datasets_chest_xray.py
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import torch

from src.datasets.chest_xray import ChestXRayDataset
from src.datasets.transforms import get_chest_xray_transforms


def _create_dummy_chest_root(tmp_path: Path) -> Path:
    root = tmp_path / "chest"
    images_dir = root / "images"
    images_dir.mkdir(parents=True, exist_ok=True)

    # Three tiny grayscale images
    for name in ["cxr1.png", "cxr2.png", "cxr3.png"]:
        img = np.full((16, 16), 150, dtype=np.uint8)
        cv2.imwrite(str(images_dir / name), img)

    # Metadata: NIH + PadChest mix, multi-label strings
    df = pd.DataFrame(
        {
            "image_path": [
                "images/cxr1.png",
                "images/cxr2.png",
                "images/cxr3.png",
            ],
            "labels": [
                "Cardiomegaly|Edema",
                "No Finding",
                "Pneumonia",
            ],
            "dataset": ["NIH", "NIH", "PadChest"],
            "split": ["train", "val", "test"],
        }
    )
    csv_path = root / "metadata.csv"
    df.to_csv(csv_path, index=False)
    return root


def test_chest_xray_multilabel_and_stats(tmp_path: Path) -> None:
    root = _create_dummy_chest_root(tmp_path)
    csv_path = root / "metadata.csv"

    t_train = get_chest_xray_transforms(split="train", image_size=32)
    t_val = get_chest_xray_transforms(split="val", image_size=32)

    ds_train = ChestXRayDataset(
        root=root,
        split="train",
        transforms=t_train,
        csv_path=csv_path,
        allowed_datasets=["NIH", "PadChest"],
    )
    ds_val = ChestXRayDataset(
        root=root,
        split="val",
        transforms=t_val,
        csv_path=csv_path,
        allowed_datasets=["NIH", "PadChest"],
    )

    assert len(ds_train) == 1
    assert len(ds_val) == 1
    assert ds_train.num_classes >= 2

    x, y, meta = ds_train[0]
    assert isinstance(x, torch.Tensor)
    assert x.shape[0] == 3  # 3-channel CXR
    assert isinstance(y, torch.Tensor)
    assert y.ndim == 1
    assert meta["split"] == "train"

    stats = ds_train.compute_class_statistics()
    assert stats["dataset"] == "ChestXRay"
    assert stats["num_samples"] == 1
    assert len(stats["class_names"]) == ds_train.num_classes

    rates = ds_train.get_positive_rates()
    assert isinstance(rates, dict)
    assert all(0.0 <= v <= 1.0 for v in rates.values())
