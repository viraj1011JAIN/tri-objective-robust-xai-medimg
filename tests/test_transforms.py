# tests/test_transforms.py
from __future__ import annotations

import numpy as np
import torch

from src.datasets.transforms import (
    build_transforms,
    get_chest_xray_transforms,
    get_derm7pt_transforms,
    get_isic_transforms,
)


def _dummy_image(height: int = 32, width: int = 32) -> np.ndarray:
    return np.full((height, width, 3), 128, dtype=np.uint8)


def test_isic_transforms_train_and_val() -> None:
    img = _dummy_image()
    t_train = get_isic_transforms(split="train", image_size=32)
    t_val = get_isic_transforms(split="val", image_size=32)

    out_train = t_train(image=img)
    out_val = t_val(image=img)

    assert "image" in out_train and "image" in out_val
    assert isinstance(out_train["image"], torch.Tensor)
    assert isinstance(out_val["image"], torch.Tensor)
    assert out_train["image"].shape[0] == 3
    assert out_val["image"].shape[0] == 3


def test_derm7pt_transforms_alias() -> None:
    img = _dummy_image()
    t = get_derm7pt_transforms(split="train", image_size=32)
    out = t(image=img)
    assert isinstance(out["image"], torch.Tensor)


def test_chest_xray_transforms_train_and_val() -> None:
    img = _dummy_image()
    t_train = get_chest_xray_transforms(split="train", image_size=32)
    t_val = get_chest_xray_transforms(split="val", image_size=32)

    out_train = t_train(image=img)
    out_val = t_val(image=img)
    assert isinstance(out_train["image"], torch.Tensor)
    assert isinstance(out_val["image"], torch.Tensor)


def test_build_transforms_factory() -> None:
    img = _dummy_image()
    for ds_name in ["isic", "derm7pt", "chest_xray"]:
        t_train = build_transforms(dataset=ds_name, split="train", image_size=32)
        out = t_train(image=img)
        assert isinstance(out["image"], torch.Tensor)
