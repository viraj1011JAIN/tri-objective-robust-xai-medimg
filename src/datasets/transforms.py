# src/datasets/transforms.py
"""
Dataset-specific augmentation and preprocessing pipelines.

Uses Albumentations 2.x + ToTensorV2 so that outputs are torch.Tensors
with the expected shapes. All transforms expect numpy arrays as input
(compatible with Albumentations 2.x requirements).
"""

from __future__ import annotations

from typing import Literal

import albumentations as A
import cv2
from albumentations.pytorch import ToTensorV2

# Standard ImageNet statistics – good default for pretrained backbones
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def _normalize_rgb() -> A.BasicTransform:
    """Normalization for RGB dermoscopy / ISIC images."""
    return A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def _normalize_cxr() -> A.BasicTransform:
    """
    Normalization for chest X-rays.

    We keep 3-channel stats so that grayscale images replicated to 3 channels
    are still valid inputs for ImageNet-pretrained models.
    """
    return A.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)


def get_isic_transforms(
    split: Literal["train", "val", "test"], image_size: int = 224
) -> A.Compose:
    """
    Augmentations for ISIC dermoscopy images.

    Train: random resized crop + flips + light color / geometric jitter.
    Val/Test: deterministic resize + normalize.

    Parameters
    ----------
    split:
        "train", "val", or "test".
    image_size:
        Target image size (square).

    Returns
    -------
    A.Compose
        Albumentations pipeline that expects `image=` keyword argument
        and returns dict with "image" key containing a torch.Tensor.
    """
    split_lower = split.lower()
    if split_lower == "train":
        transforms = [
            A.RandomResizedCrop(
                size=(image_size, image_size),  # Albumentations 2.x API: size=(H, W)
                scale=(0.8, 1.0),
                ratio=(0.75, 1.3333),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            A.RandomBrightnessContrast(p=0.2),
            A.ShiftScaleRotate(
                shift_limit=0.05,
                scale_limit=0.1,
                rotate_limit=15,
                border_mode=cv2.BORDER_REFLECT,
                p=0.2,
            ),
            _normalize_rgb(),
            ToTensorV2(),
        ]
    else:
        transforms = [
            A.Resize(height=image_size, width=image_size),
            _normalize_rgb(),
            ToTensorV2(),
        ]
    return A.Compose(transforms)


def get_derm7pt_transforms(
    split: Literal["train", "val", "test"], image_size: int = 224
) -> A.Compose:
    """
    Derm7pt uses the same augmentation strategy as ISIC.

    This keeps experiments comparable between the two dermoscopy datasets.

    Parameters
    ----------
    split:
        "train", "val", or "test".
    image_size:
        Target image size (square).

    Returns
    -------
    A.Compose
        Albumentations pipeline.
    """
    return get_isic_transforms(split=split, image_size=image_size)


def get_chest_xray_transforms(
    split: Literal["train", "val", "test"], image_size: int = 224
) -> A.Compose:
    """
    Augmentations for chest X-ray images (NIH + PadChest).

    We keep augmentations conservative to avoid breaking anatomical priors
    (no vertical flip, limited rotation).

    Parameters
    ----------
    split:
        "train", "val", or "test".
    image_size:
        Target image size (square).

    Returns
    -------
    A.Compose
        Albumentations pipeline.
    """
    split_lower = split.lower()
    if split_lower == "train":
        transforms = [
            A.RandomResizedCrop(
                size=(image_size, image_size),  # Albumentations 2.x API: size=(H, W)
                scale=(0.9, 1.0),
                ratio=(0.75, 1.3333),
                p=1.0,
            ),
            A.HorizontalFlip(p=0.5),
            A.RandomBrightnessContrast(brightness_limit=0.1, contrast_limit=0.1, p=0.2),
            _normalize_cxr(),
            ToTensorV2(),
        ]
    else:
        transforms = [
            A.Resize(height=image_size, width=image_size),
            _normalize_cxr(),
            ToTensorV2(),
        ]
    return A.Compose(transforms)


def build_transforms(
    dataset: str,
    split: Literal["train", "val", "test"],
    image_size: int = 224,
) -> A.Compose:
    """
    Factory used by tests and training code.

    Parameters
    ----------
    dataset:
        One of {"isic", "derm7pt", "chest_xray", "nih_cxr", "padchest", ...}.
        The check is case-insensitive and allows a few common aliases.
    split:
        "train", "val", or "test".
    image_size:
        Final side length after preprocessing.

    Returns
    -------
    albumentations.Compose
        A transform pipeline that outputs a dict with an "image" key
        containing a torch.Tensor of shape (C, H, W).

    Examples
    --------
    >>> t = build_transforms("isic", "train", 224)
    >>> import numpy as np
    >>> img = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    >>> out = t(image=img)
    >>> out["image"].shape
    torch.Size([3, 224, 224])
    """
    key = dataset.lower()

    if key in {"isic", "isic_2018", "isic_2019", "isic_2020"}:
        return get_isic_transforms(split=split, image_size=image_size)

    if key in {"derm7pt", "derm"}:
        return get_derm7pt_transforms(split=split, image_size=image_size)

    if key in {"chest_xray", "cxr", "nih_cxr", "padchest"}:
        return get_chest_xray_transforms(split=split, image_size=image_size)

    raise ValueError(f"Unknown dataset '{dataset}' for build_transforms().")
