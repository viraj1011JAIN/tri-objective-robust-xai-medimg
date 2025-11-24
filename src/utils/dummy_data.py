"""
Dummy Data Loader for Shadow Execution (Phase 4.3).

Generates synthetic data for testing the tri-objective pipeline
while waiting for real datasets (ISIC, NIH ChestX-ray).

This allows us to:
1. Test the training loop end-to-end
2. Verify loss computation is bug-free
3. Ensure MLflow logging works
4. Debug CUDA/memory issues
5. Validate checkpoint saving/loading

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Project: Tri-Objective Robust XAI for Medical Imaging
Target: A1+ Grade | Publication-Ready (NeurIPS/MICCAI/TMI)
Deadline: November 28, 2025

Usage
-----
>>> from src.utils.dummy_data import create_dummy_dataloader
>>>
>>> # Multi-class classification (ISIC)
>>> train_loader = create_dummy_dataloader(
...     num_classes=7,
...     task_type="multi_class",
...     batch_size=32,
...     num_batches=100,
... )
>>>
>>> # Multi-label classification (NIH)
>>> train_loader = create_dummy_dataloader(
...     num_classes=14,
...     task_type="multi_label",
...     batch_size=32,
...     num_batches=100,
... )
>>>
>>> # Test training loop
>>> for images, labels in train_loader:
...     outputs = model(images)
...     loss = criterion(outputs, labels)
...     loss.backward()
"""

from __future__ import annotations

import logging
from typing import Iterator, Optional, Tuple

import torch
from torch import Tensor
from torch.utils.data import DataLoader, Dataset

logger = logging.getLogger(__name__)


class DummyMedicalImageDataset(Dataset):
    """
    Dummy dataset for medical image classification.

    Generates random images (224x224x3) and labels on-the-fly.
    Useful for testing pipelines without real data.

    Parameters
    ----------
    num_samples : int
        Number of samples in the dataset
    num_classes : int
        Number of classes
    task_type : str
        "multi_class" (ISIC-style) or "multi_label" (NIH-style)
    image_size : int
        Image size (default: 224)
    seed : int
        Random seed for reproducibility

    Examples
    --------
    >>> dataset = DummyMedicalImageDataset(
    ...     num_samples=1000,
    ...     num_classes=7,
    ...     task_type="multi_class",
    ... )
    >>> images, labels = dataset[0]
    >>> images.shape
    torch.Size([3, 224, 224])
    >>> labels.shape
    torch.Size([])
    """

    def __init__(
        self,
        num_samples: int = 1000,
        num_classes: int = 7,
        task_type: str = "multi_class",
        image_size: int = 224,
        seed: int = 42,
    ):
        self.num_samples = num_samples
        self.num_classes = num_classes
        self.task_type = task_type
        self.image_size = image_size
        self.seed = seed

        # Validate task type
        if task_type not in ["multi_class", "multi_label"]:
            raise ValueError(
                f"task_type must be 'multi_class' or 'multi_label', got {task_type}"
            )

        logger.info(
            f"Created DummyMedicalImageDataset: "
            f"N={num_samples}, C={num_classes}, task={task_type}"
        )

    def __len__(self) -> int:
        return self.num_samples

    def __getitem__(self, idx: int) -> Tuple[Tensor, Tensor]:
        """
        Get a single sample.

        Parameters
        ----------
        idx : int
            Sample index

        Returns
        -------
        image : Tensor
            Random image, shape (3, H, W)
        label : Tensor
            Label (class index for multi-class, binary vector for multi-label)
        """
        # Set seed for reproducibility
        torch.manual_seed(self.seed + idx)

        # Generate random image (normalized to [0, 1])
        image = torch.rand(3, self.image_size, self.image_size)

        # Generate label
        if self.task_type == "multi_class":
            # Single class label (0 to num_classes-1)
            label = torch.randint(0, self.num_classes, (1,)).squeeze()
        else:
            # Multi-label binary vector
            # Average 2-3 positive labels per sample (realistic for CXR)
            num_positive = torch.randint(1, 4, (1,)).item()
            label = torch.zeros(self.num_classes)
            positive_indices = torch.randperm(self.num_classes)[:num_positive]
            label[positive_indices] = 1.0

        return image, label


def create_dummy_dataloader(
    num_samples: int = 1000,
    num_classes: int = 7,
    task_type: str = "multi_class",
    batch_size: int = 32,
    num_workers: int = 0,
    shuffle: bool = True,
    pin_memory: bool = True,
    seed: int = 42,
    image_size: int = 224,
) -> DataLoader:
    """
    Create a dummy data loader for testing.

    Parameters
    ----------
    num_samples : int
        Number of samples in the dataset
    num_classes : int
        Number of classes
    task_type : str
        "multi_class" or "multi_label"
    batch_size : int
        Batch size
    num_workers : int
        Number of worker processes
    shuffle : bool
        Whether to shuffle the data
    pin_memory : bool
        Whether to pin memory (faster GPU transfer)
    seed : int
        Random seed
    image_size : int
        Image size (default: 224)

    Returns
    -------
    dataloader : DataLoader
        PyTorch data loader

    Examples
    --------
    >>> # ISIC-style (multi-class)
    >>> train_loader = create_dummy_dataloader(
    ...     num_samples=1000,
    ...     num_classes=7,
    ...     task_type="multi_class",
    ...     batch_size=32,
    ... )
    >>>
    >>> # NIH-style (multi-label)
    >>> train_loader = create_dummy_dataloader(
    ...     num_samples=1000,
    ...     num_classes=14,
    ...     task_type="multi_label",
    ...     batch_size=32,
    ... )
    """
    dataset = DummyMedicalImageDataset(
        num_samples=num_samples,
        num_classes=num_classes,
        task_type=task_type,
        image_size=image_size,
        seed=seed,
    )

    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        drop_last=True,  # Ensure consistent batch sizes
    )

    logger.info(
        f"Created dummy DataLoader: " f"{len(dataloader)} batches of size {batch_size}"
    )

    return dataloader


def test_dummy_dataloader():
    """
    Test the dummy data loader.

    Verifies:
    1. Shapes are correct
    2. Data types are correct
    3. Labels are in valid range
    4. Multiple iterations work
    """
    print("=" * 80)
    print("Testing Dummy Data Loader")
    print("=" * 80)

    # Test multi-class
    print("\n[1] Multi-class (ISIC-style)")
    train_loader = create_dummy_dataloader(
        num_samples=100,
        num_classes=7,
        task_type="multi_class",
        batch_size=16,
    )

    for i, (images, labels) in enumerate(train_loader):
        print(f"  Batch {i}: images {images.shape}, labels {labels.shape}")
        print(f"  Label range: [{labels.min():.0f}, {labels.max():.0f}]")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")

        # Verify shapes
        assert images.shape == (16, 3, 224, 224), f"Wrong image shape: {images.shape}"
        assert labels.shape == (16,), f"Wrong label shape: {labels.shape}"
        assert labels.min() >= 0 and labels.max() < 7, "Labels out of range"
        assert images.min() >= 0 and images.max() <= 1, "Images not normalized"

        if i >= 2:  # Test 3 batches
            break

    print("  ✓ Multi-class test passed")

    # Test multi-label
    print("\n[2] Multi-label (NIH-style)")
    train_loader = create_dummy_dataloader(
        num_samples=100,
        num_classes=14,
        task_type="multi_label",
        batch_size=16,
    )

    for i, (images, labels) in enumerate(train_loader):
        print(f"  Batch {i}: images {images.shape}, labels {labels.shape}")
        print(f"  Avg positive labels: {labels.sum(dim=1).mean():.2f}")

        # Verify shapes
        assert images.shape == (16, 3, 224, 224), f"Wrong image shape: {images.shape}"
        assert labels.shape == (16, 14), f"Wrong label shape: {labels.shape}"
        assert labels.min() >= 0 and labels.max() <= 1, "Labels not binary"

        if i >= 2:
            break

    print("  ✓ Multi-label test passed")

    print("\n" + "=" * 80)
    print("All tests passed! ✓")
    print("=" * 80)


if __name__ == "__main__":
    # Run tests
    test_dummy_dataloader()
