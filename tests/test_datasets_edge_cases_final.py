# tests/test_datasets_edge_cases_final.py
"""
Final edge case tests to achieve 100% coverage.
Targets specific missing branches identified in coverage report.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pandas as pd
import pytest
import torch

from src.datasets.base_dataset import BaseMedicalDataset, Sample
from src.datasets.chest_xray import ChestXRayDataset
from src.datasets.derm7pt import Derm7ptDataset
from src.datasets.isic import ISICDataset


# =============================================================================
# Hit missing branches in base_dataset.py
# =============================================================================
def test_stack_labels_empty_samples_error(tmp_path: Path):
    """Test _stack_labels with no samples."""

    class EmptyDataset(BaseMedicalDataset):
        def _load_metadata(self):
            self.samples = [Sample(tmp_path / "img.jpg", torch.tensor(0), {})]
            self.class_names = ["class0"]

    ds = EmptyDataset(root=tmp_path, split="train")
    ds.samples = []  # Force empty after init

    with pytest.raises(ValueError, match="Cannot stack labels"):
        ds._stack_labels()


def test_finalize_metadata_infer_classes_single_label(tmp_path: Path):
    """Test _finalize_metadata when class_names is empty (single-label)."""
    img_path = tmp_path / "img.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    class InferClassDataset(BaseMedicalDataset):
        def _load_metadata(self):
            self.samples = [
                Sample(img_path, torch.tensor(0), {}),
                Sample(img_path, torch.tensor(1), {}),
                Sample(img_path, torch.tensor(2), {}),
            ]
            self.class_names = []  # Empty - will be inferred

    ds = InferClassDataset(root=tmp_path, split="train")
    assert ds.num_classes == 3  # Inferred from max label + 1


def test_finalize_metadata_infer_classes_multi_label(tmp_path: Path):
    """Test _finalize_metadata when class_names is empty (multi-label)."""
    img_path = tmp_path / "img.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    class InferMultiLabelDataset(BaseMedicalDataset):
        def _load_metadata(self):
            self.samples = [
                Sample(img_path, torch.tensor([1.0, 0.0, 1.0, 0.0]), {}),
                Sample(img_path, torch.tensor([0.0, 1.0, 0.0, 1.0]), {}),
            ]
            self.class_names = []  # Empty - will be inferred

    ds = InferMultiLabelDataset(root=tmp_path, split="train")
    assert ds.num_classes == 4  # Inferred from label tensor shape


def test_resolve_path_with_parent_directory(tmp_path: Path):
    """Test path resolution with parent directory structure."""
    parent_dir = tmp_path / "parent"
    images_dir = parent_dir / "images"
    images_dir.mkdir(parents=True)

    img_path = images_dir / "test.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    class ParentDirDataset(BaseMedicalDataset):
        def _load_metadata(self):
            # Reference with relative path
            self.samples = [Sample(Path("images/test.jpg"), torch.tensor(0), {})]
            self.class_names = ["class0"]

    ds = ParentDirDataset(root=parent_dir, split="train")
    image, label, meta = ds[0]
    assert isinstance(image, np.ndarray)


def test_compute_class_statistics_no_samples(tmp_path: Path):
    """Test compute_class_statistics with edge case of minimal samples."""
    img_path = tmp_path / "img.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    class MinimalDataset(BaseMedicalDataset):
        DATASET_NAME = "Minimal"

        def _load_metadata(self):
            self.samples = [Sample(img_path, torch.tensor(0), {})]
            self.class_names = []  # Will be inferred

    ds = MinimalDataset(root=tmp_path, split="train")
    stats = ds.compute_class_statistics()

    assert stats["num_samples"] == 1
    assert len(stats["class_names"]) == 1


# =============================================================================
# Hit missing branches in isic.py
# =============================================================================
def test_isic_explicit_column_names(tmp_path: Path):
    """Test ISIC with explicitly provided column names."""
    csv_path = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "test.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    df = pd.DataFrame(
        {
            "my_split": ["train", "val"],
            "my_image": ["test", "test"],
            "my_label": ["class0", "class1"],
        }
    )
    df.to_csv(csv_path, index=False)

    ds = ISICDataset(
        root=tmp_path,
        split="train",
        csv_path=csv_path,
        split_column="my_split",
        image_column="my_image",
        label_column="my_label",
    )

    assert len(ds) == 1
    assert ds.num_classes == 2  # Both classes in vocabulary


def test_isic_with_file_column(tmp_path: Path):
    """Test ISIC with 'file' column name."""
    csv_path = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "test.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    df = pd.DataFrame(
        {
            "split": ["train", "val"],
            "file": ["test", "test"],
            "diagnosis": ["melanoma", "nevus"],
        }
    )
    df.to_csv(csv_path, index=False)

    ds = ISICDataset(root=tmp_path, split="train", csv_path=csv_path)
    assert len(ds) == 1


# =============================================================================
# Hit missing branches in derm7pt.py
# =============================================================================
def test_derm7pt_explicit_column_names(tmp_path: Path):
    """Test Derm7pt with explicitly provided column names."""
    csv_path = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "test.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    df = pd.DataFrame(
        {
            "my_split": ["train", "val"],
            "my_image": ["test", "test"],
            "my_diagnosis": ["melanoma", "nevus"],
            "concept_pigment": [1, 0],
        }
    )
    df.to_csv(csv_path, index=False)

    ds = Derm7ptDataset(
        root=tmp_path,
        split="train",
        csv_path=csv_path,
        split_column="my_split",
        image_column="my_image",
        label_column="my_diagnosis",
        concept_columns=["concept_pigment"],
    )

    assert len(ds) == 1
    assert ds.num_classes == 2
    assert len(ds.concept_names) == 1


def test_derm7pt_with_file_column(tmp_path: Path):
    """Test Derm7pt with 'file' column name."""
    csv_path = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "test.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    df = pd.DataFrame(
        {
            "split": ["train", "val"],
            "file": ["test", "test"],
            "diagnosis": ["melanoma", "nevus"],
        }
    )
    df.to_csv(csv_path, index=False)

    ds = Derm7ptDataset(root=tmp_path, split="train", csv_path=csv_path)
    assert len(ds) == 1


def test_derm7pt_missing_columns(tmp_path: Path):
    """Test Derm7pt missing split/image/label columns."""
    csv_path = tmp_path / "metadata.csv"

    # Missing split column
    df = pd.DataFrame({"image_id": ["img1"], "diagnosis": ["melanoma"]})
    df.to_csv(csv_path, index=False)

    with pytest.raises(KeyError, match="missing split column"):
        Derm7ptDataset(root=tmp_path, split="train", csv_path=csv_path)

    # Missing image column
    df = pd.DataFrame({"split": ["train"], "diagnosis": ["melanoma"]})
    df.to_csv(csv_path, index=False)

    with pytest.raises(KeyError, match="lacks an image column"):
        Derm7ptDataset(root=tmp_path, split="train", csv_path=csv_path)

    # Missing label column
    df = pd.DataFrame({"split": ["train"], "image_id": ["img1"]})
    df.to_csv(csv_path, index=False)

    with pytest.raises(KeyError, match="lacks a label column"):
        Derm7ptDataset(root=tmp_path, split="train", csv_path=csv_path)


def test_derm7pt_concept_columns_with_nan_values(tmp_path: Path):
    """Test Derm7pt handling NaN values in concept columns."""
    csv_path = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "test.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    df = pd.DataFrame(
        {
            "split": ["train", "val"],
            "image_id": ["test", "test"],
            "diagnosis": ["melanoma", "nevus"],
            "concept_a": [1.0, np.nan],  # NaN value
            "concept_b": [-1, 1],  # -1 value
        }
    )
    df.to_csv(csv_path, index=False)

    ds = Derm7ptDataset(root=tmp_path, split="train", csv_path=csv_path)

    # Check concept matrix handles NaN and -1
    concepts = ds.concept_matrix
    assert not np.isnan(concepts).any()
    assert (concepts >= 0).all()


# =============================================================================
# Hit missing branches in chest_xray.py
# =============================================================================
def test_chest_xray_without_dataset_column(tmp_path: Path):
    """Test ChestXRay without dataset column."""
    csv_path = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "cxr.png"
    img = np.full((16, 16), 150, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    df = pd.DataFrame(
        {
            "image_path": ["images/cxr.png"],
            "labels": ["Pneumonia"],
            "split": ["train"],
        }
    )
    df.to_csv(csv_path, index=False)

    ds = ChestXRayDataset(
        root=tmp_path,
        split="train",
        csv_path=csv_path,
        dataset_column=None,  # No dataset column
        transforms=None,
    )

    assert len(ds) == 1
    image, label, meta = ds[0]
    assert meta["dataset"] is None


def test_chest_xray_dataset_filtering(tmp_path: Path):
    """Test ChestXRay dataset filtering logic."""
    csv_path = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "cxr.png"
    img = np.full((16, 16), 150, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    df = pd.DataFrame(
        {
            "image_path": ["images/cxr.png", "images/cxr.png"],
            "labels": ["Pneumonia", "Edema"],
            "split": ["train", "train"],
            "dataset": ["NIH", "PadChest"],
        }
    )
    df.to_csv(csv_path, index=False)

    # Only allow NIH
    ds = ChestXRayDataset(
        root=tmp_path,
        split="train",
        csv_path=csv_path,
        allowed_datasets=["NIH"],
        transforms=None,
    )

    assert len(ds) == 1  # Only NIH sample


# =============================================================================
# Additional transform coverage
# =============================================================================
def test_getitem_transform_exception_handling(tmp_path: Path):
    """Test __getitem__ with transform that raises unexpected exception."""
    img_path = tmp_path / "img.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    def broken_transform(image):
        raise RuntimeError("Unexpected transform error")

    class DummyDataset(BaseMedicalDataset):
        def _load_metadata(self):
            self.samples = [Sample(img_path, torch.tensor(0), {})]
            self.class_names = ["class0"]

    ds = DummyDataset(root=tmp_path, split="train", transform=broken_transform)

    with pytest.raises(RuntimeError, match="Unexpected transform error"):
        _ = ds[0]
