# tests/test_datasets_final_coverage_precision.py
"""
Precision tests targeting the exact missing lines in coverage report.
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
# Target: base_dataset.py lines 266, 270-273 (_load_image path resolution)
# =============================================================================
def test_load_image_path_not_file_triggers_resolution(tmp_path: Path):
    """Test _load_image when path doesn't exist and triggers resolution."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create actual file with extension
    actual_file = images_dir / "myfile.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(actual_file), img)

    class PathResolutionDataset(BaseMedicalDataset):
        def _load_metadata(self):
            # Store path WITHOUT extension (will trigger resolution)
            self.samples = [Sample(tmp_path / "myfile", torch.tensor(0), {})]
            self.class_names = ["class0"]

    ds = PathResolutionDataset(root=tmp_path, split="train")

    # This should trigger the path resolution in _load_image
    image, label, meta = ds[0]
    assert isinstance(image, np.ndarray)
    assert image.shape == (16, 16, 3)


def test_resolve_image_path_no_suffix_no_matches_falls_back_to_first_candidate(
    tmp_path: Path,
):
    """
    Force the branch in BaseMedicalDataset._resolve_image_path where:
    - raw path has NO suffix
    - no candidate path exists on disk
    - no recursive rglob match is found
    and the method falls back to returning the first candidate.
    """

    class DummyDataset(BaseMedicalDataset):
        def _load_metadata(self) -> None:
            # Minimal valid metadata so _finalize_metadata() works
            self.samples = [
                Sample(
                    image_path=self.root / "placeholder.png",
                    label=torch.tensor(0),
                    meta={},
                )
            ]
            self.class_names = ["neg"]

    ds = DummyDataset(root=tmp_path, split="train")

    raw = "ghost"  # no suffix, and we deliberately create no such file
    resolved = ds._resolve_image_path(raw)

    # No match under root, so we must fall back to the first candidate
    assert resolved == ds.root / raw


# =============================================================================
# Target: base_dataset.py line 251->264 (exception handling in __getitem__)
# =============================================================================
def test_getitem_transform_keyerror_fallback(tmp_path: Path):
    """Test __getitem__ transform that raises KeyError (triggers fallback)."""
    img_path = tmp_path / "img.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    def transform_raises_keyerror(image):
        # Simulate transform that raises KeyError
        raise KeyError("Missing key")

    class KeyErrorDataset(BaseMedicalDataset):
        def _load_metadata(self):
            self.samples = [Sample(img_path, torch.tensor(0), {})]
            self.class_names = ["class0"]

    ds = KeyErrorDataset(
        root=tmp_path, split="train", transform=transform_raises_keyerror
    )

    # Should catch KeyError and try fallback
    with pytest.raises(KeyError):
        _ = ds[0]


# =============================================================================
# Target: base_dataset.py line 326 (validate when all files exist)
# =============================================================================
def test_validate_all_files_exist(tmp_path: Path):
    """Test validate when all files exist (is_valid = True)."""
    img_path = tmp_path / "exists.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    class ValidDataset(BaseMedicalDataset):
        def _load_metadata(self):
            self.samples = [Sample(img_path, torch.tensor(0), {})]
            self.class_names = ["class0"]

    ds = ValidDataset(root=tmp_path, split="train")
    summary = ds.validate(strict=False)

    assert summary["is_valid"] is True
    assert summary["num_missing_files"] == 0
    assert len(summary["missing_files"]) == 0


# =============================================================================
# Target: chest_xray.py lines 137-138 (get dataset column value)
# =============================================================================
def test_chest_xray_with_dataset_column_value(tmp_path: Path):
    """Test ChestXRay retrieving dataset column value in metadata."""
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
            "dataset": ["NIH"],  # This column value should be retrieved
        }
    )
    df.to_csv(csv_path, index=False)

    ds = ChestXRayDataset(
        root=tmp_path,
        split="train",
        csv_path=csv_path,
        transforms=None,
    )

    # Access metadata to ensure dataset column is read
    image, label, meta = ds[0]
    assert meta["dataset"] == "NIH"


# =============================================================================
# Target: derm7pt.py lines 141-144 (concept column value retrieval)
# =============================================================================
def test_derm7pt_concept_in_row_index(tmp_path: Path):
    """Test Derm7pt concept column value retrieval from row."""
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
            "concept_network": [1, 0],
            "concept_streaks": [0, 1],
        }
    )
    df.to_csv(csv_path, index=False)

    ds = Derm7ptDataset(root=tmp_path, split="train", csv_path=csv_path)

    # Ensure concepts are retrieved and stored in metadata
    image, label, meta = ds[0]
    assert "concepts" in meta
    assert "concept_network" in meta["concepts"]
    assert "concept_streaks" in meta["concepts"]


def test_derm7pt_empty_concept_columns(tmp_path: Path):
    """Test Derm7pt when concept_columns list exists but is empty."""
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
        }
    )
    df.to_csv(csv_path, index=False)

    # Explicitly pass empty concept_columns list
    ds = Derm7ptDataset(
        root=tmp_path,
        split="train",
        csv_path=csv_path,
        concept_columns=[],  # Explicitly empty
    )

    assert len(ds.concept_names) == 0
    assert ds.concept_matrix.shape[1] == 0


def test_derm7pt_concept_not_in_concepts_dict(tmp_path: Path):
    """Test Derm7pt when no concepts dict is created."""
    csv_path = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "test.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    df = pd.DataFrame(
        {
            "split": ["train"],
            "image_id": ["test"],
            "diagnosis": ["melanoma"],
        }
    )
    df.to_csv(csv_path, index=False)

    ds = Derm7ptDataset(root=tmp_path, split="train", csv_path=csv_path)

    # No concepts in metadata
    image, label, meta = ds[0]
    assert "concepts" not in meta


# =============================================================================
# Target: derm7pt.py line 201 (compute_class_statistics with no concepts)
# =============================================================================
def test_derm7pt_statistics_without_concepts(tmp_path: Path):
    """Test compute_class_statistics when there are no concepts."""
    csv_path = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "test.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    df = pd.DataFrame(
        {
            "split": ["train", "val"],
            "image_id": ["test", "test2"],
            "diagnosis": ["melanoma", "nevus"],
        }
    )
    df.to_csv(csv_path, index=False)

    ds = Derm7ptDataset(root=tmp_path, split="train", csv_path=csv_path)
    stats = ds.compute_class_statistics()

    # Should have num_concepts = 0
    assert stats["num_concepts"] == 0
    assert "concept_names" not in stats  # Not added when 0 concepts


def test_derm7pt_statistics_with_concepts(tmp_path: Path):
    """Test compute_class_statistics when concepts exist."""
    csv_path = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "test.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    df = pd.DataFrame(
        {
            "split": ["train", "val"],
            "image_id": ["test", "test2"],
            "diagnosis": ["melanoma", "nevus"],
            "pigment": [1, 0],
        }
    )
    df.to_csv(csv_path, index=False)

    ds = Derm7ptDataset(root=tmp_path, split="train", csv_path=csv_path)
    stats = ds.compute_class_statistics()

    # Should have concept_names when concepts > 0
    assert stats["num_concepts"] == 1
    assert "concept_names" in stats
    assert len(stats["concept_names"]) == 1


# =============================================================================
# Target: isic.py lines 121-123 (empty split handling)
# =============================================================================
def test_isic_empty_split_after_vocabulary_building(tmp_path: Path):
    """Test ISIC when split is empty after vocabulary is built from all data."""
    csv_path = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "test.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    # Only val and test splits, no train
    df = pd.DataFrame(
        {
            "split": ["val", "test"],
            "image_id": ["test", "test"],
            "label": ["class0", "class1"],
        }
    )
    df.to_csv(csv_path, index=False)

    # Try to load train split (which doesn't exist)
    with pytest.raises(ValueError, match="no samples found"):
        ISICDataset(root=tmp_path, split="train", csv_path=csv_path)


# =============================================================================
# Additional edge cases for complete coverage
# =============================================================================
def test_chest_xray_none_label_handling(tmp_path: Path):
    """Test ChestXRay handling of 'None' string in labels."""
    csv_path = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "cxr.png"
    img = np.full((16, 16), 150, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    df = pd.DataFrame(
        {
            "image_path": ["images/cxr.png"],
            "labels": ["None"],  # String "None"
            "split": ["train"],
        }
    )
    df.to_csv(csv_path, index=False)

    ds = ChestXRayDataset(
        root=tmp_path,
        split="train",
        csv_path=csv_path,
        transforms=None,
    )

    # Should handle "None" as no findings
    assert len(ds) == 1
    image, label, meta = ds[0]
    assert label.sum().item() == 0


def test_base_dataset_class_statistics_without_dataset_name(tmp_path: Path):
    """Test compute_class_statistics when DATASET_NAME is not defined."""
    img_path = tmp_path / "img.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    class NoNameDataset(BaseMedicalDataset):
        # No DATASET_NAME attribute

        def _load_metadata(self):
            self.samples = [Sample(img_path, torch.tensor(0), {})]
            self.class_names = ["class0"]

    ds = NoNameDataset(root=tmp_path, split="train")
    stats = ds.compute_class_statistics()

    # Should use class name as fallback
    assert stats["dataset"] == "NoNameDataset"


def test_resolve_path_with_various_extensions(tmp_path: Path):
    """Test path resolution tries various image extensions."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create file with uncommon extension
    img_path = images_dir / "test.tiff"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    class TiffDataset(BaseMedicalDataset):
        def _load_metadata(self):
            # Reference without extension
            self.samples = [Sample(Path("test"), torch.tensor(0), {})]
            self.class_names = ["class0"]

    ds = TiffDataset(root=tmp_path, split="train")
    resolved = ds._resolve_image_path("test")

    # Should find test.tiff
    assert resolved.is_file()
    assert resolved.suffix == ".tiff"
