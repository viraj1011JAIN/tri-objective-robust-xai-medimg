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
# Target: base_dataset.py __getitem__ transform branches
# =============================================================================
def test_getitem_no_transform(tmp_path: Path) -> None:
    """Test __getitem__ without any transform."""
    img_path = tmp_path / "img.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    class NoTransformDataset(BaseMedicalDataset):
        def _load_metadata(self) -> None:
            self.samples = [Sample(img_path, torch.tensor(0), {})]
            self.class_names = ["class0"]

    ds = NoTransformDataset(root=tmp_path, split="train", transform=None)
    image, label, meta = ds[0]
    assert isinstance(image, np.ndarray)


def test_getitem_transform_dict_with_image_key(tmp_path: Path) -> None:
    """Transform returns dict WITH 'image' key."""
    img_path = tmp_path / "img.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    class AlbumentationsDataset(BaseMedicalDataset):
        def _load_metadata(self) -> None:
            self.samples = [Sample(img_path, torch.tensor(0), {})]
            self.class_names = ["class0"]

    def albu_transform(image: np.ndarray) -> dict:
        return {"image": image * 2, "mask": None}

    ds = AlbumentationsDataset(root=tmp_path, split="train", transform=albu_transform)
    image, _, _ = ds[0]
    assert isinstance(image, np.ndarray)
    assert image[0, 0, 0] == 254


def test_getitem_transform_dict_without_image_key(tmp_path: Path) -> None:
    """Transform returns dict WITHOUT 'image' key."""
    img_path = tmp_path / "img.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    class DictNoImageDataset(BaseMedicalDataset):
        def _load_metadata(self) -> None:
            self.samples = [Sample(img_path, torch.tensor(0), {})]
            self.class_names = ["class0"]

    def no_image_key_transform(image: np.ndarray) -> dict:
        return {"output": image + 1, "meta": "test"}

    ds = DictNoImageDataset(
        root=tmp_path, split="train", transform=no_image_key_transform
    )
    result, _, _ = ds[0]
    assert isinstance(result, dict)
    assert "output" in result


def test_getitem_transform_returns_array(tmp_path: Path) -> None:
    """Transform returns array."""
    img_path = tmp_path / "img.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    class ArrayTransformDataset(BaseMedicalDataset):
        def _load_metadata(self) -> None:
            self.samples = [Sample(img_path, torch.tensor(0), {})]
            self.class_names = ["class0"]

    def array_transform(image: np.ndarray) -> np.ndarray:
        return image + 1

    ds = ArrayTransformDataset(root=tmp_path, split="train", transform=array_transform)
    image, _, _ = ds[0]
    assert isinstance(image, np.ndarray)
    assert image[0, 0, 0] == 128


def test_getitem_transform_returns_tensor(tmp_path: Path) -> None:
    """Transform returns tensor."""
    img_path = tmp_path / "img.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    class TensorTransformDataset(BaseMedicalDataset):
        def _load_metadata(self) -> None:
            self.samples = [Sample(img_path, torch.tensor(0), {})]
            self.class_names = ["class0"]

    def tensor_transform(image: np.ndarray) -> torch.Tensor:
        return torch.from_numpy(image)

    ds = TensorTransformDataset(
        root=tmp_path, split="train", transform=tensor_transform
    )
    image, _, _ = ds[0]
    assert isinstance(image, torch.Tensor)


def test_getitem_transform_typeerror_fallback(tmp_path: Path) -> None:
    """Transform raises TypeError, falls back to positional call."""
    img_path = tmp_path / "img.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    call_count = [0]

    class FallbackDataset(BaseMedicalDataset):
        def _load_metadata(self) -> None:
            self.samples = [Sample(img_path, torch.tensor(0), {})]
            self.class_names = ["class0"]

    def fallback_transform(img=None, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            raise TypeError("no keyword")
        return img + 10

    ds = FallbackDataset(root=tmp_path, split="train", transform=fallback_transform)
    image, _, _ = ds[0]
    assert image[0, 0, 0] == 137


def test_getitem_transform_keyerror(tmp_path: Path) -> None:
    """Transform raises KeyError."""
    img_path = tmp_path / "img.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    class KeyErrorDataset(BaseMedicalDataset):
        def _load_metadata(self) -> None:
            self.samples = [Sample(img_path, torch.tensor(0), {})]
            self.class_names = ["class0"]

    def keyerror_transform(image: np.ndarray) -> np.ndarray:
        raise KeyError("fail")

    ds = KeyErrorDataset(root=tmp_path, split="train", transform=keyerror_transform)
    with pytest.raises(KeyError):
        _ = ds[0]


# =============================================================================
# Target: base_dataset.py path resolution
# =============================================================================
def test_load_image_path_resolution(tmp_path: Path) -> None:
    """Test _load_image when path triggers resolution."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    actual_file = images_dir / "myfile.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(actual_file), img)

    class PathResolutionDataset(BaseMedicalDataset):
        def _load_metadata(self) -> None:
            self.samples = [Sample(tmp_path / "myfile", torch.tensor(0), {})]
            self.class_names = ["class0"]

    ds = PathResolutionDataset(root=tmp_path, split="train")
    image, label, meta = ds[0]
    assert isinstance(image, np.ndarray)
    assert image.shape == (16, 16, 3)


def test_resolve_path_absolute(tmp_path: Path) -> None:
    """Test _resolve_image_path with absolute path."""
    img_path = tmp_path / "abs.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    class AbsDataset(BaseMedicalDataset):
        def _load_metadata(self) -> None:
            self.samples = [Sample(img_path, torch.tensor(0), {})]
            self.class_names = ["class0"]

    ds = AbsDataset(root=tmp_path, split="train")
    resolved = ds._resolve_image_path(str(img_path))
    assert resolved.is_file()


def test_resolve_path_extension_search(tmp_path: Path) -> None:
    """Test path resolution with extension search."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "test.tiff"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    class ExtDataset(BaseMedicalDataset):
        def _load_metadata(self) -> None:
            self.samples = [Sample(Path("images/test"), torch.tensor(0), {})]
            self.class_names = ["class0"]

    ds = ExtDataset(root=tmp_path, split="train")
    resolved = ds._resolve_image_path("images/test")
    assert resolved.is_file()
    assert resolved.suffix == ".tiff"


def test_resolve_path_rglob_fallback(tmp_path: Path) -> None:
    """Test rglob fallback for nested files."""
    nested_dir = tmp_path / "deep" / "nested"
    nested_dir.mkdir(parents=True)

    img_path = nested_dir / "hidden.png"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    class NestedDataset(BaseMedicalDataset):
        def _load_metadata(self) -> None:
            self.samples = [Sample(Path("hidden.png"), torch.tensor(0), {})]
            self.class_names = ["class0"]

    ds = NestedDataset(root=tmp_path, split="train")
    resolved = ds._resolve_image_path("hidden.png")
    assert resolved.is_file()


# =============================================================================
# Target: base_dataset.py validate branches
# =============================================================================
def test_validate_valid_strict_false(tmp_path: Path) -> None:
    """validate with all files present, strict=False."""
    img_path = tmp_path / "ok.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    class ValidDataset(BaseMedicalDataset):
        def _load_metadata(self) -> None:
            self.samples = [Sample(img_path, torch.tensor(0), {})]
            self.class_names = ["class0"]

    ds = ValidDataset(root=tmp_path, split="train")
    summary = ds.validate(strict=False)
    assert summary["is_valid"] is True
    assert summary["num_missing_files"] == 0


def test_validate_valid_strict_true(tmp_path: Path) -> None:
    """validate with all files present, strict=True."""
    img_path = tmp_path / "ok.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    class ValidDataset(BaseMedicalDataset):
        def _load_metadata(self) -> None:
            self.samples = [Sample(img_path, torch.tensor(0), {})]
            self.class_names = ["class0"]

    ds = ValidDataset(root=tmp_path, split="train")
    summary = ds.validate(strict=True)
    assert summary["is_valid"] is True


def test_validate_missing_strict_false(tmp_path: Path) -> None:
    """validate with missing files, strict=False."""
    missing = tmp_path / "missing.jpg"

    class MissingDataset(BaseMedicalDataset):
        def _load_metadata(self) -> None:
            self.samples = [Sample(missing, torch.tensor(0), {})]
            self.class_names = ["class0"]

    ds = MissingDataset(root=tmp_path, split="train")
    summary = ds.validate(strict=False)
    assert summary["is_valid"] is False
    assert summary["num_missing_files"] == 1


def test_validate_missing_strict_true(tmp_path: Path) -> None:
    """validate with missing files, strict=True raises."""
    missing = tmp_path / "missing.jpg"

    class MissingDataset(BaseMedicalDataset):
        def _load_metadata(self) -> None:
            self.samples = [Sample(missing, torch.tensor(0), {})]
            self.class_names = ["class0"]

    ds = MissingDataset(root=tmp_path, split="train")
    with pytest.raises(FileNotFoundError):
        ds.validate(strict=True)


def test_base_dataset_statistics_no_dataset_name(tmp_path: Path) -> None:
    """compute_class_statistics without DATASET_NAME attribute."""
    img_path = tmp_path / "img.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    class NoNameDataset(BaseMedicalDataset):
        def _load_metadata(self) -> None:
            self.samples = [Sample(img_path, torch.tensor(0), {})]
            self.class_names = ["class0"]

    ds = NoNameDataset(root=tmp_path, split="train")
    stats = ds.compute_class_statistics()
    assert stats["dataset"] == "NoNameDataset"


# =============================================================================
# Target: chest_xray.py
# =============================================================================
def test_chest_xray_dataset_column(tmp_path: Path) -> None:
    """ChestXRay with dataset column value."""
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
            "dataset": ["NIH"],
        }
    )
    df.to_csv(csv_path, index=False)

    ds = ChestXRayDataset(
        root=tmp_path, split="train", csv_path=csv_path, transforms=None
    )
    _, _, meta = ds[0]
    assert meta["dataset"] == "NIH"


def test_chest_xray_get_positive_rates(tmp_path: Path) -> None:
    """ChestXRay get_positive_rates method."""
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
        root=tmp_path, split="train", csv_path=csv_path, transforms=None
    )
    rates = ds.get_positive_rates()
    assert rates["Pneumonia"] == 1.0


def test_chest_xray_none_label(tmp_path: Path) -> None:
    """ChestXRay with 'None' label string."""
    csv_path = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "cxr.png"
    img = np.full((16, 16), 150, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    df = pd.DataFrame(
        {
            "image_path": ["images/cxr.png"],
            "labels": ["None"],
            "split": ["train"],
        }
    )
    df.to_csv(csv_path, index=False)

    ds = ChestXRayDataset(
        root=tmp_path, split="train", csv_path=csv_path, transforms=None
    )
    _, label, _ = ds[0]
    assert label.sum().item() == 0


def test_chest_xray_empty_split(tmp_path: Path) -> None:
    """ChestXRay with empty split."""
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

    with pytest.raises(ValueError, match="no samples found"):
        ChestXRayDataset(root=tmp_path, split="val", csv_path=csv_path, transforms=None)


# =============================================================================
# Target: derm7pt.py
# =============================================================================
def test_derm7pt_stats_with_concepts(tmp_path: Path) -> None:
    """Derm7pt compute_class_statistics WITH concepts."""
    csv_path = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "d1.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    df = pd.DataFrame(
        {
            "split": ["train"],
            "image_id": ["d1"],
            "diagnosis": ["melanoma"],
            "pigment": [1],
            "network": [0],
        }
    )
    df.to_csv(csv_path, index=False)

    ds = Derm7ptDataset(root=tmp_path, split="train", csv_path=csv_path)
    stats = ds.compute_class_statistics()

    assert stats["num_concepts"] == 2
    assert "concept_names" in stats
    assert len(stats["concept_names"]) == 2


def test_derm7pt_stats_without_concepts(tmp_path: Path) -> None:
    """Derm7pt compute_class_statistics WITHOUT concepts."""
    csv_path = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "d1.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    df = pd.DataFrame(
        {
            "split": ["train"],
            "image_id": ["d1"],
            "diagnosis": ["melanoma"],
        }
    )
    df.to_csv(csv_path, index=False)

    ds = Derm7ptDataset(root=tmp_path, split="train", csv_path=csv_path)
    stats = ds.compute_class_statistics()

    assert stats["num_concepts"] == 0
    assert "concept_names" not in stats


def test_derm7pt_with_concepts_meta(tmp_path: Path) -> None:
    """Derm7pt with concepts in metadata."""
    csv_path = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "d1.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    df = pd.DataFrame(
        {
            "split": ["train"],
            "image_id": ["d1"],
            "diagnosis": ["melanoma"],
            "pigment": [1],
        }
    )
    df.to_csv(csv_path, index=False)

    ds = Derm7ptDataset(root=tmp_path, split="train", csv_path=csv_path)
    _, _, meta = ds[0]
    assert "concepts" in meta
    assert meta["concepts"]["pigment"] == 1


def test_derm7pt_without_concepts_meta(tmp_path: Path) -> None:
    """Derm7pt without concepts in metadata."""
    csv_path = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "d1.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    df = pd.DataFrame(
        {
            "split": ["train"],
            "image_id": ["d1"],
            "diagnosis": ["melanoma"],
        }
    )
    df.to_csv(csv_path, index=False)

    ds = Derm7ptDataset(root=tmp_path, split="train", csv_path=csv_path)
    _, _, meta = ds[0]
    assert "concepts" not in meta


def test_derm7pt_empty_concept_columns(tmp_path: Path) -> None:
    """Derm7pt with explicitly empty concept_columns."""
    csv_path = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "d1.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    df = pd.DataFrame(
        {
            "split": ["train"],
            "image_id": ["d1"],
            "diagnosis": ["melanoma"],
        }
    )
    df.to_csv(csv_path, index=False)

    ds = Derm7ptDataset(
        root=tmp_path, split="train", csv_path=csv_path, concept_columns=[]
    )
    assert len(ds.concept_names) == 0
    assert ds.concept_matrix.shape[1] == 0


def test_derm7pt_nan_negative_one_concepts(tmp_path: Path) -> None:
    """Derm7pt handling NaN and -1 in concepts."""
    csv_path = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img1 = images_dir / "d1.jpg"
    img2 = images_dir / "d2.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img1), img)
    cv2.imwrite(str(img2), img)

    df = pd.DataFrame(
        {
            "split": ["train", "train"],
            "image_id": ["d1", "d2"],
            "diagnosis": ["melanoma", "nevus"],
            "pigment": [np.nan, -1],
        }
    )
    df.to_csv(csv_path, index=False)

    ds = Derm7ptDataset(root=tmp_path, split="train", csv_path=csv_path)
    _, _, meta1 = ds[0]
    _, _, meta2 = ds[1]

    assert meta1["concepts"]["pigment"] == 0
    assert meta2["concepts"]["pigment"] == 0


def test_derm7pt_empty_split(tmp_path: Path) -> None:
    """Derm7pt with empty split."""
    csv_path = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "d1.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    df = pd.DataFrame(
        {
            "split": ["val"],
            "image_id": ["d1"],
            "diagnosis": ["melanoma"],
        }
    )
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="no samples found"):
        Derm7ptDataset(root=tmp_path, split="train", csv_path=csv_path)


def test_derm7pt_concept_matrix_branches(tmp_path: Path) -> None:
    """Exercise concept_matrix property branches."""
    csv_path = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "d1.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    df = pd.DataFrame(
        {
            "split": ["train"],
            "image_id": ["d1"],
            "diagnosis": ["melanoma"],
            "pigment": [1],
        }
    )
    df.to_csv(csv_path, index=False)

    ds = Derm7ptDataset(root=tmp_path, split="train", csv_path=csv_path)

    mat = ds.concept_matrix
    assert mat.shape == (1, 1)

    ds._concept_matrix = None  # type: ignore[assignment]
    mat = ds.concept_matrix
    assert mat.size == 0

    ds._concept_matrix = np.zeros((0, 1), dtype=np.float32)  # type: ignore[assignment]
    mat = ds.concept_matrix
    assert mat.shape == (0, 1)


# =============================================================================
# Target: isic.py
# =============================================================================
def test_isic_empty_split(tmp_path: Path) -> None:
    """ISIC with empty split."""
    csv_path = tmp_path / "metadata.csv"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    img_path = images_dir / "test.jpg"
    img = np.full((16, 16, 3), 127, dtype=np.uint8)
    cv2.imwrite(str(img_path), img)

    df = pd.DataFrame(
        {
            "split": ["val", "test"],
            "image_id": ["test", "test"],
            "label": ["class0", "class1"],
        }
    )
    df.to_csv(csv_path, index=False)

    with pytest.raises(ValueError, match="no samples found"):
        ISICDataset(root=tmp_path, split="train", csv_path=csv_path)
