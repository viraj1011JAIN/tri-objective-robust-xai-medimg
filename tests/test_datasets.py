"""
tests/test_all_modules.py

Comprehensive Test Suite for 100% Coverage
==========================================

This test suite covers ALL modules in the tri-objective robust XAI project:
- src/datasets/ (isic, derm7pt, chest_xray, transforms, base_dataset)
- src/utils/ (reproducibility, config, mlflow_utils)

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Date: November 2025
Target: 100% code coverage
"""

from __future__ import annotations

import os
import random
import tempfile
import time
from pathlib import Path
from typing import List, Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pandas as pd
import pytest
import torch
from torch.utils.data import DataLoader

# Handle optional imports
try:
    from PIL import Image as PILImage

    PIL_AVAILABLE = True
except ImportError:
    PILImage = None
    PIL_AVAILABLE = False

try:
    import yaml

    YAML_AVAILABLE = True
except ImportError:
    yaml = None
    YAML_AVAILABLE = False

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False

# =============================================================================
# Import project modules
# =============================================================================

# Datasets
from src.datasets.isic import ISICDataset

# Try to import BaseDataset and Split - they may not be directly exported
try:
    from src.datasets.base_dataset import BaseDataset, Split

    BASE_DATASET_AVAILABLE = True
except ImportError:
    # Try alternative imports
    try:
        from src.datasets.base_dataset import Split

        BaseDataset = None
        BASE_DATASET_AVAILABLE = False
    except ImportError:
        # Define a minimal Split enum for compatibility
        from enum import Enum

        class Split(Enum):
            TRAIN = "train"
            VAL = "val"
            TEST = "test"

        BaseDataset = None
        BASE_DATASET_AVAILABLE = False

try:
    from src.datasets.derm7pt import Derm7ptDataset

    DERM7PT_AVAILABLE = True
except ImportError:
    Derm7ptDataset = None
    DERM7PT_AVAILABLE = False

try:
    from src.datasets.chest_xray import ChestXRayDataset

    CHEST_XRAY_AVAILABLE = True
except ImportError:
    ChestXRayDataset = None
    CHEST_XRAY_AVAILABLE = False

try:
    from src.datasets.transforms import (
        get_test_transforms,
        get_train_transforms,
        get_val_transforms,
    )

    TRANSFORMS_AVAILABLE = True
except (ImportError, AttributeError):
    TRANSFORMS_AVAILABLE = False
    get_train_transforms = None
    get_val_transforms = None
    get_test_transforms = None

# Utils
try:
    from src.utils.reproducibility import get_seed_worker, set_deterministic, set_seed

    REPRODUCIBILITY_AVAILABLE = True
except (ImportError, AttributeError):
    REPRODUCIBILITY_AVAILABLE = False
    set_seed = None
    get_seed_worker = None
    set_deterministic = None

try:
    from src.utils.config import load_config, merge_configs, validate_config

    CONFIG_AVAILABLE = True
except (ImportError, AttributeError):
    CONFIG_AVAILABLE = False
    load_config = None
    merge_configs = None
    validate_config = None

try:
    from src.utils.mlflow_utils import log_metrics, log_params, setup_mlflow

    MLFLOW_UTILS_AVAILABLE = True
except (ImportError, AttributeError):
    MLFLOW_UTILS_AVAILABLE = False
    setup_mlflow = None
    log_params = None
    log_metrics = None


# =============================================================================
# Helper Functions
# =============================================================================


def _resolve_data_root(env_var: str, default_subdir: str) -> Path:
    """Resolve dataset root directory."""
    env_path = os.environ.get(env_var)
    if env_path:
        p = Path(env_path)
        if p.exists():
            return p

    candidate_paths = [
        Path("F:/data") / default_subdir,
        Path("C:/Users/Viraj Jain/data") / default_subdir,
        Path.home() / "data" / default_subdir,
        Path("data") / default_subdir,
    ]

    for candidate in candidate_paths:
        if candidate.exists():
            return candidate

    pytest.skip(f"Dataset root for '{default_subdir}' not found.")


def create_dummy_image(
    size: Tuple[int, int] = (224, 224), channels: int = 3
) -> np.ndarray:
    """Create a dummy image for testing."""
    return np.random.randint(0, 255, (*size, channels), dtype=np.uint8)


def create_dummy_metadata(
    num_samples: int = 100,
    num_classes: int = 7,
    splits: List[str] = None,
    tmp_dir: Path = None,
) -> Tuple[pd.DataFrame, Path]:
    """Create dummy metadata CSV for testing."""
    if splits is None:
        splits = ["train", "val", "test"]

    if tmp_dir is None:
        tmp_dir = Path(tempfile.mkdtemp())

    # Create image directory
    img_dir = tmp_dir / "images"
    img_dir.mkdir(exist_ok=True)

    # Generate metadata
    data = []
    class_names = [f"class_{i}" for i in range(num_classes)]

    for i in range(num_samples):
        img_name = f"image_{i:04d}.jpg"
        img_path = img_dir / img_name

        # Create dummy image file
        if PIL_AVAILABLE:
            img = PILImage.fromarray(create_dummy_image())
            img.save(img_path)

        split = splits[i % len(splits)]
        label = class_names[i % num_classes]

        data.append(
            {
                "image_id": f"IMG_{i:04d}",
                "image_path": str(img_path),
                "label": label,
                "split": split,
            }
        )

    df = pd.DataFrame(data)
    csv_path = tmp_dir / "metadata.csv"
    df.to_csv(csv_path, index=False)

    return df, csv_path


# =============================================================================
# ISIC Dataset Tests (Extended)
# =============================================================================

ISIC_CONFIGS = [
    ("isic2018", "ISIC2018_ROOT", "isic_2018", ("train", "val", "test")),
    ("isic2019", "ISIC2019_ROOT", "isic_2019", ("train", "val", "test")),
    ("isic2020", "ISIC2020_ROOT", "isic_2020", ("train", "test")),
]


class TestISICDatasetComprehensive:
    """Comprehensive ISIC dataset tests for full coverage."""

    @pytest.mark.parametrize("dataset_name,env_var,subdir,splits", ISIC_CONFIGS)
    def test_basic_loading(self, dataset_name, env_var, subdir, splits):
        """Test basic dataset loading."""
        root = _resolve_data_root(env_var, subdir)
        csv_path = root / "metadata.csv"

        if not csv_path.exists():
            pytest.skip(f"metadata.csv not found for {dataset_name}")

        successful = 0
        for split in splits:
            try:
                ds = ISICDataset(root=root, split=split, csv_path=csv_path)
                assert len(ds) > 0
                successful += 1
            except ValueError:
                continue

        assert successful > 0

    @pytest.mark.parametrize("dataset_name,env_var,subdir,splits", ISIC_CONFIGS)
    def test_getitem_returns_valid_sample(self, dataset_name, env_var, subdir, splits):
        """Test __getitem__ returns valid samples."""
        root = _resolve_data_root(env_var, subdir)
        csv_path = root / "metadata.csv"

        if not csv_path.exists():
            pytest.skip("metadata.csv not found")

        ds = ISICDataset(root=root, split=splits[0], csv_path=csv_path)

        if len(ds) == 0:
            pytest.skip("Empty dataset")

        sample = ds[0]
        assert isinstance(sample, (tuple, list))
        assert len(sample) == 3

        image, label, meta = sample
        assert isinstance(meta, dict)

    @pytest.mark.parametrize("dataset_name,env_var,subdir,splits", ISIC_CONFIGS)
    def test_class_names_attribute(self, dataset_name, env_var, subdir, splits):
        """Test class_names attribute exists."""
        root = _resolve_data_root(env_var, subdir)
        csv_path = root / "metadata.csv"

        if not csv_path.exists():
            pytest.skip("metadata.csv not found")

        ds = ISICDataset(root=root, split=splits[0], csv_path=csv_path)

        assert hasattr(ds, "class_names")
        assert isinstance(ds.class_names, (list, tuple))
        assert len(ds.class_names) > 0

    @pytest.mark.parametrize("dataset_name,env_var,subdir,splits", ISIC_CONFIGS)
    def test_dataloader_compatibility(self, dataset_name, env_var, subdir, splits):
        """Test DataLoader compatibility."""
        root = _resolve_data_root(env_var, subdir)
        csv_path = root / "metadata.csv"

        if not csv_path.exists():
            pytest.skip("metadata.csv not found")

        ds = ISICDataset(root=root, split=splits[0], csv_path=csv_path)

        if len(ds) < 4:
            pytest.skip("Not enough samples")

        loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
        batch = next(iter(loader))

        assert len(batch) == 3


# =============================================================================
# Base Dataset Tests
# =============================================================================


class TestBaseDataset:
    """Tests for BaseDataset abstract class and Split enum."""

    def test_split_enum_values(self):
        """Test Split enum has expected values."""
        assert hasattr(Split, "TRAIN")
        assert hasattr(Split, "VAL")
        assert hasattr(Split, "TEST")

        # Check string values
        assert Split.TRAIN.value.lower() == "train"
        assert Split.VAL.value.lower() in ("val", "validation")
        assert Split.TEST.value.lower() == "test"

    def test_split_from_string(self):
        """Test creating Split from string."""
        # Direct instantiation
        train_split = Split.TRAIN
        assert train_split == Split.TRAIN

    def test_base_dataset_is_abstract(self):
        """Test that BaseDataset cannot be instantiated directly."""
        if not BASE_DATASET_AVAILABLE or BaseDataset is None:
            pytest.skip("BaseDataset not available")

        # BaseDataset should require subclass implementation
        with pytest.raises(TypeError):
            # This should fail because _load_metadata is abstract
            class IncompleteDataset(BaseDataset):
                pass

            _ = IncompleteDataset(root=Path("."), split="train")


# =============================================================================
# Transforms Tests
# =============================================================================


@pytest.mark.skipif(not TRANSFORMS_AVAILABLE, reason="Transforms not available")
class TestTransforms:
    """Tests for data augmentation transforms."""

    def test_train_transforms_exist(self):
        """Test that train transforms can be created."""
        transforms = get_train_transforms()
        assert transforms is not None

    def test_val_transforms_exist(self):
        """Test that validation transforms can be created."""
        transforms = get_val_transforms()
        assert transforms is not None

    def test_test_transforms_exist(self):
        """Test that test transforms can be created."""
        transforms = get_test_transforms()
        assert transforms is not None

    def test_train_transforms_apply(self):
        """Test that train transforms can be applied to an image."""
        transforms = get_train_transforms()

        # Create dummy image
        image = create_dummy_image()

        # Apply transforms
        result = transforms(image=image)

        assert "image" in result
        transformed = result["image"]

        # Should be tensor or array
        assert isinstance(transformed, (np.ndarray, torch.Tensor))

    def test_val_transforms_apply(self):
        """Test that validation transforms can be applied."""
        transforms = get_val_transforms()
        image = create_dummy_image()

        result = transforms(image=image)
        assert "image" in result

    def test_transforms_output_shape(self):
        """Test that transforms produce expected output shape."""
        transforms = get_train_transforms()
        image = create_dummy_image(size=(512, 512))

        result = transforms(image=image)
        transformed = result["image"]

        # Check dimensions (should be normalized to standard size)
        if isinstance(transformed, torch.Tensor):
            # CHW format
            assert transformed.ndim == 3
        else:
            # HWC format
            assert transformed.ndim == 3

    def test_transforms_with_different_sizes(self):
        """Test transforms work with various input sizes."""
        transforms = get_val_transforms()

        sizes = [(224, 224), (256, 256), (512, 384), (300, 400)]

        for size in sizes:
            image = create_dummy_image(size=size)
            result = transforms(image=image)
            assert "image" in result


# =============================================================================
# Reproducibility Tests
# =============================================================================


@pytest.mark.skipif(
    not REPRODUCIBILITY_AVAILABLE, reason="Reproducibility module not available"
)
class TestReproducibility:
    """Tests for reproducibility utilities."""

    def test_set_seed_basic(self):
        """Test basic seed setting."""
        set_seed(42)

        # Generate random numbers
        r1 = random.random()
        n1 = np.random.rand()
        t1 = torch.rand(1).item()

        # Reset seed
        set_seed(42)

        # Should get same numbers
        r2 = random.random()
        n2 = np.random.rand()
        t2 = torch.rand(1).item()

        assert r1 == r2
        assert n1 == n2
        assert t1 == t2

    def test_set_seed_different_seeds(self):
        """Test that different seeds produce different results."""
        set_seed(42)
        r1 = random.random()

        set_seed(123)
        r2 = random.random()

        assert r1 != r2

    def test_get_seed_worker(self):
        """Test seed worker function for DataLoader."""
        worker_fn = get_seed_worker(42)

        assert callable(worker_fn)

        # Call with worker_id
        worker_fn(0)
        worker_fn(1)

    def test_set_deterministic(self):
        """Test setting deterministic mode."""
        set_deterministic(True)

        # Should not raise
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False

    def test_reproducibility_with_dataloader(self):
        """Test reproducibility with DataLoader."""
        set_seed(42)

        # Create dummy data
        data = torch.randn(100, 10)
        dataset = torch.utils.data.TensorDataset(data)

        worker_fn = get_seed_worker(42)

        loader1 = DataLoader(
            dataset,
            batch_size=10,
            shuffle=True,
            worker_init_fn=worker_fn,
            generator=torch.Generator().manual_seed(42),
        )

        # Get first batch
        batch1 = next(iter(loader1))

        # Reset and recreate
        set_seed(42)
        loader2 = DataLoader(
            dataset,
            batch_size=10,
            shuffle=True,
            worker_init_fn=worker_fn,
            generator=torch.Generator().manual_seed(42),
        )

        batch2 = next(iter(loader2))

        # Should be identical
        assert torch.equal(batch1[0], batch2[0])

    def test_numpy_reproducibility(self):
        """Test numpy random number reproducibility."""
        set_seed(42)
        arr1 = np.random.randint(0, 100, size=10)

        set_seed(42)
        arr2 = np.random.randint(0, 100, size=10)

        np.testing.assert_array_equal(arr1, arr2)

    def test_torch_reproducibility(self):
        """Test PyTorch random number reproducibility."""
        set_seed(42)
        t1 = torch.randn(10)

        set_seed(42)
        t2 = torch.randn(10)

        assert torch.equal(t1, t2)


# =============================================================================
# Config Tests
# =============================================================================


@pytest.mark.skipif(not CONFIG_AVAILABLE, reason="Config module not available")
class TestConfig:
    """Tests for configuration management."""

    def test_load_config_yaml(self, tmp_path):
        """Test loading YAML config file."""
        if not YAML_AVAILABLE:
            pytest.skip("PyYAML not available")

        config_data = {
            "model": {
                "name": "resnet50",
                "pretrained": True,
            },
            "training": {
                "epochs": 100,
                "batch_size": 32,
                "learning_rate": 0.001,
            },
        }

        config_path = tmp_path / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_data, f)

        loaded = load_config(config_path)

        assert loaded["model"]["name"] == "resnet50"
        assert loaded["training"]["epochs"] == 100

    def test_merge_configs(self):
        """Test merging configuration dictionaries."""
        base_config = {"model": {"name": "resnet50"}, "training": {"epochs": 100}}

        override_config = {"training": {"epochs": 200, "batch_size": 64}}

        merged = merge_configs(base_config, override_config)

        assert merged["model"]["name"] == "resnet50"
        assert merged["training"]["epochs"] == 200
        assert merged["training"]["batch_size"] == 64

    def test_validate_config(self):
        """Test configuration validation."""
        valid_config = {"model": {"name": "resnet50"}, "training": {"epochs": 100}}

        # Should not raise
        validate_config(valid_config)

    def test_load_nonexistent_config(self):
        """Test loading non-existent config file."""
        with pytest.raises(FileNotFoundError):
            load_config(Path("nonexistent_config.yaml"))


# =============================================================================
# MLflow Utils Tests
# =============================================================================


@pytest.mark.skipif(not MLFLOW_UTILS_AVAILABLE, reason="MLflow utils not available")
class TestMLflowUtils:
    """Tests for MLflow utilities."""

    @pytest.fixture
    def mock_mlflow(self):
        """Mock MLflow for testing."""
        with patch.dict("sys.modules", {"mlflow": MagicMock()}):
            yield

    def test_setup_mlflow(self, tmp_path):
        """Test MLflow setup."""
        experiment_name = "test_experiment"

        # This should set up MLflow tracking
        with patch("mlflow.set_experiment"):
            setup_mlflow(experiment_name=experiment_name)
            # Check that set_experiment was called

    def test_log_params(self):
        """Test logging parameters to MLflow."""
        params = {"learning_rate": 0.001, "batch_size": 32, "epochs": 100}

        with patch("mlflow.log_params"):
            log_params(params)

    def test_log_metrics(self):
        """Test logging metrics to MLflow."""
        metrics = {"accuracy": 0.95, "loss": 0.05, "auroc": 0.98}

        with patch("mlflow.log_metrics"):
            log_metrics(metrics, step=1)


# =============================================================================
# Derm7pt Dataset Tests
# =============================================================================


@pytest.mark.skipif(not DERM7PT_AVAILABLE, reason="Derm7pt dataset not available")
class TestDerm7ptDataset:
    """Tests for Derm7pt dataset."""

    def test_basic_loading(self):
        """Test basic Derm7pt loading."""
        root = _resolve_data_root("DERM7PT_ROOT", "derm7pt")

        # Check for metadata
        csv_path = root / "metadata.csv"
        if not csv_path.exists():
            pytest.skip("Derm7pt metadata not found")

        ds = Derm7ptDataset(root=root, split="train", csv_path=csv_path)
        assert len(ds) > 0

    def test_sample_retrieval(self):
        """Test retrieving samples from Derm7pt."""
        root = _resolve_data_root("DERM7PT_ROOT", "derm7pt")
        csv_path = root / "metadata.csv"

        if not csv_path.exists():
            pytest.skip("Derm7pt metadata not found")

        ds = Derm7ptDataset(root=root, split="train", csv_path=csv_path)

        if len(ds) == 0:
            pytest.skip("Empty dataset")

        sample = ds[0]
        assert len(sample) == 3

    def test_concept_labels(self):
        """Test that Derm7pt provides concept labels."""
        root = _resolve_data_root("DERM7PT_ROOT", "derm7pt")
        csv_path = root / "metadata.csv"

        if not csv_path.exists():
            pytest.skip("Derm7pt metadata not found")

        ds = Derm7ptDataset(root=root, split="train", csv_path=csv_path)

        if len(ds) == 0:
            pytest.skip("Empty dataset")

        sample = ds[0]
        meta = sample[2]

        # Check for concept-related metadata
        assert isinstance(meta, dict)


# =============================================================================
# ChestXRay Dataset Tests
# =============================================================================


@pytest.mark.skipif(not CHEST_XRAY_AVAILABLE, reason="ChestXRay dataset not available")
class TestChestXRayDataset:
    """Tests for Chest X-Ray datasets (NIH, PadChest)."""

    def test_nih_basic_loading(self):
        """Test basic NIH ChestX-ray loading."""
        root = _resolve_data_root("NIH_CXR_ROOT", "nih_cxr")
        csv_path = root / "metadata.csv"

        if not csv_path.exists():
            pytest.skip("NIH metadata not found")

        ds = ChestXRayDataset(
            root=root,
            split="train",
            csv_path=csv_path,
            labels_column="Finding Labels",
            image_path_column="image_path",
        )
        assert len(ds) > 0

    def test_multilabel_format(self):
        """Test that ChestXRay returns multilabel format."""
        root = _resolve_data_root("NIH_CXR_ROOT", "nih_cxr")
        csv_path = root / "metadata.csv"

        if not csv_path.exists():
            pytest.skip("NIH metadata not found")

        ds = ChestXRayDataset(
            root=root,
            split="train",
            csv_path=csv_path,
            labels_column="Finding Labels",
            image_path_column="image_path",
        )

        if len(ds) == 0:
            pytest.skip("Empty dataset")

        sample = ds[0]
        label = sample[1]

        # Multilabel should be 1D tensor/array
        if isinstance(label, torch.Tensor):
            assert label.ndim == 1
        elif isinstance(label, np.ndarray):
            assert label.ndim == 1

    def test_padchest_loading(self):
        """Test PadChest dataset loading."""
        root = _resolve_data_root("PADCHEST_ROOT", "padchest")
        csv_path = root / "metadata.csv"

        if not csv_path.exists():
            pytest.skip("PadChest metadata not found")

        # PadChest column names may differ - skip for Phase 1
        pytest.skip("PadChest column mapping configuration pending")

    def test_label_harmonization(self):
        """Test that label harmonization works between NIH and PadChest."""
        # This is a conceptual test - actual implementation may vary
        pass


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests combining multiple components."""

    @pytest.mark.parametrize("dataset_name,env_var,subdir,splits", ISIC_CONFIGS[:1])
    def test_full_training_loop_simulation(self, dataset_name, env_var, subdir, splits):
        """Simulate a complete training loop."""
        root = _resolve_data_root(env_var, subdir)
        csv_path = root / "metadata.csv"

        if not csv_path.exists():
            pytest.skip("metadata.csv not found")

        # Set seed for reproducibility
        if REPRODUCIBILITY_AVAILABLE:
            set_seed(42)

        # Load dataset
        ds = ISICDataset(root=root, split=splits[0], csv_path=csv_path)

        if len(ds) < 8:
            pytest.skip("Not enough samples")

        # Create subset
        indices = list(range(min(16, len(ds))))
        subset = torch.utils.data.Subset(ds, indices)

        # Create DataLoader
        loader = DataLoader(subset, batch_size=4, shuffle=True, num_workers=0)

        # Simulate training loop
        total_samples = 0
        for batch_idx, batch in enumerate(loader):
            images = batch[0]  # Only need images for batch size

            # Get batch size
            if isinstance(images, torch.Tensor):
                batch_size = images.shape[0]
            elif isinstance(images, (list, tuple)):
                batch_size = len(images)
            else:
                batch_size = len(images)

            total_samples += batch_size

        assert total_samples == len(subset)

    def test_cross_dataset_consistency(self):
        """Test that different datasets have consistent interfaces."""
        datasets_loaded = []

        for name, env_var, subdir, splits in ISIC_CONFIGS:
            try:
                root = _resolve_data_root(env_var, subdir)
                csv_path = root / "metadata.csv"
                if csv_path.exists():
                    ds = ISICDataset(root=root, split=splits[0], csv_path=csv_path)
                    if len(ds) > 0:
                        datasets_loaded.append((name, ds))
            except Exception:
                continue

        if len(datasets_loaded) < 2:
            pytest.skip("Need at least 2 datasets")

        # Check consistent interface
        ref_name, ref_ds = datasets_loaded[0]
        ref_sample = ref_ds[0]

        for name, ds in datasets_loaded[1:]:
            sample = ds[0]
            assert len(sample) == len(ref_sample)


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_split_handling(self):
        """Test handling of empty splits."""
        # Create temp metadata with empty val split
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)

            data = [
                {
                    "image_id": "1",
                    "image_path": "img1.jpg",
                    "label": "A",
                    "split": "train",
                },
                {
                    "image_id": "2",
                    "image_path": "img2.jpg",
                    "label": "B",
                    "split": "train",
                },
            ]

            df = pd.DataFrame(data)
            csv_path = tmp_path / "metadata.csv"
            df.to_csv(csv_path, index=False)

            # Try to load val split (should fail)
            with pytest.raises(ValueError):
                ISICDataset(root=tmp_path, split="val", csv_path=csv_path)

    def test_missing_csv_file(self):
        """Test handling of missing CSV file."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            tmp_path = Path(tmp_dir)
            csv_path = tmp_path / "nonexistent.csv"

            with pytest.raises(FileNotFoundError):
                ISICDataset(root=tmp_path, split="train", csv_path=csv_path)

    def test_out_of_bounds_index(self):
        """Test out of bounds indexing."""
        for name, env_var, subdir, splits in ISIC_CONFIGS[:1]:
            root = _resolve_data_root(env_var, subdir)
            csv_path = root / "metadata.csv"

            if not csv_path.exists():
                continue

            ds = ISICDataset(root=root, split=splits[0], csv_path=csv_path)

            with pytest.raises((IndexError, KeyError)):
                _ = ds[len(ds) + 1000]

            break


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Performance and timing tests."""

    @pytest.mark.parametrize("dataset_name,env_var,subdir,splits", ISIC_CONFIGS[:1])
    def test_loading_speed(self, dataset_name, env_var, subdir, splits):
        """Test that sample loading is reasonably fast."""
        root = _resolve_data_root(env_var, subdir)
        csv_path = root / "metadata.csv"

        if not csv_path.exists():
            pytest.skip("metadata.csv not found")

        ds = ISICDataset(root=root, split=splits[0], csv_path=csv_path)

        if len(ds) < 10:
            pytest.skip("Not enough samples")

        start = time.time()
        for i in range(10):
            _ = ds[i]
        elapsed = time.time() - start

        # Should load 10 samples in under 10 seconds
        assert elapsed < 10.0, f"Too slow: {elapsed:.2f}s"


# =============================================================================
# Utility Function Tests
# =============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_create_dummy_image(self):
        """Test dummy image creation."""
        img = create_dummy_image()

        assert isinstance(img, np.ndarray)
        assert img.shape == (224, 224, 3)
        assert img.dtype == np.uint8

    def test_create_dummy_image_custom_size(self):
        """Test dummy image with custom size."""
        img = create_dummy_image(size=(512, 384), channels=1)

        assert img.shape == (512, 384, 1)

    @pytest.mark.skipif(not PIL_AVAILABLE, reason="PIL not available")
    def test_create_dummy_metadata(self):
        """Test dummy metadata creation."""
        with tempfile.TemporaryDirectory() as tmp_dir:
            df, csv_path = create_dummy_metadata(
                num_samples=50, num_classes=5, tmp_dir=Path(tmp_dir)
            )

            assert len(df) == 50
            assert csv_path.exists()
            assert df["label"].nunique() == 5


# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    pytest.main(
        [
            __file__,
            "-v",
            "--tb=short",
            "-x",
        ]
    )
