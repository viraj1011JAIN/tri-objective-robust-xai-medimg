"""
Publication-Grade Pytest Configuration for Tri-Objective Robust XAI Medical Imaging.

This conftest.py provides shared fixtures, custom markers, and session configuration
for comprehensive testing of the tri-objective adversarial training framework.

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Project: Tri-Objective Robust XAI for Medical Imaging
Target: A1+ Grade | Publication-Ready (NeurIPS/MICCAI/TMI)

Location: tests/conftest.py
"""

from __future__ import annotations

import os
import shutil
import sys
import tempfile
import warnings
from pathlib import Path
from typing import Any, Callable, Dict, Generator, List, Optional, Tuple

import mlflow
import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.utils.data import DataLoader, TensorDataset

# ==============================================================================
# PATH CONFIGURATION
# ==============================================================================

# Ensure project root is on sys.path so `import src` works in tests.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# Also add src directory for direct imports
SRC_PATH = ROOT / "src"
if str(SRC_PATH) not in sys.path:
    sys.path.insert(0, str(SRC_PATH))


# ==============================================================================
# PYTEST CONFIGURATION HOOKS
# ==============================================================================


def pytest_configure(config: pytest.Config) -> None:
    """
    Register custom markers and configure pytest settings.

    Custom markers for tri-objective XAI project:
    - gpu: Tests requiring CUDA
    - slow: Tests taking >10 seconds
    - integration: End-to-end integration tests
    - reproducibility: Determinism validation tests
    - medical: Medical imaging specific tests
    - rq1: Tests related to Research Question 1 (Robustness)
    - rq2: Tests related to Research Question 2 (Explainability)
    - rq3: Tests related to Research Question 3 (Selective Prediction)
    """
    # Set CUBLAS_WORKSPACE_CONFIG for deterministic CUDA operations
    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    config.addinivalue_line("markers", "gpu: mark tests that require a GPU")
    config.addinivalue_line("markers", "slow: mark slow tests (>10 seconds)")
    config.addinivalue_line(
        "markers",
        "integration: mark integration tests touching external systems",
    )
    config.addinivalue_line(
        "markers", "reproducibility: mark tests for deterministic behavior validation"
    )
    config.addinivalue_line(
        "markers", "medical: mark tests specific to medical imaging scenarios"
    )
    config.addinivalue_line(
        "markers", "rq1: mark tests related to RQ1 (Robustness & Generalization)"
    )
    config.addinivalue_line(
        "markers", "rq2: mark tests related to RQ2 (Explainability)"
    )
    config.addinivalue_line(
        "markers", "rq3: mark tests related to RQ3 (Selective Prediction)"
    )


def pytest_collection_modifyitems(
    config: pytest.Config, items: List[pytest.Item]
) -> None:
    """
    Modify test collection based on available resources.

    - Skip GPU tests if CUDA not available
    - Auto-mark tests in certain modules
    """
    # Skip GPU tests if CUDA not available
    if not torch.cuda.is_available():
        skip_gpu = pytest.mark.skip(reason="CUDA not available")
        for item in items:
            if "gpu" in item.keywords:
                item.add_marker(skip_gpu)

    # Auto-mark slow tests in performance modules
    for item in items:
        if "performance" in str(item.fspath).lower():
            item.add_marker(pytest.mark.slow)
        if "integration" in str(item.fspath).lower():
            item.add_marker(pytest.mark.integration)


def pytest_report_header(config: pytest.Config) -> List[str]:
    """
    Add custom header to pytest report showing environment info.
    """
    headers = [
        "=" * 60,
        "Tri-Objective Robust XAI for Medical Imaging - Test Suite",
        "=" * 60,
        f"PyTorch: {torch.__version__}",
        f"NumPy: {np.__version__}",
        f"CUDA available: {torch.cuda.is_available()}",
    ]

    if torch.cuda.is_available():
        headers.append(f"CUDA device: {torch.cuda.get_device_name(0)}")
        mem_gb = torch.cuda.get_device_properties(0).total_memory / 1e9
        headers.append(f"CUDA memory: {mem_gb:.1f} GB")

    headers.append("=" * 60)
    return headers


# ==============================================================================
# GLOBAL PATH FIXTURES
# ==============================================================================


@pytest.fixture(scope="session")
def project_root() -> Path:
    """Absolute path to the project root."""
    return ROOT


@pytest.fixture(scope="session")
def data_root(project_root: Path) -> Path:
    """Top-level data directory."""
    return project_root / "data"


@pytest.fixture(scope="session")
def configs_root(project_root: Path) -> Path:
    """Top-level configs directory."""
    return project_root / "configs"


@pytest.fixture(scope="session")
def results_root(project_root: Path) -> Path:
    """Top-level results directory."""
    return project_root / "results"


# ==============================================================================
# DEVICE AND SEED FIXTURES
# ==============================================================================


@pytest.fixture(scope="session")
def device() -> torch.device:
    """
    Return optimal compute device (CUDA > MPS > CPU).

    Returns:
        torch.device: Best available device for testing.
    """
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


@pytest.fixture(scope="session")
def gpu_available() -> bool:
    """Check if GPU acceleration is available."""
    return torch.cuda.is_available()


@pytest.fixture(scope="session")
def random_seed() -> int:
    """Global random seed for reproducibility (matches dissertation protocol)."""
    return 42


@pytest.fixture(autouse=True)
def set_random_seeds(random_seed: int) -> None:
    """
    Set all random seeds before each test for determinism.

    This is critical for multi-seed experiments (n=3) in the dissertation.
    Seeds: Python random, NumPy, PyTorch (CPU and CUDA).
    """
    import random

    random.seed(random_seed)

    torch.manual_seed(random_seed)
    np.random.seed(random_seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)

    # Deterministic operations
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Set hash seed for Python
    os.environ["PYTHONHASHSEED"] = str(random_seed)

    # Set CUBLAS workspace config for CUDA determinism
    if torch.cuda.is_available():
        os.environ.setdefault("CUBLAS_WORKSPACE_CONFIG", ":4096:8")

    # Disable torch deterministic algorithms to avoid CuBLAS errors in tests
    if hasattr(torch, "use_deterministic_algorithms"):
        torch.use_deterministic_algorithms(False)


@pytest.fixture
def deterministic_mode() -> Generator[None, None, None]:
    """
    Enable fully deterministic mode for reproducibility tests.

    Warning: May impact performance.
    """
    old_deterministic = (
        torch.backends.cudnn.deterministic if hasattr(torch.backends, "cudnn") else True
    )
    old_benchmark = (
        torch.backends.cudnn.benchmark if hasattr(torch.backends, "cudnn") else False
    )

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    if hasattr(torch, "use_deterministic_algorithms"):
        try:
            torch.use_deterministic_algorithms(True)
        except Exception:
            pass

    yield

    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.deterministic = old_deterministic
        torch.backends.cudnn.benchmark = old_benchmark


# ==============================================================================
# TEMPORARY DIRECTORY FIXTURES
# ==============================================================================


@pytest.fixture(scope="session")
def temp_dir() -> Generator[Path, None, None]:
    """
    Temporary directory for test artifacts (cleaned up at end).

    Yields:
        Path: Temporary directory path.
    """
    tmp = tempfile.mkdtemp(prefix="triobj_test_")
    path = Path(tmp)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def temp_checkpoint(tmp_path: Path) -> Path:
    """
    Provide temporary path for checkpoint testing.

    Returns:
        Path: Temporary file path for saving checkpoints.
    """
    return tmp_path / "test_checkpoint.pt"


# ==============================================================================
# DATASET CONFIGURATION FIXTURES
# ==============================================================================


@pytest.fixture(scope="session")
def isic_num_classes() -> int:
    """Number of classes for ISIC 2018 dermoscopy dataset."""
    return 7


@pytest.fixture(scope="session")
def nih_num_classes() -> int:
    """Number of classes for NIH ChestX-ray14 dataset (multi-label)."""
    return 14


@pytest.fixture(scope="session")
def dermoscopy_classes() -> int:
    """Alias for ISIC classes."""
    return 7


@pytest.fixture(scope="session")
def chest_xray_classes() -> int:
    """Alias for NIH classes."""
    return 14


@pytest.fixture(scope="session")
def binary_classes() -> int:
    """Number of classes for binary classification."""
    return 2


# ==============================================================================
# INPUT TENSOR FIXTURES
# ==============================================================================


@pytest.fixture
def batch_rgb(device: torch.device) -> Tensor:
    """
    Standard RGB batch for testing (ImageNet-style input).

    Shape: (4, 3, 224, 224)
    """
    return torch.randn(4, 3, 224, 224, device=device, dtype=torch.float32)


@pytest.fixture
def batch_grayscale(device: torch.device) -> Tensor:
    """
    Grayscale batch (medical imaging common case).

    Shape: (4, 1, 224, 224)
    """
    return torch.randn(4, 1, 224, 224, device=device, dtype=torch.float32)


@pytest.fixture
def single_image(device: torch.device) -> Tensor:
    """
    Single image for inference testing.

    Shape: (1, 3, 224, 224)
    """
    return torch.randn(1, 3, 224, 224, device=device, dtype=torch.float32)


@pytest.fixture
def large_batch(device: torch.device) -> Tensor:
    """
    Large batch for memory stress testing.

    Shape: (64, 3, 224, 224)
    """
    return torch.randn(64, 3, 224, 224, device=device, dtype=torch.float32)


@pytest.fixture
def normalized_batch(device: torch.device) -> Tensor:
    """
    ImageNet-normalized batch (realistic preprocessing).

    Shape: (4, 3, 224, 224)
    Values normalized with ImageNet mean/std.
    """
    mean = torch.tensor([0.485, 0.456, 0.406], device=device).view(1, 3, 1, 1)
    std = torch.tensor([0.229, 0.224, 0.225], device=device).view(1, 3, 1, 1)

    raw = torch.rand(4, 3, 224, 224, device=device)
    normalized = (raw - mean) / std
    return normalized


@pytest.fixture
def adversarial_perturbation(device: torch.device) -> Tensor:
    """
    Small adversarial perturbation for stability testing.

    Shape: (4, 3, 224, 224)
    Magnitude: epsilon = 2/255
    """
    epsilon = 2 / 255
    return torch.randn(4, 3, 224, 224, device=device) * epsilon


# ==============================================================================
# LABEL FIXTURES
# ==============================================================================


@pytest.fixture
def dermoscopy_labels(device: torch.device, isic_num_classes: int) -> Tensor:
    """
    Random dermoscopy labels for testing.

    Shape: (4,)
    """
    return torch.randint(0, isic_num_classes, (4,), device=device)


@pytest.fixture
def multilabel_targets(device: torch.device, nih_num_classes: int) -> Tensor:
    """
    Random multi-label targets for chest X-ray.

    Shape: (4, 14)
    """
    return torch.randint(0, 2, (4, nih_num_classes), device=device).float()


# ==============================================================================
# DATALOADER FIXTURES
# ==============================================================================


@pytest.fixture
def dummy_dataloader(device: torch.device) -> DataLoader:
    """
    Small DataLoader for quick training/validation tests.

    Returns:
        DataLoader with 8 samples, batch_size=4.
    """
    batch_size = 4
    images = torch.randn(batch_size * 2, 3, 224, 224)
    labels = torch.randint(0, 7, (batch_size * 2,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@pytest.fixture
def sample_batch() -> Dict[str, Tensor]:
    """
    Sample batch dictionary used by some unit tests.

    Returns:
        Dict with images, labels, indices.
    """
    batch_size = 4
    return {
        "images": torch.randn(batch_size, 3, 224, 224),
        "labels": torch.randint(0, 7, (batch_size,)),
        "indices": torch.arange(batch_size),
    }


@pytest.fixture
def medical_dataloader(device: torch.device, isic_num_classes: int) -> DataLoader:
    """
    DataLoader mimicking ISIC dermoscopy data structure.

    Returns:
        DataLoader with 16 samples, batch_size=4.
    """
    n_samples = 16
    images = torch.randn(n_samples, 3, 224, 224)
    labels = torch.randint(0, isic_num_classes, (n_samples,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=4, shuffle=False)


# ==============================================================================
# TOY MODEL FIXTURES
# ==============================================================================


class TinyCNN(nn.Module):
    """
    Minimal CNN for unit tests.

    Use when you need a real model without loading heavy backbones.
    """

    def __init__(self, num_classes: int = 7, in_channels: int = 3) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(in_channels, 16, kernel_size=3, padding=1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(32, num_classes)
        self._feature_maps: Dict[str, Tensor] = {}

    def forward(self, x: Tensor) -> Tensor:
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)

    def get_feature_maps(self, x: Tensor) -> Dict[str, Tensor]:
        """Extract feature maps for Grad-CAM compatibility."""
        features = {}
        for i, layer in enumerate(self.features):
            x = layer(x)
            if isinstance(layer, nn.ReLU):
                features[f"relu_{i}"] = x.clone()
        return features


@pytest.fixture
def toy_model(device: torch.device, isic_num_classes: int) -> nn.Module:
    """
    Tiny CNN moved to the appropriate device.

    Use when you need a real model object in tests without loading heavy
    backbones like ResNet or ViT.
    """
    model = TinyCNN(num_classes=isic_num_classes)
    model.to(device)
    model.eval()
    return model


@pytest.fixture
def trainable_model(device: torch.device, isic_num_classes: int) -> nn.Module:
    """
    Tiny CNN in training mode.
    """
    model = TinyCNN(num_classes=isic_num_classes)
    model.to(device)
    model.train()
    return model


# ==============================================================================
# MODEL FACTORY FIXTURE
# ==============================================================================


@pytest.fixture
def model_factory(device: torch.device) -> Callable:
    """
    Factory for creating model instances dynamically.

    Usage:
        model = model_factory("resnet50", num_classes=7)
    """

    def _factory(
        model_name: str,
        num_classes: int = 7,
        pretrained: bool = True,
        eval_mode: bool = True,
    ) -> nn.Module:
        """
        Create model instance.

        Args:
            model_name: One of "resnet50", "efficientnet_b0", "vit_b16", "tiny"
            num_classes: Number of output classes
            pretrained: Use pretrained weights
            eval_mode: Set to eval mode

        Returns:
            Model instance on device
        """
        if model_name == "tiny":
            model = TinyCNN(num_classes=num_classes)
        else:
            try:
                from models.efficientnet import EfficientNetB0Classifier
                from models.resnet import ResNet50Classifier
                from models.vit import ViTB16Classifier
            except ImportError:
                try:
                    from src.models.efficientnet import EfficientNetB0Classifier
                    from src.models.resnet import ResNet50Classifier
                    from src.models.vit import ViTB16Classifier
                except ImportError as e:
                    pytest.skip(f"Model imports not available: {e}")

            model_map = {
                "resnet50": ResNet50Classifier,
                "efficientnet_b0": EfficientNetB0Classifier,
                "vit_b16": ViTB16Classifier,
            }

            if model_name not in model_map:
                raise ValueError(
                    f"Unknown model: {model_name}. "
                    f"Choose from {list(model_map.keys()) + ['tiny']}"
                )

            model = model_map[model_name](
                num_classes=num_classes, pretrained=pretrained
            )

        model = model.to(device)

        if eval_mode:
            model.eval()
        else:
            model.train()

        return model

    return _factory


# ==============================================================================
# CONFIGURATION FIXTURES
# ==============================================================================


@pytest.fixture(scope="session")
def sample_experiment_config(data_root: Path) -> Dict[str, Any]:
    """
    In-memory configuration structure that mirrors ExperimentConfig schema.

    Safe to use in tests that need a realistic config object without hitting disk.
    """
    device_str = "cuda" if torch.cuda.is_available() else "cpu"

    return {
        "experiment": {
            "name": "unit-test-experiment",
            "description": "Minimal config used for unit tests.",
            "project_name": "tri-objective-robust-xai-medimg",
            "tags": {"kind": "unit-test", "dataset": "debug"},
        },
        "dataset": {
            "name": "ISIC2018",
            "root": str((data_root / "processed").as_posix()),
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
        },
        "model": {
            "name": "resnet50",
            "num_classes": 7,
            "pretrained": False,
        },
        "training": {
            "max_epochs": 1,
            "device": device_str,
            "eval_every_n_epochs": 1,
            "log_every_n_steps": 10,
            "gradient_clip_val": 0.0,
            "learning_rate": 1e-3,
            "weight_decay": 0.0,
        },
        "loss": {
            "lambda_rob": 0.3,
            "lambda_expl": 0.1,
            "trades_beta": 6.0,
            "pgd_epsilon": 8 / 255,
            "pgd_steps": 7,
        },
        "reproducibility": {
            "seed": 42,
        },
    }


@pytest.fixture
def tri_objective_config() -> Dict[str, Any]:
    """
    Configuration for tri-objective loss testing.
    """
    return {
        "lambda_rob": 0.3,
        "lambda_expl": 0.1,
        "gamma": 0.5,
        "trades_beta": 6.0,
        "pgd_epsilon": 8 / 255,
        "pgd_steps": 7,
        "ssim_epsilon": 2 / 255,
        "tau_artifact": 0.3,
        "tau_medical": 0.5,
    }


# ==============================================================================
# MLFLOW FIXTURES
# ==============================================================================


@pytest.fixture
def mlflow_test_uri(temp_dir: Path) -> Generator[str, None, None]:
    """
    Configure MLflow to use an isolated local directory for tracking.

    Important: Uses plain path (not file:// URI) for Windows compatibility.

    Yields:
        str: Path to MLflow tracking directory.
    """
    tracking_dir = temp_dir / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)

    # Use a simple local path
    mlflow.set_tracking_uri(str(tracking_dir))

    try:
        yield str(tracking_dir)
    finally:
        # End any active run and reset tracking URI
        try:
            mlflow.end_run()
        except Exception:
            pass
        mlflow.set_tracking_uri("")


@pytest.fixture
def mlflow_experiment(mlflow_test_uri: str) -> Generator[str, None, None]:
    """
    Create isolated MLflow experiment for testing.

    Yields:
        str: Experiment name.
    """
    exp_name = "test-experiment"
    mlflow.set_experiment(exp_name)

    yield exp_name

    try:
        mlflow.end_run()
    except Exception:
        pass


# ==============================================================================
# ASSERTION HELPERS
# ==============================================================================


@pytest.fixture
def assert_tensor_valid() -> Callable:
    """
    Fixture providing tensor validation helper.

    Usage:
        assert_tensor_valid(tensor, "output", check_nan=True)
    """

    def _validate(
        tensor: Tensor,
        name: str = "tensor",
        check_nan: bool = True,
        check_inf: bool = True,
        check_requires_grad: Optional[bool] = None,
        expected_shape: Optional[Tuple[int, ...]] = None,
    ) -> None:
        """
        Validate tensor properties.

        Args:
            tensor: Tensor to validate
            name: Name for error messages
            check_nan: Check for NaN values
            check_inf: Check for Inf values
            check_requires_grad: Check requires_grad if specified
            expected_shape: Check shape if specified
        """
        assert isinstance(
            tensor, Tensor
        ), f"{name} should be Tensor, got {type(tensor)}"

        if check_nan:
            assert not torch.isnan(tensor).any(), f"NaN values in {name}"

        if check_inf:
            assert not torch.isinf(tensor).any(), f"Inf values in {name}"

        if check_requires_grad is not None:
            assert tensor.requires_grad == check_requires_grad, (
                f"{name} requires_grad={tensor.requires_grad}, "
                f"expected {check_requires_grad}"
            )

        if expected_shape is not None:
            assert (
                tensor.shape == expected_shape
            ), f"{name} shape {tensor.shape}, expected {expected_shape}"

    return _validate


@pytest.fixture
def assert_gradients_exist() -> Callable:
    """
    Fixture to check gradients were computed.
    """

    def _check(model: nn.Module, min_params_with_grad: int = 1) -> None:
        """Check that model parameters received gradients."""
        params_with_grad = sum(
            1
            for p in model.parameters()
            if p.grad is not None and p.grad.abs().sum() > 0
        )
        assert params_with_grad >= min_params_with_grad, (
            f"Only {params_with_grad} parameters have gradients, "
            f"expected at least {min_params_with_grad}"
        )

    return _check


# ==============================================================================
# TIMING HELPERS
# ==============================================================================


@pytest.fixture
def timer() -> Callable:
    """
    Fixture providing timing context manager.

    Usage:
        with timer() as t:
            result = model(x)
        print(f"Elapsed: {t.elapsed:.3f} ms")
    """
    import time

    class Timer:
        def __init__(self):
            self.start = None
            self.end = None
            self.elapsed = None

        def __enter__(self):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.start = time.perf_counter()
            return self

        def __exit__(self, *args):
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            self.end = time.perf_counter()
            self.elapsed = (self.end - self.start) * 1000  # ms

    return Timer


# ==============================================================================
# CLEANUP FIXTURES
# ==============================================================================


@pytest.fixture(autouse=True)
def cleanup_gpu() -> Generator[None, None, None]:
    """
    Automatically clean up GPU memory after each test.
    """
    yield

    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ==============================================================================
# WARNINGS CONFIGURATION
# ==============================================================================


@pytest.fixture(autouse=True)
def suppress_warnings() -> Generator[None, None, None]:
    """
    Suppress common warnings during tests.
    """
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore", message=".*non-deterministic.*", category=UserWarning
        )
        warnings.filterwarnings(
            "ignore", message=".*pretrained.*", category=UserWarning
        )
        warnings.filterwarnings("ignore", category=DeprecationWarning)
        yield


# ==============================================================================
# ADVERSARIAL ATTACK FIXTURES (RQ1)
# ==============================================================================


@pytest.fixture
def fgsm_config() -> Dict[str, Any]:
    """Configuration for FGSM attack testing."""
    return {
        "epsilon": 8 / 255,
        "clip_min": 0.0,
        "clip_max": 1.0,
    }


@pytest.fixture
def pgd_config() -> Dict[str, Any]:
    """Configuration for PGD attack testing."""
    return {
        "epsilon": 8 / 255,
        "alpha": 2 / 255,
        "num_steps": 10,
        "random_start": True,
        "clip_min": 0.0,
        "clip_max": 1.0,
    }


# ==============================================================================
# EXPLAINABILITY FIXTURES (RQ2)
# ==============================================================================


@pytest.fixture
def gradcam_target_layer() -> str:
    """Default target layer for Grad-CAM on ResNet-50."""
    return "layer4"


@pytest.fixture
def tcav_config() -> Dict[str, Any]:
    """Configuration for TCAV testing."""
    return {
        "artifact_concepts": ["ruler", "hair", "ink", "borders"],
        "medical_concepts": ["asymmetry", "pigment_network", "blue_white_veil"],
        "tau_artifact": 0.3,
        "tau_medical": 0.5,
        "lambda_medical": 0.5,
    }


# ==============================================================================
# SELECTIVE PREDICTION FIXTURES (RQ3)
# ==============================================================================


@pytest.fixture
def selective_config() -> Dict[str, Any]:
    """Configuration for selective prediction testing."""
    return {
        "tau_conf": 0.7,
        "tau_stab": 0.6,
        "target_coverage": 0.9,
    }
