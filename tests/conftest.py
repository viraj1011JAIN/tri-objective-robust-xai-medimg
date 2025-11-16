"""Pytest configuration and shared fixtures for tri-objective XAI project."""

from __future__ import annotations

import shutil
import sys
import tempfile
from pathlib import Path
from typing import Any, Dict

import mlflow
import numpy as np
import pytest
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset

# Ensure project root is on sys.path so `import src` works in tests.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:  # pragma: no cover - trivial
    sys.path.insert(0, str(ROOT))


def pytest_configure(config: pytest.Config) -> None:
    """Register custom markers used in the test suite."""
    config.addinivalue_line("markers", "gpu: mark tests that require a GPU")
    config.addinivalue_line("markers", "slow: mark slow tests")
    config.addinivalue_line(
        "markers",
        "integration: mark integration tests touching external systems",
    )


# ---------------------------------------------------------------------------
# Global path fixtures
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Device and seed fixtures
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture(scope="session")
def random_seed() -> int:
    """Global random seed for tests."""
    return 42


@pytest.fixture(autouse=True)
def set_random_seeds(random_seed: int) -> None:
    """Set all random seeds before each test for determinism."""
    torch.manual_seed(random_seed)
    np.random.seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(random_seed)
    if hasattr(torch.backends, "cudnn"):  # pragma: no branch
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# ---------------------------------------------------------------------------
# Temporary directory for test artifacts
# ---------------------------------------------------------------------------


@pytest.fixture(scope="session")
def temp_dir() -> Path:
    """Temporary directory for test artifacts (cleaned up at end)."""
    tmp = tempfile.mkdtemp()
    path = Path(tmp)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


# ---------------------------------------------------------------------------
# Data / model fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def dummy_dataloader(device: torch.device) -> DataLoader:
    """Small DataLoader for quick training/validation tests."""
    batch_size = 4
    images = torch.randn(batch_size * 2, 3, 32, 32, device=device)
    labels = torch.randint(0, 10, (batch_size * 2,), device=device)
    # Use CPU tensors for DataLoader workers
    dataset = TensorDataset(images.cpu(), labels.cpu())
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@pytest.fixture
def sample_batch() -> Dict[str, torch.Tensor]:
    """Small CIFAR-like batch used by some unit tests."""
    batch_size = 4
    return {
        "images": torch.randn(batch_size, 3, 32, 32),
        "labels": torch.randint(0, 10, (batch_size,)),
        "indices": torch.arange(batch_size),
    }


class TinyCNN(nn.Module):
    """Minimal CNN for unit tests (2-class logits)."""

    def __init__(self, num_classes: int = 2) -> None:
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 8, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
        )
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        x = self.features(x)
        x = torch.flatten(x, 1)
        return self.classifier(x)


@pytest.fixture
def toy_model(device: torch.device) -> nn.Module:
    """
    Tiny CNN moved to the appropriate device.

    Use this when you need a real model object in tests without loading heavy
    backbones like ResNet or ViT.
    """
    model = TinyCNN(num_classes=2)
    model.to(device)
    model.eval()
    return model


@pytest.fixture(scope="session")
def sample_experiment_config(data_root: Path) -> Dict[str, Any]:
    """
    In-memory configuration structure that mirrors your ExperimentConfig schema.

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
            "name": "DebugDataset",
            "root": str((data_root / "processed").as_posix()),
            "batch_size": 4,
            "num_workers": 0,
            "pin_memory": False,
        },
        "model": {
            "name": "resnet50",
            "num_classes": 2,
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
        "reproducibility": {
            "seed": 42,
        },
    }


# ---------------------------------------------------------------------------
# MLflow fixture
# ---------------------------------------------------------------------------


@pytest.fixture
def mlflow_test_uri(temp_dir: Path) -> str:
    """Configure MLflow to use an isolated local directory for tracking.

    Important: on Windows we pass a plain path (not file:// URI), otherwise
    MLflow can throw 'not a valid remote uri' errors.
    """
    tracking_dir = temp_dir / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)

    # Use a simple local path, which MLflow treats as a FileStore
    mlflow.set_tracking_uri(str(tracking_dir))

    try:
        yield str(tracking_dir)
    finally:
        # End any active run and reset tracking URI back to default
        try:
            mlflow.end_run()
        except Exception:
            pass
        mlflow.set_tracking_uri("")
