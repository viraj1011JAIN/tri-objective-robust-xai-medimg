"""Pytest configuration and shared fixtures for tri-objective XAI project."""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path
from typing import Dict

import mlflow
import numpy as np
import pytest
import torch
from torch.utils.data import DataLoader, TensorDataset


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
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@pytest.fixture(scope="session")
def temp_dir() -> Path:
    """Temporary directory for test artifacts (cleaned up at end)."""
    tmp = tempfile.mkdtemp()
    path = Path(tmp)
    try:
        yield path
    finally:
        shutil.rmtree(path, ignore_errors=True)


@pytest.fixture
def dummy_dataloader(device: torch.device) -> DataLoader:
    """Small DataLoader for quick training / validation tests."""
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


@pytest.fixture
def mlflow_test_uri(temp_dir: Path) -> str:
    """Configure MLflow to use an isolated local directory for tracking.

    Important: on Windows we pass a plain path (not file:// URI), otherwise
    mlflow can throw 'not a valid remote uri' errors.
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
