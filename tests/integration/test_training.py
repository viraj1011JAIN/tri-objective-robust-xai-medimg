"""Integration tests for simple training loop and MLflow logging."""

from __future__ import annotations

from pathlib import Path

import mlflow
import pytest
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset


@pytest.fixture
def mlflow_tmp_uri(tmp_path: Path) -> str:
    """Set MLflow tracking to an isolated local file:// URI."""
    tracking_dir = tmp_path / "mlruns"
    tracking_dir.mkdir(parents=True, exist_ok=True)
    uri = tracking_dir.as_uri()  # e.g. file:///C:/Users/...
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.end_run()


class TestTrainingLoop:
    def test_single_epoch_training(self, device: torch.device) -> None:
        """Smoke test: one epoch of training runs without errors."""
        # Tiny dummy dataset
        x = torch.randn(32, 3, 32, 32)
        y = torch.randint(0, 10, (32,))
        dataset = TensorDataset(x, y)
        loader = DataLoader(dataset, batch_size=8, shuffle=True)

        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 32 * 32, 64),
            nn.ReLU(),
            nn.Linear(64, 10),
        ).to(device)

        optimizer = optim.SGD(model.parameters(), lr=1e-2)
        criterion = nn.CrossEntropyLoss()

        model.train()
        total_loss = 0.0
        num_batches = 0

        for xb, yb in loader:
            xb, yb = xb.to(device), yb.to(device)
            optimizer.zero_grad()
            out = model(xb)
            loss = criterion(out, yb)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            num_batches += 1

        avg_loss = total_loss / num_batches
        assert num_batches > 0
        assert avg_loss >= 0.0
        assert torch.isfinite(torch.tensor(avg_loss))


class TestMLflowIntegration:
    def test_mlflow_logging(self, mlflow_tmp_uri: str) -> None:
        """Test that MLflow can log params and metrics with a local file:// store."""
        mlflow.set_experiment("test_experiment")

        with mlflow.start_run() as run:
            mlflow.log_param("learning_rate", 1e-3)
            mlflow.log_param("batch_size", 32)
            for step in range(3):
                mlflow.log_metric("train_loss", 1.0 - 0.1 * step, step=step)

        # We only require that a run was created successfully
        assert run.info.run_id
