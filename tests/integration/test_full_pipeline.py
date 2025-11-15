"""End-to-end integration test: tiny CNN + MLflow + checkpointing."""

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
    uri = tracking_dir.as_uri()
    mlflow.set_tracking_uri(uri)
    yield uri
    mlflow.end_run()


class TestFullPipeline:
    def test_end_to_end_training_pipeline(
        self,
        device: torch.device,
        tmp_path: Path,
        mlflow_tmp_uri: str,
    ) -> None:
        """Train a tiny CNN, log with MLflow, and save/load a checkpoint."""
        # ---------- 1. Dummy data ----------
        x_train = torch.randn(64, 3, 32, 32)
        y_train = torch.randint(0, 10, (64,))
        x_val = torch.randn(16, 3, 32, 32)
        y_val = torch.randint(0, 10, (16,))

        train_loader = DataLoader(
            TensorDataset(x_train, y_train), batch_size=16, shuffle=True
        )
        val_loader = DataLoader(
            TensorDataset(x_val, y_val), batch_size=16, shuffle=False
        )

        # ---------- 2. Model ----------
        def make_model() -> nn.Module:
            return nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
                nn.Flatten(),
                nn.Linear(32 * 8 * 8, 64),
                nn.ReLU(),
                nn.Linear(64, 10),
            )

        model = make_model().to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=1e-3)

        num_epochs = 2
        best_val_loss = float("inf")
        checkpoint_path = tmp_path / "best_model.pth"

        # ---------- 3. MLflow ----------
        mlflow.set_experiment("integration_test")

        with mlflow.start_run() as run:
            for epoch in range(num_epochs):
                # ----- Train -----
                model.train()
                train_loss = 0.0
                train_correct = 0
                train_total = 0

                for xb, yb in train_loader:
                    xb, yb = xb.to(device), yb.to(device)
                    optimizer.zero_grad()
                    out = model(xb)
                    loss = criterion(out, yb)
                    loss.backward()
                    optimizer.step()

                    train_loss += loss.item()
                    preds = out.argmax(dim=1)
                    train_correct += (preds == yb).sum().item()
                    train_total += yb.size(0)

                avg_train_loss = train_loss / len(train_loader)
                train_acc = train_correct / max(train_total, 1)

                # ----- Validate -----
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0

                with torch.no_grad():
                    for xb, yb in val_loader:
                        xb, yb = xb.to(device), yb.to(device)
                        out = model(xb)
                        loss = criterion(out, yb)
                        val_loss += loss.item()
                        preds = out.argmax(dim=1)
                        val_correct += (preds == yb).sum().item()
                        val_total += yb.size(0)

                avg_val_loss = val_loss / len(val_loader)
                val_acc = val_correct / max(val_total, 1)

                # Log metrics
                mlflow.log_metric("train_loss", avg_train_loss, step=epoch)
                mlflow.log_metric("train_accuracy", train_acc, step=epoch)
                mlflow.log_metric("val_loss", avg_val_loss, step=epoch)
                mlflow.log_metric("val_accuracy", val_acc, step=epoch)

                # Save best checkpoint
                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    torch.save(
                        {
                            "epoch": epoch,
                            "model_state_dict": model.state_dict(),
                            "optimizer_state_dict": optimizer.state_dict(),
                            "val_loss": best_val_loss,
                        },
                        checkpoint_path,
                    )

            # Only require that a run exists and logging did not explode
            assert run.info.run_id

        # ---------- 4. Checkpoint verification ----------
        assert checkpoint_path.exists()
        checkpoint = torch.load(checkpoint_path, map_location=device)
        assert "model_state_dict" in checkpoint
        assert "optimizer_state_dict" in checkpoint

        new_model = make_model().to(device)
        new_model.load_state_dict(checkpoint["model_state_dict"])

        new_model.eval()
        model.eval()
        test_input = torch.randn(1, 3, 32, 32, device=device)

        with torch.no_grad():
            out1 = model(test_input)
            out2 = new_model(test_input)

        assert out1.shape == out2.shape
        assert torch.isfinite(out1).all()
        assert torch.isfinite(out2).all()
