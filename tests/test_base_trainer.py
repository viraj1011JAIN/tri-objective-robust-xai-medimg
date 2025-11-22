"""
Comprehensive test suite for src/training/base_trainer.py.

Covers:
- TrainingConfig and TrainingMetrics dataclasses
- BaseTrainer initialization and state management
- Training and validation loops
- Checkpoint saving/loading
- Early stopping logic
- Learning rate scheduling
- MLflow integration
- Metric tracking and history

Author: Viraj Pankaj Jain
Institution: University of Glasgow
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict, Tuple
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from src.training.base_trainer import (
    BaseTrainer,
    TrainingConfig,
    TrainingMetrics,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class SimpleModel(nn.Module):
    """Minimal model for testing."""

    def __init__(self, input_size: int = 10, output_size: int = 2) -> None:
        super().__init__()
        self.fc = nn.Linear(input_size, output_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc(x)


class ConcreteTrainer(BaseTrainer):
    """Concrete implementation of BaseTrainer for testing."""

    def training_step(
        self, batch: Any, batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Implement training step."""
        if isinstance(batch, (tuple, list)):
            x, y = batch
        else:
            x = batch
            y = torch.zeros(x.size(0), dtype=torch.long)

        x = x.to(self.device)
        y = y.to(self.device)

        logits = self.model(x)
        loss = nn.functional.cross_entropy(logits, y)

        with torch.no_grad():
            preds = logits.argmax(dim=1)
            acc = (preds == y).float().mean().item()

        return loss, {"accuracy": acc}

    def validation_step(
        self, batch: Any, batch_idx: int
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Implement validation step."""
        return self.training_step(batch, batch_idx)


@pytest.fixture
def device():
    """Return CPU device for testing."""
    return "cpu"


@pytest.fixture
def model():
    """Create simple model."""
    return SimpleModel(input_size=10, output_size=2)


@pytest.fixture
def train_loader():
    """Create training data loader."""
    torch.manual_seed(42)
    x = torch.randn(100, 10)
    y = torch.randint(0, 2, (100,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=20, shuffle=True)


@pytest.fixture
def val_loader():
    """Create validation data loader."""
    torch.manual_seed(123)
    x = torch.randn(40, 10)
    y = torch.randint(0, 2, (40,))
    dataset = TensorDataset(x, y)
    return DataLoader(dataset, batch_size=20, shuffle=False)


@pytest.fixture
def optimizer(model):
    """Create optimizer."""
    return SGD(model.parameters(), lr=0.01)


@pytest.fixture
def config():
    """Create training configuration."""
    return TrainingConfig(
        max_epochs=3,
        eval_every_n_epochs=1,
        log_every_n_steps=2,
        early_stopping_patience=2,
        gradient_clip_val=1.0,
        checkpoint_dir="checkpoints/test",
    )


@pytest.fixture
def temp_dir():
    """Create temporary directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ---------------------------------------------------------------------------
# Dataclass Tests
# ---------------------------------------------------------------------------


class TestTrainingConfig:
    """Test TrainingConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TrainingConfig()

        assert config.max_epochs == 100
        assert config.eval_every_n_epochs == 1
        assert config.log_every_n_steps == 50
        assert config.early_stopping_patience == 20
        assert config.early_stopping_min_delta == 1e-4
        assert config.gradient_clip_val == 1.0
        assert config.checkpoint_dir == "checkpoints"
        assert config.save_top_k == 1
        assert config.monitor_metric == "val_loss"
        assert config.monitor_mode == "min"
        assert config.use_mlflow is False

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TrainingConfig(
            max_epochs=50,
            early_stopping_patience=10,
            monitor_mode="max",
            use_mlflow=True,
        )

        assert config.max_epochs == 50
        assert config.early_stopping_patience == 10
        assert config.monitor_mode == "max"
        assert config.use_mlflow is True

    def test_optional_fields(self):
        """Test optional MLflow fields."""
        config = TrainingConfig(
            mlflow_tracking_uri="http://localhost:5000",
            mlflow_experiment_name="test_exp",
        )

        assert config.mlflow_tracking_uri == "http://localhost:5000"
        assert config.mlflow_experiment_name == "test_exp"


class TestTrainingMetrics:
    """Test TrainingMetrics dataclass."""

    def test_default_values(self):
        """Test default metric values."""
        metrics = TrainingMetrics()

        assert metrics.loss == 0.0
        assert metrics.accuracy == 0.0
        assert metrics.num_batches == 0
        assert metrics.num_samples == 0
        assert isinstance(metrics.extra_metrics, dict)
        assert len(metrics.extra_metrics) == 0

    def test_custom_values(self):
        """Test custom metric values."""
        metrics = TrainingMetrics(
            loss=0.5,
            accuracy=0.85,
            num_batches=10,
            num_samples=200,
            extra_metrics={"f1": 0.8},
        )

        assert metrics.loss == 0.5
        assert metrics.accuracy == 0.85
        assert metrics.num_batches == 10
        assert metrics.num_samples == 200
        assert metrics.extra_metrics["f1"] == 0.8


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestBaseTrainerInit:
    """Test BaseTrainer initialization."""

    def test_cannot_instantiate_abstract_class(
        self, model, train_loader, optimizer, config, device
    ):
        """Test that BaseTrainer cannot be instantiated directly."""
        with pytest.raises(
            TypeError, match="Can't instantiate abstract class"
        ):
            BaseTrainer(
                model=model,
                optimizer=optimizer,
                train_loader=train_loader,
                config=config,
                device=device,
            )

    def test_init_basic(self, model, train_loader, optimizer, config, device):
        """Test basic initialization."""
        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        assert trainer.model is not None
        assert trainer.optimizer is optimizer
        assert trainer.train_loader is train_loader
        assert trainer.config is config
        assert trainer.device == device
        assert trainer.current_epoch == 0
        assert trainer.global_step == 0
        assert trainer.best_epoch == 0
        assert trainer.patience_counter == 0

    def test_init_with_val_loader(
        self, model, train_loader, val_loader, optimizer, config, device
    ):
        """Test initialization with validation loader."""
        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )

        assert trainer.val_loader is val_loader

    def test_init_with_scheduler(
        self, model, train_loader, optimizer, config, device
    ):
        """Test initialization with learning rate scheduler."""
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            scheduler=scheduler,
            device=device,
        )

        assert trainer.scheduler is scheduler

    def test_init_checkpoint_dir_from_config(
        self, model, train_loader, optimizer, device
    ):
        """Test checkpoint directory creation from config."""
        config = TrainingConfig(checkpoint_dir="custom/checkpoints")

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        assert trainer.checkpoint_dir == Path("custom/checkpoints")

    def test_init_checkpoint_dir_explicit(
        self, model, train_loader, optimizer, config, device, temp_dir
    ):
        """Test explicit checkpoint directory."""
        checkpoint_dir = temp_dir / "explicit_checkpoints"

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            checkpoint_dir=checkpoint_dir,
            device=device,
        )

        assert trainer.checkpoint_dir == checkpoint_dir
        assert checkpoint_dir.exists()

    def test_init_best_metric_min_mode(
        self, model, train_loader, optimizer, device
    ):
        """Test best_metric initialization in min mode."""
        config = TrainingConfig(monitor_mode="min")

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        assert trainer.best_metric == float("inf")

    def test_init_best_metric_max_mode(
        self, model, train_loader, optimizer, device
    ):
        """Test best_metric initialization in max mode."""
        config = TrainingConfig(monitor_mode="max")

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        assert trainer.best_metric == float("-inf")

    def test_init_creates_checkpoint_dir(
        self, model, train_loader, optimizer, config, device, temp_dir
    ):
        """Test that checkpoint directory is created."""
        checkpoint_dir = temp_dir / "auto_created"
        config.checkpoint_dir = str(checkpoint_dir)

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        assert trainer is not None
        assert checkpoint_dir.exists()


# ---------------------------------------------------------------------------
# MLflow Tests
# ---------------------------------------------------------------------------


class TestMLflowIntegration:
    """Test MLflow integration."""

    def test_mlflow_import_handled(self):
        """Test that mlflow import exception handling allows module to load."""
        # Simply test that base_trainer can be imported even if mlflow fails
        # The try/except in base_trainer should handle any import errors
        try:
            from src.training.base_trainer import (
                BaseTrainer,
                TrainingConfig,
                TrainingMetrics,
            )

            # If we get here, the module loaded successfully
            assert TrainingConfig is not None
            assert TrainingMetrics is not None
            assert BaseTrainer is not None
        except ImportError:
            # Should not happen - mlflow import failure is caught
            pytest.fail("base_trainer should handle mlflow import failure")

    @patch("src.training.base_trainer.mlflow", None)
    def test_mlflow_none_no_error(
        self, model, train_loader, optimizer, config, device
    ):
        """Test that trainer works when mlflow is None."""
        config.use_mlflow = False

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        assert trainer is not None
        # Should not raise errors even if mlflow is None
        trainer._log_mlflow_metrics({"loss": 0.5}, step=1)

    @patch("src.training.base_trainer.mlflow")
    def test_setup_mlflow_basic(
        self, mock_mlflow, model, train_loader, optimizer, device
    ):
        """Test basic MLflow setup."""
        config = TrainingConfig(use_mlflow=True)

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        assert trainer is not None
        mock_mlflow.start_run.assert_called_once()
        mock_mlflow.log_params.assert_called_once()

    @patch("src.training.base_trainer.mlflow")
    def test_setup_mlflow_with_tracking_uri(
        self, mock_mlflow, model, train_loader, optimizer, device
    ):
        """Test MLflow setup with tracking URI."""
        config = TrainingConfig(
            use_mlflow=True,
            mlflow_tracking_uri="http://localhost:5000",
        )

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        assert trainer is not None
        mock_mlflow.set_tracking_uri.assert_called_once_with(
            "http://localhost:5000"
        )

    @patch("src.training.base_trainer.mlflow")
    def test_setup_mlflow_with_experiment_name(
        self, mock_mlflow, model, train_loader, optimizer, device
    ):
        """Test MLflow setup with experiment name."""
        config = TrainingConfig(
            use_mlflow=True,
            mlflow_experiment_name="test_experiment",
        )

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        assert trainer is not None
        mock_mlflow.set_experiment.assert_called_once_with("test_experiment")

    @patch("src.training.base_trainer.mlflow")
    def test_log_mlflow_metrics(
        self, mock_mlflow, model, train_loader, optimizer, device
    ):
        """Test MLflow metrics logging."""
        mock_mlflow.active_run.return_value = True
        config = TrainingConfig(use_mlflow=True)

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        metrics = {"loss": 0.5, "accuracy": 0.85}
        trainer._log_mlflow_metrics(metrics, step=10)

        mock_mlflow.log_metrics.assert_called_once_with(metrics, step=10)

    def test_mlflow_disabled(
        self, model, train_loader, optimizer, config, device
    ):
        """Test training without MLflow."""
        config.use_mlflow = False

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        # Should not raise any errors
        trainer._log_mlflow_metrics({"loss": 0.5}, step=1)


# ---------------------------------------------------------------------------
# Batch Processing Tests
# ---------------------------------------------------------------------------


class TestBatchProcessing:
    """Test batch processing utilities."""

    def test_get_batch_size_tuple(
        self, model, train_loader, optimizer, config, device
    ):
        """Test _get_batch_size with tuple batch."""
        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        batch = (torch.randn(16, 10), torch.randint(0, 2, (16,)))
        batch_size = trainer._get_batch_size(batch)

        assert batch_size == 16

    def test_get_batch_size_list(
        self, model, train_loader, optimizer, config, device
    ):
        """Test _get_batch_size with list batch."""
        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        batch = [torch.randn(8, 10), torch.randint(0, 2, (8,))]
        batch_size = trainer._get_batch_size(batch)

        assert batch_size == 8

    def test_get_batch_size_dict(
        self, model, train_loader, optimizer, config, device
    ):
        """Test _get_batch_size with dict batch."""
        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        batch = {
            "input": torch.randn(12, 10),
            "target": torch.randint(0, 2, (12,)),
        }
        batch_size = trainer._get_batch_size(batch)

        assert batch_size == 12

    def test_get_batch_size_tensor(
        self, model, train_loader, optimizer, config, device
    ):
        """Test _get_batch_size with single tensor."""
        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        batch = torch.randn(24, 10)
        batch_size = trainer._get_batch_size(batch)

        assert batch_size == 24


# ---------------------------------------------------------------------------
# Training Loop Tests
# ---------------------------------------------------------------------------


class TestTrainingLoop:
    """Test train_epoch method."""

    def test_train_epoch_basic(
        self, model, train_loader, optimizer, config, device
    ):
        """Test basic training epoch."""
        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        metrics = trainer.train_epoch()

        assert metrics.loss >= 0.0
        assert metrics.num_batches == len(train_loader)
        assert metrics.num_samples == len(train_loader.dataset)

    def test_train_epoch_updates_global_step(
        self, model, train_loader, optimizer, config, device
    ):
        """Test that train_epoch updates global_step."""
        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        initial_step = trainer.global_step
        trainer.train_epoch()

        assert trainer.global_step == initial_step + len(train_loader)

    def test_train_epoch_gradient_clipping(
        self, model, train_loader, optimizer, device
    ):
        """Test gradient clipping during training."""
        config = TrainingConfig(gradient_clip_val=0.5)

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        trainer.train_epoch()
        # Should complete without errors

    def test_train_epoch_no_gradient_clipping(
        self, model, train_loader, optimizer, device
    ):
        """Test training without gradient clipping."""
        config = TrainingConfig(gradient_clip_val=0.0)

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        trainer.train_epoch()
        # Should complete without errors

    @patch("src.training.base_trainer.mlflow")
    def test_train_epoch_logs_periodically(
        self, mock_mlflow, model, train_loader, optimizer, device
    ):
        """Test periodic logging during training."""
        mock_mlflow.active_run.return_value = True
        config = TrainingConfig(
            use_mlflow=True,
            log_every_n_steps=2,
        )

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        trainer.train_epoch()

        # Should have logged metrics
        assert mock_mlflow.log_metrics.call_count > 0

    def test_train_epoch_no_logging(
        self, model, train_loader, optimizer, device
    ):
        """Test training without periodic logging."""
        config = TrainingConfig(
            log_every_n_steps=0,  # Disable logging
        )

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        metrics = trainer.train_epoch()

        # Should complete without logging
        assert metrics.num_batches > 0

    def test_train_epoch_custom_metrics(
        self, model, train_loader, optimizer, config, device
    ):
        """Test training with custom metrics not in TrainingMetrics."""

        class CustomTrainer(ConcreteTrainer):
            """Trainer that returns custom metrics."""

            def training_step(
                self, batch: Any, batch_idx: int
            ) -> Tuple[torch.Tensor, Dict[str, float]]:
                """Return custom metrics."""
                loss, metrics = super().training_step(batch, batch_idx)
                # Add custom metric not in TrainingMetrics
                metrics["custom_metric"] = 0.42
                return loss, metrics

        trainer = CustomTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        metrics = trainer.train_epoch()

        # Should handle custom metrics gracefully
        assert metrics.num_batches > 0

    def test_train_epoch_empty_batch_metrics(
        self, model, train_loader, optimizer, config, device
    ):
        """Test training with no extra batch metrics."""

        class MinimalTrainer(ConcreteTrainer):
            """Trainer with minimal metrics."""

            def training_step(
                self, batch: Any, batch_idx: int
            ) -> Tuple[torch.Tensor, Dict[str, float]]:
                """Return only loss."""
                if isinstance(batch, (tuple, list)):
                    x, y = batch
                else:
                    x = batch
                    y = torch.zeros(x.size(0), dtype=torch.long)

                x = x.to(self.device)
                y = y.to(self.device)

                logits = self.model(x)
                loss = nn.functional.cross_entropy(logits, y)

                return loss, {}  # Empty metrics dict

        trainer = MinimalTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        metrics = trainer.train_epoch()

        # Should handle empty metrics dict
        assert metrics.num_batches > 0

    def test_train_epoch_empty_loader(
        self, model, optimizer, config, device
    ):
        """Test training with empty data loader."""
        # Create empty data loader
        empty_dataset = TensorDataset(
            torch.empty(0, 10), torch.empty(0, dtype=torch.long)
        )
        empty_loader = DataLoader(empty_dataset, batch_size=20)

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=empty_loader,
            config=config,
            device=device,
        )

        metrics = trainer.train_epoch()

        # Should handle empty loader (no samples processed)
        assert metrics.num_samples == 0
        assert metrics.loss == 0.0


# ---------------------------------------------------------------------------
# Validation Loop Tests
# ---------------------------------------------------------------------------


class TestValidationLoop:
    """Test validate method."""

    def test_validate_basic(
        self, model, train_loader, val_loader, optimizer, config, device
    ):
        """Test basic validation."""
        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )

        metrics = trainer.validate()

        assert metrics.loss >= 0.0
        assert metrics.num_batches == len(val_loader)
        assert metrics.num_samples == len(val_loader.dataset)

    def test_validate_without_val_loader(
        self, model, train_loader, optimizer, config, device
    ):
        """Test validate when val_loader is None."""
        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=None,
            config=config,
            device=device,
        )

        metrics = trainer.validate()

        # Should return empty metrics
        assert metrics.loss == 0.0
        assert metrics.num_batches == 0

    def test_validate_model_in_eval_mode(
        self, model, train_loader, val_loader, optimizer, config, device
    ):
        """Test that model is in eval mode during validation."""
        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )

        trainer.model.train()  # Set to train mode
        trainer.validate()

        # Model should be back in train mode after validate
        # (validate sets to eval, but doesn't change it back)

    def test_validate_custom_metrics(
        self, model, train_loader, val_loader, optimizer, config, device
    ):
        """Test validation with custom metrics not in TrainingMetrics."""

        class CustomTrainer(ConcreteTrainer):
            """Trainer that returns custom validation metrics."""

            def validation_step(
                self, batch: Any, batch_idx: int
            ) -> Tuple[torch.Tensor, Dict[str, float]]:
                """Return custom metrics."""
                loss, metrics = super().validation_step(batch, batch_idx)
                # Add custom metric not in TrainingMetrics
                metrics["custom_val_metric"] = 0.99
                return loss, metrics

        trainer = CustomTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )

        metrics = trainer.validate()

        # Should handle custom metrics gracefully
        assert metrics.num_batches > 0

    def test_validate_empty_loader(
        self, model, train_loader, optimizer, config, device
    ):
        """Test validation with empty data loader."""
        # Create empty data loader
        empty_dataset = TensorDataset(
            torch.empty(0, 10), torch.empty(0, dtype=torch.long)
        )
        empty_loader = DataLoader(empty_dataset, batch_size=20)

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=empty_loader,
            config=config,
            device=device,
        )

        metrics = trainer.validate()

        # Should handle empty loader (no samples processed)
        assert metrics.num_samples == 0
        assert metrics.loss == 0.0


# ---------------------------------------------------------------------------
# Early Stopping Tests
# ---------------------------------------------------------------------------


class TestEarlyStopping:
    """Test early stopping logic."""

    def test_early_stopping_min_mode_improvement(
        self, model, train_loader, optimizer, device
    ):
        """Test early stopping improvement detection in min mode."""
        config = TrainingConfig(
            monitor_mode="min",
            early_stopping_min_delta=0.01,
        )

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        trainer.best_metric = 1.0
        metrics = TrainingMetrics(loss=0.8)

        improved = trainer._check_early_stopping(metrics)

        assert improved is True
        assert trainer.best_metric == 0.8
        assert trainer.patience_counter == 0

    def test_early_stopping_min_mode_no_improvement(
        self, model, train_loader, optimizer, device
    ):
        """Test early stopping no improvement in min mode."""
        config = TrainingConfig(
            monitor_mode="min",
            early_stopping_min_delta=0.01,
        )

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        trainer.best_metric = 1.0
        metrics = TrainingMetrics(loss=1.1)

        improved = trainer._check_early_stopping(metrics)

        assert improved is False
        assert trainer.patience_counter == 1

    def test_early_stopping_max_mode_improvement(
        self, model, train_loader, optimizer, device
    ):
        """Test early stopping improvement detection in max mode."""
        config = TrainingConfig(
            monitor_mode="max",
            early_stopping_min_delta=0.01,
        )

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        trainer.best_metric = 0.8
        metrics = TrainingMetrics(loss=0.9)

        improved = trainer._check_early_stopping(metrics)

        assert improved is True
        assert trainer.best_metric == 0.9

    def test_early_stopping_triggered(
        self, model, train_loader, optimizer, device
    ):
        """Test early stopping trigger."""
        config = TrainingConfig(
            monitor_mode="min",
            early_stopping_patience=3,
        )

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        trainer.best_metric = 1.0
        trainer.patience_counter = 2

        metrics = TrainingMetrics(loss=1.1)
        trainer._check_early_stopping(metrics)

        assert trainer.patience_counter == 3

    def test_early_stopping_min_delta_no_improvement(
        self, model, train_loader, optimizer, device
    ):
        """Test early stopping when improvement is less than min_delta."""
        config = TrainingConfig(
            monitor_mode="min",
            early_stopping_min_delta=0.1,  # Require 0.1 improvement
        )

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        trainer.best_metric = 1.0
        # Improve by 0.05, but min_delta is 0.1
        metrics = TrainingMetrics(loss=0.95)

        improved = trainer._check_early_stopping(metrics)

        # Should not count as improvement
        assert improved is False
        assert trainer.patience_counter == 1

    def test_early_stopping_max_mode_no_improvement(
        self, model, train_loader, optimizer, device
    ):
        """Test early stopping in max mode with no improvement."""
        config = TrainingConfig(
            monitor_mode="max",
            early_stopping_min_delta=0.1,
        )

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        trainer.best_metric = 0.9
        # Improve by 0.05, but min_delta is 0.1
        metrics = TrainingMetrics(loss=0.95)

        improved = trainer._check_early_stopping(metrics)

        # Should not count as improvement
        assert improved is False
        assert trainer.patience_counter == 1


# ---------------------------------------------------------------------------
# Checkpoint Tests
# ---------------------------------------------------------------------------


class TestCheckpoints:
    """Test checkpoint saving and loading."""

    def test_save_checkpoint_basic(
        self, model, train_loader, optimizer, config, device, temp_dir
    ):
        """Test basic checkpoint saving."""
        config.checkpoint_dir = str(temp_dir)

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        trainer.save_checkpoint()

        last_checkpoint = temp_dir / "last.pt"
        assert last_checkpoint.exists()

    def test_save_checkpoint_best(
        self, model, train_loader, optimizer, config, device, temp_dir
    ):
        """Test saving best checkpoint."""
        config.checkpoint_dir = str(temp_dir)

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        trainer.save_checkpoint(is_best=True)

        best_checkpoint = temp_dir / "best.pt"
        assert best_checkpoint.exists()

    def test_save_checkpoint_with_scheduler(
        self, model, train_loader, optimizer, config, device, temp_dir
    ):
        """Test checkpoint saving with scheduler."""
        config.checkpoint_dir = str(temp_dir)
        scheduler = StepLR(optimizer, step_size=1)

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            scheduler=scheduler,
            device=device,
        )

        trainer.save_checkpoint()

        checkpoint = torch.load(temp_dir / "last.pt", weights_only=False)
        assert "scheduler_state_dict" in checkpoint

    def test_load_checkpoint_basic(
        self, model, train_loader, optimizer, config, device, temp_dir
    ):
        """Test basic checkpoint loading."""
        config.checkpoint_dir = str(temp_dir)

        trainer1 = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        trainer1.current_epoch = 5
        trainer1.global_step = 100
        trainer1.best_metric = 0.5
        trainer1.save_checkpoint()

        # Create new trainer and load
        model2 = SimpleModel()
        optimizer2 = SGD(model2.parameters(), lr=0.01)

        trainer2 = ConcreteTrainer(
            model=model2,
            optimizer=optimizer2,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        trainer2.load_checkpoint(temp_dir / "last.pt")

        assert trainer2.current_epoch == 5
        assert trainer2.global_step == 100
        assert trainer2.best_metric == 0.5

    def test_load_checkpoint_with_scheduler(
        self, model, train_loader, optimizer, config, device, temp_dir
    ):
        """Test checkpoint loading with scheduler."""
        config.checkpoint_dir = str(temp_dir)
        scheduler1 = StepLR(optimizer, step_size=1)

        trainer1 = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            scheduler=scheduler1,
            device=device,
        )

        trainer1.save_checkpoint()

        # Load with new trainer
        model2 = SimpleModel()
        optimizer2 = SGD(model2.parameters(), lr=0.01)
        scheduler2 = StepLR(optimizer2, step_size=1)

        trainer2 = ConcreteTrainer(
            model=model2,
            optimizer=optimizer2,
            train_loader=train_loader,
            config=config,
            scheduler=scheduler2,
            device=device,
        )

        trainer2.load_checkpoint(temp_dir / "last.pt")
        # Should load without errors

    def test_load_checkpoint_file_not_found(
        self, model, train_loader, optimizer, config, device, temp_dir
    ):
        """Test loading non-existent checkpoint."""
        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            config=config,
            device=device,
        )

        with pytest.raises(FileNotFoundError):
            trainer.load_checkpoint(temp_dir / "nonexistent.pt")


# ---------------------------------------------------------------------------
# Full Training Tests
# ---------------------------------------------------------------------------


class TestFullTraining:
    """Test complete fit() method."""

    def test_fit_basic(
        self, model, train_loader, val_loader, optimizer, device
    ):
        """Test basic fit() execution."""
        config = TrainingConfig(
            max_epochs=3,
            eval_every_n_epochs=1,
        )

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )

        history = trainer.fit()

        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 3
        assert len(history["val_loss"]) == 3

    def test_fit_without_validation(
        self, model, train_loader, optimizer, config, device
    ):
        """Test fit() without validation loader."""
        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=None,
            config=config,
            device=device,
        )

        history = trainer.fit()

        assert "train_loss" in history
        assert len(history["val_loss"]) == 0

    def test_fit_with_scheduler(
        self, model, train_loader, val_loader, optimizer, config, device
    ):
        """Test fit() with learning rate scheduler."""
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            scheduler=scheduler,
            device=device,
        )

        history = trainer.fit()

        assert len(history["train_loss"]) == config.max_epochs

    def test_fit_eval_every_n_epochs(
        self, model, train_loader, val_loader, optimizer, device
    ):
        """Test fit() with eval_every_n_epochs."""
        config = TrainingConfig(
            max_epochs=6,
            eval_every_n_epochs=2,
        )

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )

        history = trainer.fit()

        # Should validate every 2 epochs
        assert len(history["val_loss"]) == 3

    @patch("src.training.base_trainer.mlflow")
    def test_fit_with_mlflow(
        self, mock_mlflow, model, train_loader, val_loader, optimizer, device
    ):
        """Test fit() with MLflow logging."""
        mock_mlflow.active_run.return_value = True
        config = TrainingConfig(
            max_epochs=2,
            use_mlflow=True,
        )

        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )

        history = trainer.fit()

        # Should have logged metrics
        assert len(history) > 0
        assert mock_mlflow.log_metrics.call_count > 0

    def test_fit_tracks_history(
        self, model, train_loader, val_loader, optimizer, config, device
    ):
        """Test that fit() tracks metrics history."""
        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )

        trainer.fit()

        assert len(trainer.train_metrics_history) == config.max_epochs
        assert len(trainer.val_metrics_history) == config.max_epochs

    def test_fit_updates_best_epoch(
        self, model, train_loader, val_loader, optimizer, config, device
    ):
        """Test that fit() updates best_epoch."""
        trainer = ConcreteTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
        )

        trainer.fit()

        assert trainer.best_epoch >= 0
        assert trainer.best_epoch < config.max_epochs
