"""
Comprehensive test suite for src/training/baseline_trainer.py.

Covers:
- BaselineTrainer initialization with all parameter combinations
- training_step and validation_step
- train_epoch and validate with metric computation
- get_temperature and get_loss_statistics
- Integration with TaskLoss and CalibrationLoss
- Class weight handling
- Multi-class and multi-label tasks
- Focal loss variants

Author: Viraj Pankaj Jain
Institution: University of Glasgow
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any, Dict

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from src.training.base_trainer import TrainingConfig
from src.training.baseline_trainer import BaselineTrainer


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


class SimpleCNN(nn.Module):
    """Minimal CNN for testing."""

    def __init__(self, num_classes: int = 7, in_channels: int = 3) -> None:
        super().__init__()
        self.conv = nn.Conv2d(in_channels, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


@pytest.fixture
def device():
    """Return CPU device for testing."""
    return torch.device("cpu")


@pytest.fixture
def simple_model(device):
    """Create a simple CNN model."""
    model = SimpleCNN(num_classes=7, in_channels=3)
    return model.to(device)


@pytest.fixture
def train_loader():
    """Create synthetic training data."""
    torch.manual_seed(42)
    images = torch.randn(64, 3, 32, 32)
    labels = torch.randint(0, 7, (64,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=16, shuffle=True)


@pytest.fixture
def val_loader():
    """Create synthetic validation data."""
    torch.manual_seed(123)
    images = torch.randn(32, 3, 32, 32)
    labels = torch.randint(0, 7, (32,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=16, shuffle=False)


@pytest.fixture
def optimizer(simple_model):
    """Create optimizer."""
    return SGD(simple_model.parameters(), lr=0.01)


@pytest.fixture
def config():
    """Create training configuration."""
    return TrainingConfig(
        max_epochs=2,
        eval_every_n_epochs=1,
        log_every_n_steps=10,
        early_stopping_patience=5,
        gradient_clip_val=1.0,
        checkpoint_dir="checkpoints/test",
    )


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary checkpoint directory."""
    with tempfile.TemporaryDirectory() as tmpdir:
        yield Path(tmpdir)


# ---------------------------------------------------------------------------
# Initialization Tests
# ---------------------------------------------------------------------------


class TestBaselineTrainerInit:
    """Test BaselineTrainer initialization."""

    def test_init_with_defaults(
        self, simple_model, train_loader, val_loader, optimizer, config, device
    ):
        """Test initialization with default parameters."""
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        assert trainer.num_classes == 7
        assert trainer.task_type == "multi_class"
        assert trainer.use_focal_loss is False
        assert trainer.focal_gamma == 2.0
        assert trainer.use_calibration is False
        assert isinstance(trainer.criterion, nn.Module)

    def test_init_with_focal_loss(
        self, simple_model, train_loader, val_loader, optimizer, config, device
    ):
        """Test initialization with focal loss."""
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
            use_focal_loss=True,
            focal_gamma=3.0,
        )

        assert trainer.use_focal_loss is True
        assert trainer.focal_gamma == 3.0

    def test_init_with_calibration_loss(
        self, simple_model, train_loader, val_loader, optimizer, config, device
    ):
        """Test initialization with calibration loss."""
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
            use_calibration=True,
            init_temperature=1.5,
            label_smoothing=0.1,
        )

        assert trainer.use_calibration is True
        # Should use CalibrationLoss
        assert hasattr(trainer.criterion, "get_temperature")

    def test_init_with_class_weights(
        self, simple_model, train_loader, val_loader, optimizer, config, device
    ):
        """Test initialization with class weights."""
        class_weights = torch.tensor([1.0, 2.0, 1.5, 1.0, 3.0, 1.0, 2.5])

        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
            class_weights=class_weights,
        )

        # Class weights should be on correct device
        assert trainer.criterion is not None

    def test_init_multi_label_task(
        self, device, train_loader, val_loader, optimizer, config
    ):
        """Test initialization with multi-label task type."""
        model = SimpleCNN(num_classes=14, in_channels=1)
        model = model.to(device)

        trainer = BaselineTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=14,
            device=device,
            task_type="multi_label",
        )

        assert trainer.task_type == "multi_label"
        assert trainer.num_classes == 14

    def test_init_with_scheduler(
        self,
        simple_model,
        train_loader,
        val_loader,
        optimizer,
        config,
        device,
    ):
        """Test initialization with learning rate scheduler."""
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
            scheduler=scheduler,
        )

        assert trainer.scheduler is not None

    def test_init_with_checkpoint_dir(
        self,
        simple_model,
        train_loader,
        val_loader,
        optimizer,
        config,
        device,
        temp_checkpoint_dir,
    ):
        """Test initialization with checkpoint directory."""
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
            checkpoint_dir=temp_checkpoint_dir,
        )

        assert trainer.checkpoint_dir == temp_checkpoint_dir

    def test_init_type_conversions(
        self, simple_model, train_loader, val_loader, optimizer, config, device
    ):
        """Test that initialization properly converts types."""
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes="7",  # String should convert to int
            device=device,
            use_focal_loss="True",  # String should convert to bool
            focal_gamma="2.5",  # String should convert to float
            use_calibration=1,  # Int should convert to bool
        )

        assert trainer.num_classes == 7
        assert isinstance(trainer.num_classes, int)
        assert trainer.use_focal_loss is True
        assert trainer.focal_gamma == 2.5
        assert trainer.use_calibration is True


# ---------------------------------------------------------------------------
# Training Step Tests
# ---------------------------------------------------------------------------


class TestTrainingStep:
    """Test training_step method."""

    def test_training_step_basic(
        self, simple_model, train_loader, optimizer, config, device
    ):
        """Test basic training step execution."""
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=None,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        batch = next(iter(train_loader))
        loss, metrics = trainer.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_training_step_accumulates_predictions(
        self, simple_model, train_loader, optimizer, config, device
    ):
        """Test that training_step accumulates predictions."""
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=None,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        assert len(trainer.train_predictions) == 0
        assert len(trainer.train_targets) == 0

        batch = next(iter(train_loader))
        trainer.training_step(batch, batch_idx=0)

        assert len(trainer.train_predictions) == 1
        assert len(trainer.train_targets) == 1

    def test_training_step_with_focal_loss(
        self, simple_model, train_loader, optimizer, config, device
    ):
        """Test training step with focal loss."""
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=None,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
            use_focal_loss=True,
            focal_gamma=2.0,
        )

        batch = next(iter(train_loader))
        loss, metrics = trainer.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)


# ---------------------------------------------------------------------------
# Validation Step Tests
# ---------------------------------------------------------------------------


class TestValidationStep:
    """Test validation_step method."""

    def test_validation_step_basic(
        self, simple_model, val_loader, optimizer, config, device
    ):
        """Test basic validation step execution."""
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=val_loader,  # Use same for simplicity
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        batch = next(iter(val_loader))
        loss, metrics = trainer.validation_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_validation_step_accumulates_predictions(
        self, simple_model, val_loader, optimizer, config, device
    ):
        """Test that validation_step accumulates predictions."""
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=val_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        assert len(trainer.val_predictions) == 0
        assert len(trainer.val_targets) == 0

        batch = next(iter(val_loader))
        trainer.validation_step(batch, batch_idx=0)

        assert len(trainer.val_predictions) == 1
        assert len(trainer.val_targets) == 1


# ---------------------------------------------------------------------------
# Epoch-Level Tests
# ---------------------------------------------------------------------------


class TestEpochMethods:
    """Test train_epoch and validate methods."""

    def test_train_epoch_computes_accuracy(
        self, simple_model, train_loader, optimizer, config, device
    ):
        """Test that train_epoch computes epoch-level accuracy."""
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=None,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        metrics = trainer.train_epoch()

        assert metrics.accuracy >= 0.0
        assert metrics.accuracy <= 1.0
        assert metrics.loss >= 0.0
        assert metrics.num_batches > 0

    def test_train_epoch_clears_predictions(
        self, simple_model, train_loader, optimizer, config, device
    ):
        """Test that train_epoch clears previous predictions."""
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=None,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        # Add dummy predictions
        trainer.train_predictions.append(torch.tensor([1, 2, 3]))
        trainer.train_targets.append(torch.tensor([1, 2, 3]))

        metrics = trainer.train_epoch()

        # After train_epoch, predictions should be fresh
        assert len(trainer.train_predictions) == len(train_loader)

    def test_validate_computes_accuracy(
        self, simple_model, train_loader, val_loader, optimizer, config, device
    ):
        """Test that validate computes epoch-level accuracy."""
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        metrics = trainer.validate()

        assert metrics.accuracy >= 0.0
        assert metrics.accuracy <= 1.0
        assert metrics.loss >= 0.0

    def test_validate_clears_predictions(
        self, simple_model, train_loader, val_loader, optimizer, config, device
    ):
        """Test that validate clears previous predictions."""
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        # Add dummy predictions
        trainer.val_predictions.append(torch.tensor([1, 2, 3]))
        trainer.val_targets.append(torch.tensor([1, 2, 3]))

        metrics = trainer.validate()

        # After validate, predictions should be fresh
        assert len(trainer.val_predictions) == len(val_loader)


# ---------------------------------------------------------------------------
# Temperature and Statistics Tests
# ---------------------------------------------------------------------------


class TestTemperatureAndStatistics:
    """Test get_temperature and get_loss_statistics methods."""

    def test_get_temperature_with_calibration(
        self, simple_model, train_loader, optimizer, config, device
    ):
        """Test get_temperature with calibration loss."""
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=None,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
            use_calibration=True,
            init_temperature=2.0,
        )

        temp = trainer.get_temperature()
        assert temp is not None
        assert isinstance(temp, float)
        assert temp > 0.0

    def test_get_temperature_without_calibration(
        self, simple_model, train_loader, optimizer, config, device
    ):
        """Test get_temperature without calibration loss."""
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=None,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
            use_calibration=False,
        )

        temp = trainer.get_temperature()
        assert temp is None

    def test_get_loss_statistics_with_calibration(
        self, simple_model, train_loader, optimizer, config, device
    ):
        """Test get_loss_statistics with calibration loss."""
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=None,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
            use_calibration=True,
        )

        # Run one training step to generate statistics
        batch = next(iter(train_loader))
        trainer.training_step(batch, batch_idx=0)

        stats = trainer.get_loss_statistics()
        assert isinstance(stats, dict)

    def test_get_loss_statistics_with_task_loss(
        self, simple_model, train_loader, optimizer, config, device
    ):
        """Test get_loss_statistics with TaskLoss (has get_statistics)."""
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=None,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
            use_calibration=False,
        )

        # Run one step to generate statistics
        batch = next(iter(train_loader))
        trainer.training_step(batch, batch_idx=0)

        stats = trainer.get_loss_statistics()
        # TaskLoss has get_statistics method
        assert isinstance(stats, dict)

    def test_get_loss_statistics_without_method(self):
        """Test get_loss_statistics when criterion lacks get_statistics."""
        import torch

        # Create a custom criterion without get_statistics
        class DummyCriterion(nn.Module):
            def forward(self, pred, target):
                return torch.tensor(0.5)

        model = SimpleCNN(num_classes=7)
        images = torch.randn(8, 3, 32, 32)
        labels = torch.randint(0, 7, (8,))
        train_loader = DataLoader(
            TensorDataset(images, labels),
            batch_size=4,
        )

        trainer = BaselineTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=None,
            optimizer=SGD(model.parameters(), lr=0.01),
            config=TrainingConfig(
                max_epochs=1,
                eval_every_n_epochs=1,
                log_every_n_steps=10,
                early_stopping_patience=5,
                gradient_clip_val=1.0,
                checkpoint_dir="checkpoints/test",
            ),
            num_classes=7,
            device=torch.device("cpu"),
        )

        # Replace criterion with one lacking get_statistics
        trainer.criterion = DummyCriterion()

        stats = trainer.get_loss_statistics()
        # Should return empty dict when method doesn't exist
        assert stats == {}


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests for complete training workflows."""

    def test_full_training_loop(
        self, simple_model, train_loader, val_loader, optimizer, config, device
    ):
        """Test complete training loop with fit()."""
        config = TrainingConfig(
            max_epochs=2,
            eval_every_n_epochs=1,
            log_every_n_steps=10,
            early_stopping_patience=10,
            gradient_clip_val=1.0,
            checkpoint_dir="checkpoints/test",
        )

        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        history = trainer.fit()

        assert "train_loss" in history
        assert "val_loss" in history
        assert len(history["train_loss"]) == 2
        assert len(history["val_loss"]) == 2

    def test_training_with_scheduler(
        self, simple_model, train_loader, val_loader, optimizer, config, device
    ):
        """Test training with learning rate scheduler."""
        scheduler = StepLR(optimizer, step_size=1, gamma=0.5)

        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
            scheduler=scheduler,
        )

        trainer.train_epoch()

        # Learning rate should change if scheduler steps
        # (depends on when scheduler.step() is called in base_trainer)
        # Just verify training completes without error

    def test_training_with_different_batch_sizes(
        self, simple_model, optimizer, config, device
    ):
        """Test training with different batch sizes."""
        torch.manual_seed(42)
        images = torch.randn(100, 3, 32, 32)
        labels = torch.randint(0, 7, (100,))
        dataset = TensorDataset(images, labels)

        for batch_size in [8, 16, 32]:
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

            trainer = BaselineTrainer(
                model=simple_model,
                train_loader=loader,
                val_loader=None,
                optimizer=optimizer,
                config=config,
                num_classes=7,
                device=device,
            )

            metrics = trainer.train_epoch()
            assert metrics.num_batches > 0

    def test_criterion_on_correct_device(
        self, simple_model, train_loader, optimizer, config, device
    ):
        """Test that criterion is moved to correct device."""
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=None,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        # Check criterion parameters are on correct device
        for param in trainer.criterion.parameters():
            assert param.device == device

    def test_empty_predictions_handling(
        self, simple_model, optimizer, config, device
    ):
        """Test handling of empty predictions lists."""
        # Create empty loader
        empty_images = torch.empty(0, 3, 32, 32)
        empty_labels = torch.empty(0).long()
        dataset = TensorDataset(empty_images, empty_labels)
        empty_loader = DataLoader(dataset, batch_size=16)

        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=empty_loader,
            val_loader=None,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        # Should handle empty predictions gracefully
        metrics = trainer.train_epoch()

        # Accuracy should remain at initial value when no predictions
        assert metrics.accuracy == 0.0

    def test_validate_with_empty_predictions(
        self, simple_model, optimizer, config, device
    ):
        """Test validate with empty predictions list."""
        # Create empty validation loader
        empty_images = torch.empty(0, 3, 32, 32)
        empty_labels = torch.empty(0).long()
        dataset = TensorDataset(empty_images, empty_labels)
        empty_val_loader = DataLoader(dataset, batch_size=16)

        # Use non-empty train loader
        torch.manual_seed(42)
        train_images = torch.randn(32, 3, 32, 32)
        train_labels = torch.randint(0, 7, (32,))
        train_dataset = TensorDataset(train_images, train_labels)
        train_loader = DataLoader(train_dataset, batch_size=16)

        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=empty_val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        # Should handle empty val_predictions gracefully
        metrics = trainer.validate()

        # Accuracy should remain at initial value when no predictions
        assert metrics.accuracy == 0.0
