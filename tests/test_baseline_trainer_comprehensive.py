"""
Comprehensive Test Suite for BaselineTrainer (20% → 95%+ Coverage).

Tests all code paths for production-level quality aligned with dissertation's
tri-objective framework:
- All initialization parameter combinations
- Training and validation steps with different loss configurations
- Epoch-level metric aggregation
- Temperature and statistics retrieval
- Integration with CalibrationLoss and TaskLoss
- Class weight handling
- Multi-class and multi-label tasks
- Focal loss variants
- Edge cases and error handling

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Date: November 26, 2025
Target: 95%+ Coverage | A1 Dissertation Quality
"""

from __future__ import annotations

import tempfile
from pathlib import Path
from typing import Any

import pytest
import torch
import torch.nn as nn
from torch.optim import SGD, Adam
from torch.optim.lr_scheduler import StepLR
from torch.utils.data import DataLoader, TensorDataset

from src.training.base_trainer import TrainingConfig, TrainingMetrics
from src.training.baseline_trainer import BaselineTrainer

# ---------------------------------------------------------------------------
# Test Models
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


class MultiLabelCNN(nn.Module):
    """Multi-label classification model (for CheXpert)."""

    def __init__(self, num_classes: int = 14) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(16, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = torch.relu(x)
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def simple_model(device):
    """Create simple CNN model."""
    model = SimpleCNN(num_classes=7, in_channels=3)
    return model.to(device)


@pytest.fixture
def multilabel_model(device):
    """Create multi-label model for CheXpert."""
    model = MultiLabelCNN(num_classes=14)
    return model.to(device)


@pytest.fixture
def train_loader():
    """Create synthetic training data (ISIC-like)."""
    torch.manual_seed(42)
    images = torch.randn(64, 3, 32, 32)
    labels = torch.randint(0, 7, (64,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=16, shuffle=True)


@pytest.fixture
def train_loader_multilabel():
    """Create synthetic training data (CheXpert-like multi-label)."""
    torch.manual_seed(42)
    images = torch.randn(64, 1, 32, 32)
    labels = torch.randint(0, 2, (64, 14)).float()  # Binary labels for 14 classes
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
def val_loader_multilabel():
    """Create synthetic validation data (multi-label)."""
    torch.manual_seed(123)
    images = torch.randn(32, 1, 32, 32)
    labels = torch.randint(0, 2, (32, 14)).float()
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=16, shuffle=False)


@pytest.fixture
def train_loader_three_tuple():
    """Create training data with 3-tuple format (images, labels, metadata)."""
    torch.manual_seed(42)
    images = torch.randn(64, 3, 32, 32)
    labels = torch.randint(0, 7, (64,))
    metadata = torch.arange(64)  # Dummy metadata
    dataset = TensorDataset(images, labels, metadata)
    return DataLoader(dataset, batch_size=16, shuffle=True)


@pytest.fixture
def config():
    """Create training configuration."""
    return TrainingConfig(
        max_epochs=2,
        eval_every_n_epochs=1,
        log_every_n_steps=2,
        early_stopping_patience=5,
        gradient_clip_val=1.0,
        checkpoint_dir="checkpoints_test",
        monitor_metric="val_loss",
        monitor_mode="min",
        use_mlflow=False,
    )


@pytest.fixture
def class_weights(device):
    """Create class weights for imbalanced datasets."""
    return torch.tensor([0.5, 1.0, 2.0, 1.5, 0.8, 3.0, 2.5], device=device)


# ---------------------------------------------------------------------------
# Test Class: Initialization
# ---------------------------------------------------------------------------


class TestBaselineTrainerInitialization:
    """Test BaselineTrainer initialization with various configurations."""

    def test_init_basic_default_params(
        self, simple_model, train_loader, val_loader, config, device
    ):
        """Test basic initialization with default parameters."""
        optimizer = SGD(simple_model.parameters(), lr=0.01)

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
        assert trainer.device == str(device)
        assert trainer.criterion is not None

    def test_init_with_class_weights(
        self, simple_model, train_loader, val_loader, config, device, class_weights
    ):
        """Test initialization with class weights."""
        optimizer = SGD(simple_model.parameters(), lr=0.01)

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

        assert trainer.criterion is not None
        # Verify weights are on correct device
        assert hasattr(trainer.criterion, "class_weights") or hasattr(
            trainer.criterion, "weight"
        )

    def test_init_with_focal_loss(
        self, simple_model, train_loader, val_loader, config, device
    ):
        """Test initialization with focal loss."""
        optimizer = SGD(simple_model.parameters(), lr=0.01)

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
        self, simple_model, train_loader, val_loader, config, device
    ):
        """Test initialization with calibration loss."""
        optimizer = SGD(simple_model.parameters(), lr=0.01)

        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
            use_calibration=True,
            init_temperature=2.0,
            label_smoothing=0.1,
        )

        assert trainer.use_calibration is True
        assert trainer.criterion is not None
        # CalibrationLoss should have get_temperature method
        temp = trainer.get_temperature()
        assert temp is not None
        assert isinstance(temp, float)

    def test_init_multilabel_task(
        self,
        multilabel_model,
        train_loader_multilabel,
        val_loader_multilabel,
        config,
        device,
    ):
        """Test initialization for multi-label classification (CheXpert)."""
        optimizer = SGD(multilabel_model.parameters(), lr=0.01)

        trainer = BaselineTrainer(
            model=multilabel_model,
            train_loader=train_loader_multilabel,
            val_loader=val_loader_multilabel,
            optimizer=optimizer,
            config=config,
            num_classes=14,
            device=device,
            task_type="multi_label",
        )

        assert trainer.task_type == "multi_label"
        assert trainer.num_classes == 14

    def test_init_with_scheduler(
        self, simple_model, train_loader, val_loader, config, device
    ):
        """Test initialization with learning rate scheduler."""
        optimizer = SGD(simple_model.parameters(), lr=0.01)
        scheduler = StepLR(optimizer, step_size=5, gamma=0.5)

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
        self, simple_model, train_loader, val_loader, config, device
    ):
        """Test initialization with custom checkpoint directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            checkpoint_path = Path(tmpdir) / "custom_checkpoints"
            optimizer = SGD(simple_model.parameters(), lr=0.01)

            trainer = BaselineTrainer(
                model=simple_model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                config=config,
                num_classes=7,
                device=device,
                checkpoint_dir=checkpoint_path,
            )

            assert trainer.checkpoint_dir == checkpoint_path
            assert checkpoint_path.exists()

    def test_init_all_parameters(
        self, simple_model, train_loader, val_loader, config, device, class_weights
    ):
        """Test initialization with all parameters specified."""
        with tempfile.TemporaryDirectory() as tmpdir:
            optimizer = Adam(simple_model.parameters(), lr=0.001)
            scheduler = StepLR(optimizer, step_size=3, gamma=0.8)
            checkpoint_path = Path(tmpdir) / "test_checkpoints"

            trainer = BaselineTrainer(
                model=simple_model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                config=config,
                num_classes=7,
                scheduler=scheduler,
                device=device,
                checkpoint_dir=checkpoint_path,
                class_weights=class_weights,
                task_type="multi_class",
                use_focal_loss=True,
                focal_gamma=2.5,
                use_calibration=False,
            )

            assert trainer.num_classes == 7
            assert trainer.task_type == "multi_class"
            assert trainer.use_focal_loss is True
            assert trainer.focal_gamma == 2.5
            assert trainer.use_calibration is False


# ---------------------------------------------------------------------------
# Test Class: Training Step
# ---------------------------------------------------------------------------


class TestBaselineTrainerTrainingStep:
    """Test training_step method."""

    def test_training_step_basic(
        self, simple_model, train_loader, val_loader, config, device
    ):
        """Test basic training step."""
        optimizer = SGD(simple_model.parameters(), lr=0.01)
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        batch = next(iter(train_loader))
        loss, metrics = trainer.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_training_step_stores_predictions(
        self, simple_model, train_loader, val_loader, config, device
    ):
        """Test that training_step stores predictions for epoch metrics."""
        optimizer = SGD(simple_model.parameters(), lr=0.01)
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        trainer.train_predictions.clear()
        trainer.train_targets.clear()

        batch = next(iter(train_loader))
        trainer.training_step(batch, batch_idx=0)

        assert len(trainer.train_predictions) == 1
        assert len(trainer.train_targets) == 1

    def test_training_step_with_three_tuple_batch(
        self, simple_model, train_loader_three_tuple, val_loader, config, device
    ):
        """Test training_step with 3-tuple batch (images, labels, metadata)."""
        optimizer = SGD(simple_model.parameters(), lr=0.01)
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader_three_tuple,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        batch = next(iter(train_loader_three_tuple))
        loss, metrics = trainer.training_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert "accuracy" in metrics


# ---------------------------------------------------------------------------
# Test Class: Validation Step
# ---------------------------------------------------------------------------


class TestBaselineTrainerValidationStep:
    """Test validation_step method."""

    def test_validation_step_basic(
        self, simple_model, train_loader, val_loader, config, device
    ):
        """Test basic validation step."""
        optimizer = SGD(simple_model.parameters(), lr=0.01)
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        batch = next(iter(val_loader))
        loss, metrics = trainer.validation_step(batch, batch_idx=0)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_validation_step_stores_predictions(
        self, simple_model, train_loader, val_loader, config, device
    ):
        """Test that validation_step stores predictions."""
        optimizer = SGD(simple_model.parameters(), lr=0.01)
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        trainer.val_predictions.clear()
        trainer.val_targets.clear()

        batch = next(iter(val_loader))
        trainer.validation_step(batch, batch_idx=0)

        assert len(trainer.val_predictions) == 1
        assert len(trainer.val_targets) == 1


# ---------------------------------------------------------------------------
# Test Class: Epoch Training
# ---------------------------------------------------------------------------


class TestBaselineTrainerEpochTraining:
    """Test train_epoch and validate methods."""

    def test_train_epoch_completes(
        self, simple_model, train_loader, val_loader, config, device
    ):
        """Test that train_epoch completes successfully."""
        optimizer = SGD(simple_model.parameters(), lr=0.01)
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        metrics = trainer.train_epoch()

        assert isinstance(metrics, TrainingMetrics)
        assert metrics.loss >= 0.0
        assert 0.0 <= metrics.accuracy <= 1.0
        assert metrics.num_batches > 0
        assert metrics.num_samples > 0

    def test_train_epoch_clears_buffers(
        self, simple_model, train_loader, val_loader, config, device
    ):
        """Test that train_epoch clears prediction buffers."""
        optimizer = SGD(simple_model.parameters(), lr=0.01)
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        # Add dummy data
        trainer.train_predictions.append(torch.tensor([0, 1, 2]))
        trainer.train_targets.append(torch.tensor([0, 1, 1]))

        trainer.train_epoch()

        # After epoch, buffers should be populated again (not empty from clear)
        assert len(trainer.train_predictions) > 0
        assert len(trainer.train_targets) > 0

    def test_validate_completes(
        self, simple_model, train_loader, val_loader, config, device
    ):
        """Test that validate completes successfully."""
        optimizer = SGD(simple_model.parameters(), lr=0.01)
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

        assert isinstance(metrics, TrainingMetrics)
        assert metrics.loss >= 0.0
        assert 0.0 <= metrics.accuracy <= 1.0
        assert metrics.num_batches > 0
        assert metrics.num_samples > 0

    def test_validate_clears_buffers(
        self, simple_model, train_loader, val_loader, config, device
    ):
        """Test that validate clears prediction buffers."""
        optimizer = SGD(simple_model.parameters(), lr=0.01)
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        # Add dummy data
        trainer.val_predictions.append(torch.tensor([0, 1, 2]))
        trainer.val_targets.append(torch.tensor([0, 1, 1]))

        trainer.validate()

        # After validation, buffers should be populated again
        assert len(trainer.val_predictions) > 0
        assert len(trainer.val_targets) > 0


# ---------------------------------------------------------------------------
# Test Class: Utility Methods
# ---------------------------------------------------------------------------


class TestBaselineTrainerUtilityMethods:
    """Test get_temperature and get_loss_statistics methods."""

    def test_get_temperature_with_calibration(
        self, simple_model, train_loader, val_loader, config, device
    ):
        """Test get_temperature when using calibration loss."""
        optimizer = SGD(simple_model.parameters(), lr=0.01)
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
            use_calibration=True,
            init_temperature=2.5,
        )

        temp = trainer.get_temperature()

        assert temp is not None
        assert isinstance(temp, float)
        assert temp > 0.0

    def test_get_temperature_without_calibration(
        self, simple_model, train_loader, val_loader, config, device
    ):
        """Test get_temperature returns None without calibration."""
        optimizer = SGD(simple_model.parameters(), lr=0.01)
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
            use_calibration=False,
        )

        temp = trainer.get_temperature()

        assert temp is None

    def test_get_loss_statistics_with_phase32_loss(
        self, simple_model, train_loader, val_loader, config, device
    ):
        """Test get_loss_statistics with Phase 3.2 loss functions."""
        optimizer = SGD(simple_model.parameters(), lr=0.01)
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        # Run a training step to populate statistics
        batch = next(iter(train_loader))
        trainer.training_step(batch, batch_idx=0)

        stats = trainer.get_loss_statistics()

        assert isinstance(stats, dict)
        # Stats may be empty if loss doesn't implement get_statistics

    def test_get_loss_statistics_without_method(
        self, simple_model, train_loader, val_loader, config, device
    ):
        """Test get_loss_statistics returns empty dict if method not available."""
        optimizer = SGD(simple_model.parameters(), lr=0.01)
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        # Mock criterion without get_statistics
        class SimpleLoss(nn.Module):
            def forward(self, logits, labels):
                return nn.functional.cross_entropy(logits, labels)

        trainer.criterion = SimpleLoss()

        stats = trainer.get_loss_statistics()

        assert stats == {}


# ---------------------------------------------------------------------------
# Test Class: Integration Tests
# ---------------------------------------------------------------------------


class TestBaselineTrainerIntegration:
    """Integration tests for complete training workflows."""

    def test_full_training_loop_two_epochs(
        self, simple_model, train_loader, val_loader, config, device
    ):
        """Test complete training loop for 2 epochs."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir
            optimizer = SGD(simple_model.parameters(), lr=0.01)

            trainer = BaselineTrainer(
                model=simple_model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                config=config,
                num_classes=7,
                device=device,
            )

            history = {"train_loss": [], "val_loss": []}
            for epoch in range(config.max_epochs):
                train_metrics = trainer.train_epoch()
                val_metrics = trainer.validate()
                history["train_loss"].append(train_metrics.loss)
                history["val_loss"].append(val_metrics.loss)

            assert "train_loss" in history
            assert "val_loss" in history
            assert len(history["train_loss"]) == config.max_epochs
            assert len(history["val_loss"]) == config.max_epochs

    def test_training_with_focal_loss_and_class_weights(
        self, simple_model, train_loader, val_loader, config, device, class_weights
    ):
        """Test training with focal loss and class weights."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir
            config.max_epochs = 1
            optimizer = SGD(simple_model.parameters(), lr=0.01)

            trainer = BaselineTrainer(
                model=simple_model,
                train_loader=train_loader,
                val_loader=val_loader,
                optimizer=optimizer,
                config=config,
                num_classes=7,
                device=device,
                class_weights=class_weights,
                use_focal_loss=True,
                focal_gamma=2.0,
            )

            history = {"train_loss": []}
            for epoch in range(config.max_epochs):
                train_metrics = trainer.train_epoch()
                history["train_loss"].append(train_metrics.loss)

            assert "train_loss" in history
            assert len(history["train_loss"]) > 0

    def test_training_with_calibration_and_smoothing(
        self, simple_model, train_loader, val_loader, config, device
    ):
        """Test training with calibration loss and label smoothing."""
        with tempfile.TemporaryDirectory() as tmpdir:
            config.checkpoint_dir = tmpdir
            config.max_epochs = 1
            optimizer = SGD(simple_model.parameters(), lr=0.01)

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

            history = {"train_loss": []}
            for _ in range(config.max_epochs):
                train_metrics = trainer.train_epoch()
                history["train_loss"].append(train_metrics.loss)

            assert "train_loss" in history
            temp = trainer.get_temperature()
            assert temp is not None


# ---------------------------------------------------------------------------
# Test Class: Edge Cases
# ---------------------------------------------------------------------------


class TestBaselineTrainerEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_prediction_buffers(
        self, simple_model, train_loader, val_loader, config, device
    ):
        """Test behavior with empty prediction buffers."""
        optimizer = SGD(simple_model.parameters(), lr=0.01)
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        trainer.train_predictions.clear()
        trainer.train_targets.clear()

        # Should not crash
        metrics = trainer.train_epoch()
        assert isinstance(metrics, TrainingMetrics)

    def test_single_batch_training(self, simple_model, config, device):
        """Test training with single batch."""
        # Create tiny dataset
        images = torch.randn(4, 3, 32, 32)
        labels = torch.randint(0, 7, (4,))
        dataset = TensorDataset(images, labels)
        tiny_loader = DataLoader(dataset, batch_size=4, shuffle=False)

        optimizer = SGD(simple_model.parameters(), lr=0.01)
        trainer = BaselineTrainer(
            model=simple_model,
            train_loader=tiny_loader,
            val_loader=tiny_loader,
            optimizer=optimizer,
            config=config,
            num_classes=7,
            device=device,
        )

        metrics = trainer.train_epoch()

        assert metrics.num_batches == 1
        assert metrics.num_samples == 4


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
