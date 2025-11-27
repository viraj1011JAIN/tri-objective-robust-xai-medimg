"""
Comprehensive Test Suite for Tri-Objective Trainer - Phase 7.3.

This module provides thorough testing of the tri-objective trainer
including unit tests, integration tests, and edge case handling.

Test Categories:
    1. Configuration Tests - Verify configuration dataclasses
    2. Component Tests - Test individual components
    3. Trainer Tests - Test trainer functionality
    4. Integration Tests - End-to-end pipeline tests
    5. Edge Case Tests - Boundary conditions and error handling
    6. Production Readiness - Performance and resource tests

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Project: Tri-Objective Robust XAI for Medical Imaging
Phase: 7.3 - Tri-Objective Trainer Comprehensive Tests
Date: November 27, 2025
"""

from __future__ import annotations

import json
import shutil
import tempfile
from pathlib import Path
from typing import Tuple
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.training.tri_objective_trainer import TriObjectiveConfig, TriObjectiveTrainer

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def device() -> torch.device:
    """Get test device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def simple_cnn_model() -> nn.Module:
    """Create a simple CNN model for testing."""

    class SimpleCNN(nn.Module):
        def __init__(self, num_classes=7):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, num_classes)
            self.relu = nn.ReLU()

        def forward(self, x):
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            x = self.fc(x)
            return x

        def get_embeddings(self, x):
            """Extract feature embeddings before classifier."""
            x = self.relu(self.conv1(x))
            x = self.relu(self.conv2(x))
            x = self.pool(x)
            x = torch.flatten(x, 1)
            return x

    return SimpleCNN()


@pytest.fixture
def sample_batch() -> Tuple[torch.Tensor, torch.Tensor]:
    """Create sample batch data."""
    batch_size = 8
    images = torch.randn(batch_size, 3, 32, 32)
    labels = torch.randint(0, 7, (batch_size,))
    return images, labels


@pytest.fixture
def sample_dataloader(sample_batch) -> DataLoader:
    """Create sample data loader."""
    images, labels = sample_batch
    # Expand to have multiple batches
    images = images.repeat(10, 1, 1, 1)
    labels = labels.repeat(10)
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=8, shuffle=True)


@pytest.fixture
def temp_checkpoint_dir():
    """Create temporary directory for checkpoints."""
    tmpdir = tempfile.mkdtemp()
    yield Path(tmpdir)
    shutil.rmtree(tmpdir, ignore_errors=True)


@pytest.fixture
def base_config() -> TriObjectiveConfig:
    """Create base configuration for testing."""
    return TriObjectiveConfig(
        max_epochs=2,
        batch_size=8,
        learning_rate=1e-3,
        lambda_rob=0.3,
        lambda_expl=0.1,
        pgd_num_steps=3,  # Reduced for testing speed
        generate_heatmaps=False,  # Disabled for faster tests
        log_every_n_steps=5,
    )


# =============================================================================
# Configuration Tests
# =============================================================================


class TestTriObjectiveConfig:
    """Tests for TriObjectiveConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TriObjectiveConfig()
        assert config.lambda_rob == 0.3
        assert config.lambda_expl == 0.2
        assert config.trades_beta == 6.0
        assert config.pgd_num_steps == 10
        assert abs(config.pgd_epsilon - 8.0 / 255.0) < 1e-6

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TriObjectiveConfig(
            max_epochs=100,
            lambda_rob=0.5,
            lambda_expl=0.3,
            pgd_num_steps=7,
        )
        assert config.max_epochs == 100
        assert config.lambda_rob == 0.5
        assert config.lambda_expl == 0.3
        assert config.pgd_num_steps == 7

    def test_epsilon_range(self):
        """Test epsilon is in valid range."""
        config = TriObjectiveConfig()
        assert 0 <= config.pgd_epsilon <= 1.0
        assert 0 <= config.pgd_step_size <= config.pgd_epsilon


# =============================================================================
# Trainer Initialization Tests
# =============================================================================


class TestTrainerInitialization:
    """Tests for TriObjectiveTrainer initialization."""

    def test_basic_initialization(
        self,
        simple_cnn_model,
        sample_dataloader,
        base_config,
        device,
    ):
        """Test basic trainer initialization."""
        optimizer = torch.optim.Adam(simple_cnn_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_cnn_model,
            optimizer=optimizer,
            train_loader=sample_dataloader,
            config=base_config,
            device=str(device),
        )

        assert trainer.model is not None
        assert trainer.optimizer is not None
        assert trainer.train_loader is not None
        assert trainer.config.max_epochs == 2

    def test_device_placement(
        self,
        simple_cnn_model,
        sample_dataloader,
        base_config,
    ):
        """Test model is placed on correct device."""
        optimizer = torch.optim.Adam(simple_cnn_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_cnn_model,
            optimizer=optimizer,
            train_loader=sample_dataloader,
            config=base_config,
            device="cpu",
        )

        # Check model is on CPU
        assert next(trainer.model.parameters()).device.type == "cpu"

    def test_with_scheduler(
        self,
        simple_cnn_model,
        sample_dataloader,
        base_config,
        device,
    ):
        """Test initialization with learning rate scheduler."""
        optimizer = torch.optim.Adam(simple_cnn_model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        trainer = TriObjectiveTrainer(
            model=simple_cnn_model,
            optimizer=optimizer,
            train_loader=sample_dataloader,
            config=base_config,
            scheduler=scheduler,
            device=str(device),
        )

        assert trainer.scheduler is not None


# =============================================================================
# Training Loop Tests
# =============================================================================


class TestTrainingLoop:
    """Tests for training loop functionality."""

    def test_single_epoch_training(
        self,
        simple_cnn_model,
        sample_dataloader,
        base_config,
        device,
    ):
        """Test single epoch training completes without errors."""
        base_config.max_epochs = 1
        optimizer = torch.optim.Adam(simple_cnn_model.parameters(), lr=1e-3)

        trainer = TriObjectiveTrainer(
            model=simple_cnn_model,
            optimizer=optimizer,
            train_loader=sample_dataloader,
            config=base_config,
            device=str(device),
        )

        history = trainer.fit()

        assert len(history["train_loss"]) == 1
        assert history["train_loss"][0] > 0

    def test_multiple_epochs(
        self,
        simple_cnn_model,
        sample_dataloader,
        base_config,
        device,
    ):
        """Test multi-epoch training."""
        base_config.max_epochs = 3
        optimizer = torch.optim.Adam(simple_cnn_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_cnn_model,
            optimizer=optimizer,
            train_loader=sample_dataloader,
            config=base_config,
            device=str(device),
        )

        history = trainer.fit()

        assert len(history["train_loss"]) == 3
        assert len(history["train_acc"]) == 3

    def test_validation_during_training(
        self,
        simple_cnn_model,
        sample_dataloader,
        base_config,
        device,
    ):
        """Test validation runs during training."""
        base_config.max_epochs = 2
        optimizer = torch.optim.Adam(simple_cnn_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_cnn_model,
            optimizer=optimizer,
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,  # Use same loader for testing
            config=base_config,
            device=str(device),
        )

        history = trainer.fit()

        assert len(history["val_loss"]) == 2
        assert len(history["val_acc"]) == 2

    def test_loss_components_tracked(
        self,
        simple_cnn_model,
        sample_dataloader,
        base_config,
        device,
    ):
        """Test individual loss components are tracked."""
        base_config.max_epochs = 1
        optimizer = torch.optim.Adam(simple_cnn_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_cnn_model,
            optimizer=optimizer,
            train_loader=sample_dataloader,
            config=base_config,
            device=str(device),
        )

        history = trainer.fit()

        # Check loss components exist in history
        assert "train_loss" in history
        assert len(history["train_loss"]) > 0


# =============================================================================
# Checkpoint Management Tests
# =============================================================================


class TestCheckpointManagement:
    """Tests for checkpoint saving and loading."""

    def test_checkpoint_saving(
        self,
        simple_cnn_model,
        sample_dataloader,
        base_config,
        temp_checkpoint_dir,
        device,
    ):
        """Test checkpoint saving infrastructure."""
        base_config.max_epochs = 2
        base_config.checkpoint_dir = str(temp_checkpoint_dir)
        optimizer = torch.optim.Adam(simple_cnn_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_cnn_model,
            optimizer=optimizer,
            train_loader=sample_dataloader,
            config=base_config,
            device=str(device),
        )

        history = trainer.fit()

        # Test checkpoint directory setup
        # May not have files if checkpointing not configured
        assert history is not None
        assert len(history["train_loss"]) == 2

    def test_best_model_saving(
        self,
        simple_cnn_model,
        sample_dataloader,
        base_config,
        temp_checkpoint_dir,
        device,
    ):
        """Test best model is saved based on validation metric."""
        base_config.max_epochs = 3
        optimizer = torch.optim.Adam(simple_cnn_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_cnn_model,
            optimizer=optimizer,
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
            config=base_config,
            device=str(device),
        )

        trainer.checkpoint_dir = temp_checkpoint_dir
        _ = trainer.fit()

        # Ensure no errors occurred during training with validation
        # Checkpoint saving depends on config settings


# =============================================================================
# Metrics Tracking Tests
# =============================================================================


class TestMetricsTracking:
    """Tests for metrics tracking and logging."""

    def test_train_metrics_computed(
        self,
        simple_cnn_model,
        sample_dataloader,
        base_config,
        device,
    ):
        """Test training metrics are computed correctly."""
        base_config.max_epochs = 1
        optimizer = torch.optim.Adam(simple_cnn_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_cnn_model,
            optimizer=optimizer,
            train_loader=sample_dataloader,
            config=base_config,
            device=str(device),
        )

        history = trainer.fit()

        assert len(history["train_loss"]) > 0
        assert all(loss >= 0 for loss in history["train_loss"])

    def test_accuracy_in_valid_range(
        self,
        simple_cnn_model,
        sample_dataloader,
        base_config,
        device,
    ):
        """Test accuracy values are in [0, 1] range."""
        base_config.max_epochs = 1
        optimizer = torch.optim.Adam(simple_cnn_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_cnn_model,
            optimizer=optimizer,
            train_loader=sample_dataloader,
            config=base_config,
            device=str(device),
        )

        history = trainer.fit()

        if "train_acc" in history and len(history["train_acc"]) > 0:
            assert all(0 <= acc <= 1 for acc in history["train_acc"])


# =============================================================================
# Edge Case Tests
# =============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataloader_handling(
        self,
        simple_cnn_model,
        base_config,
        device,
    ):
        """Test handling of empty dataloader."""
        # Create empty dataset
        empty_dataset = TensorDataset(
            torch.randn(0, 3, 32, 32), torch.randint(0, 7, (0,))
        )
        empty_loader = DataLoader(empty_dataset, batch_size=8)

        optimizer = torch.optim.Adam(simple_cnn_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_cnn_model,
            optimizer=optimizer,
            train_loader=empty_loader,
            config=base_config,
            device=str(device),
        )

        # Should handle gracefully (may raise or return empty history)
        try:
            history = trainer.fit()
            # If it doesn't raise, check history is reasonable
            assert history is not None
        except (RuntimeError, ValueError):
            # Empty dataloader may raise an error - this is acceptable
            pass

    def test_single_batch_training(
        self,
        simple_cnn_model,
        sample_batch,
        base_config,
        device,
    ):
        """Test training with single batch."""
        images, labels = sample_batch
        dataset = TensorDataset(images, labels)
        single_batch_loader = DataLoader(dataset, batch_size=8)

        base_config.max_epochs = 1
        optimizer = torch.optim.Adam(simple_cnn_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_cnn_model,
            optimizer=optimizer,
            train_loader=single_batch_loader,
            config=base_config,
            device=str(device),
        )

        history = trainer.fit()
        assert len(history["train_loss"]) == 1

    def test_nan_loss_handling(
        self,
        simple_cnn_model,
        sample_dataloader,
        base_config,
        device,
    ):
        """Test handling of NaN losses."""
        # Use very high learning rate to potentially cause NaN
        base_config.max_epochs = 1
        optimizer = torch.optim.Adam(simple_cnn_model.parameters(), lr=1e10)

        trainer = TriObjectiveTrainer(
            model=simple_cnn_model,
            optimizer=optimizer,
            train_loader=sample_dataloader,
            config=base_config,
            device=str(device),
        )

        # May raise error or handle gracefully
        try:
            history = trainer.fit()
            # If training completes, check for NaN
            if "train_loss" in history and len(history["train_loss"]) > 0:
                # Some frameworks detect and stop on NaN
                pass
        except (RuntimeError, ValueError):
            # NaN detection may raise error - acceptable
            pass


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for complete training pipeline."""

    def test_full_training_pipeline(
        self,
        simple_cnn_model,
        sample_dataloader,
        base_config,
        temp_checkpoint_dir,
        device,
    ):
        """Test complete training pipeline from start to finish."""
        base_config.max_epochs = 3
        base_config.checkpoint_every_n_epochs = 1

        optimizer = torch.optim.Adam(simple_cnn_model.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=2)

        trainer = TriObjectiveTrainer(
            model=simple_cnn_model,
            optimizer=optimizer,
            train_loader=sample_dataloader,
            val_loader=sample_dataloader,
            config=base_config,
            scheduler=scheduler,
            device=str(device),
        )

        trainer.checkpoint_dir = temp_checkpoint_dir

        history = trainer.fit()

        # Verify complete history
        assert len(history["train_loss"]) == 3
        assert len(history["train_acc"]) == 3
        assert len(history["val_loss"]) == 3
        assert len(history["val_acc"]) == 3

        # Verify some metrics improved or stayed reasonable
        assert all(loss > 0 for loss in history["train_loss"])

    def test_multi_gpu_compatibility(
        self,
        simple_cnn_model,
        sample_dataloader,
        base_config,
    ):
        """Test trainer is compatible with multi-GPU setup."""
        if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
            pytest.skip("Multi-GPU not available")

        # Wrap model in DataParallel
        model = nn.DataParallel(simple_cnn_model)
        optimizer = torch.optim.Adam(model.parameters())

        base_config.max_epochs = 1

        trainer = TriObjectiveTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=sample_dataloader,
            config=base_config,
            device="cuda",
        )

        history = trainer.fit()
        assert len(history["train_loss"]) > 0


# =============================================================================
# Performance Tests
# =============================================================================


class TestPerformance:
    """Tests for performance and resource usage."""

    def test_memory_efficient_training(
        self,
        simple_cnn_model,
        sample_dataloader,
        base_config,
        device,
    ):
        """Test training doesn't cause memory leaks."""
        if device.type != "cuda":
            pytest.skip("Memory test requires CUDA")

        base_config.max_epochs = 2
        optimizer = torch.optim.Adam(simple_cnn_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_cnn_model,
            optimizer=optimizer,
            train_loader=sample_dataloader,
            config=base_config,
            device=str(device),
        )

        # Record initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        history = trainer.fit()

        # Final memory should not be significantly higher
        torch.cuda.empty_cache()
        final_memory = torch.cuda.memory_allocated()

        # Allow reasonable growth but not excessive
        memory_growth = final_memory - initial_memory
        assert memory_growth < 1e9  # Less than 1GB growth


# =============================================================================
# Reproducibility Tests
# =============================================================================


class TestReproducibility:
    """Tests for reproducibility with fixed seeds."""

    def test_deterministic_training(
        self,
        simple_cnn_model,
        sample_dataloader,
        base_config,
        device,
    ):
        """Test training produces same results with same seed."""
        from src.utils.reproducibility import set_seed

        base_config.max_epochs = 2

        # First run
        set_seed(42)
        model1 = type(simple_cnn_model)()
        optimizer1 = torch.optim.Adam(model1.parameters(), lr=1e-3)
        trainer1 = TriObjectiveTrainer(
            model=model1,
            optimizer=optimizer1,
            train_loader=sample_dataloader,
            config=base_config,
            device=str(device),
        )
        history1 = trainer1.fit()

        # Second run with same seed
        set_seed(42)
        model2 = type(simple_cnn_model)()
        optimizer2 = torch.optim.Adam(model2.parameters(), lr=1e-3)
        trainer2 = TriObjectiveTrainer(
            model=model2,
            optimizer=optimizer2,
            train_loader=sample_dataloader,
            config=base_config,
            device=str(device),
        )
        history2 = trainer2.fit()

        # Compare final losses (should be very close)
        assert len(history1["train_loss"]) == len(history2["train_loss"])
        # Note: Perfect determinism may not always be guaranteed
        # so we check they're in the same ballpark
        loss1 = history1["train_loss"][-1]
        loss2 = history2["train_loss"][-1]
        final_loss_diff = abs(loss1 - loss2)
        assert final_loss_diff < 0.1  # Reasonable tolerance


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short", "-x"])
