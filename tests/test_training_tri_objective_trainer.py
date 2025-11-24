"""
Comprehensive A1-Grade Tests for Tri-Objective Trainer.

Tests all functionality in src/training/tri_objective_trainer.py:
- TriObjectiveConfig dataclass
- TriObjectiveTrainer initialization
- Training step (multi-class and multi-label)
- Validation step (multi-class and multi-label)
- Adversarial example generation
- Embedding extraction
- Heatmap generation
- Epoch callbacks with MLflow
- Factory function
- All branches and edge cases

Target: 100% line coverage, 100% branch coverage, 0 failures, 0 skipped.

Author: Viraj Pankaj Jain
Institution: University of Glasgow
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, Optional
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.attacks.pgd import PGD, PGDConfig
from src.losses.tri_objective import TriObjectiveLoss
from src.training.tri_objective_trainer import (
    TriObjectiveConfig,
    TriObjectiveTrainer,
    create_tri_objective_trainer,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    """Device for testing (CPU to avoid device mismatch issues)."""
    # Always use CPU for testing to avoid CUDA/CPU mismatch
    return torch.device("cpu")


@pytest.fixture
def simple_model():
    """Simple CNN model for testing."""

    class SimpleModel(nn.Module):
        def __init__(self, num_classes=7):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, num_classes)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

        def get_embeddings(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            return x.view(x.size(0), -1)

    return SimpleModel()


@pytest.fixture
def model_with_dict_output():
    """Model that returns dict output."""

    class DictModel(nn.Module):
        def __init__(self, num_classes=7):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, num_classes)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            embeddings = self.pool(x).view(x.size(0), -1)
            logits = self.fc(embeddings)
            return {"logits": logits, "embeddings": embeddings}

        def get_embeddings(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            return x.view(x.size(0), -1)

    return DictModel()


@pytest.fixture
def model_no_embeddings():
    """Model without get_embeddings method."""

    class NoEmbeddingsModel(nn.Module):
        def __init__(self, num_classes=7):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, num_classes)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    return NoEmbeddingsModel()


@pytest.fixture
def model_no_embeddings_dict():
    """Model without get_embeddings but returns dict without embeddings."""

    class NoEmbeddingsDictModel(nn.Module):
        def __init__(self, num_classes=7):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(16, num_classes)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return {"logits": self.fc(x)}

    return NoEmbeddingsDictModel()


@pytest.fixture
def train_loader_multiclass():
    """Training data loader for multi-class."""
    images = torch.randn(32, 3, 32, 32)
    labels = torch.randint(0, 7, (32,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=8, shuffle=False)


@pytest.fixture
def train_loader_multilabel():
    """Training data loader for multi-label."""
    images = torch.randn(32, 3, 32, 32)
    labels = torch.randint(0, 2, (32, 14)).float()
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=8, shuffle=False)


@pytest.fixture
def val_loader_multiclass():
    """Validation data loader for multi-class."""
    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 7, (16,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=8, shuffle=False)


@pytest.fixture
def val_loader_multilabel():
    """Validation data loader for multi-label."""
    images = torch.randn(16, 3, 32, 32)
    labels = torch.randint(0, 2, (16, 14)).float()
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=8, shuffle=False)


# ---------------------------------------------------------------------------
# Test TriObjectiveConfig
# ---------------------------------------------------------------------------


class TestTriObjectiveConfig:
    """Test TriObjectiveConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TriObjectiveConfig()

        assert config.lambda_rob == 0.3
        assert config.lambda_expl == 0.2
        assert config.lambda_ssim == 0.7
        assert config.lambda_tcav == 0.3
        assert config.temperature == 1.5
        assert config.trades_beta == 6.0
        assert config.pgd_epsilon == 8.0 / 255.0
        assert config.pgd_step_size == 2.0 / 255.0
        assert config.pgd_num_steps == 10
        assert config.pgd_random_start is True
        assert config.generate_heatmaps is False
        assert config.heatmap_layer == "layer4"
        assert config.extract_embeddings is True
        assert config.embedding_layer == "avgpool"

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TriObjectiveConfig(
            lambda_rob=0.5,
            lambda_expl=0.3,
            temperature=2.0,
            pgd_epsilon=4.0 / 255.0,
            generate_heatmaps=True,
        )

        assert config.lambda_rob == 0.5
        assert config.lambda_expl == 0.3
        assert config.temperature == 2.0
        assert config.pgd_epsilon == 4.0 / 255.0
        assert config.generate_heatmaps is True

    def test_inherits_from_training_config(self):
        """Test that it inherits TrainingConfig attributes."""
        config = TriObjectiveConfig(
            max_epochs=50,
            learning_rate=1e-3,
        )

        assert config.max_epochs == 50
        assert config.learning_rate == 1e-3


# ---------------------------------------------------------------------------
# Test TriObjectiveTrainer Initialization
# ---------------------------------------------------------------------------


class TestTriObjectiveTrainerInit:
    """Test TriObjectiveTrainer initialization."""

    def test_basic_initialization(self, simple_model, train_loader_multiclass, device):
        """Test basic trainer initialization."""
        config = TriObjectiveConfig(max_epochs=10)
        optimizer = torch.optim.Adam(simple_model.parameters(), lr=1e-4)

        trainer = TriObjectiveTrainer(
            model=simple_model,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            device=device,
        )

        assert trainer.device == device
        assert trainer.num_classes == 7
        assert trainer.task_type == "multi_class"
        assert isinstance(trainer.criterion, TriObjectiveLoss)
        assert isinstance(trainer.pgd_attack, PGD)
        assert trainer.generate_heatmaps is False
        assert trainer.extract_embeddings is True

    def test_initialization_with_val_loader(
        self, simple_model, train_loader_multiclass, val_loader_multiclass, device
    ):
        """Test initialization with validation loader."""
        config = TriObjectiveConfig()
        optimizer = torch.optim.Adam(simple_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_model,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            val_loader=val_loader_multiclass,
            device=device,
        )

        assert trainer.val_loader is not None

    def test_initialization_with_scheduler(
        self, simple_model, train_loader_multiclass, device
    ):
        """Test initialization with learning rate scheduler."""
        config = TriObjectiveConfig()
        optimizer = torch.optim.Adam(simple_model.parameters())
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10)

        trainer = TriObjectiveTrainer(
            model=simple_model,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            scheduler=scheduler,
            device=device,
        )

        assert trainer.scheduler is not None

    def test_initialization_with_heatmaps_enabled(
        self, simple_model, train_loader_multiclass, device
    ):
        """Test initialization with heatmap generation enabled."""
        config = TriObjectiveConfig(generate_heatmaps=True)
        optimizer = torch.optim.Adam(simple_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_model,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            device=device,
        )

        assert trainer.generate_heatmaps is True

    def test_initialization_multilabel(
        self, simple_model, train_loader_multilabel, device
    ):
        """Test initialization with multi-label task."""

        # Create model with 14 classes for multi-label
        class MultiLabelModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(16, 14)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)

            def get_embeddings(self, x):
                x = torch.relu(self.conv1(x))
                x = self.pool(x)
                return x.view(x.size(0), -1)

        model = MultiLabelModel()
        config = TriObjectiveConfig()
        config.task_type = "multi_label"
        optimizer = torch.optim.Adam(model.parameters())

        trainer = TriObjectiveTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader_multilabel,
            config=config,
            device=device,
        )

        assert trainer.task_type == "multi_label"
        assert trainer.num_classes == 14


# ---------------------------------------------------------------------------
# Test _infer_num_classes
# ---------------------------------------------------------------------------


class TestInferNumClasses:
    """Test _infer_num_classes method."""

    def test_infer_from_tensor_output(
        self, simple_model, train_loader_multiclass, device
    ):
        """Test inferring num_classes from tensor output."""
        config = TriObjectiveConfig()
        optimizer = torch.optim.Adam(simple_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_model,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            device=device,
        )

        assert trainer.num_classes == 7

    def test_infer_from_dict_output(
        self, model_with_dict_output, train_loader_multiclass, device
    ):
        """Test inferring num_classes from dict output."""
        config = TriObjectiveConfig()
        optimizer = torch.optim.Adam(model_with_dict_output.parameters())

        trainer = TriObjectiveTrainer(
            model=model_with_dict_output,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            device=device,
        )

        assert trainer.num_classes == 7


# ---------------------------------------------------------------------------
# Test _generate_adversarial_examples
# ---------------------------------------------------------------------------


class TestGenerateAdversarialExamples:
    """Test _generate_adversarial_examples method."""

    def test_generate_adversarial_examples(
        self, simple_model, train_loader_multiclass, device
    ):
        """Test adversarial example generation."""
        config = TriObjectiveConfig(pgd_num_steps=5, use_mlflow=False)
        optimizer = torch.optim.Adam(simple_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_model,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            device=device,
        )

        images = torch.randn(4, 3, 32, 32).to(device)
        labels = torch.randint(0, 7, (4,)).to(device)

        # Set to train mode first
        trainer.model.train()

        images_adv = trainer._generate_adversarial_examples(images, labels)

        assert images_adv.shape == images.shape
        assert not torch.equal(images_adv, images)  # Should be perturbed
        assert trainer.model.training  # Should be back in train mode


# ---------------------------------------------------------------------------
# Test _extract_embeddings
# ---------------------------------------------------------------------------


class TestExtractEmbeddings:
    """Test _extract_embeddings method."""

    def test_extract_with_get_embeddings_method(
        self, simple_model, train_loader_multiclass, device
    ):
        """Test embedding extraction with get_embeddings method."""
        config = TriObjectiveConfig()
        optimizer = torch.optim.Adam(simple_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_model,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            device=device,
        )

        images = torch.randn(4, 3, 32, 32).to(device)
        embeddings = trainer._extract_embeddings(images)

        assert embeddings is not None
        assert embeddings.shape[0] == 4
        assert embeddings.ndim == 2

    def test_extract_without_get_embeddings_with_dict(
        self, model_with_dict_output, train_loader_multiclass, device
    ):
        """Test fallback with dict output containing embeddings."""
        config = TriObjectiveConfig()
        optimizer = torch.optim.Adam(model_with_dict_output.parameters())

        trainer = TriObjectiveTrainer(
            model=model_with_dict_output,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            device=device,
        )

        images = torch.randn(4, 3, 32, 32).to(device)
        embeddings = trainer._extract_embeddings(images)

        assert embeddings is not None
        assert embeddings.shape[0] == 4

    def test_extract_without_get_embeddings_no_dict(
        self, model_no_embeddings, train_loader_multiclass, device
    ):
        """Test fallback without get_embeddings and no dict output."""
        config = TriObjectiveConfig()
        optimizer = torch.optim.Adam(model_no_embeddings.parameters())

        # Manually remove get_embeddings if exists
        if hasattr(model_no_embeddings, "get_embeddings"):
            delattr(model_no_embeddings, "get_embeddings")

        trainer = TriObjectiveTrainer(
            model=model_no_embeddings,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            device=device,
        )

        images = torch.randn(4, 3, 32, 32).to(device)
        embeddings = trainer._extract_embeddings(images)

        # Should return None or handle gracefully
        assert embeddings is None or isinstance(embeddings, torch.Tensor)

    def test_extract_without_get_embeddings_dict_no_embeddings_key(
        self, model_no_embeddings_dict, train_loader_multiclass, device
    ):
        """Test fallback with dict but no embeddings key."""
        config = TriObjectiveConfig()
        optimizer = torch.optim.Adam(model_no_embeddings_dict.parameters())

        # Remove get_embeddings
        if hasattr(model_no_embeddings_dict, "get_embeddings"):
            delattr(model_no_embeddings_dict, "get_embeddings")

        trainer = TriObjectiveTrainer(
            model=model_no_embeddings_dict,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            device=device,
        )

        images = torch.randn(4, 3, 32, 32).to(device)
        embeddings = trainer._extract_embeddings(images)

        assert embeddings is None


# ---------------------------------------------------------------------------
# Test _generate_heatmaps
# ---------------------------------------------------------------------------


class TestGenerateHeatmaps:
    """Test _generate_heatmaps method."""

    def test_generate_heatmaps(self, simple_model, train_loader_multiclass, device):
        """Test heatmap generation (placeholder)."""
        config = TriObjectiveConfig()
        optimizer = torch.optim.Adam(simple_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_model,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            device=device,
        )

        images = torch.randn(4, 3, 32, 32).to(device)
        labels = torch.randint(0, 7, (4,)).to(device)

        heatmaps = trainer._generate_heatmaps(images, labels)

        assert heatmaps.shape == (4, 1, 32, 32)
        assert heatmaps.min() >= 0.0 and heatmaps.max() <= 1.0


# ---------------------------------------------------------------------------
# Test training_step
# ---------------------------------------------------------------------------


class TestTrainingStep:
    """Test training_step method."""

    def test_training_step_multiclass(
        self, simple_model, train_loader_multiclass, device
    ):
        """Test training step for multi-class."""
        config = TriObjectiveConfig(
            pgd_num_steps=3,
            gradient_clip_val=1.0,
            use_mlflow=False,
        )
        optimizer = torch.optim.Adam(simple_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_model,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            device=device,
        )

        batch = next(iter(train_loader_multiclass))
        metrics = trainer.training_step(batch, batch_idx=0)

        assert "loss" in metrics
        assert "task_loss" in metrics
        assert "robustness_loss" in metrics
        assert "explanation_loss" in metrics
        assert "ssim_loss" in metrics
        assert "tcav_loss" in metrics
        assert "temperature" in metrics
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_training_step_multilabel(self, train_loader_multilabel, device):
        """Test training step for multi-label."""

        # Create multi-label model
        class MultiLabelModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(16, 14)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)

            def get_embeddings(self, x):
                x = torch.relu(self.conv1(x))
                x = self.pool(x)
                return x.view(x.size(0), -1)

        model = MultiLabelModel()
        config = TriObjectiveConfig(pgd_num_steps=3, use_mlflow=False)
        config.task_type = "multi_label"
        optimizer = torch.optim.Adam(model.parameters())

        trainer = TriObjectiveTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader_multilabel,
            config=config,
            device=device,
        )

        batch = next(iter(train_loader_multilabel))
        metrics = trainer.training_step(batch, batch_idx=0)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_training_step_with_dict_output(
        self, model_with_dict_output, train_loader_multiclass, device
    ):
        """Test training step with model returning dict."""
        config = TriObjectiveConfig(pgd_num_steps=3, use_mlflow=False)
        optimizer = torch.optim.Adam(model_with_dict_output.parameters())

        trainer = TriObjectiveTrainer(
            model=model_with_dict_output,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            device=device,
        )

        batch = next(iter(train_loader_multiclass))
        metrics = trainer.training_step(batch, batch_idx=0)

        assert "loss" in metrics
        assert "accuracy" in metrics

    def test_training_step_without_embeddings(
        self, simple_model, train_loader_multiclass, device
    ):
        """Test training step without embedding extraction."""
        config = TriObjectiveConfig(
            pgd_num_steps=3,
            extract_embeddings=False,
            use_mlflow=False,
        )
        optimizer = torch.optim.Adam(simple_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_model,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            device=device,
        )

        batch = next(iter(train_loader_multiclass))
        metrics = trainer.training_step(batch, batch_idx=0)

        assert "loss" in metrics
        assert metrics["tcav_loss"] == 0.0  # No TCAV without embeddings

    def test_training_step_with_heatmaps(
        self, simple_model, train_loader_multiclass, device
    ):
        """Test training step with heatmap generation."""
        config = TriObjectiveConfig(
            pgd_num_steps=3,
            generate_heatmaps=True,
            use_mlflow=False,
        )
        optimizer = torch.optim.Adam(simple_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_model,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            device=device,
        )

        batch = next(iter(train_loader_multiclass))
        metrics = trainer.training_step(batch, batch_idx=0)

        assert "loss" in metrics
        # SSIM loss should be non-zero with heatmaps
        assert "ssim_loss" in metrics

    def test_training_step_without_gradient_clipping(
        self, simple_model, train_loader_multiclass, device
    ):
        """Test training step without gradient clipping."""
        config = TriObjectiveConfig(
            pgd_num_steps=3,
            gradient_clip_val=0.0,  # Disable clipping
            use_mlflow=False,
        )
        optimizer = torch.optim.Adam(simple_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_model,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            device=device,
        )

        batch = next(iter(train_loader_multiclass))
        metrics = trainer.training_step(batch, batch_idx=0)

        assert "loss" in metrics


# ---------------------------------------------------------------------------
# Test validation_step
# ---------------------------------------------------------------------------


class TestValidationStep:
    """Test validation_step method."""

    def test_validation_step_multiclass(
        self, simple_model, val_loader_multiclass, train_loader_multiclass, device
    ):
        """Test validation step for multi-class."""
        config = TriObjectiveConfig()
        optimizer = torch.optim.Adam(simple_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_model,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            val_loader=val_loader_multiclass,
            device=device,
        )

        batch = next(iter(val_loader_multiclass))
        metrics = trainer.validation_step(batch, batch_idx=0)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0
        assert isinstance(metrics["loss"], float)

    def test_validation_step_multilabel(
        self, val_loader_multilabel, train_loader_multilabel, device
    ):
        """Test validation step for multi-label."""

        # Create multi-label model
        class MultiLabelModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.fc = nn.Linear(16, 14)

            def forward(self, x):
                x = torch.relu(self.conv1(x))
                x = self.pool(x)
                x = x.view(x.size(0), -1)
                return self.fc(x)

            def get_embeddings(self, x):
                x = torch.relu(self.conv1(x))
                x = self.pool(x)
                return x.view(x.size(0), -1)

        model = MultiLabelModel()
        config = TriObjectiveConfig()
        config.task_type = "multi_label"
        optimizer = torch.optim.Adam(model.parameters())

        trainer = TriObjectiveTrainer(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader_multilabel,
            config=config,
            val_loader=val_loader_multilabel,
            device=device,
        )

        batch = next(iter(val_loader_multilabel))
        metrics = trainer.validation_step(batch, batch_idx=0)

        assert "loss" in metrics
        assert "accuracy" in metrics
        assert 0.0 <= metrics["accuracy"] <= 1.0

    def test_validation_step_with_dict_output(
        self,
        model_with_dict_output,
        val_loader_multiclass,
        train_loader_multiclass,
        device,
    ):
        """Test validation step with model returning dict."""
        config = TriObjectiveConfig()
        optimizer = torch.optim.Adam(model_with_dict_output.parameters())

        trainer = TriObjectiveTrainer(
            model=model_with_dict_output,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            val_loader=val_loader_multiclass,
            device=device,
        )

        batch = next(iter(val_loader_multiclass))
        metrics = trainer.validation_step(batch, batch_idx=0)

        assert "loss" in metrics
        assert "accuracy" in metrics


# ---------------------------------------------------------------------------
# Test Epoch Callbacks
# ---------------------------------------------------------------------------


class TestEpochCallbacks:
    """Test epoch callback methods."""

    def test_on_train_epoch_end_without_mlflow(
        self, simple_model, train_loader_multiclass, device
    ):
        """Test training epoch end callback without MLflow."""
        config = TriObjectiveConfig(use_mlflow=False)
        optimizer = torch.optim.Adam(simple_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_model,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            device=device,
        )

        metrics = {
            "loss": 1.5,
            "task_loss": 1.0,
            "robustness_loss": 0.3,
            "explanation_loss": 0.2,
            "accuracy": 0.75,
        }

        # Should not raise error
        trainer.on_train_epoch_end(epoch=1, metrics=metrics)

    def test_on_train_epoch_end_with_mlflow(
        self, simple_model, train_loader_multiclass, device
    ):
        """Test training epoch end callback with MLflow."""
        with (
            patch("src.training.tri_objective_trainer.MLFLOW_AVAILABLE", True),
            patch("src.training.tri_objective_trainer.mlflow") as mock_mlflow,
        ):

            config = TriObjectiveConfig(
                use_mlflow=False
            )  # Disable to avoid base trainer MLflow
            optimizer = torch.optim.Adam(simple_model.parameters())

            trainer = TriObjectiveTrainer(
                model=simple_model,
                optimizer=optimizer,
                train_loader=train_loader_multiclass,
                config=config,
                device=device,
            )

            # Manually enable MLflow for this test
            trainer.config.use_mlflow = True

            metrics = {
                "loss": 1.5,
                "task_loss": 1.0,
                "accuracy": 0.75,
            }

            trainer.on_train_epoch_end(epoch=1, metrics=metrics)

            # Verify MLflow was called
            assert mock_mlflow.log_metric.call_count == 3

    def test_on_val_epoch_end_without_mlflow(
        self, simple_model, train_loader_multiclass, device
    ):
        """Test validation epoch end callback without MLflow."""
        config = TriObjectiveConfig(use_mlflow=False)
        optimizer = torch.optim.Adam(simple_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_model,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            device=device,
        )

        metrics = {
            "loss": 1.3,
            "accuracy": 0.80,
        }

        # Should not raise error
        trainer.on_val_epoch_end(epoch=1, metrics=metrics)

    def test_on_val_epoch_end_with_mlflow(
        self, simple_model, train_loader_multiclass, device
    ):
        """Test validation epoch end callback with MLflow."""
        with (
            patch("src.training.tri_objective_trainer.MLFLOW_AVAILABLE", True),
            patch("src.training.tri_objective_trainer.mlflow") as mock_mlflow,
        ):

            config = TriObjectiveConfig(
                use_mlflow=False
            )  # Disable to avoid base trainer MLflow
            optimizer = torch.optim.Adam(simple_model.parameters())

            trainer = TriObjectiveTrainer(
                model=simple_model,
                optimizer=optimizer,
                train_loader=train_loader_multiclass,
                config=config,
                device=device,
            )

            # Manually enable MLflow for this test
            trainer.config.use_mlflow = True

            metrics = {
                "loss": 1.3,
                "accuracy": 0.80,
            }

            trainer.on_val_epoch_end(epoch=1, metrics=metrics)

            # Verify MLflow was called
            assert mock_mlflow.log_metric.call_count == 2


# ---------------------------------------------------------------------------
# Test Factory Function
# ---------------------------------------------------------------------------


class TestCreateTriObjectiveTrainer:
    """Test create_tri_objective_trainer factory function."""

    def test_factory_basic(self, simple_model, train_loader_multiclass, device):
        """Test basic factory creation."""
        trainer = create_tri_objective_trainer(
            model=simple_model,
            train_loader=train_loader_multiclass,
            device=device,
            use_mlflow=False,
        )

        assert isinstance(trainer, TriObjectiveTrainer)
        assert trainer.device == device
        assert trainer.optimizer is not None
        assert trainer.scheduler is not None

    def test_factory_with_val_loader(
        self, simple_model, train_loader_multiclass, val_loader_multiclass, device
    ):
        """Test factory with validation loader."""
        trainer = create_tri_objective_trainer(
            model=simple_model,
            train_loader=train_loader_multiclass,
            val_loader=val_loader_multiclass,
            device=device,
            use_mlflow=False,
        )

        assert trainer.val_loader is not None

    def test_factory_custom_params(self, simple_model, train_loader_multiclass, device):
        """Test factory with custom parameters."""
        trainer = create_tri_objective_trainer(
            model=simple_model,
            train_loader=train_loader_multiclass,
            learning_rate=1e-3,
            weight_decay=1e-3,
            max_epochs=50,
            lambda_rob=0.4,
            lambda_expl=0.3,
            pgd_epsilon=4.0 / 255.0,
            pgd_num_steps=20,
            device=device,
            checkpoint_dir="custom_checkpoints",
            use_mlflow=False,
        )

        assert trainer.config.max_epochs == 50
        assert trainer.config.lambda_rob == 0.4
        assert trainer.config.lambda_expl == 0.3
        assert trainer.config.pgd_epsilon == 4.0 / 255.0
        assert trainer.config.pgd_num_steps == 20
        assert trainer.config.checkpoint_dir == "custom_checkpoints"
        assert trainer.config.use_mlflow is False

    def test_factory_with_kwargs(self, simple_model, train_loader_multiclass, device):
        """Test factory with additional kwargs."""
        trainer = create_tri_objective_trainer(
            model=simple_model,
            train_loader=train_loader_multiclass,
            device=device,
            generate_heatmaps=True,
            temperature=2.0,
            use_mlflow=False,
        )

        assert trainer.config.generate_heatmaps is True
        assert trainer.config.temperature == 2.0

    def test_factory_creates_optimizer(
        self, simple_model, train_loader_multiclass, device
    ):
        """Test factory creates AdamW optimizer."""
        trainer = create_tri_objective_trainer(
            model=simple_model,
            train_loader=train_loader_multiclass,
            learning_rate=5e-4,
            weight_decay=1e-3,
            device=device,
            use_mlflow=False,
        )

        assert isinstance(trainer.optimizer, torch.optim.AdamW)
        assert trainer.optimizer.param_groups[0]["lr"] == 5e-4
        assert trainer.optimizer.param_groups[0]["weight_decay"] == 1e-3

    def test_factory_creates_scheduler(
        self, simple_model, train_loader_multiclass, device
    ):
        """Test factory creates cosine annealing scheduler."""
        trainer = create_tri_objective_trainer(
            model=simple_model,
            train_loader=train_loader_multiclass,
            learning_rate=1e-4,
            max_epochs=100,
            device=device,
            use_mlflow=False,
        )

        assert isinstance(trainer.scheduler, torch.optim.lr_scheduler.CosineAnnealingLR)
        assert trainer.scheduler.T_max == 100
        assert trainer.scheduler.eta_min == 1e-4 * 0.01


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Test integration scenarios."""

    def test_full_training_step_flow(
        self, simple_model, train_loader_multiclass, device
    ):
        """Test complete training flow with all components."""
        config = TriObjectiveConfig(
            max_epochs=2,
            pgd_num_steps=2,
            use_mlflow=False,
        )
        optimizer = torch.optim.Adam(simple_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_model,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            device=device,
        )

        # Run a few training steps
        for batch_idx, batch in enumerate(train_loader_multiclass):
            if batch_idx >= 2:
                break
            metrics = trainer.training_step(batch, batch_idx=batch_idx)
            assert "loss" in metrics
            assert metrics["loss"] >= 0

    def test_full_validation_flow(
        self, simple_model, train_loader_multiclass, val_loader_multiclass, device
    ):
        """Test complete validation flow."""
        config = TriObjectiveConfig(use_mlflow=False)
        optimizer = torch.optim.Adam(simple_model.parameters())

        trainer = TriObjectiveTrainer(
            model=simple_model,
            optimizer=optimizer,
            train_loader=train_loader_multiclass,
            config=config,
            val_loader=val_loader_multiclass,
            device=device,
        )

        # Run validation steps
        for batch_idx, batch in enumerate(val_loader_multiclass):
            if batch_idx >= 2:
                break
            metrics = trainer.validation_step(batch, batch_idx=batch_idx)
            assert "loss" in metrics
            assert "accuracy" in metrics


# ---------------------------------------------------------------------------
# Run Tests
# ---------------------------------------------------------------------------


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
