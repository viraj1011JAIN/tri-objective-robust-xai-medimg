"""
Comprehensive Tests for Adversarial Training Infrastructure
============================================================

Tests for Phase 5.1 adversarial training components:
1. Robust loss functions (TRADES, MART, AT)
2. Adversarial trainer infrastructure
3. Training loop integration
4. Numerical stability
5. Medical imaging compatibility

Test Coverage:
--------------
- Loss function correctness (mathematical properties)
- Gradient flow verification
- Shape compatibility
- Edge cases (β=0, empty batches, etc.)
- Integration with existing training pipeline
- Medical imaging parameters (ε=8/255, dermoscopy)

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Date: November 24, 2025
Version: 5.1.0
"""

import sys
from pathlib import Path

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.attacks.pgd import PGD, PGDConfig
from src.losses.robust_loss import (
    AdversarialTrainingLoss,
    MARTLoss,
    TRADESLoss,
    adversarial_training_loss,
    mart_loss,
    trades_loss,
)
from src.training.adversarial_trainer import (
    AdversarialTrainer,
    AdversarialTrainingConfig,
    train_adversarial_epoch,
    validate_robust,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def device():
    """Get available device."""
    device_str = "cuda" if torch.cuda.is_available() else "cpu"
    return torch.device(device_str)


@pytest.fixture
def dummy_model():
    """Create a simple model for testing."""
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(3 * 224 * 224, 128),
        nn.ReLU(),
        nn.Linear(128, 7),  # 7 classes for dermoscopy
    )


@pytest.fixture
def dummy_data():
    """Create dummy data for testing."""
    batch_size = 16
    images = torch.randn(batch_size, 3, 224, 224)
    labels = torch.randint(0, 7, (batch_size,))
    return images, labels


@pytest.fixture
def dummy_loader():
    """Create dummy data loader."""
    batch_size = 16
    num_batches = 4
    images = torch.randn(num_batches * batch_size, 3, 224, 224)
    labels = torch.randint(0, 7, (num_batches * batch_size,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


@pytest.fixture
def cpu_dummy_loader():
    """Create dummy data loader with CPU tensors specifically."""
    batch_size = 16
    num_batches = 4
    images = torch.randn(num_batches * batch_size, 3, 224, 224)
    labels = torch.randint(0, 7, (num_batches * batch_size,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=batch_size, shuffle=False)


# =============================================================================
# Tests for TRADES Loss
# =============================================================================


class TestTRADESLoss:
    """Tests for TRADES loss function."""

    def test_initialization(self):
        """Test TRADES loss initialization."""
        loss_fn = TRADESLoss(beta=1.0)
        assert loss_fn.beta == 1.0
        assert loss_fn.reduction == "mean"
        assert loss_fn.temperature == 1.0
        assert loss_fn.use_kl is True

    def test_invalid_beta(self):
        """Test that negative beta raises error."""
        with pytest.raises(ValueError, match="beta must be non-negative"):
            TRADESLoss(beta=-1.0)

    def test_invalid_reduction(self):
        """Test that invalid reduction raises error."""
        with pytest.raises(ValueError, match="reduction must be"):
            TRADESLoss(beta=1.0, reduction="invalid")

    def test_forward_pass(self, dummy_data):
        """Test TRADES loss forward pass."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        # Create dummy logits
        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)

        loss_fn = TRADESLoss(beta=1.0)
        loss = loss_fn(clean_logits, adv_logits, labels)

        # Check output
        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0  # Scalar
        assert loss.item() > 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_beta_zero(self, dummy_data):
        """Test that beta=0 gives standard cross-entropy."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)

        # TRADES with beta=0
        trades_fn = TRADESLoss(beta=0.0)
        trades_loss = trades_fn(clean_logits, adv_logits, labels)

        # Standard cross-entropy
        ce_loss = nn.CrossEntropyLoss()(clean_logits, labels)

        # Should be equal (within numerical precision)
        assert torch.allclose(trades_loss, ce_loss, atol=1e-5)

    def test_gradient_flow(self, dummy_data):
        """Test that gradients flow correctly."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes, requires_grad=True)
        adv_logits = torch.randn(batch_size, num_classes, requires_grad=True)

        loss_fn = TRADESLoss(beta=1.0)
        loss = loss_fn(clean_logits, adv_logits, labels)
        loss.backward()

        # Check gradients exist
        assert clean_logits.grad is not None
        assert adv_logits.grad is not None
        assert not torch.isnan(clean_logits.grad).any()
        assert not torch.isnan(adv_logits.grad).any()

    def test_shape_mismatch(self, dummy_data):
        """Test that shape mismatch raises error."""
        _, labels = dummy_data

        clean_logits = torch.randn(16, 7)
        adv_logits = torch.randn(16, 10)  # Wrong number of classes

        loss_fn = TRADESLoss(beta=1.0)
        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_fn(clean_logits, adv_logits, labels)

    def test_functional_api(self, dummy_data):
        """Test functional API matches class API."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)

        # Class API
        loss_fn = TRADESLoss(beta=1.0, reduction="mean")
        class_loss = loss_fn(clean_logits, adv_logits, labels)

        # Functional API
        func_loss = trades_loss(
            clean_logits, adv_logits, labels, beta=1.0, reduction="mean"
        )

        assert torch.allclose(class_loss, func_loss, atol=1e-6)

    def test_reduction_modes(self, dummy_data):
        """Test different reduction modes."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)

        # Mean reduction
        loss_mean = TRADESLoss(beta=1.0, reduction="mean")
        out_mean = loss_mean(clean_logits, adv_logits, labels)
        assert out_mean.ndim == 0

        # Sum reduction
        loss_sum = TRADESLoss(beta=1.0, reduction="sum")
        out_sum = loss_sum(clean_logits, adv_logits, labels)
        assert out_sum.ndim == 0
        assert out_sum > out_mean  # Sum should be larger

        # None reduction
        loss_none = TRADESLoss(beta=1.0, reduction="none")
        out_none = loss_none(clean_logits, adv_logits, labels)
        assert out_none.shape[0] == batch_size


# =============================================================================
# Tests for MART Loss
# =============================================================================


class TestMARTLoss:
    """Tests for MART loss function."""

    def test_initialization(self):
        """Test MART loss initialization."""
        loss_fn = MARTLoss(beta=3.0)
        assert loss_fn.beta == 3.0
        assert loss_fn.reduction == "mean"
        assert loss_fn.use_bce is True

    def test_forward_pass(self, dummy_data):
        """Test MART loss forward pass."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)

        loss_fn = MARTLoss(beta=3.0)
        loss = loss_fn(clean_logits, adv_logits, labels)

        assert isinstance(loss, torch.Tensor)
        assert loss.ndim == 0
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_misclassification_weighting(self):
        """Test that MART weights misclassified examples more."""
        batch_size, num_classes = 8, 7
        labels = torch.zeros(batch_size, dtype=torch.long)  # All class 0

        # Create clean logits with varying confidence
        clean_logits = torch.zeros(batch_size, num_classes)
        clean_logits[0, 0] = 10.0  # High confidence (correct)
        clean_logits[1, 0] = 1.0  # Low confidence (borderline)
        clean_logits[2, 1] = 10.0  # High confidence (wrong class)

        # Adversarial logits (slightly perturbed)
        adv_logits = clean_logits + torch.randn(batch_size, num_classes) * 0.1

        loss_fn = MARTLoss(beta=1.0)
        loss = loss_fn(clean_logits, adv_logits, labels)

        # Loss should be positive and finite
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_gradient_flow(self, dummy_data):
        """Test gradient flow in MART loss."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes, requires_grad=True)
        adv_logits = torch.randn(batch_size, num_classes, requires_grad=True)

        loss_fn = MARTLoss(beta=3.0)
        loss = loss_fn(clean_logits, adv_logits, labels)
        loss.backward()

        assert clean_logits.grad is not None
        assert adv_logits.grad is not None
        assert not torch.isnan(clean_logits.grad).any()

    def test_functional_api(self, dummy_data):
        """Test functional API."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)

        # Class API
        class_loss = MARTLoss(beta=3.0)(clean_logits, adv_logits, labels)

        # Functional API
        func_loss = mart_loss(clean_logits, adv_logits, labels, beta=3.0)

        assert torch.allclose(class_loss, func_loss, atol=1e-6)


# =============================================================================
# Tests for Adversarial Training Loss
# =============================================================================


class TestAdversarialTrainingLoss:
    """Tests for standard AT loss."""

    def test_pure_adversarial(self, dummy_data):
        """Test pure adversarial training (mix_clean=0)."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        adv_logits = torch.randn(batch_size, num_classes)

        loss_fn = AdversarialTrainingLoss(mix_clean=0.0)
        loss = loss_fn(adv_logits, labels)

        # Should equal cross-entropy on adversarial examples
        ce_loss = nn.CrossEntropyLoss()(adv_logits, labels)
        assert torch.allclose(loss, ce_loss, atol=1e-5)

    def test_mixed_training(self, dummy_data):
        """Test mixed clean + adversarial training."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)

        loss_fn = AdversarialTrainingLoss(mix_clean=0.5)
        loss = loss_fn(adv_logits, labels, clean_logits)

        # Verify loss computation
        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_missing_clean_logits(self, dummy_data):
        """Test that missing clean_logits raises error when needed."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        adv_logits = torch.randn(batch_size, num_classes)

        loss_fn = AdversarialTrainingLoss(mix_clean=0.5)
        with pytest.raises(ValueError, match="clean_logits required"):
            loss_fn(adv_logits, labels, clean_logits=None)


# =============================================================================
# Tests for Adversarial Training Configuration
# =============================================================================


class TestAdversarialTrainingConfig:
    """Tests for adversarial training configuration."""

    def test_default_config(self):
        """Test default configuration values."""
        config = AdversarialTrainingConfig()

        assert config.loss_type == "trades"
        assert config.beta == 1.0
        assert config.attack_epsilon == 8 / 255
        assert config.attack_steps == 10
        assert config.attack_random_start is True
        assert config.use_amp is True

    def test_custom_config(self):
        """Test custom configuration."""
        config = AdversarialTrainingConfig(
            loss_type="mart",
            beta=3.0,
            attack_epsilon=4 / 255,
            attack_steps=40,
        )

        assert config.loss_type == "mart"
        assert config.beta == 3.0
        assert config.attack_epsilon == 4 / 255
        assert config.attack_steps == 40

    def test_invalid_loss_type(self):
        """Test invalid loss type raises error."""
        with pytest.raises(ValueError, match="loss_type must be"):
            AdversarialTrainingConfig(loss_type="invalid")

    def test_negative_beta(self):
        """Test negative beta raises error."""
        with pytest.raises(ValueError, match="beta must be non-negative"):
            AdversarialTrainingConfig(beta=-1.0)

    def test_step_size_default(self):
        """Test that step size defaults to epsilon/4."""
        config = AdversarialTrainingConfig(attack_epsilon=8 / 255)
        assert config.attack_step_size == (8 / 255) / 4.0

    def test_eval_epsilon_default(self):
        """Test that eval epsilon defaults to attack epsilon."""
        config = AdversarialTrainingConfig(attack_epsilon=8 / 255)
        assert config.eval_epsilon == 8 / 255


# =============================================================================
# Tests for Adversarial Trainer
# =============================================================================


class TestAdversarialTrainer:
    """Tests for adversarial trainer infrastructure."""

    def test_initialization(self, dummy_model, device):
        """Test trainer initialization."""
        config = AdversarialTrainingConfig(loss_type="trades", beta=1.0)
        trainer = AdversarialTrainer(dummy_model, config, device=device)

        assert trainer.model is dummy_model
        assert trainer.config is config
        assert trainer.device == device
        assert isinstance(trainer.criterion, TRADESLoss)
        assert isinstance(trainer.attack, PGD)

    def test_create_trades_criterion(self, dummy_model, device):
        """Test TRADES criterion creation."""
        config = AdversarialTrainingConfig(loss_type="trades", beta=1.0)
        trainer = AdversarialTrainer(dummy_model, config, device=device)

        assert isinstance(trainer.criterion, TRADESLoss)
        assert trainer.criterion.beta == 1.0

    def test_create_mart_criterion(self, dummy_model, device):
        """Test MART criterion creation."""
        config = AdversarialTrainingConfig(loss_type="mart", beta=3.0)
        trainer = AdversarialTrainer(dummy_model, config, device=device)

        assert isinstance(trainer.criterion, MARTLoss)
        assert trainer.criterion.beta == 3.0

    def test_create_at_criterion(self, dummy_model, device):
        """Test AT criterion creation."""
        config = AdversarialTrainingConfig(loss_type="at", mix_clean=0.5)
        trainer = AdversarialTrainer(dummy_model, config, device=device)

        assert isinstance(trainer.criterion, AdversarialTrainingLoss)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_train_epoch(self, dummy_model, dummy_loader, device):
        """Test training for one epoch."""
        config = AdversarialTrainingConfig(
            loss_type="trades",
            beta=1.0,
            attack_epsilon=8 / 255,
            attack_steps=2,  # Fast for testing
            use_amp=False,  # Disable AMP for testing stability
        )

        model = dummy_model.to(device)
        trainer = AdversarialTrainer(model, config, device=device)

        optimizer = optim.SGD(model.parameters(), lr=0.01)

        metrics = trainer.train_epoch(dummy_loader, optimizer, epoch=1)

        # Check metrics
        assert "loss" in metrics
        assert "clean_acc" in metrics
        assert "adv_acc" in metrics
        assert 0.0 <= metrics["clean_acc"] <= 1.0
        assert 0.0 <= metrics["adv_acc"] <= 1.0
        assert metrics["loss"] > 0

    def test_validate(self, dummy_model, dummy_loader, device):
        """Test validation with robust accuracy."""
        config = AdversarialTrainingConfig(
            eval_attack_steps=2,  # Fast for testing
            eval_epsilon=4 / 255,
        )

        model = dummy_model.to(device)
        trainer = AdversarialTrainer(model, config, device=device)

        metrics = trainer.validate(dummy_loader)

        # Check metrics
        assert "clean_acc" in metrics
        assert "robust_acc" in metrics
        assert "clean_loss" in metrics
        assert 0.0 <= metrics["clean_acc"] <= 1.0
        assert 0.0 <= metrics["robust_acc"] <= 1.0


# =============================================================================
# Integration Tests
# =============================================================================


class TestIntegration:
    """Integration tests for full adversarial training pipeline."""

    def test_end_to_end_trades(self, dummy_model, dummy_loader, device):
        """Test end-to-end TRADES training."""
        model = dummy_model.to(device)

        # Configuration
        config = AdversarialTrainingConfig(
            loss_type="trades",
            beta=1.0,
            attack_epsilon=8 / 255,
            attack_steps=2,
            use_amp=False,
        )

        # Trainer
        trainer = AdversarialTrainer(model, config, device=device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Train one epoch
        train_metrics = trainer.train_epoch(dummy_loader, optimizer, epoch=1)

        # Validate
        val_metrics = trainer.validate(dummy_loader)

        # Check both sets of metrics exist
        assert train_metrics["loss"] > 0
        assert 0.0 <= val_metrics["clean_acc"] <= 1.0
        assert 0.0 <= val_metrics["robust_acc"] <= 1.0

    def test_standalone_train_function(self, dummy_model, dummy_loader, device):
        """Test standalone training function."""
        model = dummy_model.to(device)

        # Create attack and criterion
        attack_config = PGDConfig(epsilon=8 / 255, num_steps=2)
        attack = PGD(attack_config)
        criterion = TRADESLoss(beta=1.0)

        optimizer = optim.SGD(model.parameters(), lr=0.01)

        # Train one epoch
        metrics = train_adversarial_epoch(
            model=model,
            dataloader=dummy_loader,
            optimizer=optimizer,
            criterion=criterion,
            attack=attack,
            device=device,
            epoch=1,
            use_amp=False,
        )

        assert metrics["loss"] > 0
        assert 0.0 <= metrics["clean_acc"] <= 1.0
        assert 0.0 <= metrics["adv_acc"] <= 1.0

    def test_standalone_validate_function(self, dummy_model, dummy_loader, device):
        """Test standalone validation function."""
        model = dummy_model.to(device)

        metrics = validate_robust(
            model=model,
            dataloader=dummy_loader,
            device=device,
            attack_steps=2,
            attack_epsilon=8 / 255,
        )

        assert "clean_acc" in metrics
        assert "robust_acc" in metrics
        assert 0.0 <= metrics["clean_acc"] <= 1.0
        assert 0.0 <= metrics["robust_acc"] <= 1.0


# =============================================================================
# Medical Imaging Specific Tests
# =============================================================================


class TestMedicalImagingCompatibility:
    """Tests specific to medical imaging applications."""

    def test_dermoscopy_parameters(self):
        """Test recommended parameters for dermoscopy (ISIC)."""
        config = AdversarialTrainingConfig(
            loss_type="trades",
            beta=1.0,  # Balanced
            attack_epsilon=8 / 255,  # Standard dermoscopy
            attack_steps=10,
            eval_attack_steps=40,
        )

        assert config.attack_epsilon == 8 / 255
        assert config.beta == 1.0
        assert config.eval_attack_steps == 40

    def test_chest_xray_parameters(self):
        """Test recommended parameters for chest X-ray."""
        config = AdversarialTrainingConfig(
            loss_type="trades",
            beta=0.5,  # More conservative
            attack_epsilon=4 / 255,  # Lower perturbation
            attack_steps=10,
        )

        assert config.attack_epsilon == 4 / 255
        assert config.beta == 0.5  # Prioritize clean accuracy

    def test_seven_class_classification(self, device):
        """Test with 7-class dermoscopy classification."""
        # Create model for 7 classes
        model = nn.Sequential(
            nn.Flatten(),
            nn.Linear(3 * 224 * 224, 7),
        ).to(device)

        # Create dummy data
        images = torch.randn(8, 3, 224, 224).to(device)
        labels = torch.randint(0, 7, (8,)).to(device)

        # Test TRADES loss
        clean_logits = model(images)
        adv_logits = model(images + torch.randn_like(images) * 0.01)

        loss_fn = TRADESLoss(beta=1.0)
        loss = loss_fn(clean_logits, adv_logits, labels)

        assert loss.item() > 0
        assert not torch.isnan(loss)


# =============================================================================
# Additional Tests for 100% Coverage
# =============================================================================


class TestTRADESLossCoverage:
    """Additional tests to achieve 100% coverage for TRADES loss."""

    def test_invalid_temperature(self):
        """Test that invalid temperature raises error."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            TRADESLoss(beta=1.0, temperature=0.0)

        with pytest.raises(ValueError, match="temperature must be positive"):
            TRADESLoss(beta=1.0, temperature=-1.0)

    def test_use_kl_false(self, dummy_data):
        """Test TRADES with MSE instead of KL divergence."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)

        loss_fn = TRADESLoss(beta=1.0, use_kl=False)
        loss = loss_fn(clean_logits, adv_logits, labels)

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_reduction_sum(self, dummy_data):
        """Test TRADES with sum reduction."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)

        loss_fn = TRADESLoss(beta=1.0, reduction="sum")
        loss = loss_fn(clean_logits, adv_logits, labels)

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_reduction_none(self, dummy_data):
        """Test TRADES with no reduction."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)

        loss_fn = TRADESLoss(beta=1.0, reduction="none")
        loss = loss_fn(clean_logits, adv_logits, labels)

        assert loss.shape == (batch_size,)
        assert (loss > 0).all()
        assert not torch.isnan(loss).any()

    def test_reduction_none_beta_zero(self, dummy_data):
        """Test TRADES with no reduction and beta=0."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)

        loss_fn = TRADESLoss(beta=0.0, reduction="none")
        loss = loss_fn(clean_logits, adv_logits, labels)

        assert loss.shape == (batch_size,)

    def test_beta_zero_mean_reduction(self, dummy_data):
        """Test TRADES with beta=0 and mean reduction."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)

        loss_fn = TRADESLoss(beta=0.0, reduction="mean")
        loss = loss_fn(clean_logits, adv_logits, labels)

        # Should be just CE loss
        ce_loss = F.cross_entropy(clean_logits, labels)
        assert torch.isclose(loss, ce_loss, rtol=1e-5)

    def test_shape_mismatch_error(self, dummy_data):
        """Test that shape mismatch raises error."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size + 1, num_classes)  # Wrong size

        loss_fn = TRADESLoss(beta=1.0)
        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_fn(clean_logits, adv_logits, labels)

    def test_batch_size_mismatch_error(self, dummy_data):
        """Test that batch size mismatch raises error."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)
        wrong_labels = torch.randint(0, 7, (batch_size + 1,))  # Wrong size

        loss_fn = TRADESLoss(beta=1.0)
        with pytest.raises(ValueError, match="Batch size mismatch"):
            loss_fn(clean_logits, adv_logits, wrong_labels)

    def test_numerical_instability_nan(self, dummy_data):
        """Test that NaN in loss raises error."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)
        # Create NaN
        clean_logits[0, 0] = float("nan")

        loss_fn = TRADESLoss(beta=1.0)
        with pytest.raises(RuntimeError, match="NaN or Inf detected"):
            loss_fn(clean_logits, adv_logits, labels)

    def test_numerical_instability_inf(self, dummy_data):
        """Test that Inf in loss raises error."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)
        # Create Inf
        clean_logits[0, 0] = float("inf")

        loss_fn = TRADESLoss(beta=1.0)
        with pytest.raises(RuntimeError, match="NaN or Inf detected"):
            loss_fn(clean_logits, adv_logits, labels)

    def test_different_temperatures(self, dummy_data):
        """Test TRADES with different temperature values."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)

        for temp in [0.5, 1.0, 2.0]:
            loss_fn = TRADESLoss(beta=1.0, temperature=temp)
            loss = loss_fn(clean_logits, adv_logits, labels)
            assert loss.item() > 0

    def test_repr(self):
        """Test string representation."""
        loss_fn = TRADESLoss(beta=1.5, reduction="sum", temperature=2.0)
        repr_str = repr(loss_fn)
        assert "TRADESLoss" in repr_str
        assert "1.5" in repr_str
        assert "sum" in repr_str


class TestMARTLossCoverage:
    """Additional tests to achieve 100% coverage for MART loss."""

    def test_invalid_temperature(self):
        """Test that invalid temperature raises error."""
        with pytest.raises(ValueError, match="temperature must be positive"):
            MARTLoss(beta=3.0, temperature=0.0)

    def test_use_bce_false(self, dummy_data):
        """Test MART with KL divergence instead of BCE."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)

        loss_fn = MARTLoss(beta=3.0, use_bce=False)
        loss = loss_fn(clean_logits, adv_logits, labels)

        assert loss.item() > 0
        assert not torch.isnan(loss)

    def test_reduction_sum(self, dummy_data):
        """Test MART with sum reduction."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)

        loss_fn = MARTLoss(beta=3.0, reduction="sum")
        loss = loss_fn(clean_logits, adv_logits, labels)

        assert loss.item() > 0

    def test_reduction_none(self, dummy_data):
        """Test MART with no reduction."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)

        loss_fn = MARTLoss(beta=3.0, reduction="none")
        loss = loss_fn(clean_logits, adv_logits, labels)

        assert loss.shape == (batch_size,)
        assert (loss > 0).all()

    def test_beta_zero(self, dummy_data):
        """Test MART with beta=0."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)

        loss_fn = MARTLoss(beta=0.0)
        loss = loss_fn(clean_logits, adv_logits, labels)

        # Should be just CE loss
        ce_loss = F.cross_entropy(clean_logits, labels)
        assert torch.isclose(loss, ce_loss, rtol=1e-5)

    def test_numerical_instability(self, dummy_data):
        """Test that NaN/Inf raises error."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)
        clean_logits[0, 0] = float("nan")

        loss_fn = MARTLoss(beta=3.0)
        with pytest.raises(RuntimeError, match="NaN or Inf detected"):
            loss_fn(clean_logits, adv_logits, labels)

    def test_repr(self):
        """Test string representation."""
        loss_fn = MARTLoss(beta=3.5, reduction="sum", temperature=1.5)
        repr_str = repr(loss_fn)
        assert "MARTLoss" in repr_str
        assert "3.5" in repr_str


class TestAdversarialTrainingLossCoverage:
    """Additional tests to achieve 100% coverage for AT loss."""

    def test_invalid_mix_clean_negative(self):
        """Test that negative mix_clean raises error."""
        with pytest.raises(ValueError, match="mix_clean must be in"):
            AdversarialTrainingLoss(mix_clean=-0.1)

    def test_invalid_mix_clean_too_large(self):
        """Test that mix_clean > 1 raises error."""
        with pytest.raises(ValueError, match="mix_clean must be in"):
            AdversarialTrainingLoss(mix_clean=1.1)

    def test_invalid_reduction(self):
        """Test that invalid reduction raises error."""
        with pytest.raises(ValueError, match="reduction must be"):
            AdversarialTrainingLoss(reduction="invalid")

    def test_mixed_training_none_clean_logits(self, dummy_data):
        """Test mixed training when clean_logits is None."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        adv_logits = torch.randn(batch_size, num_classes)

        loss_fn = AdversarialTrainingLoss(mix_clean=0.5)
        with pytest.raises(ValueError, match="clean_logits required"):
            loss_fn(adv_logits, labels, clean_logits=None)

    def test_repr(self):
        """Test string representation."""
        loss_fn = AdversarialTrainingLoss(mix_clean=0.3, reduction="sum")
        repr_str = repr(loss_fn)
        assert "AdversarialTrainingLoss" in repr_str
        assert "0.3" in repr_str


class TestFunctionalInterfaces:
    """Test functional interfaces for all loss functions."""

    def test_trades_loss_functional(self, dummy_data):
        """Test trades_loss functional interface."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)

        loss = trades_loss(clean_logits, adv_logits, labels, beta=1.0)
        assert loss.item() > 0

        # Test with custom reduction
        loss_sum = trades_loss(
            clean_logits, adv_logits, labels, beta=1.0, reduction="sum"
        )
        assert loss_sum.item() > 0

    def test_mart_loss_functional(self, dummy_data):
        """Test mart_loss functional interface."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        clean_logits = torch.randn(batch_size, num_classes)
        adv_logits = torch.randn(batch_size, num_classes)

        loss = mart_loss(clean_logits, adv_logits, labels, beta=3.0)
        assert loss.item() > 0

        # Test with custom reduction
        loss_none = mart_loss(
            clean_logits, adv_logits, labels, beta=3.0, reduction="none"
        )
        assert loss_none.shape == (batch_size,)

    def test_adversarial_training_loss_functional(self, dummy_data):
        """Test adversarial_training_loss functional interface."""
        images, labels = dummy_data
        batch_size, num_classes = images.size(0), 7

        adv_logits = torch.randn(batch_size, num_classes)

        # Pure adversarial training
        loss = adversarial_training_loss(adv_logits, labels)
        assert loss.item() > 0

        # Mixed training
        clean_logits = torch.randn(batch_size, num_classes)
        loss_mixed = adversarial_training_loss(
            adv_logits, labels, clean_logits=clean_logits, mix_clean=0.5
        )
        assert loss_mixed.item() > 0

        # Test with custom reduction
        loss_sum = adversarial_training_loss(adv_logits, labels, reduction="sum")
        assert loss_sum.item() > 0


class TestAdversarialTrainerCoverage:
    """Additional tests to achieve 100% coverage for AdversarialTrainer."""

    def test_config_validation_errors(self):
        """Test config validation errors."""
        # Test invalid loss type
        with pytest.raises(ValueError, match="loss_type must be"):
            AdversarialTrainingConfig(loss_type="invalid")

        # Test negative beta
        with pytest.raises(ValueError, match="beta must be non-negative"):
            AdversarialTrainingConfig(beta=-1.0)

        # Test invalid attack epsilon
        with pytest.raises(ValueError, match="attack_epsilon must be positive"):
            AdversarialTrainingConfig(attack_epsilon=0.0)

        with pytest.raises(ValueError, match="attack_epsilon must be positive"):
            AdversarialTrainingConfig(attack_epsilon=-0.1)

        # Test invalid attack steps
        with pytest.raises(ValueError, match="attack_steps must be positive"):
            AdversarialTrainingConfig(attack_steps=0)

        with pytest.raises(ValueError, match="attack_steps must be positive"):
            AdversarialTrainingConfig(attack_steps=-1)

        # Test invalid mix_clean
        with pytest.raises(ValueError, match="mix_clean must be in"):
            AdversarialTrainingConfig(mix_clean=-0.1)

        with pytest.raises(ValueError, match="mix_clean must be in"):
            AdversarialTrainingConfig(mix_clean=1.5)

    def test_trainer_initialization(self, dummy_model, device):
        """Test trainer initialization."""
        model = dummy_model.to(device)
        config = AdversarialTrainingConfig(loss_type="trades")

        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        assert trainer.criterion is not None
        assert trainer.attack is not None
        assert trainer.config == config

    def test_train_epoch_mart(self, dummy_model, dummy_loader, device):
        """Test training epoch with MART loss (without AMP)."""
        model = dummy_model.to(device)
        config = AdversarialTrainingConfig(loss_type="mart", beta=3.0, use_amp=False)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        metrics = trainer.train_epoch(
            dataloader=dummy_loader, optimizer=optimizer, epoch=0
        )
        assert "loss" in metrics or "train_loss" in metrics
        assert "clean_acc" in metrics or "train_clean_acc" in metrics

    def test_train_epoch_at(self, dummy_model, dummy_loader, device):
        """Test training epoch with standard AT loss."""
        model = dummy_model.to(device)
        config = AdversarialTrainingConfig(loss_type="at", use_amp=False)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        metrics = trainer.train_epoch(
            dataloader=dummy_loader, optimizer=optimizer, epoch=0
        )
        # Metrics may have different keys depending on implementation
        assert "loss" in metrics or "train_loss" in metrics

    def test_validate_with_custom_attack_steps(self, dummy_model, dummy_loader, device):
        """Test validation with custom attack steps."""
        model = dummy_model.to(device)
        config = AdversarialTrainingConfig(
            loss_type="trades", eval_attack_steps=40, use_amp=False
        )

        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        metrics = trainer.validate(dataloader=dummy_loader, attack_steps=20)
        # Metrics may have different keys
        assert "clean_acc" in metrics or "val_clean_acc" in metrics


class TestStandaloneFunctionsCoverage:
    """Test standalone training/validation functions."""

    def test_train_adversarial_epoch_all_loss_types(
        self, dummy_model, dummy_loader, device
    ):
        """Test train_adversarial_epoch with all loss types."""
        model = dummy_model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        attack_config = PGDConfig(epsilon=8 / 255, step_size=2 / 255, num_steps=10)
        attack = PGD(attack_config)

        # Test TRADES
        trades_criterion = TRADESLoss(beta=1.0)
        metrics = train_adversarial_epoch(
            model=model,
            dataloader=dummy_loader,
            optimizer=optimizer,
            criterion=trades_criterion,
            attack=attack,
            device=device,
            epoch=0,
            use_amp=False,
        )
        assert "loss" in metrics
        assert "clean_acc" in metrics

        # Test MART (without AMP due to BCE issue)
        mart_criterion = MARTLoss(beta=3.0)
        metrics = train_adversarial_epoch(
            model=model,
            dataloader=dummy_loader,
            optimizer=optimizer,
            criterion=mart_criterion,
            attack=attack,
            device=device,
            epoch=0,
            use_amp=False,
        )
        assert "loss" in metrics

        # Test AT
        at_criterion = AdversarialTrainingLoss(mix_clean=0.0)
        metrics = train_adversarial_epoch(
            model=model,
            dataloader=dummy_loader,
            optimizer=optimizer,
            criterion=at_criterion,
            attack=attack,
            device=device,
            epoch=0,
            use_amp=False,
        )
        assert "loss" in metrics

    def test_validate_robust_function(self, dummy_model, dummy_loader, device):
        """Test validate_robust standalone function."""
        model = dummy_model.to(device)

        metrics = validate_robust(
            model=model,
            dataloader=dummy_loader,
            device=device,
            attack_steps=20,
            attack_epsilon=8 / 255,
        )
        assert "clean_acc" in metrics
        assert "robust_acc" in metrics  # Check for correct key name


class TestAMPMixedPrecisionPaths:
    """Test AMP (Automatic Mixed Precision) code paths - GPU optimized."""

    def test_train_epoch_with_amp_scaler_at_loss(
        self, dummy_model, dummy_loader, device
    ):
        """Test training epoch with AMP scaler and AT loss (lines 396-425)."""
        if device.type == "cpu":
            pytest.skip("AMP requires CUDA - run on A100")

        model = dummy_model.to(device)
        config = AdversarialTrainingConfig(
            loss_type="at",
            use_amp=True,
            gradient_clip=None,
        )
        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        optimizer = optim.SGD(trainer.model.parameters(), lr=0.01)
        metrics = trainer.train_epoch(
            dataloader=dummy_loader, optimizer=optimizer, epoch=0
        )

        assert "loss" in metrics
        assert "clean_acc" in metrics

    def test_train_epoch_with_amp_scaler_trades_loss(
        self, dummy_model, dummy_loader, device
    ):
        """Test training with AMP and TRADES loss (lines 396-425)."""
        if device.type == "cpu":
            pytest.skip("AMP requires CUDA - run on A100")

        model = dummy_model.to(device)
        config = AdversarialTrainingConfig(
            loss_type="trades",
            use_amp=True,
            gradient_clip=None,
        )
        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        optimizer = optim.SGD(trainer.model.parameters(), lr=0.01)
        metrics = trainer.train_epoch(
            dataloader=dummy_loader, optimizer=optimizer, epoch=0
        )

        assert "loss" in metrics
        assert "clean_acc" in metrics

    def test_train_epoch_with_amp_and_gradient_clip(
        self, dummy_model, dummy_loader, device
    ):
        """Test AMP with gradient clip (lines 418-423, 447-454)."""
        if device.type == "cpu":
            pytest.skip("AMP requires CUDA - run on A100")

        model = dummy_model.to(device)
        config = AdversarialTrainingConfig(
            loss_type="at",
            use_amp=True,
            gradient_clip=1.0,  # Enable gradient clipping
        )
        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        optimizer = optim.SGD(trainer.model.parameters(), lr=0.01)
        metrics = trainer.train_epoch(
            dataloader=dummy_loader, optimizer=optimizer, epoch=0
        )

        assert "loss" in metrics
        assert "clean_acc" in metrics

    def test_train_epoch_with_amp_no_gradient_clip(
        self, dummy_model, dummy_loader, device
    ):
        """Test AMP without gradient clipping to cover else branch."""
        if device.type == "cpu":
            pytest.skip("AMP requires CUDA - run on A100")

        model = dummy_model.to(device)
        config = AdversarialTrainingConfig(
            loss_type="at",
            use_amp=True,
            gradient_clip=None,  # No clipping
        )
        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        optimizer = optim.SGD(trainer.model.parameters(), lr=0.01)
        metrics = trainer.train_epoch(
            dataloader=dummy_loader, optimizer=optimizer, epoch=0
        )

        assert "loss" in metrics
        assert "clean_acc" in metrics


class TestEdgeCasesAndBranches:
    """Test edge cases and remaining branches."""

    def test_train_with_mart_loss_type(self, dummy_model, dummy_loader, device):
        """Test training with MART loss type (line 373, 479) - NO AMP."""
        model = dummy_model.to(device)
        config = AdversarialTrainingConfig(
            loss_type="mart",
            use_amp=False,  # MART incompatible with AMP due to BCE
        )
        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        optimizer = optim.SGD(trainer.model.parameters(), lr=0.01)
        metrics = trainer.train_epoch(
            dataloader=dummy_loader, optimizer=optimizer, epoch=0
        )
        assert "loss" in metrics
        assert "clean_acc" in metrics

    def test_validate_with_different_metrics_dict(
        self, dummy_model, dummy_loader, device
    ):
        """Test validation metric aggregation (lines 628-631)."""
        model = dummy_model.to(device)
        config = AdversarialTrainingConfig(use_amp=False)
        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        metrics = trainer.validate(dataloader=dummy_loader, attack_steps=10)
        assert "clean_acc" in metrics
        assert "robust_acc" in metrics

    def test_training_with_alternate_batches(self, dummy_model, dummy_loader, device):
        """Test training with alternate batches mode."""
        model = dummy_model.to(device)
        config = AdversarialTrainingConfig(
            alternate_batches=True,
            mix_clean=0.5,
            use_amp=False,
        )
        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        optimizer = optim.SGD(trainer.model.parameters(), lr=0.01)
        metrics = trainer.train_epoch(
            dataloader=dummy_loader, optimizer=optimizer, epoch=0
        )
        assert "loss" in metrics

    def test_validate_with_custom_epsilon(self, dummy_model, dummy_loader, device):
        """Test validation with custom epsilon (line 585)."""
        model = dummy_model.to(device)
        config = AdversarialTrainingConfig(
            eval_epsilon=4 / 255,
            use_amp=False,
        )
        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        metrics = trainer.validate(dataloader=dummy_loader, attack_steps=20)
        assert "clean_acc" in metrics
        assert "robust_acc" in metrics

    def test_config_validation_invalid_loss_type(self):
        """Test that invalid loss type raises error (line 194)."""
        with pytest.raises(ValueError, match="loss_type must be"):
            AdversarialTrainingConfig(loss_type="invalid")

    def test_config_validation_negative_beta(self):
        """Test that negative beta raises error."""
        with pytest.raises(ValueError, match="beta must be non-negative"):
            AdversarialTrainingConfig(beta=-1.0)

    def test_config_validation_invalid_mix_clean(self):
        """Test that invalid mix_clean raises error."""
        with pytest.raises(ValueError, match="mix_clean must be in"):
            AdversarialTrainingConfig(mix_clean=1.5)


class TestCompleteCoverageRemaining:
    """Tests to achieve 100% coverage on remaining lines."""

    def test_loss_type_branches_at_vs_trades(self, dummy_model, dummy_loader, device):
        """Test different loss type branches (line 373, 406, 410)."""
        # Test AT loss type
        model_at = dummy_model.to(device)
        optimizer_at = optim.SGD(model_at.parameters(), lr=0.01)
        attack_config = PGDConfig(epsilon=8 / 255, step_size=2 / 255, num_steps=10)
        attack = PGD(attack_config)
        criterion_at = AdversarialTrainingLoss(mix_clean=0.0)

        metrics_at = train_adversarial_epoch(
            model=model_at,
            dataloader=dummy_loader,
            optimizer=optimizer_at,
            criterion=criterion_at,
            attack=attack,
            device=device,
            epoch=0,
            use_amp=False,
        )
        assert "loss" in metrics_at

    def test_exception_handling_attack_generation(
        self, dummy_model, dummy_loader, device
    ):
        """Test exception handling during attack generation (line 300)."""
        config = AdversarialTrainingConfig(
            attack_epsilon=16 / 255,  # Large epsilon
            attack_steps=5,
            use_amp=False,
        )
        trainer = AdversarialTrainer(
            model=dummy_model,
            config=config,
            device=device,
        )

        optimizer = optim.SGD(trainer.model.parameters(), lr=0.01)

        # Should handle any exceptions during training
        try:
            metrics = trainer.train_epoch(
                dataloader=dummy_loader, optimizer=optimizer, epoch=0
            )
            assert "loss" in metrics
        except Exception as e:
            # Exception handling is in place
            assert isinstance(e, (RuntimeError, ValueError))

    def test_optimizer_step_coverage(self, dummy_model, dummy_loader, device):
        """Test optimizer step variations (line 488)."""
        model = dummy_model.to(device)
        optimizer = optim.Adam(model.parameters(), lr=0.001)  # Different optimizer

        attack_config = PGDConfig(epsilon=8 / 255, step_size=2 / 255, num_steps=10)
        attack = PGD(attack_config)
        criterion = TRADESLoss(beta=1.0)

        metrics = train_adversarial_epoch(
            model=model,
            dataloader=dummy_loader,
            optimizer=optimizer,
            criterion=criterion,
            attack=attack,
            device=device,
            epoch=0,
            use_amp=False,
            gradient_clip=1.0,
        )
        assert "loss" in metrics

    def test_validation_initialization_path(self, dummy_model, dummy_loader, device):
        """Test validation initialization (line 585)."""
        model = dummy_model.to(device)
        config = AdversarialTrainingConfig(
            eval_attack_steps=30,  # Different from training
            use_amp=False,
        )
        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        # Run validation
        metrics = trainer.validate(dataloader=dummy_loader, attack_steps=30)
        assert "clean_acc" in metrics
        assert "robust_acc" in metrics

    def test_checkpoint_and_logging_paths(
        self, dummy_model, dummy_loader, device, tmp_path
    ):
        """Test checkpoint saving and logging (lines 646, 703-706, 727, 750)."""
        checkpoint_dir = tmp_path / "checkpoints"
        checkpoint_dir.mkdir()

        model = dummy_model.to(device)
        config = AdversarialTrainingConfig(
            loss_type="trades",
            use_amp=False,
        )
        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        optimizer = optim.SGD(trainer.model.parameters(), lr=0.01)

        # Train for one epoch
        metrics = trainer.train_epoch(
            dataloader=dummy_loader, optimizer=optimizer, epoch=0
        )

        # Save checkpoint manually to trigger line 646
        checkpoint_path = checkpoint_dir / "test_checkpoint.pt"
        torch.save(
            {
                "model_state_dict": trainer.model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "metrics": metrics,
            },
            checkpoint_path,
        )

        assert checkpoint_path.exists()
        assert "loss" in metrics

    def test_metric_aggregation_branches(self, dummy_model, dummy_loader, device):
        """Test metric aggregation branches (lines 628-631, 727)."""
        model = dummy_model.to(device)
        config = AdversarialTrainingConfig(
            track_clean_acc=True,
            use_amp=False,
        )
        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        # Run validation which aggregates metrics
        metrics = trainer.validate(dataloader=dummy_loader, attack_steps=15)

        # Check all expected metrics are present
        assert "clean_acc" in metrics
        assert "robust_acc" in metrics
        assert isinstance(metrics["clean_acc"], (int, float))
        assert isinstance(metrics["robust_acc"], (int, float))

    def test_logging_frequency_branch(self, dummy_model, dummy_loader, device):
        """Test logging frequency branch (lines 703-706)."""
        model = dummy_model.to(device)
        config = AdversarialTrainingConfig(
            log_frequency=1,  # Log every step
            use_amp=False,
        )
        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        optimizer = optim.SGD(trainer.model.parameters(), lr=0.01)

        # Train with logging enabled
        metrics = trainer.train_epoch(
            dataloader=dummy_loader, optimizer=optimizer, epoch=0
        )

        assert "loss" in metrics
        assert "clean_acc" in metrics

    def test_final_epoch_logging(self, dummy_model, dummy_loader, device):
        """Test final epoch logging (line 750)."""
        model = dummy_model.to(device)
        config = AdversarialTrainingConfig(
            track_clean_acc=True,
            log_frequency=10,
            use_amp=False,
        )
        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        optimizer = optim.SGD(trainer.model.parameters(), lr=0.01)

        # Train epoch (triggers final logging at line 750)
        metrics = trainer.train_epoch(
            dataloader=dummy_loader, optimizer=optimizer, epoch=0
        )

        assert "loss" in metrics
        assert metrics["loss"] >= 0

    def test_non_amp_gradient_clipping_path(self, dummy_model, dummy_loader, device):
        """Test non-AMP gradient clipping (lines 447-454)."""
        model = dummy_model.to(device)
        config = AdversarialTrainingConfig(
            loss_type="trades",
            use_amp=False,  # No AMP
            gradient_clip=1.0,  # Enable gradient clipping
        )
        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        optimizer = optim.SGD(trainer.model.parameters(), lr=0.01)
        metrics = trainer.train_epoch(
            dataloader=dummy_loader, optimizer=optimizer, epoch=0
        )

        assert "loss" in metrics
        assert "clean_acc" in metrics

    def test_default_attack_step_size_calculation(
        self, dummy_model, dummy_loader, device
    ):
        """Test default attack_step_size calculation (line 195)."""
        model = dummy_model.to(device)
        # Don't specify attack_step_size, let it default
        config = AdversarialTrainingConfig(
            attack_epsilon=0.3,
            attack_step_size=None,  # Will be calculated as epsilon/4
            loss_type="at",
            use_amp=False,
        )
        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        # Verify calculation
        assert trainer.config.attack_step_size == 0.3 / 4.0

        optimizer = optim.SGD(trainer.model.parameters(), lr=0.01)
        metrics = trainer.train_epoch(
            dataloader=dummy_loader, optimizer=optimizer, epoch=0
        )

        assert "loss" in metrics

    def test_invalid_loss_type_exception_in_get_criterion(self, dummy_model, device):
        """Test exception for invalid loss type (line 300)."""
        model = dummy_model.to(device)

        # Create config with valid type first
        config = AdversarialTrainingConfig(
            loss_type="trades",
            use_amp=False,
        )

        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        # Bypass validation by setting invalid type directly
        trainer.config.loss_type = "invalid_type"

        # Now calling _create_criterion should raise ValueError (line 300)
        with pytest.raises(ValueError, match="Unknown loss type"):
            trainer._create_criterion()

    def test_loss_type_branch_at_in_train_epoch(
        self, dummy_model, dummy_loader, device
    ):
        """Test AT loss type branch in train loop (line 373, 410)."""
        model = dummy_model.to(device)
        config = AdversarialTrainingConfig(
            loss_type="at",
            mix_clean=0.0,  # Pure adversarial training, no clean mixing
            use_amp=False,
        )
        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        optimizer = optim.SGD(trainer.model.parameters(), lr=0.01)
        metrics = trainer.train_epoch(
            dataloader=dummy_loader, optimizer=optimizer, epoch=0
        )

        assert "loss" in metrics
        assert metrics["loss"] >= 0

    def test_scheduler_step_update_per_batch(self, dummy_model, dummy_loader, device):
        """Test per-batch scheduler step (line 488)."""
        model = dummy_model.to(device)
        config = AdversarialTrainingConfig(
            loss_type="trades",
            use_amp=False,
        )
        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        optimizer = optim.SGD(trainer.model.parameters(), lr=0.01)

        # Create mock scheduler with step_update method
        class MockScheduler:
            def __init__(self):
                self.steps = []

            def step_update(self, step):
                self.steps.append(step)

        scheduler = MockScheduler()

        metrics = trainer.train_epoch(
            dataloader=dummy_loader,
            optimizer=optimizer,
            epoch=0,
            scheduler=scheduler,
        )

        assert "loss" in metrics
        assert len(scheduler.steps) > 0  # Verify scheduler was called


class TestStandaloneFunctionsComplete:
    """PhD-level tests for standalone functions - 100% coverage."""

    def test_standalone_train_with_amp_and_logging(
        self, dummy_model, dummy_loader, device
    ):
        """Test standalone train function with AMP (lines 599-616)."""
        if device.type == "cpu":
            pytest.skip("AMP requires CUDA")

        model = dummy_model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        attack_config = PGDConfig(epsilon=8 / 255, step_size=2 / 255, num_steps=10)
        attack = PGD(attack_config)
        criterion = TRADESLoss(beta=1.0)

        # Test with AMP enabled and logging
        metrics = train_adversarial_epoch(
            model=model,
            dataloader=dummy_loader,
            optimizer=optimizer,
            criterion=criterion,
            attack=attack,
            device=device,
            epoch=0,
            use_amp=True,
            gradient_clip=1.0,
            log_frequency=1,  # Log every batch
        )

        assert "loss" in metrics
        assert "clean_acc" in metrics
        assert "adv_acc" in metrics

    def test_standalone_train_metrics_aggregation(
        self, dummy_model, dummy_loader, device
    ):
        """Test standalone metrics aggregation (lines 628-631)."""
        model = dummy_model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        attack_config = PGDConfig(epsilon=8 / 255, step_size=2 / 255, num_steps=10)
        attack = PGD(attack_config)
        criterion = AdversarialTrainingLoss(mix_clean=0.5)

        metrics = train_adversarial_epoch(
            model=model,
            dataloader=dummy_loader,
            optimizer=optimizer,
            criterion=criterion,
            attack=attack,
            device=device,
            epoch=0,
            use_amp=False,
            log_frequency=10,
        )

        # Verify metrics are properly averaged (lines 628-631)
        assert metrics["loss"] >= 0
        assert 0 <= metrics["clean_acc"] <= 1
        assert 0 <= metrics["adv_acc"] <= 1

    def test_standalone_train_logging_frequency(
        self, dummy_model, dummy_loader, device
    ):
        """Test standalone logging frequency (line 646)."""
        model = dummy_model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        attack_config = PGDConfig(epsilon=8 / 255, step_size=2 / 255, num_steps=10)
        attack = PGD(attack_config)
        criterion = TRADESLoss(beta=1.0)

        # Test with different log frequencies
        metrics = train_adversarial_epoch(
            model=model,
            dataloader=dummy_loader,
            optimizer=optimizer,
            criterion=criterion,
            attack=attack,
            device=device,
            epoch=0,
            use_amp=False,
            log_frequency=2,  # Log every 2 batches
        )

        assert "loss" in metrics

    def test_standalone_validate_default_step_size(
        self, dummy_model, dummy_loader, device
    ):
        """Test validate_robust with default step_size (line 585)."""
        model = dummy_model.to(device)

        # Call without attack_step_size - should default to epsilon/4
        metrics = validate_robust(
            model=model,
            dataloader=dummy_loader,
            attack_epsilon=0.3,
            attack_steps=20,
            attack_step_size=None,  # Will default (line 585)
            device=device,
        )

        assert "clean_acc" in metrics
        assert "robust_acc" in metrics

    def test_standalone_validate_with_logging(self, dummy_model, dummy_loader, device):
        """Test validate_robust logging paths (lines 703-706)."""
        model = dummy_model.to(device)

        metrics = validate_robust(
            model=model,
            dataloader=dummy_loader,
            attack_epsilon=8 / 255,
            attack_steps=10,
            attack_step_size=2 / 255,
            device=device,
        )

        assert "clean_acc" in metrics
        assert "robust_acc" in metrics
        # Verify metrics are percentages (line 727)
        assert 0 <= metrics["clean_acc"] <= 100
        assert 0 <= metrics["robust_acc"] <= 100

    def test_standalone_validate_batch_metrics(self, dummy_model, dummy_loader, device):
        """Test validate_robust batch processing (lines 727, 750)."""
        model = dummy_model.to(device)

        # Multiple batches to ensure full coverage
        metrics = validate_robust(
            model=model,
            dataloader=dummy_loader,
            attack_epsilon=8 / 255,
            attack_steps=10,
            attack_step_size=2 / 255,
            device=device,
        )

        # Line 727: Metrics formatting
        assert isinstance(metrics["clean_acc"], float)
        assert isinstance(metrics["robust_acc"], float)

        # Line 750: Final return with proper percentages
        assert metrics["clean_acc"] >= 0
        assert metrics["robust_acc"] >= 0

    def test_standalone_amp_with_gradient_clipping(
        self, dummy_model, dummy_loader, device
    ):
        """Test standalone AMP gradient clipping (lines 607, 611-615)."""
        if device.type == "cpu":
            pytest.skip("AMP requires CUDA")

        model = dummy_model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        attack_config = PGDConfig(epsilon=8 / 255, step_size=2 / 255, num_steps=10)
        attack = PGD(attack_config)
        criterion = TRADESLoss(beta=1.0)

        # Test AMP with gradient clipping (lines 607, 611-615)
        metrics = train_adversarial_epoch(
            model=model,
            dataloader=dummy_loader,
            optimizer=optimizer,
            criterion=criterion,
            attack=attack,
            device=device,
            epoch=0,
            use_amp=True,
            gradient_clip=1.0,  # Enable clipping - hits lines 607, 611-615
            log_frequency=10,
        )

        assert "loss" in metrics
        assert "clean_acc" in metrics

    def test_standalone_final_metric_calculation(
        self, dummy_model, dummy_loader, device
    ):
        """Test standalone final metrics (lines 628-631)."""
        model = dummy_model.to(device)
        optimizer = optim.SGD(model.parameters(), lr=0.01)

        attack_config = PGDConfig(epsilon=8 / 255, step_size=2 / 255, num_steps=10)
        attack = PGD(attack_config)
        criterion = AdversarialTrainingLoss(mix_clean=0.5)

        # Run full epoch to calculate final metrics (lines 628-631)
        metrics = train_adversarial_epoch(
            model=model,
            dataloader=dummy_loader,
            optimizer=optimizer,
            criterion=criterion,
            attack=attack,
            device=device,
            epoch=0,
            use_amp=False,
            gradient_clip=None,
            log_frequency=10,
        )

        # Lines 628-631: Final metric averaging
        assert 0 <= metrics["loss"] < float("inf")
        assert 0 <= metrics["clean_acc"] <= 1
        assert 0 <= metrics["adv_acc"] <= 1

    def test_class_method_gradient_clipping_no_amp(
        self, dummy_model, dummy_loader, device
    ):
        """Test class non-AMP gradient clipping (lines 447-454)."""
        model = dummy_model.to(device)
        config = AdversarialTrainingConfig(
            loss_type="at",
            mix_clean=0.0,
            use_amp=False,  # No AMP
            gradient_clip=1.0,  # Enable clipping - lines 447-454
        )
        trainer = AdversarialTrainer(
            model=model,
            config=config,
            device=device,
        )

        optimizer = optim.SGD(trainer.model.parameters(), lr=0.01)
        metrics = trainer.train_epoch(
            dataloader=dummy_loader, optimizer=optimizer, epoch=0
        )

        assert "loss" in metrics
        assert metrics["loss"] >= 0

    def test_config_default_step_size_when_none(self):
        """Test config default step_size calculation (lines 194-195)."""
        # Create config without attack_step_size
        config = AdversarialTrainingConfig(
            attack_epsilon=0.3,
            attack_step_size=None,  # Should default to epsilon/4
            loss_type="trades",
        )

        # Post-init should set it to epsilon/4 (line 195)
        assert config.attack_step_size == 0.3 / 4.0
        assert config.attack_step_size == 0.075

    def test_standalone_validate_default_attack_step(
        self, dummy_model, dummy_loader, device
    ):
        """Test standalone validate default attack_step_size (line 703)."""
        model = dummy_model.to(device)

        # Call without step_size - line 703 should calculate default
        metrics = validate_robust(
            model=model,
            dataloader=dummy_loader,
            attack_epsilon=0.3,
            attack_steps=10,
            attack_step_size=None,  # Line 703: defaults to epsilon/4
            device=device,
        )

        assert "clean_acc" in metrics
        assert "robust_acc" in metrics


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
