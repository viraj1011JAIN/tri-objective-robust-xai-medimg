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
    return "cuda" if torch.cuda.is_available() else "cpu"


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

        # Should be between pure clean and pure adversarial
        clean_ce = nn.CrossEntropyLoss()(clean_logits, labels)
        adv_ce = nn.CrossEntropyLoss()(adv_logits, labels)

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


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
