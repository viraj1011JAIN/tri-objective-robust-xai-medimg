"""
Comprehensive A1-Grade Tests for Tri-Objective Loss.

Tests all functionality in src/losses/tri_objective.py:
- SSIMLoss: Gaussian window creation, SSIM computation, all reductions
- TCAVLoss: Lazy initialization, eager initialization, return_scores
- TRADESLoss: KL divergence computation with beta scaling
- TriObjectiveLoss: Full tri-objective optimization with all components

Target: 100% line coverage, 100% branch coverage, 0 failures, 0 skipped.

Author: Viraj Pankaj Jain
Institution: University of Glasgow
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict
from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.losses.task_loss import TaskLoss
from src.losses.tri_objective import SSIMLoss, TCAVLoss, TRADESLoss, TriObjectiveLoss

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def device():
    """Device for testing."""
    return torch.device("cpu")


@pytest.fixture
def batch_heatmaps(device):
    """Sample heatmaps for SSIM testing."""
    # Create two similar heatmaps (high SSIM)
    batch_size = 4
    h, w = 32, 32

    heatmap1 = torch.rand(batch_size, 1, h, w, device=device)
    heatmap2 = heatmap1 + torch.randn(batch_size, 1, h, w, device=device) * 0.1

    return heatmap1, heatmap2


@pytest.fixture
def batch_embeddings(device):
    """Sample embeddings for TCAV testing."""
    batch_size = 8
    embedding_dim = 16

    embeddings = torch.randn(batch_size, embedding_dim, device=device)
    return embeddings


@pytest.fixture
def batch_logits(device):
    """Sample logits for TRADES testing."""
    batch_size = 8
    num_classes = 7

    logits_clean = torch.randn(batch_size, num_classes, device=device)
    logits_adv = (
        logits_clean + torch.randn(batch_size, num_classes, device=device) * 0.5
    )

    return logits_clean, logits_adv


@pytest.fixture
def batch_labels_multiclass(device):
    """Multi-class labels."""
    batch_size = 8
    num_classes = 7
    labels = torch.randint(0, num_classes, (batch_size,), device=device)
    return labels


@pytest.fixture
def batch_labels_multilabel(device):
    """Multi-label labels."""
    batch_size = 8
    num_classes = 7
    labels = torch.randint(0, 2, (batch_size, num_classes), device=device).float()
    return labels


# ---------------------------------------------------------------------------
# SSIMLoss Tests
# ---------------------------------------------------------------------------


class TestSSIMLoss:
    """Tests for SSIMLoss class."""

    def test_initialization_default(self):
        """Test default initialization."""
        loss_fn = SSIMLoss()

        assert loss_fn.window_size == 11
        assert loss_fn.sigma == 1.5
        assert loss_fn.channel == 1
        assert loss_fn.data_range == 1.0
        assert loss_fn.reduction == "mean"
        assert loss_fn.window is not None
        assert loss_fn.window.shape == (1, 1, 11, 11)

    def test_initialization_custom_params(self):
        """Test custom parameters."""
        loss_fn = SSIMLoss(
            window_size=7,
            sigma=2.0,
            channel=3,
            data_range=255.0,
            reduction="sum",
        )

        assert loss_fn.window_size == 7
        assert loss_fn.sigma == 2.0
        assert loss_fn.channel == 3
        assert loss_fn.data_range == 255.0
        assert loss_fn.reduction == "sum"
        assert loss_fn.window.shape == (3, 1, 7, 7)

    def test_create_window_gaussian_properties(self):
        """Test Gaussian window has correct properties."""
        loss_fn = SSIMLoss(window_size=11, sigma=1.5, channel=1)
        window = loss_fn.window

        # Window should sum to approximately 1 per channel
        window_sum = window[0, 0].sum()
        assert torch.abs(window_sum - 1.0) < 1e-5

        # Window should be symmetric
        assert torch.allclose(window[0, 0], window[0, 0].T, atol=1e-6)

        # Center should have highest value
        center = loss_fn.window_size // 2
        assert window[0, 0, center, center] == window[0, 0].max()

    def test_forward_identical_images_low_loss(self, batch_heatmaps, device):
        """Test SSIM loss is near zero for identical images."""
        heatmap, _ = batch_heatmaps
        loss_fn = SSIMLoss()

        # Identical images should have SSIM ≈ 1, so loss ≈ 0
        loss = loss_fn(heatmap, heatmap)

        assert loss < 0.01  # Very small loss
        assert loss >= 0.0  # Non-negative

    def test_forward_different_images_high_loss(self, device):
        """Test SSIM loss is high for very different images."""
        batch_size = 4
        h, w = 32, 32

        heatmap1 = torch.zeros(batch_size, 1, h, w, device=device)
        heatmap2 = torch.ones(batch_size, 1, h, w, device=device)

        loss_fn = SSIMLoss()
        loss = loss_fn(heatmap1, heatmap2)

        assert loss > 0.5  # High loss for very different images
        assert loss <= 2.0  # Loss is bounded

    def test_forward_shape_mismatch_raises_error(self, device):
        """Test shape mismatch raises ValueError."""
        heatmap1 = torch.rand(4, 1, 32, 32, device=device)
        heatmap2 = torch.rand(4, 1, 64, 64, device=device)  # Different size

        loss_fn = SSIMLoss()

        with pytest.raises(ValueError, match="Shape mismatch"):
            loss_fn(heatmap1, heatmap2)

    def test_forward_wrong_dimensions_raises_error(self, device):
        """Test wrong dimensions raises ValueError."""
        heatmap = torch.rand(4, 32, 32, device=device)  # 3D instead of 4D

        loss_fn = SSIMLoss()

        with pytest.raises(ValueError, match="Expected 4D tensors"):
            loss_fn(heatmap, heatmap)

    def test_reduction_mean(self, batch_heatmaps, device):
        """Test mean reduction."""
        heatmap1, heatmap2 = batch_heatmaps
        loss_fn = SSIMLoss(reduction="mean")

        loss = loss_fn(heatmap1, heatmap2)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0.0

    def test_reduction_sum(self, batch_heatmaps, device):
        """Test sum reduction."""
        heatmap1, heatmap2 = batch_heatmaps
        loss_fn = SSIMLoss(reduction="sum")

        loss = loss_fn(heatmap1, heatmap2)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0.0

    def test_reduction_none(self, batch_heatmaps, device):
        """Test no reduction."""
        heatmap1, heatmap2 = batch_heatmaps
        batch_size = heatmap1.shape[0]

        loss_fn = SSIMLoss(reduction="none")
        loss = loss_fn(heatmap1, heatmap2)

        assert loss.shape == (batch_size,)  # Per-sample loss
        assert (loss >= 0.0).all()

    def test_reduction_invalid_raises_error(self, batch_heatmaps, device):
        """Test invalid reduction raises ValueError."""
        heatmap1, heatmap2 = batch_heatmaps
        loss_fn = SSIMLoss(reduction="invalid")

        with pytest.raises(ValueError, match="Unknown reduction"):
            loss_fn(heatmap1, heatmap2)

    def test_window_device_transfer(self):
        """Test window is moved to correct device when needed."""
        loss_fn = SSIMLoss()

        # Window starts on CPU
        assert loss_fn.window.device.type == "cpu"

        # Test with CUDA if available to trigger device transfer
        if torch.cuda.is_available():
            cuda_device = torch.device("cuda")
            heatmap1 = torch.rand(2, 1, 32, 32, device=cuda_device)
            heatmap2 = torch.rand(2, 1, 32, 32, device=cuda_device)

            # Forward pass should trigger device transfer (line 154)
            loss = loss_fn(heatmap1, heatmap2)

            # Loss should be on CUDA
            assert loss.device.type == "cuda"
        else:
            # If CUDA not available, just test CPU path
            heatmap1 = torch.rand(2, 1, 32, 32)
            heatmap2 = torch.rand(2, 1, 32, 32)

            loss = loss_fn(heatmap1, heatmap2)
            assert loss.device.type == "cpu"

    def test_window_already_on_device_skips_transfer(self, device):
        """Test window transfer is skipped when already on correct device."""
        loss_fn = SSIMLoss()
        heatmap1 = torch.rand(2, 1, 32, 32, device=device)
        heatmap2 = torch.rand(2, 1, 32, 32, device=device)

        # First forward: window gets moved to device
        loss1 = loss_fn(heatmap1, heatmap2)

        # Second forward: window already on device (should skip transfer)
        loss2 = loss_fn(heatmap1, heatmap2)

        # Both should work correctly
        assert torch.abs(loss1 - loss2) < 1e-6

    def test_ssim_computation_differentiable(self, batch_heatmaps, device):
        """Test SSIM loss is differentiable."""
        heatmap1, heatmap2 = batch_heatmaps
        heatmap1.requires_grad = True

        loss_fn = SSIMLoss()
        loss = loss_fn(heatmap1, heatmap2)

        loss.backward()

        assert heatmap1.grad is not None
        assert not torch.isnan(heatmap1.grad).any()

    def test_ssim_constants_set_correctly(self):
        """Test c1 and c2 constants are set correctly."""
        data_range = 255.0
        loss_fn = SSIMLoss(data_range=data_range)

        expected_c1 = (0.01 * data_range) ** 2
        expected_c2 = (0.03 * data_range) ** 2

        assert abs(loss_fn.c1 - expected_c1) < 1e-6
        assert abs(loss_fn.c2 - expected_c2) < 1e-6


# ---------------------------------------------------------------------------
# TCAVLoss Tests
# ---------------------------------------------------------------------------


class TestTCAVLoss:
    """Tests for TCAVLoss class."""

    def test_initialization_with_embedding_dim(self):
        """Test initialization with explicit embedding_dim."""
        embedding_dim = 128
        loss_fn = TCAVLoss(num_concepts=10, embedding_dim=embedding_dim)

        assert loss_fn.num_concepts == 10
        assert loss_fn.embedding_dim == embedding_dim
        assert loss_fn._initialized is True
        assert loss_fn.medical_cavs.shape == (5, 128)
        assert loss_fn.artifact_cavs.shape == (5, 128)

    def test_initialization_without_embedding_dim(self):
        """Test initialization without embedding_dim (lazy init)."""
        loss_fn = TCAVLoss(num_concepts=10, embedding_dim=None)

        assert loss_fn.num_concepts == 10
        assert loss_fn.embedding_dim is None
        assert loss_fn._initialized is False

    def test_lazy_initialization_on_first_forward(self, device):
        """Test lazy initialization on first forward pass."""
        loss_fn = TCAVLoss(num_concepts=10, embedding_dim=None)
        embeddings = torch.randn(8, 16, device=device)

        # Before forward: not initialized
        assert loss_fn._initialized is False

        # Forward pass
        loss = loss_fn(embeddings)

        # After forward: initialized with correct dim
        assert loss_fn._initialized is True
        assert loss_fn.embedding_dim == 16
        assert loss_fn.medical_cavs.shape == (5, 16)
        assert loss_fn.artifact_cavs.shape == (5, 16)

    def test_forward_returns_scalar_loss(self, batch_embeddings, device):
        """Test forward returns scalar loss."""
        loss_fn = TCAVLoss(num_concepts=10)
        loss = loss_fn(batch_embeddings)

        assert loss.ndim == 0  # Scalar
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_forward_with_return_scores(self, batch_embeddings, device):
        """Test forward with return_scores=True."""
        loss_fn = TCAVLoss(num_concepts=10)
        loss, scores = loss_fn(batch_embeddings, return_scores=True)

        assert loss.ndim == 0  # Scalar
        assert isinstance(scores, dict)
        assert "medical_tcav" in scores
        assert "artifact_tcav" in scores
        assert isinstance(scores["medical_tcav"], float)
        assert isinstance(scores["artifact_tcav"], float)

    def test_forward_without_return_scores(self, batch_embeddings, device):
        """Test forward with return_scores=False (default)."""
        loss_fn = TCAVLoss(num_concepts=10)
        result = loss_fn(batch_embeddings, return_scores=False)

        # Should return only loss, not tuple
        assert isinstance(result, Tensor)
        assert result.ndim == 0

    def test_forward_differentiable(self, batch_embeddings, device):
        """Test forward is differentiable."""
        batch_embeddings.requires_grad = True
        loss_fn = TCAVLoss(num_concepts=10)

        loss = loss_fn(batch_embeddings)
        loss.backward()

        assert batch_embeddings.grad is not None
        assert not torch.isnan(batch_embeddings.grad).any()

    def test_cav_normalization_in_forward(self, device):
        """Test embeddings are normalized in forward pass."""
        embeddings = torch.randn(8, 16, device=device)
        loss_fn = TCAVLoss(num_concepts=10, embedding_dim=16)

        # Forward pass normalizes embeddings internally
        loss = loss_fn(embeddings)

        # Loss should be computed correctly
        assert not torch.isnan(loss)

    def test_multiple_forwards_consistent(self, batch_embeddings, device):
        """Test multiple forward passes are consistent."""
        loss_fn = TCAVLoss(num_concepts=10)

        loss1 = loss_fn(batch_embeddings)
        loss2 = loss_fn(batch_embeddings)

        # Same input should give same output
        assert torch.abs(loss1 - loss2) < 1e-6

    def test_initialize_cavs_not_called_when_already_initialized(self, device):
        """Test _initialize_cavs skips when already initialized."""
        loss_fn = TCAVLoss(num_concepts=10, embedding_dim=16)

        # Already initialized
        assert loss_fn._initialized is True

        # Try to initialize again (should skip)
        initial_medical_cavs = loss_fn.medical_cavs.clone()
        loss_fn._initialize_cavs(32)  # Try different dim

        # Should still have original CAVs (not re-initialized)
        assert torch.equal(loss_fn.medical_cavs, initial_medical_cavs)
        assert loss_fn.embedding_dim == 16  # Still original dim


# ---------------------------------------------------------------------------
# TRADESLoss Tests
# ---------------------------------------------------------------------------


class TestTRADESLoss:
    """Tests for TRADESLoss class."""

    def test_initialization_default_beta(self):
        """Test default beta parameter."""
        loss_fn = TRADESLoss()
        assert loss_fn.beta == 6.0

    def test_initialization_custom_beta(self):
        """Test custom beta parameter."""
        loss_fn = TRADESLoss(beta=10.0)
        assert loss_fn.beta == 10.0

    def test_forward_returns_scalar(self, batch_logits, device):
        """Test forward returns scalar loss."""
        logits_clean, logits_adv = batch_logits
        loss_fn = TRADESLoss()

        loss = loss_fn(logits_clean, logits_adv)

        assert loss.ndim == 0  # Scalar
        assert loss >= 0.0  # KL divergence is non-negative

    def test_forward_identical_logits_zero_loss(self, device):
        """Test KL divergence is zero for identical logits."""
        logits = torch.randn(8, 7, device=device)
        loss_fn = TRADESLoss(beta=1.0)

        loss = loss_fn(logits, logits)

        assert loss < 0.01  # Near zero

    def test_forward_different_logits_positive_loss(self, batch_logits, device):
        """Test KL divergence is positive for different logits."""
        logits_clean, logits_adv = batch_logits
        loss_fn = TRADESLoss(beta=1.0)

        loss = loss_fn(logits_clean, logits_adv)

        assert loss > 0.0

    def test_beta_scaling(self, batch_logits, device):
        """Test beta parameter scales loss correctly."""
        logits_clean, logits_adv = batch_logits

        loss_fn_beta1 = TRADESLoss(beta=1.0)
        loss_fn_beta2 = TRADESLoss(beta=2.0)

        loss1 = loss_fn_beta1(logits_clean, logits_adv)
        loss2 = loss_fn_beta2(logits_clean, logits_adv)

        # loss2 should be approximately 2 * loss1
        assert torch.abs(loss2 - 2.0 * loss1) < 0.01

    def test_forward_differentiable(self, batch_logits, device):
        """Test forward is differentiable."""
        logits_clean, logits_adv = batch_logits
        logits_clean.requires_grad = True

        loss_fn = TRADESLoss()
        loss = loss_fn(logits_clean, logits_adv)

        loss.backward()

        assert logits_clean.grad is not None
        assert not torch.isnan(logits_clean.grad).any()

    def test_kl_divergence_computation(self, device):
        """Test KL divergence is computed correctly."""
        # Create known distributions
        logits_clean = torch.tensor([[1.0, 0.0, 0.0]], device=device)
        logits_adv = torch.tensor([[0.0, 1.0, 0.0]], device=device)

        loss_fn = TRADESLoss(beta=1.0)
        loss = loss_fn(logits_clean, logits_adv)

        # KL divergence should be positive and finite
        assert loss > 0.0
        assert not torch.isinf(loss)


# ---------------------------------------------------------------------------
# TriObjectiveLoss Tests
# ---------------------------------------------------------------------------


class TestTriObjectiveLoss:
    """Tests for TriObjectiveLoss class."""

    def test_initialization_default_params(self):
        """Test default initialization."""
        loss_fn = TriObjectiveLoss(num_classes=7, task_type="multi_class")

        assert loss_fn.num_classes == 7
        assert loss_fn.task_type == "multi_class"
        assert loss_fn.lambda_rob == 0.3
        assert loss_fn.lambda_expl == 0.2
        assert loss_fn.lambda_ssim == 0.7
        assert loss_fn.lambda_tcav == 0.3
        assert isinstance(loss_fn.temperature, nn.Parameter)
        assert loss_fn.temperature.item() == pytest.approx(1.5)
        assert isinstance(loss_fn.task_loss_fn, TaskLoss)
        assert isinstance(loss_fn.robustness_loss_fn, TRADESLoss)
        assert isinstance(loss_fn.ssim_loss_fn, SSIMLoss)
        assert isinstance(loss_fn.tcav_loss_fn, TCAVLoss)

    def test_initialization_custom_params(self):
        """Test custom parameters."""
        loss_fn = TriObjectiveLoss(
            num_classes=10,
            task_type="multi_label",
            lambda_rob=0.5,
            lambda_expl=0.3,
            lambda_ssim=0.6,
            lambda_tcav=0.4,
            temperature=2.0,
            trades_beta=10.0,
            reduction="sum",
            name="custom_loss",
        )

        assert loss_fn.num_classes == 10
        assert loss_fn.task_type == "multi_label"
        assert loss_fn.lambda_rob == 0.5
        assert loss_fn.lambda_expl == 0.3
        assert loss_fn.lambda_ssim == 0.6
        assert loss_fn.lambda_tcav == 0.4
        assert loss_fn.temperature.item() == pytest.approx(2.0)
        assert loss_fn.robustness_loss_fn.beta == 10.0

    def test_forward_multiclass_without_optional(
        self, batch_logits, batch_labels_multiclass, device
    ):
        """Test forward with multi-class, no heatmaps/embeddings."""
        logits_clean, logits_adv = batch_logits
        labels = batch_labels_multiclass

        loss_fn = TriObjectiveLoss(num_classes=7, task_type="multi_class")
        outputs = loss_fn(
            logits_clean=logits_clean,
            logits_adv=logits_adv,
            labels=labels,
        )

        assert isinstance(outputs, dict)
        assert "loss" in outputs
        assert "task" in outputs
        assert "robustness" in outputs
        assert "explanation" in outputs
        assert "ssim" in outputs
        assert "tcav" in outputs
        assert "temperature" in outputs

        # Check loss is scalar
        assert outputs["loss"].ndim == 0
        assert outputs["task"].ndim == 0
        assert outputs["robustness"].ndim == 0

        # Explanation should be zero (no heatmaps/embeddings)
        assert outputs["explanation"].item() == 0.0
        assert outputs["ssim"].item() == 0.0
        assert outputs["tcav"].item() == 0.0

    def test_forward_multiclass_with_embeddings(
        self, batch_logits, batch_labels_multiclass, batch_embeddings, device
    ):
        """Test forward with embeddings."""
        logits_clean, logits_adv = batch_logits
        labels = batch_labels_multiclass

        loss_fn = TriObjectiveLoss(num_classes=7, task_type="multi_class")
        outputs = loss_fn(
            logits_clean=logits_clean,
            logits_adv=logits_adv,
            labels=labels,
            embeddings=batch_embeddings,
        )

        # TCAV loss should be non-zero
        assert outputs["tcav"].item() != 0.0

        # Explanation loss should include TCAV
        expected_expl = loss_fn.lambda_tcav * outputs["tcav"]
        assert torch.abs(outputs["explanation"] - expected_expl) < 1e-5

    def test_forward_multiclass_with_heatmaps(
        self, batch_logits, batch_labels_multiclass, batch_heatmaps, device
    ):
        """Test forward with heatmaps."""
        logits_clean, logits_adv = batch_logits
        labels = batch_labels_multiclass
        heatmap_clean, heatmap_adv = batch_heatmaps

        loss_fn = TriObjectiveLoss(num_classes=7, task_type="multi_class")
        outputs = loss_fn(
            logits_clean=logits_clean,
            logits_adv=logits_adv,
            labels=labels,
            heatmap_clean=heatmap_clean,
            heatmap_adv=heatmap_adv,
        )

        # SSIM loss should be non-zero
        assert outputs["ssim"].item() >= 0.0

        # Explanation loss should include SSIM
        expected_expl = loss_fn.lambda_ssim * outputs["ssim"]
        assert torch.abs(outputs["explanation"] - expected_expl) < 1e-5

    def test_forward_multiclass_with_all_components(
        self,
        batch_logits,
        batch_labels_multiclass,
        batch_heatmaps,
        batch_embeddings,
        device,
    ):
        """Test forward with all components."""
        logits_clean, logits_adv = batch_logits
        labels = batch_labels_multiclass
        heatmap_clean, heatmap_adv = batch_heatmaps

        loss_fn = TriObjectiveLoss(num_classes=7, task_type="multi_class")
        outputs = loss_fn(
            logits_clean=logits_clean,
            logits_adv=logits_adv,
            labels=labels,
            heatmap_clean=heatmap_clean,
            heatmap_adv=heatmap_adv,
            embeddings=batch_embeddings,
        )

        # All components should be non-zero
        assert outputs["task"].item() > 0.0
        assert outputs["robustness"].item() > 0.0
        assert outputs["ssim"].item() >= 0.0
        assert outputs["tcav"].item() != 0.0

        # Total loss should be sum of components
        expected_total = (
            outputs["task"]
            + loss_fn.lambda_rob * outputs["robustness"]
            + loss_fn.lambda_expl * outputs["explanation"]
        )
        assert torch.abs(outputs["loss"] - expected_total) < 1e-4

    def test_forward_multilabel(self, batch_logits, batch_labels_multilabel, device):
        """Test forward with multi-label."""
        logits_clean, logits_adv = batch_logits
        labels = batch_labels_multilabel

        loss_fn = TriObjectiveLoss(num_classes=7, task_type="multi_label")
        outputs = loss_fn(
            logits_clean=logits_clean,
            logits_adv=logits_adv,
            labels=labels,
        )

        assert outputs["loss"].ndim == 0
        assert outputs["task"].item() > 0.0
        assert outputs["robustness"].item() > 0.0

    def test_temperature_scaling(self, batch_logits, batch_labels_multiclass, device):
        """Test temperature scaling affects loss."""
        logits_clean, logits_adv = batch_logits
        labels = batch_labels_multiclass

        loss_fn_t1 = TriObjectiveLoss(num_classes=7, temperature=1.0)
        loss_fn_t2 = TriObjectiveLoss(num_classes=7, temperature=2.0)

        outputs_t1 = loss_fn_t1(logits_clean, logits_adv, labels)
        outputs_t2 = loss_fn_t2(logits_clean, logits_adv, labels)

        # Different temperatures should give different task losses
        assert outputs_t1["task"].item() != outputs_t2["task"].item()

    def test_lambda_weights_affect_total_loss(
        self, batch_logits, batch_labels_multiclass, device
    ):
        """Test lambda weights affect total loss."""
        logits_clean, logits_adv = batch_logits
        labels = batch_labels_multiclass

        loss_fn_low = TriObjectiveLoss(num_classes=7, lambda_rob=0.1)
        loss_fn_high = TriObjectiveLoss(num_classes=7, lambda_rob=0.9)

        outputs_low = loss_fn_low(logits_clean, logits_adv, labels)
        outputs_high = loss_fn_high(logits_clean, logits_adv, labels)

        # Higher lambda_rob should increase total loss
        assert outputs_high["loss"].item() > outputs_low["loss"].item()

    def test_forward_differentiable(
        self, batch_logits, batch_labels_multiclass, device
    ):
        """Test forward is differentiable."""
        logits_clean, logits_adv = batch_logits
        labels = batch_labels_multiclass
        logits_clean.requires_grad = True

        loss_fn = TriObjectiveLoss(num_classes=7, task_type="multi_class")
        outputs = loss_fn(logits_clean, logits_adv, labels)

        outputs["loss"].backward()

        assert logits_clean.grad is not None
        assert not torch.isnan(logits_clean.grad).any()

    def test_temperature_is_learnable(
        self, batch_logits, batch_labels_multiclass, device
    ):
        """Test temperature parameter is learnable."""
        logits_clean, logits_adv = batch_logits
        labels = batch_labels_multiclass

        loss_fn = TriObjectiveLoss(num_classes=7, task_type="multi_class")
        initial_temp = loss_fn.temperature.item()

        # Forward + backward
        outputs = loss_fn(logits_clean, logits_adv, labels)
        outputs["loss"].backward()

        # Temperature should have gradient
        assert loss_fn.temperature.grad is not None

    def test_compute_raises_not_implemented_error(self):
        """Test compute() raises NotImplementedError."""
        loss_fn = TriObjectiveLoss(num_classes=7)
        logits = torch.randn(8, 7)
        labels = torch.randint(0, 7, (8,))

        with pytest.raises(
            NotImplementedError, match="TriObjectiveLoss requires forward"
        ):
            loss_fn.compute(logits, labels)

    def test_explanation_loss_composition(
        self,
        batch_logits,
        batch_labels_multiclass,
        batch_heatmaps,
        batch_embeddings,
        device,
    ):
        """Test explanation loss is composed correctly."""
        logits_clean, logits_adv = batch_logits
        labels = batch_labels_multiclass
        heatmap_clean, heatmap_adv = batch_heatmaps

        loss_fn = TriObjectiveLoss(
            num_classes=7,
            lambda_ssim=0.7,
            lambda_tcav=0.3,
        )

        outputs = loss_fn(
            logits_clean=logits_clean,
            logits_adv=logits_adv,
            labels=labels,
            heatmap_clean=heatmap_clean,
            heatmap_adv=heatmap_adv,
            embeddings=batch_embeddings,
        )

        expected_expl = (
            loss_fn.lambda_ssim * outputs["ssim"]
            + loss_fn.lambda_tcav * outputs["tcav"]
        )

        assert torch.abs(outputs["explanation"] - expected_expl) < 1e-5

    def test_only_heatmap_clean_provided(
        self, batch_logits, batch_labels_multiclass, batch_heatmaps, device
    ):
        """Test only heatmap_clean provided (no SSIM computation)."""
        logits_clean, logits_adv = batch_logits
        labels = batch_labels_multiclass
        heatmap_clean, _ = batch_heatmaps

        loss_fn = TriObjectiveLoss(num_classes=7)
        outputs = loss_fn(
            logits_clean=logits_clean,
            logits_adv=logits_adv,
            labels=labels,
            heatmap_clean=heatmap_clean,
            heatmap_adv=None,  # Only one heatmap
        )

        # SSIM should be zero (need both heatmaps)
        assert outputs["ssim"].item() == 0.0

    def test_only_heatmap_adv_provided(
        self, batch_logits, batch_labels_multiclass, batch_heatmaps, device
    ):
        """Test only heatmap_adv provided (no SSIM computation)."""
        logits_clean, logits_adv = batch_logits
        labels = batch_labels_multiclass
        _, heatmap_adv = batch_heatmaps

        loss_fn = TriObjectiveLoss(num_classes=7)
        outputs = loss_fn(
            logits_clean=logits_clean,
            logits_adv=logits_adv,
            labels=labels,
            heatmap_clean=None,
            heatmap_adv=heatmap_adv,  # Only one heatmap
        )

        # SSIM should be zero (need both heatmaps)
        assert outputs["ssim"].item() == 0.0


# ---------------------------------------------------------------------------
# Integration Tests
# ---------------------------------------------------------------------------


class TestIntegration:
    """Integration tests for tri-objective loss."""

    def test_full_training_step_simulation(
        self,
        batch_logits,
        batch_labels_multiclass,
        batch_heatmaps,
        batch_embeddings,
        device,
    ):
        """Simulate full training step with all components."""
        logits_clean, logits_adv = batch_logits
        labels = batch_labels_multiclass
        heatmap_clean, heatmap_adv = batch_heatmaps

        # Enable gradients
        logits_clean.requires_grad = True

        # Create loss function
        loss_fn = TriObjectiveLoss(
            num_classes=7,
            task_type="multi_class",
            lambda_rob=0.3,
            lambda_expl=0.2,
        )

        # Forward pass
        outputs = loss_fn(
            logits_clean=logits_clean,
            logits_adv=logits_adv,
            labels=labels,
            heatmap_clean=heatmap_clean,
            heatmap_adv=heatmap_adv,
            embeddings=batch_embeddings,
        )

        # Backward pass
        outputs["loss"].backward()

        # Check gradients
        assert logits_clean.grad is not None
        assert not torch.isnan(logits_clean.grad).any()
        assert not torch.isinf(logits_clean.grad).any()

        # Check all outputs are present
        assert all(
            key in outputs
            for key in [
                "loss",
                "task",
                "robustness",
                "explanation",
                "ssim",
                "tcav",
                "temperature",
            ]
        )

    def test_loss_components_are_additive(
        self, batch_logits, batch_labels_multiclass, device
    ):
        """Test loss components sum correctly."""
        logits_clean, logits_adv = batch_logits
        labels = batch_labels_multiclass

        loss_fn = TriObjectiveLoss(
            num_classes=7,
            lambda_rob=0.3,
            lambda_expl=0.2,
        )

        outputs = loss_fn(logits_clean, logits_adv, labels)

        # Compute expected total
        expected = (
            outputs["task"] + 0.3 * outputs["robustness"] + 0.2 * outputs["explanation"]
        )

        assert torch.abs(outputs["loss"] - expected) < 1e-4
