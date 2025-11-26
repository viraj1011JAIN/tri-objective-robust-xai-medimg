"""
Unit Tests for Explanation Loss Module.

This module provides comprehensive tests for the explanation loss components:
- SSIMStabilityLoss
- TCavConceptLoss
- ExplanationLoss (combined)

Test categories:
1. Basic functionality tests
2. Gradient flow verification
3. Edge case handling
4. Numerical stability tests
5. Configuration validation
6. Integration tests

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
"""

import math
from typing import List, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

# Import modules under test
from src.losses.explanation_loss import (
    ExplanationLoss,
    ExplanationLossConfig,
    SSIMKernelType,
    SSIMStabilityLoss,
    TCavConceptLoss,
    benchmark_computational_overhead,
    create_explanation_loss,
    verify_gradient_flow,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def device() -> torch.device:
    """Get the appropriate device for testing."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_heatmaps(device: torch.device) -> Tuple[Tensor, Tensor]:
    """Generate sample heatmaps for testing."""
    batch_size = 4
    height, width = 56, 56

    heatmap_clean = torch.rand(batch_size, 1, height, width, device=device)
    heatmap_adv = torch.rand(batch_size, 1, height, width, device=device)

    return heatmap_clean, heatmap_adv


@pytest.fixture
def identical_heatmaps(device: torch.device) -> Tuple[Tensor, Tensor]:
    """Generate identical heatmaps (SSIM should be 1.0)."""
    batch_size = 4
    height, width = 56, 56

    heatmap = torch.rand(batch_size, 1, height, width, device=device)
    return heatmap.clone(), heatmap.clone()


@pytest.fixture
def sample_cavs(device: torch.device) -> Tuple[List[Tensor], List[Tensor]]:
    """Generate sample CAVs for testing."""
    feat_dim = 2048
    num_artifact = 4  # ruler, hair, ink, borders
    num_medical = 6  # asymmetry, pigment_network, etc.

    artifact_cavs = [torch.randn(feat_dim, device=device) for _ in range(num_artifact)]
    medical_cavs = [torch.randn(feat_dim, device=device) for _ in range(num_medical)]

    return artifact_cavs, medical_cavs


@pytest.fixture
def sample_activations_gradients(device: torch.device) -> Tuple[Tensor, Tensor]:
    """Generate sample activations and gradients for TCAV testing."""
    batch_size = 8
    feat_dim = 2048

    activations = torch.randn(batch_size, feat_dim, device=device)
    gradients = torch.randn(batch_size, feat_dim, device=device)

    return activations, gradients


@pytest.fixture
def simple_cnn(device: torch.device) -> nn.Module:
    """Create a simple CNN for testing."""

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 64, 3, padding=1)
            self.conv2 = nn.Conv2d(64, 128, 3, padding=1)
            self.conv3 = nn.Conv2d(128, 256, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(256, 10)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

        def get_feature_maps(self, x):
            x = F.relu(self.conv1(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv2(x))
            x = F.max_pool2d(x, 2)
            x = F.relu(self.conv3(x))
            return x

    model = SimpleCNN().to(device)
    model.eval()
    return model


# ============================================================================
# SSIMStabilityLoss Tests
# ============================================================================


class TestSSIMStabilityLoss:
    """Tests for SSIMStabilityLoss class."""

    def test_initialization_default(self):
        """Test default initialization."""
        loss = SSIMStabilityLoss()

        assert loss.window_size == 11
        assert loss.sigma == 1.5
        assert loss.data_range == 1.0
        assert loss.k1 == 0.01
        assert loss.k2 == 0.03
        assert loss.reduction == "mean"
        assert not loss.use_ms_ssim

    def test_initialization_custom(self):
        """Test custom initialization."""
        loss = SSIMStabilityLoss(
            window_size=7,
            sigma=2.0,
            data_range=255.0,
            k1=0.02,
            k2=0.06,
            reduction="sum",
            use_ms_ssim=True,
        )

        assert loss.window_size == 7
        assert loss.sigma == 2.0
        assert loss.data_range == 255.0
        assert loss.k1 == 0.02
        assert loss.k2 == 0.06
        assert loss.reduction == "sum"
        assert loss.use_ms_ssim

    def test_initialization_invalid_window_size(self):
        """Test that invalid window size raises error."""
        with pytest.raises(ValueError, match="window_size must be odd"):
            SSIMStabilityLoss(window_size=10)

        with pytest.raises(ValueError, match="window_size must be odd and >= 3"):
            SSIMStabilityLoss(window_size=2)

    def test_initialization_invalid_sigma(self):
        """Test that invalid sigma raises error."""
        with pytest.raises(ValueError, match="sigma must be positive"):
            SSIMStabilityLoss(sigma=-1.0)

    def test_initialization_invalid_reduction(self):
        """Test that invalid reduction raises error."""
        with pytest.raises(ValueError, match="reduction must be"):
            SSIMStabilityLoss(reduction="invalid")

    def test_identical_inputs_ssim(self, identical_heatmaps: Tuple[Tensor, Tensor]):
        """Test that identical inputs give SSIM ≈ 1.0 (loss ≈ 0)."""
        heatmap1, heatmap2 = identical_heatmaps
        loss_fn = SSIMStabilityLoss()

        loss = loss_fn(heatmap1, heatmap2)

        # Loss should be very close to 0 (SSIM ≈ 1)
        assert loss.item() < 0.01, f"Expected loss ≈ 0, got {loss.item()}"

    def test_different_inputs_ssim(self, sample_heatmaps: Tuple[Tensor, Tensor]):
        """Test that different inputs give SSIM < 1.0 (loss > 0)."""
        heatmap1, heatmap2 = sample_heatmaps
        loss_fn = SSIMStabilityLoss()

        loss = loss_fn(heatmap1, heatmap2)

        # Loss should be > 0
        assert loss.item() > 0.0, f"Expected loss > 0, got {loss.item()}"
        # Loss should be bounded by 1 (SSIM in [0, 1])
        assert loss.item() <= 1.0, f"Expected loss <= 1, got {loss.item()}"

    def test_output_shape_mean_reduction(self, sample_heatmaps: Tuple[Tensor, Tensor]):
        """Test output shape with mean reduction."""
        heatmap1, heatmap2 = sample_heatmaps
        loss_fn = SSIMStabilityLoss(reduction="mean")

        loss = loss_fn(heatmap1, heatmap2)

        assert loss.dim() == 0, f"Expected scalar, got shape {loss.shape}"

    def test_output_shape_none_reduction(self, sample_heatmaps: Tuple[Tensor, Tensor]):
        """Test output shape with no reduction."""
        heatmap1, heatmap2 = sample_heatmaps
        loss_fn = SSIMStabilityLoss(reduction="none")

        loss = loss_fn(heatmap1, heatmap2)

        batch_size = heatmap1.size(0)
        assert loss.shape == (
            batch_size,
        ), f"Expected ({batch_size},), got {loss.shape}"

    def test_gradient_flow(self, sample_heatmaps: Tuple[Tensor, Tensor]):
        """Test that gradients flow correctly."""
        heatmap1, heatmap2 = sample_heatmaps
        heatmap1.requires_grad_(True)
        heatmap2.requires_grad_(True)

        loss_fn = SSIMStabilityLoss()
        loss = loss_fn(heatmap1, heatmap2)
        loss.backward()

        assert heatmap1.grad is not None, "Gradient for heatmap1 is None"
        assert heatmap2.grad is not None, "Gradient for heatmap2 is None"
        assert not torch.isnan(heatmap1.grad).any(), "NaN in heatmap1 gradients"
        assert not torch.isnan(heatmap2.grad).any(), "NaN in heatmap2 gradients"

    def test_3d_input_handling(self, device: torch.device):
        """Test that 3D inputs are handled correctly."""
        heatmap1 = torch.rand(4, 56, 56, device=device)  # (B, H, W)
        heatmap2 = torch.rand(4, 56, 56, device=device)

        loss_fn = SSIMStabilityLoss()
        loss = loss_fn(heatmap1, heatmap2)

        assert loss.dim() == 0, "Should return scalar for 3D input"

    def test_shape_mismatch_error(self, device: torch.device):
        """Test that shape mismatch raises error."""
        heatmap1 = torch.rand(4, 1, 56, 56, device=device)
        heatmap2 = torch.rand(4, 1, 32, 32, device=device)

        loss_fn = SSIMStabilityLoss()

        with pytest.raises(ValueError, match="shapes must match"):
            loss_fn(heatmap1, heatmap2)

    def test_ms_ssim(self, device: torch.device):
        """Test Multi-Scale SSIM."""
        # Need larger images for MS-SSIM
        heatmap1 = torch.rand(2, 1, 160, 160, device=device)
        heatmap2 = torch.rand(2, 1, 160, 160, device=device)

        loss_fn = SSIMStabilityLoss(use_ms_ssim=True)
        loss = loss_fn(heatmap1, heatmap2)

        assert loss.dim() == 0, "MS-SSIM should return scalar"
        assert loss.item() >= 0.0, "MS-SSIM loss should be non-negative"

    def test_gaussian_kernel(self, device: torch.device):
        """Test Gaussian kernel creation."""
        loss_fn = SSIMStabilityLoss(kernel_type="gaussian")
        heatmap1 = torch.rand(2, 1, 56, 56, device=device)
        heatmap2 = torch.rand(2, 1, 56, 56, device=device)

        loss = loss_fn(heatmap1, heatmap2)
        assert not torch.isnan(loss), "Gaussian kernel produced NaN"

    def test_uniform_kernel(self, device: torch.device):
        """Test uniform (box) kernel."""
        loss_fn = SSIMStabilityLoss(kernel_type="uniform")
        heatmap1 = torch.rand(2, 1, 56, 56, device=device)
        heatmap2 = torch.rand(2, 1, 56, 56, device=device)

        loss = loss_fn(heatmap1, heatmap2)
        assert not torch.isnan(loss), "Uniform kernel produced NaN"

    def test_symmetry(self, sample_heatmaps: Tuple[Tensor, Tensor]):
        """Test that SSIM is symmetric: SSIM(A, B) = SSIM(B, A)."""
        heatmap1, heatmap2 = sample_heatmaps
        loss_fn = SSIMStabilityLoss()

        loss_ab = loss_fn(heatmap1, heatmap2)
        loss_ba = loss_fn(heatmap2, heatmap1)

        assert torch.isclose(
            loss_ab, loss_ba, atol=1e-6
        ), f"SSIM should be symmetric: {loss_ab.item()} vs {loss_ba.item()}"


# ============================================================================
# TCavConceptLoss Tests
# ============================================================================


class TestTCavConceptLoss:
    """Tests for TCavConceptLoss class."""

    def test_initialization_default(self):
        """Test default initialization without CAVs."""
        loss = TCavConceptLoss()

        assert loss.tau_artifact == 0.3
        assert loss.tau_medical == 0.5
        assert loss.lambda_medical == 0.5
        assert loss.reduction == "mean"
        assert loss.differentiable

    def test_initialization_with_cavs(
        self,
        sample_cavs: Tuple[List[Tensor], List[Tensor]],
    ):
        """Test initialization with CAVs."""
        artifact_cavs, medical_cavs = sample_cavs

        loss = TCavConceptLoss(
            artifact_cavs=artifact_cavs,
            medical_cavs=medical_cavs,
        )

        assert len(loss._artifact_cavs) == len(artifact_cavs)
        assert len(loss._medical_cavs) == len(medical_cavs)

    def test_initialization_invalid_tau(self):
        """Test that invalid tau values raise error."""
        with pytest.raises(ValueError, match="tau_artifact must be in"):
            TCavConceptLoss(tau_artifact=1.5)

        with pytest.raises(ValueError, match="tau_medical must be in"):
            TCavConceptLoss(tau_medical=-0.1)

    def test_initialization_invalid_lambda(self):
        """Test that invalid lambda raises error."""
        with pytest.raises(ValueError, match="lambda_medical must be non-negative"):
            TCavConceptLoss(lambda_medical=-1.0)

    def test_no_cavs_warning(
        self,
        sample_activations_gradients: Tuple[Tensor, Tensor],
    ):
        """Test that empty CAVs give zero loss with warning."""
        activations, gradients = sample_activations_gradients
        loss_fn = TCavConceptLoss()

        with pytest.warns(UserWarning, match="No CAVs registered"):
            loss, metrics = loss_fn(activations, gradients)

        assert loss.item() == 0.0
        assert len(metrics) == 0

    def test_forward_with_cavs(
        self,
        sample_cavs: Tuple[List[Tensor], List[Tensor]],
        sample_activations_gradients: Tuple[Tensor, Tensor],
    ):
        """Test forward pass with CAVs."""
        artifact_cavs, medical_cavs = sample_cavs
        activations, gradients = sample_activations_gradients

        loss_fn = TCavConceptLoss(
            artifact_cavs=artifact_cavs,
            medical_cavs=medical_cavs,
        )

        loss, metrics = loss_fn(activations, gradients)

        assert isinstance(loss, Tensor), "Loss should be a tensor"
        assert loss.dim() == 0, "Loss should be a scalar"
        assert "artifact_tcav_mean" in metrics
        assert "medical_tcav_mean" in metrics
        assert "tcav_ratio" in metrics

    def test_tcav_scores_range(
        self,
        sample_cavs: Tuple[List[Tensor], List[Tensor]],
        sample_activations_gradients: Tuple[Tensor, Tensor],
    ):
        """Test that TCAV scores are in valid range."""
        artifact_cavs, medical_cavs = sample_cavs
        activations, gradients = sample_activations_gradients

        loss_fn = TCavConceptLoss(
            artifact_cavs=artifact_cavs,
            medical_cavs=medical_cavs,
        )

        _, metrics = loss_fn(activations, gradients)

        assert 0.0 <= metrics["artifact_tcav_mean"] <= 1.0
        assert 0.0 <= metrics["medical_tcav_mean"] <= 1.0

    def test_gradient_flow(
        self,
        sample_cavs: Tuple[List[Tensor], List[Tensor]],
        device: torch.device,
    ):
        """Test gradient flow through concept loss."""
        artifact_cavs, medical_cavs = sample_cavs

        activations = torch.randn(4, 2048, device=device, requires_grad=True)
        gradients = torch.randn(4, 2048, device=device, requires_grad=True)

        loss_fn = TCavConceptLoss(
            artifact_cavs=artifact_cavs,
            medical_cavs=medical_cavs,
            differentiable=True,
        )

        loss, _ = loss_fn(activations, gradients)
        loss.backward()

        assert gradients.grad is not None, "Gradient should flow to input"
        assert not torch.isnan(gradients.grad).any(), "Gradients contain NaN"

    def test_update_cavs(
        self,
        sample_cavs: Tuple[List[Tensor], List[Tensor]],
        device: torch.device,
    ):
        """Test updating CAVs after initialization."""
        loss_fn = TCavConceptLoss()

        artifact_cavs, medical_cavs = sample_cavs
        loss_fn.update_cavs(artifact_cavs=artifact_cavs, medical_cavs=medical_cavs)

        assert len(loss_fn._artifact_cavs) == len(artifact_cavs)
        assert len(loss_fn._medical_cavs) == len(medical_cavs)

    def test_soft_vs_hard_tcav(
        self,
        sample_cavs: Tuple[List[Tensor], List[Tensor]],
        sample_activations_gradients: Tuple[Tensor, Tensor],
    ):
        """Test soft (differentiable) vs hard TCAV."""
        artifact_cavs, medical_cavs = sample_cavs
        activations, gradients = sample_activations_gradients

        loss_soft = TCavConceptLoss(
            artifact_cavs=artifact_cavs,
            medical_cavs=medical_cavs,
            differentiable=True,
        )

        loss_hard = TCavConceptLoss(
            artifact_cavs=artifact_cavs,
            medical_cavs=medical_cavs,
            differentiable=False,
        )

        _, metrics_soft = loss_soft(activations, gradients)
        _, metrics_hard = loss_hard(activations, gradients)

        # Both should produce valid TCAV scores
        assert "artifact_tcav_mean" in metrics_soft
        assert "artifact_tcav_mean" in metrics_hard

    def test_penalty_when_artifact_tcav_high(
        self,
        device: torch.device,
    ):
        """Test that high artifact TCAV produces penalty."""
        feat_dim = 256

        # Create a CAV that will produce high TCAV score
        artifact_cav = torch.ones(feat_dim, device=device)

        # Create gradients aligned with CAV (high positive directional derivative)
        gradients = torch.ones(4, feat_dim, device=device)
        activations = torch.randn(4, feat_dim, device=device)

        loss_fn = TCavConceptLoss(
            artifact_cavs=[artifact_cav],
            medical_cavs=[],
            tau_artifact=0.3,  # Low threshold
        )

        loss, metrics = loss_fn(activations, gradients)

        # With aligned gradients, TCAV should be high (close to 1)
        # and penalty should be positive
        assert loss.item() > 0, "Should have penalty for high artifact TCAV"


# ============================================================================
# ExplanationLossConfig Tests
# ============================================================================


class TestExplanationLossConfig:
    """Tests for ExplanationLossConfig dataclass."""

    def test_default_config(self):
        """Test default configuration values."""
        config = ExplanationLossConfig()

        assert config.gamma == 0.5
        assert config.tau_artifact == 0.3
        assert config.tau_medical == 0.5
        assert config.lambda_medical == 0.5
        assert config.fgsm_epsilon == pytest.approx(2.0 / 255.0)
        assert not config.use_ms_ssim
        assert config.ssim_window_size == 11

    def test_custom_config(self):
        """Test custom configuration."""
        config = ExplanationLossConfig(
            gamma=0.7,
            tau_artifact=0.2,
            use_ms_ssim=True,
        )

        assert config.gamma == 0.7
        assert config.tau_artifact == 0.2
        assert config.use_ms_ssim

    def test_invalid_gamma(self):
        """Test that negative gamma raises error."""
        with pytest.raises(ValueError, match="gamma must be non-negative"):
            ExplanationLossConfig(gamma=-0.1)

    def test_invalid_tau_artifact(self):
        """Test that invalid tau_artifact raises error."""
        with pytest.raises(ValueError, match="tau_artifact must be in"):
            ExplanationLossConfig(tau_artifact=1.5)

    def test_invalid_window_size(self):
        """Test that even window size raises error."""
        with pytest.raises(ValueError, match="ssim_window_size must be odd"):
            ExplanationLossConfig(ssim_window_size=10)

    def test_invalid_reduction(self):
        """Test that invalid reduction raises error."""
        with pytest.raises(ValueError, match="reduction must be"):
            ExplanationLossConfig(reduction="invalid")


# ============================================================================
# ExplanationLoss Integration Tests
# ============================================================================


class TestExplanationLoss:
    """Integration tests for ExplanationLoss class."""

    def test_initialization(
        self,
        simple_cnn: nn.Module,
        sample_cavs: Tuple[List[Tensor], List[Tensor]],
    ):
        """Test ExplanationLoss initialization."""
        artifact_cavs, medical_cavs = sample_cavs
        config = ExplanationLossConfig()

        loss_fn = ExplanationLoss(
            model=simple_cnn,
            config=config,
            artifact_cavs=artifact_cavs,
            medical_cavs=medical_cavs,
        )

        assert loss_fn.gamma == config.gamma
        assert loss_fn.fgsm_epsilon == config.fgsm_epsilon

    def test_compute_stability_only(
        self,
        sample_heatmaps: Tuple[Tensor, Tensor],
    ):
        """Test computing only stability loss."""
        heatmap_clean, heatmap_adv = sample_heatmaps

        loss_fn = ExplanationLoss()
        loss = loss_fn.compute_stability_only(heatmap_clean, heatmap_adv)

        assert loss.dim() == 0
        assert 0.0 <= loss.item() <= 1.0

    def test_compute_concept_only(
        self,
        sample_cavs: Tuple[List[Tensor], List[Tensor]],
        sample_activations_gradients: Tuple[Tensor, Tensor],
    ):
        """Test computing only concept loss."""
        artifact_cavs, medical_cavs = sample_cavs
        activations, gradients = sample_activations_gradients

        loss_fn = ExplanationLoss(
            artifact_cavs=artifact_cavs,
            medical_cavs=medical_cavs,
        )

        loss, metrics = loss_fn.compute_concept_only(activations, gradients)

        assert isinstance(loss, Tensor)
        assert "artifact_tcav_mean" in metrics
        assert "medical_tcav_mean" in metrics

    def test_full_forward_pass(
        self,
        simple_cnn: nn.Module,
        device: torch.device,
    ):
        """Test full forward pass with model."""
        # Create CAVs matching simple_cnn output dimension (256)
        feat_dim = 256
        artifact_cavs = [torch.randn(feat_dim, device=device) for _ in range(2)]
        medical_cavs = [torch.randn(feat_dim, device=device) for _ in range(2)]

        loss_fn = ExplanationLoss(
            model=simple_cnn,
            artifact_cavs=artifact_cavs,
            medical_cavs=medical_cavs,
        )

        # Create sample input
        images = torch.rand(2, 3, 64, 64, device=device)
        labels = torch.randint(0, 10, (2,), device=device)

        loss, metrics = loss_fn(images, labels, return_components=True)

        assert isinstance(loss, Tensor)
        assert "loss_stability" in metrics
        assert "loss_concept" in metrics
        assert "loss_total" in metrics

    def test_set_model(
        self,
        simple_cnn: nn.Module,
    ):
        """Test setting model after initialization."""
        loss_fn = ExplanationLoss()

        assert loss_fn._model is None

        loss_fn.set_model(simple_cnn)

        assert loss_fn._model is simple_cnn

    def test_no_model_error(self, device: torch.device):
        """Test that forward without model raises error."""
        loss_fn = ExplanationLoss()

        images = torch.rand(2, 3, 64, 64, device=device)
        labels = torch.randint(0, 10, (2,), device=device)

        with pytest.raises(RuntimeError, match="Model not set"):
            loss_fn(images, labels)


# ============================================================================
# Factory Function Tests
# ============================================================================


class TestCreateExplanationLoss:
    """Tests for create_explanation_loss factory function."""

    def test_default_creation(self):
        """Test creating with defaults."""
        loss_fn = create_explanation_loss()

        assert isinstance(loss_fn, ExplanationLoss)
        assert loss_fn.gamma == 0.5
        assert not loss_fn.config.use_ms_ssim

    def test_custom_creation(
        self,
        simple_cnn: nn.Module,
        sample_cavs: Tuple[List[Tensor], List[Tensor]],
    ):
        """Test creating with custom parameters."""
        artifact_cavs, medical_cavs = sample_cavs

        loss_fn = create_explanation_loss(
            model=simple_cnn,
            artifact_cavs=artifact_cavs,
            medical_cavs=medical_cavs,
            gamma=0.7,
            use_ms_ssim=True,
        )

        assert loss_fn.gamma == 0.7
        assert loss_fn.config.use_ms_ssim


# ============================================================================
# Utility Function Tests
# ============================================================================


class TestUtilityFunctions:
    """Tests for utility functions."""

    def test_verify_gradient_flow(
        self,
        device: torch.device,
    ):
        """Test gradient flow verification utility."""
        # Create CAVs matching test dimension (256)
        feat_dim = 256
        artifact_cavs = [torch.randn(feat_dim, device=device) for _ in range(2)]
        medical_cavs = [torch.randn(feat_dim, device=device) for _ in range(2)]

        loss_fn = ExplanationLoss(
            artifact_cavs=artifact_cavs,
            medical_cavs=medical_cavs,
        )

        results = verify_gradient_flow(
            loss_fn, batch_size=4, image_size=56, device=device
        )

        assert "ssim_grad_flow" in results
        assert "concept_grad_flow" in results
        assert "combined_grad_flow" in results
        assert results["ssim_grad_flow"], "SSIM gradient flow failed"

    def test_benchmark_overhead(
        self,
        sample_cavs: Tuple[List[Tensor], List[Tensor]],
        device: torch.device,
    ):
        """Test computational overhead benchmark."""
        artifact_cavs, medical_cavs = sample_cavs

        loss_fn = ExplanationLoss(
            artifact_cavs=artifact_cavs,
            medical_cavs=medical_cavs,
        )

        timings = benchmark_computational_overhead(
            loss_fn,
            batch_size=4,
            image_size=56,
            num_iterations=5,
            device=device,
        )

        assert "ssim_time_ms" in timings
        assert "concept_time_ms" in timings
        assert "total_time_ms" in timings
        assert timings["total_time_ms"] > 0


# ============================================================================
# Numerical Stability Tests
# ============================================================================


class TestNumericalStability:
    """Tests for numerical stability."""

    def test_ssim_with_zeros(self, device: torch.device):
        """Test SSIM with zero inputs."""
        heatmap1 = torch.zeros(2, 1, 32, 32, device=device)
        heatmap2 = torch.zeros(2, 1, 32, 32, device=device)

        loss_fn = SSIMStabilityLoss()
        loss = loss_fn(heatmap1, heatmap2)

        assert not torch.isnan(loss), "SSIM produced NaN with zeros"
        assert not torch.isinf(loss), "SSIM produced Inf with zeros"

    def test_ssim_with_constant(self, device: torch.device):
        """Test SSIM with constant inputs."""
        heatmap1 = torch.ones(2, 1, 32, 32, device=device) * 0.5
        heatmap2 = torch.ones(2, 1, 32, 32, device=device) * 0.5

        loss_fn = SSIMStabilityLoss()
        loss = loss_fn(heatmap1, heatmap2)

        assert not torch.isnan(loss), "SSIM produced NaN with constants"
        assert loss.item() < 0.01, "Identical constant images should have SSIM ≈ 1"

    def test_ssim_with_extreme_values(self, device: torch.device):
        """Test SSIM with extreme values."""
        heatmap1 = torch.rand(2, 1, 32, 32, device=device) * 0.001  # Very small
        heatmap2 = torch.rand(2, 1, 32, 32, device=device) * 0.999  # Very large

        loss_fn = SSIMStabilityLoss()
        loss = loss_fn(heatmap1, heatmap2)

        assert not torch.isnan(loss), "SSIM produced NaN with extreme values"
        assert not torch.isinf(loss), "SSIM produced Inf with extreme values"

    def test_tcav_with_zero_gradients(
        self,
        sample_cavs: Tuple[List[Tensor], List[Tensor]],
        device: torch.device,
    ):
        """Test TCAV with zero gradients."""
        artifact_cavs, medical_cavs = sample_cavs

        activations = torch.randn(4, 2048, device=device)
        gradients = torch.zeros(4, 2048, device=device)

        loss_fn = TCavConceptLoss(
            artifact_cavs=artifact_cavs,
            medical_cavs=medical_cavs,
        )

        loss, metrics = loss_fn(activations, gradients)

        assert not torch.isnan(loss), "TCAV produced NaN with zero gradients"


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases."""

    def test_single_sample_batch(self, device: torch.device):
        """Test with batch size of 1."""
        heatmap1 = torch.rand(1, 1, 32, 32, device=device)
        heatmap2 = torch.rand(1, 1, 32, 32, device=device)

        loss_fn = SSIMStabilityLoss()
        loss = loss_fn(heatmap1, heatmap2)

        assert loss.dim() == 0

    def test_minimum_image_size(self, device: torch.device):
        """Test with minimum acceptable image size."""
        window_size = 11
        heatmap1 = torch.rand(2, 1, window_size, window_size, device=device)
        heatmap2 = torch.rand(2, 1, window_size, window_size, device=device)

        loss_fn = SSIMStabilityLoss(window_size=window_size)
        loss = loss_fn(heatmap1, heatmap2)

        assert not torch.isnan(loss)

    def test_too_small_image_error(self, device: torch.device):
        """Test that too small images raise error."""
        heatmap1 = torch.rand(2, 1, 5, 5, device=device)
        heatmap2 = torch.rand(2, 1, 5, 5, device=device)

        loss_fn = SSIMStabilityLoss(window_size=11)

        with pytest.raises(RuntimeError, match="spatial size"):
            loss_fn(heatmap1, heatmap2)

    def test_single_cav(self, device: torch.device):
        """Test with single CAV."""
        single_cav = torch.randn(256, device=device)

        loss_fn = TCavConceptLoss(
            artifact_cavs=[single_cav],
            medical_cavs=[],
        )

        activations = torch.randn(4, 256, device=device)
        gradients = torch.randn(4, 256, device=device)

        loss, metrics = loss_fn(activations, gradients)

        assert "artifact_tcav_mean" in metrics


# ============================================================================
# Performance Tests (Optional)
# ============================================================================


@pytest.mark.slow
class TestPerformance:
    """Performance tests (run with pytest --runslow)."""

    def test_large_batch_ssim(self, device: torch.device):
        """Test SSIM with large batch."""
        heatmap1 = torch.rand(32, 1, 224, 224, device=device)
        heatmap2 = torch.rand(32, 1, 224, 224, device=device)

        loss_fn = SSIMStabilityLoss()

        # Should complete without memory error
        loss = loss_fn(heatmap1, heatmap2)

        assert not torch.isnan(loss)

    def test_many_cavs(self, device: torch.device):
        """Test with many CAVs."""
        feat_dim = 2048
        num_cavs = 50

        artifact_cavs = [torch.randn(feat_dim, device=device) for _ in range(num_cavs)]
        medical_cavs = [torch.randn(feat_dim, device=device) for _ in range(num_cavs)]

        loss_fn = TCavConceptLoss(
            artifact_cavs=artifact_cavs,
            medical_cavs=medical_cavs,
        )

        activations = torch.randn(16, feat_dim, device=device)
        gradients = torch.randn(16, feat_dim, device=device)

        loss, metrics = loss_fn(activations, gradients)

        assert not torch.isnan(loss)


# ============================================================================
# Extended Coverage Tests for 100% Coverage
# ============================================================================


class TestConfigurationValidation:
    """Test configuration validation for all edge cases."""

    def test_invalid_tau_artifact_negative(self):
        """Test tau_artifact validation with negative value."""
        with pytest.raises(ValueError, match="tau_artifact must be in"):
            ExplanationLossConfig(tau_artifact=-0.1)

    def test_invalid_tau_artifact_above_one(self):
        """Test tau_artifact validation above 1."""
        with pytest.raises(ValueError, match="tau_artifact must be in"):
            ExplanationLossConfig(tau_artifact=1.5)

    def test_invalid_tau_medical_negative(self):
        """Test tau_medical validation with negative value."""
        with pytest.raises(ValueError, match="tau_medical must be in"):
            ExplanationLossConfig(tau_medical=-0.1)

    def test_invalid_tau_medical_above_one(self):
        """Test tau_medical validation above 1."""
        with pytest.raises(ValueError, match="tau_medical must be in"):
            ExplanationLossConfig(tau_medical=1.1)

    def test_invalid_lambda_medical(self):
        """Test lambda_medical validation."""
        with pytest.raises(ValueError, match="lambda_medical must be non-negative"):
            ExplanationLossConfig(lambda_medical=-0.5)

    def test_invalid_fgsm_epsilon(self):
        """Test fgsm_epsilon validation."""
        with pytest.raises(ValueError, match="fgsm_epsilon must be non-negative"):
            ExplanationLossConfig(fgsm_epsilon=-1.0)

    def test_invalid_ssim_window_size_even(self):
        """Test SSIM window size validation (even number)."""
        with pytest.raises(ValueError, match="ssim_window_size must be odd"):
            ExplanationLossConfig(ssim_window_size=10)

    def test_invalid_ssim_window_size_small(self):
        """Test SSIM window size validation (too small)."""
        with pytest.raises(ValueError, match="ssim_window_size must be odd and >= 3"):
            ExplanationLossConfig(ssim_window_size=1)

    def test_invalid_ssim_sigma_zero(self):
        """Test SSIM sigma validation with zero."""
        with pytest.raises(ValueError, match="ssim_sigma must be positive"):
            ExplanationLossConfig(ssim_sigma=0)

    def test_invalid_ssim_sigma_negative(self):
        """Test SSIM sigma validation with negative value."""
        with pytest.raises(ValueError, match="ssim_sigma must be positive"):
            ExplanationLossConfig(ssim_sigma=-1.5)

    def test_invalid_reduction(self):
        """Test reduction validation."""
        with pytest.raises(ValueError, match="reduction must be"):
            ExplanationLossConfig(reduction="invalid")

    def test_invalid_soft_temperature_zero(self):
        """Test soft_temperature validation with zero."""
        with pytest.raises(ValueError, match="soft_temperature must be positive"):
            ExplanationLossConfig(soft_temperature=0)

    def test_invalid_soft_temperature_negative(self):
        """Test soft_temperature validation with negative."""
        with pytest.raises(ValueError, match="soft_temperature must be positive"):
            ExplanationLossConfig(soft_temperature=-5.0)

    def test_tcav_reduction_validation(self):
        """Test TCavConceptLoss reduction validation."""
        with pytest.raises(ValueError, match="reduction must be"):
            TCavConceptLoss(reduction="invalid")

    def test_tcav_soft_temperature_validation(self):
        """Test TCavConceptLoss soft_temperature validation."""
        with pytest.raises(ValueError, match="soft_temperature must be positive"):
            TCavConceptLoss(soft_temperature=0)


class TestSSIMReductionModes:
    """Test SSIM with different reduction modes."""

    def test_ssim_reduction_sum(self, device: torch.device):
        """Test SSIM with reduction='sum'."""
        loss_fn = SSIMStabilityLoss(reduction="sum")

        heatmap1 = torch.randn(4, 1, 56, 56, device=device, requires_grad=True)
        heatmap2 = torch.randn(4, 1, 56, 56, device=device, requires_grad=True)

        loss = loss_fn(heatmap1, heatmap2)

        assert isinstance(loss, Tensor)
        assert loss.dim() == 0  # Scalar
        assert loss.requires_grad

    def test_ssim_reduction_none(self, device: torch.device):
        """Test SSIM with reduction='none'."""
        loss_fn = SSIMStabilityLoss(reduction="none")

        batch_size = 4
        heatmap1 = torch.randn(batch_size, 1, 56, 56, device=device, requires_grad=True)
        heatmap2 = torch.randn(batch_size, 1, 56, 56, device=device, requires_grad=True)

        loss = loss_fn(heatmap1, heatmap2)

        assert isinstance(loss, Tensor)
        assert loss.shape == (batch_size,)  # Per-sample scores
        assert loss.requires_grad


class TestMSSSIMScales:
    """Test MS-SSIM with different scales."""

    def test_ms_ssim_forward(self, device: torch.device):
        """Test MS-SSIM forward pass."""
        loss_fn = SSIMStabilityLoss(use_ms_ssim=True)

        # Use larger image for MS-SSIM (needs to be downscalable)
        heatmap1 = torch.randn(2, 1, 224, 224, device=device, requires_grad=True)
        heatmap2 = torch.randn(2, 1, 224, 224, device=device, requires_grad=True)

        loss = loss_fn(heatmap1, heatmap2)

        assert isinstance(loss, Tensor)
        assert 0 <= loss.item() <= 1
        assert loss.requires_grad


class TestCAVUpdateEdgeCases:
    """Test CAV update with various edge cases."""

    def test_update_only_artifact_cavs(self, device: torch.device):
        """Test updating only artifact CAVs."""
        feat_dim = 256
        artifact_cavs = [torch.randn(feat_dim, device=device) for _ in range(2)]
        medical_cavs = [torch.randn(feat_dim, device=device) for _ in range(2)]

        loss_fn = TCavConceptLoss(
            artifact_cavs=artifact_cavs, medical_cavs=medical_cavs
        )

        # Update only artifact CAVs
        new_artifact = [torch.randn(feat_dim, device=device) for _ in range(3)]
        loss_fn.update_cavs(artifact_cavs=new_artifact, medical_cavs=None)

        assert len(loss_fn._artifact_cavs) == 3
        assert len(loss_fn._medical_cavs) == 2  # Unchanged

    def test_update_only_medical_cavs(self, device: torch.device):
        """Test updating only medical CAVs."""
        feat_dim = 256
        artifact_cavs = [torch.randn(feat_dim, device=device) for _ in range(2)]
        medical_cavs = [torch.randn(feat_dim, device=device) for _ in range(2)]

        loss_fn = TCavConceptLoss(
            artifact_cavs=artifact_cavs, medical_cavs=medical_cavs
        )

        # Update only medical CAVs
        new_medical = [torch.randn(feat_dim, device=device) for _ in range(4)]
        loss_fn.update_cavs(artifact_cavs=None, medical_cavs=new_medical)

        assert len(loss_fn._artifact_cavs) == 2  # Unchanged
        assert len(loss_fn._medical_cavs) == 4

    def test_update_both_cavs(self, device: torch.device):
        """Test updating both artifact and medical CAVs."""
        feat_dim = 256
        artifact_cavs = [torch.randn(feat_dim, device=device) for _ in range(2)]
        medical_cavs = [torch.randn(feat_dim, device=device) for _ in range(2)]

        loss_fn = TCavConceptLoss(
            artifact_cavs=artifact_cavs, medical_cavs=medical_cavs
        )

        # Update both
        new_artifact = [torch.randn(feat_dim, device=device) for _ in range(3)]
        new_medical = [torch.randn(feat_dim, device=device) for _ in range(5)]
        loss_fn.update_cavs(artifact_cavs=new_artifact, medical_cavs=new_medical)

        assert len(loss_fn._artifact_cavs) == 3
        assert len(loss_fn._medical_cavs) == 5


class TestTCAVValidationErrors:
    """Test TCAV input validation errors."""

    def test_shape_mismatch_error(self, device: torch.device):
        """Test error when activations and gradients shapes don't match."""
        feat_dim = 256
        cavs = [torch.randn(feat_dim, device=device) for _ in range(2)]
        loss_fn = TCavConceptLoss(artifact_cavs=cavs)

        activations = torch.randn(8, feat_dim, device=device)
        # Different batch size
        gradients = torch.randn(4, feat_dim, device=device)

        with pytest.raises(ValueError, match="must have same shape"):
            loss_fn(activations, gradients)

    def test_wrong_dimensions_error(self, device: torch.device):
        """Test error when inputs are not 2D."""
        feat_dim = 256
        cavs = [torch.randn(feat_dim, device=device) for _ in range(2)]
        loss_fn = TCavConceptLoss(artifact_cavs=cavs)

        # 3D tensor instead of 2D
        activations = torch.randn(8, feat_dim, 1, device=device)
        gradients = torch.randn(8, feat_dim, 1, device=device)

        with pytest.raises(ValueError, match="Expected 2D activations"):
            loss_fn(activations, gradients)


class TestTCAVReductionAndMetrics:
    """Test TCAV with different reduction modes and metrics."""

    def test_tcav_reduction_sum(self, device: torch.device):
        """Test TCAV with reduction='sum'."""
        feat_dim = 256
        artifact_cavs = [torch.randn(feat_dim, device=device) for _ in range(2)]
        medical_cavs = [torch.randn(feat_dim, device=device) for _ in range(2)]

        loss_fn = TCavConceptLoss(
            artifact_cavs=artifact_cavs,
            medical_cavs=medical_cavs,
            reduction="sum",
        )

        activations = torch.randn(8, feat_dim, device=device, requires_grad=True)
        gradients = torch.randn(8, feat_dim, device=device)

        loss, metrics = loss_fn(activations, gradients)

        assert isinstance(loss, Tensor)
        assert isinstance(metrics, dict)
        # With sum reduction, loss is sum of all concept losses
        assert loss.item() >= 0

    def test_tcav_metrics_with_zero_artifact(self, device: torch.device):
        """Test TCAV metrics when artifact scores are zero."""
        feat_dim = 256
        artifact_cavs = [torch.randn(feat_dim, device=device) for _ in range(2)]
        medical_cavs = [torch.randn(feat_dim, device=device) for _ in range(2)]

        loss_fn = TCavConceptLoss(
            artifact_cavs=artifact_cavs,
            medical_cavs=medical_cavs,
            tau_artifact=0.0,
        )

        # Create zero gradients to get low TCAV scores
        activations = torch.randn(8, feat_dim, device=device, requires_grad=True)
        gradients = torch.zeros(8, feat_dim, device=device)

        loss, metrics = loss_fn(activations, gradients)

        assert "artifact_tcav_mean" in metrics
        assert "medical_tcav_mean" in metrics
        if "tcav_ratio" in metrics:
            assert isinstance(metrics["tcav_ratio"], float)


class TestModelLayerDetection:
    """Test model layer detection and error handling."""

    def test_invalid_layer_name(self, device: torch.device):
        """Test error when specified layer doesn't exist."""
        model = nn.Sequential(
            nn.Conv2d(3, 64, 3),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(128, 10),
        ).to(device)

        config = ExplanationLossConfig(gradcam_target_layer="nonexistent.layer")
        artifact_cavs = [torch.randn(128, device=device) for _ in range(2)]
        medical_cavs = [torch.randn(128, device=device) for _ in range(2)]

        loss_fn = ExplanationLoss(
            model=model,
            config=config,
            artifact_cavs=artifact_cavs,
            medical_cavs=medical_cavs,
        )

        images = torch.randn(2, 3, 56, 56, device=device)
        labels = torch.randint(0, 10, (2,), device=device)

        with pytest.raises(RuntimeError, match="Layer .* not found in model"):
            loss_fn(images, labels)

    def test_no_conv_layers_error(self, device: torch.device):
        """Test error when model has no convolutional layers."""
        model = nn.Sequential(
            nn.Flatten(), nn.Linear(3 * 56 * 56, 128), nn.Linear(128, 10)
        ).to(device)

        artifact_cavs = [torch.randn(128, device=device) for _ in range(2)]
        medical_cavs = [torch.randn(128, device=device) for _ in range(2)]

        loss_fn = ExplanationLoss(
            model=model, artifact_cavs=artifact_cavs, medical_cavs=medical_cavs
        )

        images = torch.randn(2, 3, 56, 56, device=device)
        labels = torch.randint(0, 10, (2,), device=device)

        with pytest.raises(RuntimeError, match="No convolutional layers found"):
            loss_fn(images, labels)


class TestModelNotSetErrors:
    """Test errors when model is not set."""

    def test_gradcam_without_model(self, device: torch.device):
        """Test error when generating Grad-CAM without model."""
        artifact_cavs = [torch.randn(256, device=device) for _ in range(2)]
        medical_cavs = [torch.randn(256, device=device) for _ in range(2)]

        loss_fn = ExplanationLoss(
            artifact_cavs=artifact_cavs, medical_cavs=medical_cavs
        )

        images = torch.randn(2, 3, 56, 56, device=device)
        labels = torch.randint(0, 10, (2,), device=device)

        with pytest.raises(RuntimeError, match="Model not set"):
            loss_fn._generate_gradcam_heatmap(images, labels)

    def test_extract_features_without_model(self, device: torch.device):
        """Test error when extracting features without model."""
        artifact_cavs = [torch.randn(256, device=device) for _ in range(2)]
        medical_cavs = [torch.randn(256, device=device) for _ in range(2)]

        loss_fn = ExplanationLoss(
            artifact_cavs=artifact_cavs, medical_cavs=medical_cavs
        )

        images = torch.randn(2, 3, 56, 56, device=device)
        labels = torch.randint(0, 10, (2,), device=device)

        with pytest.raises(RuntimeError, match="Model not set"):
            loss_fn._extract_features_for_tcav(images, labels)


class TestTrainingStateRestoration:
    """Test that model training state is properly restored."""

    def test_training_state_preserved(self, simple_cnn, device: torch.device):
        """Test that model training state is restored after forward pass."""
        feat_dim = 256
        artifact_cavs = [torch.randn(feat_dim, device=device) for _ in range(2)]
        medical_cavs = [torch.randn(feat_dim, device=device) for _ in range(2)]

        simple_cnn.train()
        assert simple_cnn.training

        loss_fn = ExplanationLoss(
            model=simple_cnn, artifact_cavs=artifact_cavs, medical_cavs=medical_cavs
        )

        images = torch.randn(4, 3, 56, 56, device=device)
        labels = torch.randint(0, 10, (4,), device=device)

        loss = loss_fn(images, labels)

        # Model should be back to training mode
        assert simple_cnn.training

    def test_eval_state_preserved(self, simple_cnn, device: torch.device):
        """Test that model eval state is preserved."""
        feat_dim = 256
        artifact_cavs = [torch.randn(feat_dim, device=device) for _ in range(2)]
        medical_cavs = [torch.randn(feat_dim, device=device) for _ in range(2)]

        simple_cnn.eval()
        assert not simple_cnn.training

        loss_fn = ExplanationLoss(
            model=simple_cnn, artifact_cavs=artifact_cavs, medical_cavs=medical_cavs
        )

        images = torch.randn(4, 3, 56, 56, device=device)
        labels = torch.randint(0, 10, (4,), device=device)

        loss = loss_fn(images, labels)

        # Model should still be in eval mode
        assert not simple_cnn.training


class TestGradientFlowEdgeCases:
    """Test gradient flow verification edge cases."""

    def test_gradient_flow_no_model(self, device: torch.device):
        """Test gradient flow when no model is set."""
        feat_dim = 256
        artifact_cavs = [torch.randn(feat_dim, device=device) for _ in range(2)]
        medical_cavs = [torch.randn(feat_dim, device=device) for _ in range(2)]

        loss_fn = ExplanationLoss(
            artifact_cavs=artifact_cavs, medical_cavs=medical_cavs
        )

        results = verify_gradient_flow(
            loss_fn, batch_size=4, image_size=56, device=device
        )

        assert results["ssim_grad_flow"]
        assert results["concept_grad_flow"]
        assert not results["combined_grad_flow"]

    def test_gradient_flow_no_cavs(self, simple_cnn, device: torch.device):
        """Test gradient flow when no CAVs are provided."""
        loss_fn = ExplanationLoss(model=simple_cnn)

        results = verify_gradient_flow(
            loss_fn, batch_size=4, image_size=56, device=device
        )

        assert results["ssim_grad_flow"]
        assert results["concept_grad_flow"]
        assert isinstance(results["combined_grad_flow"], bool)


class TestBenchmarkCUDASync:
    """Test benchmark with CUDA synchronization."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_benchmark_with_cuda(self):
        """Test benchmark on CUDA device."""
        device = torch.device("cuda")
        feat_dim = 2048
        artifact_cavs = [torch.randn(feat_dim, device=device) for _ in range(4)]
        medical_cavs = [torch.randn(feat_dim, device=device) for _ in range(6)]

        loss_fn = ExplanationLoss(
            artifact_cavs=artifact_cavs, medical_cavs=medical_cavs
        )

        results = benchmark_computational_overhead(
            loss_fn, batch_size=8, image_size=224, num_iterations=5, device=device
        )

        assert "ssim_time_ms" in results
        assert "concept_time_ms" in results
        assert results["ssim_time_ms"] > 0
        assert results["concept_time_ms"] > 0

    def test_benchmark_with_cpu(self, device: torch.device):
        """Test benchmark on CPU device."""
        if device.type == "cuda":
            device = torch.device("cpu")

        feat_dim = 2048
        artifact_cavs = [torch.randn(feat_dim, device=device) for _ in range(4)]
        medical_cavs = [torch.randn(feat_dim, device=device) for _ in range(6)]

        loss_fn = ExplanationLoss(
            artifact_cavs=artifact_cavs, medical_cavs=medical_cavs
        )

        results = benchmark_computational_overhead(
            loss_fn, batch_size=8, image_size=224, num_iterations=5, device=device
        )

        assert "ssim_time_ms" in results
        assert "concept_time_ms" in results
        assert results["ssim_time_ms"] > 0
        assert results["concept_time_ms"] > 0


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
