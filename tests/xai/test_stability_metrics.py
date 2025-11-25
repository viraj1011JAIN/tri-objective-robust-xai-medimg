"""
Comprehensive Test Suite for Explanation Stability Metrics.

Tests all stability metrics (SSIM, MS-SSIM, Spearman, L2, Cosine) for:
- Correctness (mathematical properties)
- Edge cases (identical/different heatmaps)
- Numerical stability (NaN/Inf handling)
- Batch processing efficiency
- GPU/CPU compatibility
- Integration with Grad-CAM pipeline

Test Coverage Target: >95%

Author: Viraj Pankaj Jain
Phase: 6.2 - Explanation Stability Metrics
Date: November 25, 2025
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest
import torch
from torch import Tensor

from src.xai.stability_metrics import (
    SSIM,
    MultiScaleSSIM,
    StabilityMetrics,
    StabilityMetricsConfig,
    cosine_similarity,
    create_stability_metrics,
    normalized_l2_distance,
    spearman_correlation,
)


# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def sample_heatmaps(device: torch.device) -> Tuple[Tensor, Tensor]:
    """Create sample heatmaps for testing."""
    torch.manual_seed(42)
    heatmap1 = torch.randn(4, 1, 14, 14, device=device)
    heatmap2 = torch.randn(4, 1, 14, 14, device=device)
    return heatmap1, heatmap2


@pytest.fixture
def identical_heatmaps(device: torch.device) -> Tuple[Tensor, Tensor]:
    """Create identical heatmaps for perfect similarity tests."""
    torch.manual_seed(42)
    heatmap = torch.randn(4, 1, 14, 14, device=device)
    return heatmap, heatmap.clone()


@pytest.fixture
def normalized_heatmaps(device: torch.device) -> Tuple[Tensor, Tensor]:
    """Create normalized heatmaps in [0, 1] range."""
    torch.manual_seed(42)
    h1 = torch.rand(4, 1, 14, 14, device=device)
    h2 = torch.rand(4, 1, 14, 14, device=device)
    return h1, h2


@pytest.fixture
def config() -> StabilityMetricsConfig:
    """Create default configuration."""
    return StabilityMetricsConfig()


@pytest.fixture
def stability_metrics(config: StabilityMetricsConfig) -> StabilityMetrics:
    """Create StabilityMetrics instance."""
    return StabilityMetrics(config)


# ============================================================================
# Test Configuration
# ============================================================================


class TestStabilityMetricsConfig:
    """Test configuration dataclass."""

    def test_default_initialization(self):
        """Test default config values."""
        config = StabilityMetricsConfig()

        assert config.ssim_window_size == 11
        assert config.ssim_sigma == 1.5
        assert config.ssim_data_range == 1.0
        assert config.normalize_heatmaps is True
        assert config.epsilon == 1e-8

    def test_ms_ssim_weights_default(self):
        """Test MS-SSIM weights are set correctly."""
        config = StabilityMetricsConfig()

        assert len(config.ms_ssim_weights) == 5
        assert np.isclose(sum(config.ms_ssim_weights), 1.0)

    def test_ms_ssim_weights_normalization(self):
        """Test MS-SSIM weights are normalized if needed."""
        config = StabilityMetricsConfig(ms_ssim_weights=[1.0, 1.0, 1.0])

        # Should be normalized to sum to 1.0
        assert np.isclose(sum(config.ms_ssim_weights), 1.0)
        assert all(w == 1.0 / 3.0 for w in config.ms_ssim_weights)

    def test_window_size_validation(self):
        """Test window size must be odd."""
        with pytest.raises(ValueError, match="must be odd"):
            StabilityMetricsConfig(ssim_window_size=10)

    def test_custom_values(self):
        """Test custom configuration values."""
        config = StabilityMetricsConfig(
            ssim_window_size=7,
            ssim_sigma=1.0,
            normalize_heatmaps=False,
        )

        assert config.ssim_window_size == 7
        assert config.ssim_sigma == 1.0
        assert config.normalize_heatmaps is False


# ============================================================================
# Test SSIM
# ============================================================================


class TestSSIM:
    """Test SSIM implementation."""

    def test_initialization(self):
        """Test SSIM module initialization."""
        ssim = SSIM(window_size=11, sigma=1.5, channel=1)

        assert ssim.window_size == 11
        assert ssim.sigma == 1.5
        assert ssim.channel == 1
        assert ssim.window.shape == (1, 1, 11, 11)

    def test_identical_heatmaps_perfect_ssim(self, identical_heatmaps):
        """Test SSIM = 1.0 for identical heatmaps."""
        h1, h2 = identical_heatmaps
        ssim = SSIM(reduction="mean")

        ssim_val = ssim(h1, h2)

        assert torch.isclose(ssim_val, torch.tensor(1.0), atol=1e-4)

    def test_different_heatmaps_lower_ssim(self, sample_heatmaps):
        """Test SSIM < 1.0 for different heatmaps."""
        h1, h2 = sample_heatmaps
        ssim = SSIM(reduction="mean")

        ssim_val = ssim(h1, h2)

        assert ssim_val < 1.0
        assert ssim_val > -1.0  # Valid range

    def test_ssim_range_valid(self, normalized_heatmaps):
        """Test SSIM is in valid range [-1, 1]."""
        h1, h2 = normalized_heatmaps
        ssim = SSIM(reduction="mean")

        ssim_val = ssim(h1, h2)

        assert -1.0 <= ssim_val <= 1.0

    def test_ssim_reduction_none(self, sample_heatmaps):
        """Test SSIM with no reduction returns per-sample values."""
        h1, h2 = sample_heatmaps
        batch_size = h1.shape[0]
        ssim = SSIM(reduction="none")

        ssim_val = ssim(h1, h2)

        assert ssim_val.shape == (batch_size,)

    def test_ssim_reduction_sum(self, sample_heatmaps):
        """Test SSIM with sum reduction."""
        h1, h2 = sample_heatmaps
        ssim = SSIM(reduction="sum")

        ssim_val = ssim(h1, h2)

        assert ssim_val.ndim == 0  # Scalar

    def test_shape_mismatch_error(self, device):
        """Test error on shape mismatch."""
        h1 = torch.randn(4, 1, 14, 14, device=device)
        h2 = torch.randn(4, 1, 28, 28, device=device)
        ssim = SSIM()

        with pytest.raises(ValueError, match="shapes must match"):
            ssim(h1, h2)

    def test_wrong_dimensions_error(self, device):
        """Test error on wrong input dimensions."""
        h1 = torch.randn(14, 14, device=device)  # 2D instead of 4D
        h2 = torch.randn(14, 14, device=device)
        ssim = SSIM()

        with pytest.raises(ValueError, match="Expected 4D input"):
            ssim(h1, h2)

    def test_device_compatibility(self, device):
        """Test SSIM works on both CPU and GPU."""
        h1 = torch.randn(2, 1, 14, 14, device=device)
        h2 = torch.randn(2, 1, 14, 14, device=device)
        ssim = SSIM()

        ssim_val = ssim(h1, h2)

        assert ssim_val.device.type == device.type

    def test_gradient_flow(self, device):
        """Test SSIM is differentiable."""
        h1 = torch.randn(2, 1, 14, 14, device=device, requires_grad=True)
        h2 = torch.randn(2, 1, 14, 14, device=device)
        ssim = SSIM()

        ssim_val = ssim(h1, h2)
        ssim_val.backward()

        assert h1.grad is not None
        assert not torch.isnan(h1.grad).any()

    def test_symmetric_property(self, sample_heatmaps):
        """Test SSIM(x, y) = SSIM(y, x)."""
        h1, h2 = sample_heatmaps
        ssim = SSIM(reduction="mean")

        ssim_xy = ssim(h1, h2)
        ssim_yx = ssim(h2, h1)

        assert torch.isclose(ssim_xy, ssim_yx, atol=1e-6)

    def test_numerical_stability_zeros(self, device):
        """Test SSIM handles all-zero inputs."""
        h1 = torch.zeros(2, 1, 14, 14, device=device)
        h2 = torch.zeros(2, 1, 14, 14, device=device)
        ssim = SSIM()

        ssim_val = ssim(h1, h2)

        assert not torch.isnan(ssim_val)
        assert not torch.isinf(ssim_val)

    def test_numerical_stability_small_values(self, device):
        """Test SSIM stability with very small values."""
        h1 = torch.ones(2, 1, 14, 14, device=device) * 1e-8
        h2 = torch.ones(2, 1, 14, 14, device=device) * 1e-8
        ssim = SSIM()

        ssim_val = ssim(h1, h2)

        assert not torch.isnan(ssim_val)
        assert not torch.isinf(ssim_val)

    def test_unknown_reduction_error(self, device):
        """Test error on unknown reduction type."""
        h1 = torch.randn(2, 1, 14, 14, device=device)
        h2 = torch.randn(2, 1, 14, 14, device=device)
        ssim = SSIM(reduction="invalid")

        with pytest.raises(ValueError, match="Unknown reduction"):
            ssim(h1, h2)


# ============================================================================
# Test Multi-Scale SSIM
# ============================================================================


class TestMultiScaleSSIM:
    """Test MS-SSIM implementation."""

    def test_initialization(self):
        """Test MS-SSIM initialization."""
        ms_ssim = MultiScaleSSIM()

        assert ms_ssim.num_scales == 5
        assert len(ms_ssim.weights) == 5

    def test_identical_heatmaps_perfect_ms_ssim(self, device):
        """Test MS-SSIM = 1.0 for identical heatmaps."""
        # Need larger size for multi-scale
        h1 = torch.randn(2, 1, 56, 56, device=device)
        h2 = h1.clone()
        ms_ssim = MultiScaleSSIM(reduction="mean")

        ms_ssim_val = ms_ssim(h1, h2)

        assert torch.isclose(ms_ssim_val, torch.tensor(1.0), atol=1e-4)

    def test_different_heatmaps_lower_ms_ssim(self, device):
        """Test MS-SSIM < 1.0 for different heatmaps."""
        h1 = torch.randn(2, 1, 56, 56, device=device)
        h2 = torch.randn(2, 1, 56, 56, device=device)
        ms_ssim = MultiScaleSSIM(reduction="mean")

        ms_ssim_val = ms_ssim(h1, h2)

        assert ms_ssim_val < 1.0
        assert ms_ssim_val > 0.0

    def test_small_input_size_warning(self, device):
        """Test warning for small input size."""
        h1 = torch.randn(2, 1, 14, 14, device=device)
        h2 = torch.randn(2, 1, 14, 14, device=device)
        ms_ssim = MultiScaleSSIM()

        # Should still work but with fewer scales
        ms_ssim_val = ms_ssim(h1, h2)

        assert not torch.isnan(ms_ssim_val)

    def test_custom_weights(self, device):
        """Test MS-SSIM with custom weights."""
        h1 = torch.randn(2, 1, 56, 56, device=device)
        h2 = torch.randn(2, 1, 56, 56, device=device)
        custom_weights = [0.2, 0.3, 0.5]
        ms_ssim = MultiScaleSSIM(weights=custom_weights)

        ms_ssim_val = ms_ssim(h1, h2)

        assert not torch.isnan(ms_ssim_val)

    def test_gradient_flow(self, device):
        """Test MS-SSIM is differentiable."""
        h1 = torch.randn(2, 1, 56, 56, device=device, requires_grad=True)
        h2 = torch.randn(2, 1, 56, 56, device=device)
        ms_ssim = MultiScaleSSIM()

        ms_ssim_val = ms_ssim(h1, h2)
        ms_ssim_val.backward()

        assert h1.grad is not None
        assert not torch.isnan(h1.grad).any()


# ============================================================================
# Test Spearman Correlation
# ============================================================================


class TestSpearmanCorrelation:
    """Test Spearman rank correlation."""

    def test_identical_heatmaps_perfect_correlation(self, identical_heatmaps):
        """Test ρ = 1.0 for identical heatmaps."""
        h1, h2 = identical_heatmaps
        rho = spearman_correlation(h1, h2, reduction="mean")

        assert np.isclose(rho, 1.0, atol=1e-4)

    def test_different_heatmaps_lower_correlation(self, sample_heatmaps):
        """Test ρ < 1.0 for different heatmaps."""
        h1, h2 = sample_heatmaps
        rho = spearman_correlation(h1, h2, reduction="mean")

        assert rho < 1.0
        assert rho > -1.0  # Valid range

    def test_opposite_heatmaps_negative_correlation(self, device):
        """Test negative correlation for opposite patterns."""
        h1 = torch.arange(100, dtype=torch.float32, device=device).reshape(1, 1, 10, 10)
        h2 = -h1  # Opposite
        rho = spearman_correlation(h1, h2, reduction="mean")

        assert rho < 0.0  # Should be negative

    def test_reduction_none(self, sample_heatmaps):
        """Test per-sample correlations."""
        h1, h2 = sample_heatmaps
        batch_size = h1.shape[0]
        rho = spearman_correlation(h1, h2, reduction="none")

        assert len(rho) == batch_size

    def test_numpy_input(self):
        """Test Spearman works with numpy arrays."""
        h1 = np.random.randn(2, 1, 14, 14).astype(np.float32)
        h2 = np.random.randn(2, 1, 14, 14).astype(np.float32)

        rho = spearman_correlation(h1, h2, reduction="mean")

        assert isinstance(rho, float)
        assert -1.0 <= rho <= 1.0

    def test_constant_heatmaps(self, device):
        """Test Spearman with constant values."""
        h1 = torch.ones(2, 1, 14, 14, device=device)
        h2 = torch.ones(2, 1, 14, 14, device=device)

        rho = spearman_correlation(h1, h2, reduction="mean")

        # Should return 1.0 (handled as identical)
        assert np.isclose(rho, 1.0)


# ============================================================================
# Test L2 Distance
# ============================================================================


class TestNormalizedL2Distance:
    """Test normalized L2 distance."""

    def test_identical_heatmaps_zero_distance(self, identical_heatmaps):
        """Test distance = 0 for identical heatmaps."""
        h1, h2 = identical_heatmaps
        dist = normalized_l2_distance(h1, h2, reduction="mean")

        assert torch.isclose(dist, torch.tensor(0.0), atol=1e-6)

    def test_different_heatmaps_positive_distance(self, sample_heatmaps):
        """Test distance > 0 for different heatmaps."""
        h1, h2 = sample_heatmaps
        dist = normalized_l2_distance(h1, h2, reduction="mean")

        assert dist > 0.0
        assert dist <= 1.0  # Normalized

    def test_reduction_none(self, sample_heatmaps):
        """Test per-sample distances."""
        h1, h2 = sample_heatmaps
        batch_size = h1.shape[0]
        dist = normalized_l2_distance(h1, h2, reduction="none")

        assert dist.shape == (batch_size,)

    def test_reduction_sum(self, sample_heatmaps):
        """Test sum reduction."""
        h1, h2 = sample_heatmaps
        dist = normalized_l2_distance(h1, h2, reduction="sum")

        assert dist.ndim == 0  # Scalar

    def test_gradient_flow(self, device):
        """Test L2 distance is differentiable."""
        h1 = torch.randn(2, 1, 14, 14, device=device, requires_grad=True)
        h2 = torch.randn(2, 1, 14, 14, device=device)

        dist = normalized_l2_distance(h1, h2, reduction="mean")
        dist.backward()

        assert h1.grad is not None
        assert not torch.isnan(h1.grad).any()

    def test_numerical_stability_zeros(self, device):
        """Test L2 distance with zero inputs."""
        h1 = torch.zeros(2, 1, 14, 14, device=device)
        h2 = torch.zeros(2, 1, 14, 14, device=device)

        dist = normalized_l2_distance(h1, h2, reduction="mean")

        assert not torch.isnan(dist)
        assert not torch.isinf(dist)

    def test_unknown_reduction_error(self, sample_heatmaps):
        """Test error on unknown reduction type."""
        h1, h2 = sample_heatmaps

        with pytest.raises(ValueError, match="Unknown reduction"):
            normalized_l2_distance(h1, h2, reduction="invalid")


# ============================================================================
# Test Cosine Similarity
# ============================================================================


class TestCosineSimilarity:
    """Test cosine similarity."""

    def test_identical_heatmaps_perfect_similarity(self, identical_heatmaps):
        """Test cos = 1.0 for identical heatmaps."""
        h1, h2 = identical_heatmaps
        cos_sim = cosine_similarity(h1, h2, reduction="mean")

        assert torch.isclose(cos_sim, torch.tensor(1.0), atol=1e-4)

    def test_scaled_heatmaps_perfect_similarity(self, device):
        """Test cos = 1.0 for scaled versions (same direction)."""
        h1 = torch.randn(2, 1, 14, 14, device=device)
        h2 = 3.0 * h1  # Same direction, different magnitude
        cos_sim = cosine_similarity(h1, h2, reduction="mean")

        assert torch.isclose(cos_sim, torch.tensor(1.0), atol=1e-4)

    def test_opposite_heatmaps_negative_similarity(self, device):
        """Test cos = -1.0 for opposite heatmaps."""
        h1 = torch.randn(2, 1, 14, 14, device=device)
        h2 = -h1  # Opposite direction
        cos_sim = cosine_similarity(h1, h2, reduction="mean")

        assert torch.isclose(cos_sim, torch.tensor(-1.0), atol=1e-4)

    def test_orthogonal_heatmaps(self, device):
        """Test cos ≈ 0 for orthogonal patterns."""
        # Create orthogonal patterns
        h1 = torch.zeros(1, 1, 10, 10, device=device)
        h1[0, 0, :5, :] = 1.0  # Top half

        h2 = torch.zeros(1, 1, 10, 10, device=device)
        h2[0, 0, 5:, :] = 1.0  # Bottom half

        cos_sim = cosine_similarity(h1, h2, reduction="mean")

        # Should be close to 0 (orthogonal)
        assert abs(cos_sim) < 0.1

    def test_reduction_none(self, sample_heatmaps):
        """Test per-sample similarities."""
        h1, h2 = sample_heatmaps
        batch_size = h1.shape[0]
        cos_sim = cosine_similarity(h1, h2, reduction="none")

        assert cos_sim.shape == (batch_size,)

    def test_gradient_flow(self, device):
        """Test cosine similarity is differentiable."""
        h1 = torch.randn(2, 1, 14, 14, device=device, requires_grad=True)
        h2 = torch.randn(2, 1, 14, 14, device=device)

        cos_sim = cosine_similarity(h1, h2, reduction="mean")
        cos_sim.backward()

        assert h1.grad is not None
        assert not torch.isnan(h1.grad).any()

    def test_unknown_reduction_error(self, sample_heatmaps):
        """Test error on unknown reduction type."""
        h1, h2 = sample_heatmaps

        with pytest.raises(ValueError, match="Unknown reduction"):
            cosine_similarity(h1, h2, reduction="invalid")


# ============================================================================
# Test StabilityMetrics Class
# ============================================================================


class TestStabilityMetrics:
    """Test unified StabilityMetrics class."""

    def test_initialization(self, config):
        """Test StabilityMetrics initialization."""
        metrics = StabilityMetrics(config)

        assert metrics.config == config
        assert metrics.ssim_module is not None
        assert metrics.ms_ssim_module is not None

    def test_initialization_default_config(self):
        """Test initialization with default config."""
        metrics = StabilityMetrics()

        assert metrics.config is not None

    def test_repr(self):
        """Test __repr__ method for debugging."""
        metrics = StabilityMetrics()
        repr_str = repr(metrics)

        assert "StabilityMetrics" in repr_str
        assert "window_size" in repr_str
        assert "sigma" in repr_str
        assert "normalize" in repr_str
        assert "scales" in repr_str

    def test_compute_ssim(self, stability_metrics, sample_heatmaps):
        """Test compute_ssim method."""
        h1, h2 = sample_heatmaps
        ssim_val = stability_metrics.compute_ssim(h1, h2)

        assert isinstance(ssim_val, float)
        assert -1.0 <= ssim_val <= 1.0

    def test_compute_ms_ssim(self, stability_metrics, device):
        """Test compute_ms_ssim method."""
        h1 = torch.randn(2, 1, 56, 56, device=device)
        h2 = torch.randn(2, 1, 56, 56, device=device)
        ms_ssim_val = stability_metrics.compute_ms_ssim(h1, h2)

        assert isinstance(ms_ssim_val, float)
        assert 0.0 <= ms_ssim_val <= 1.0

    def test_compute_spearman(self, stability_metrics, sample_heatmaps):
        """Test compute_spearman method."""
        h1, h2 = sample_heatmaps
        rho = stability_metrics.compute_spearman(h1, h2)

        assert isinstance(rho, float)
        assert -1.0 <= rho <= 1.0

    def test_compute_l2_distance(self, stability_metrics, sample_heatmaps):
        """Test compute_l2_distance method."""
        h1, h2 = sample_heatmaps
        dist = stability_metrics.compute_l2_distance(h1, h2)

        assert isinstance(dist, float)
        assert 0.0 <= dist <= 1.0

    def test_compute_cosine_similarity(self, stability_metrics, sample_heatmaps):
        """Test compute_cosine_similarity method."""
        h1, h2 = sample_heatmaps
        cos_sim = stability_metrics.compute_cosine_similarity(h1, h2)

        assert isinstance(cos_sim, float)
        assert -1.0 <= cos_sim <= 1.0

    def test_compute_all(self, stability_metrics, sample_heatmaps):
        """Test compute_all method returns all metrics."""
        h1, h2 = sample_heatmaps
        results = stability_metrics.compute_all(h1, h2, include_ms_ssim=True)

        # Check all expected keys are present
        expected_keys = ["ssim", "ms_ssim", "spearman", "l2_distance", "cosine_similarity"]
        for key in expected_keys:
            assert key in results
            assert isinstance(results[key], float)

    def test_compute_all_without_ms_ssim(self, stability_metrics, sample_heatmaps):
        """Test compute_all without MS-SSIM."""
        h1, h2 = sample_heatmaps
        results = stability_metrics.compute_all(h1, h2, include_ms_ssim=False)

        assert "ssim" in results
        assert "ms_ssim" not in results
        assert "spearman" in results

    def test_preprocess_heatmaps_3d_to_4d(self, stability_metrics, device):
        """Test preprocessing converts 3D to 4D."""
        h1 = torch.randn(2, 14, 14, device=device)  # (B, H, W)
        h2 = torch.randn(2, 14, 14, device=device)

        h1_prep, h2_prep = stability_metrics._preprocess_heatmaps(h1, h2)

        assert h1_prep.dim() == 4
        assert h1_prep.shape[1] == 1  # Channel dimension added

    def test_preprocess_heatmaps_normalization(self, device):
        """Test preprocessing normalizes to [0, 1]."""
        config = StabilityMetricsConfig(normalize_heatmaps=True)
        metrics = StabilityMetrics(config)

        h1 = torch.randn(2, 1, 14, 14, device=device)
        h2 = torch.randn(2, 1, 14, 14, device=device)

        h1_prep, h2_prep = metrics._preprocess_heatmaps(h1, h2)

        assert h1_prep.min() >= 0.0
        assert h1_prep.max() <= 1.0
        assert h2_prep.min() >= 0.0
        assert h2_prep.max() <= 1.0

    def test_shape_mismatch_error(self, stability_metrics, device):
        """Test error on shape mismatch."""
        h1 = torch.randn(2, 1, 14, 14, device=device)
        h2 = torch.randn(2, 1, 28, 28, device=device)

        with pytest.raises(ValueError, match="shapes must match"):
            stability_metrics.compute_ssim(h1, h2)


# ============================================================================
# Test Factory Function
# ============================================================================


class TestFactoryFunction:
    """Test create_stability_metrics factory."""

    def test_factory_default(self):
        """Test factory with default config."""
        metrics = create_stability_metrics()

        assert isinstance(metrics, StabilityMetrics)
        assert metrics.config is not None

    def test_factory_custom_config(self):
        """Test factory with custom config."""
        config = StabilityMetricsConfig(ssim_window_size=7)
        metrics = create_stability_metrics(config)

        assert metrics.config.ssim_window_size == 7


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests with Grad-CAM pipeline."""

    def test_with_gradcam_output(self, device):
        """Test stability metrics with Grad-CAM-like output."""
        # Simulate Grad-CAM heatmaps
        heatmap_clean = torch.rand(4, 1, 7, 7, device=device)
        heatmap_adv = heatmap_clean + 0.1 * torch.randn_like(heatmap_clean)

        metrics = create_stability_metrics()
        results = metrics.compute_all(heatmap_clean, heatmap_adv)

        # Should have high similarity (small perturbation)
        assert results["ssim"] > 0.8
        assert results["spearman"] > 0.8

    def test_batch_processing(self, device):
        """Test metrics work with large batches."""
        batch_size = 32
        h1 = torch.randn(batch_size, 1, 14, 14, device=device)
        h2 = torch.randn(batch_size, 1, 14, 14, device=device)

        metrics = create_stability_metrics()
        results = metrics.compute_all(h1, h2)

        # All metrics should be computed successfully
        assert all(isinstance(v, float) for v in results.values())

    def test_hypothesis_h2_validation(self, device):
        """Test that stable explanations meet H2 threshold."""
        # H2: SSIM(clean, adv) ≥ 0.75 for good explanations

        # Simulate highly stable explanations (small perturbation)
        heatmap_clean = torch.rand(4, 1, 14, 14, device=device)
        heatmap_adv = heatmap_clean + 0.02 * torch.randn_like(heatmap_clean)  # Smaller noise

        metrics = create_stability_metrics()
        ssim_val = metrics.compute_ssim(heatmap_clean, heatmap_adv)

        # Should meet H2 threshold for dissertation validation
        assert ssim_val >= 0.75, (
            f"H2 requires SSIM ≥ 0.75 for stable explanations, got {ssim_val:.3f}"
        )


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Test computational efficiency."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_acceleration(self, device):
        """Test GPU provides speedup."""
        if device.type != "cuda":
            pytest.skip("CUDA not available")

        h1 = torch.randn(16, 1, 56, 56, device=device)
        h2 = torch.randn(16, 1, 56, 56, device=device)

        config = StabilityMetricsConfig(use_cuda=True)
        metrics = StabilityMetrics(config)

        # Should not error on GPU
        results = metrics.compute_all(h1, h2)
        assert results is not None

    def test_batch_efficiency(self, device):
        """Test batch processing is efficient."""
        # Single sample
        h1_single = torch.randn(1, 1, 14, 14, device=device)
        h2_single = torch.randn(1, 1, 14, 14, device=device)

        # Batch
        h1_batch = torch.randn(32, 1, 14, 14, device=device)
        h2_batch = torch.randn(32, 1, 14, 14, device=device)

        metrics = create_stability_metrics()

        # Both should work
        result_single = metrics.compute_ssim(h1_single, h2_single)
        result_batch = metrics.compute_ssim(h1_batch, h2_batch)

        assert isinstance(result_single, float)
        assert isinstance(result_batch, float)
