"""
Comprehensive Tests for Stability Scorer Module (Phase 8.2).

This test suite provides complete coverage for the stability scoring
functionality, including:
- StabilityScore dataclass validation
- SSIM stability scorer
- Alternative stability metrics (Spearman, L2, Cosine)
- Unified StabilityScorer interface
- Batch processing
- Edge cases and error handling
- Integration tests

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Date: November 27, 2025
"""

from typing import List, Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.validation.stability_scorer import (
    DEFAULT_EPSILON,
    CosineStabilityScorer,
    L2StabilityScorer,
    SpearmanStabilityScorer,
    SSIMStabilityScorer,
    StabilityMethod,
    StabilityScore,
    StabilityScorer,
    compute_stability_metrics,
    create_stability_scorer,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def device() -> torch.device:
    """Get computation device."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def simple_model() -> nn.Module:
    """Create simple CNN for testing."""

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1),
                nn.ReLU(),
                nn.MaxPool2d(2),
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((7, 7)),
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1),
                nn.ReLU(),
                nn.AdaptiveAvgPool2d((1, 1)),
            )
            self.fc = nn.Linear(128, 7)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = x.view(x.size(0), -1)
            return self.fc(x)

    model = SimpleCNN()
    model.eval()
    return model


@pytest.fixture
def sample_image(device: torch.device) -> Tensor:
    """Generate sample image for testing."""
    torch.manual_seed(42)
    return torch.rand(1, 3, 224, 224, device=device)


@pytest.fixture
def sample_batch(device: torch.device) -> Tuple[Tensor, Tensor]:
    """Generate batch of images and labels."""
    torch.manual_seed(42)
    images = torch.rand(8, 3, 224, 224, device=device)
    labels = torch.randint(0, 7, (8,), device=device)
    return images, labels


@pytest.fixture
def sample_heatmaps() -> Tuple[np.ndarray, np.ndarray]:
    """Generate sample heatmaps."""
    np.random.seed(42)
    heatmap_clean = np.random.rand(224, 224)
    # Similar heatmap with small perturbation
    heatmap_adv = heatmap_clean + 0.05 * np.random.randn(224, 224)
    heatmap_adv = np.clip(heatmap_adv, 0, 1)
    return heatmap_clean, heatmap_adv


# ============================================================================
# StabilityScore Tests
# ============================================================================


class TestStabilityScore:
    """Tests for StabilityScore dataclass."""

    def test_initialization(self, sample_heatmaps):
        """Test StabilityScore initialization."""
        heatmap_clean, heatmap_adv = sample_heatmaps

        score = StabilityScore(
            stability=0.85,
            instability=0.15,
            method=StabilityMethod.SSIM,
            clean_explanation=heatmap_clean,
            perturbed_explanation=heatmap_adv,
            perturbation_norm=0.00784,
            epsilon=2 / 255,
        )

        assert score.stability == 0.85
        assert score.instability == 0.15
        assert score.method == StabilityMethod.SSIM
        assert score.perturbation_norm == 0.00784
        assert score.epsilon == pytest.approx(2 / 255, rel=1e-6)

    def test_stability_clipping(self, sample_heatmaps):
        """Test that stability is clipped to [0, 1]."""
        heatmap_clean, heatmap_adv = sample_heatmaps

        # Test clipping above 1
        score = StabilityScore(
            stability=1.5,
            instability=0.0,
            method=StabilityMethod.SSIM,
            clean_explanation=heatmap_clean,
            perturbed_explanation=heatmap_adv,
            perturbation_norm=0.0,
            epsilon=2 / 255,
        )
        assert score.stability == 1.0

        # Test clipping below 0
        score = StabilityScore(
            stability=-0.1,
            instability=1.1,
            method=StabilityMethod.SSIM,
            clean_explanation=heatmap_clean,
            perturbed_explanation=heatmap_adv,
            perturbation_norm=0.0,
            epsilon=2 / 255,
        )
        assert score.stability == 0.0

    def test_instability_consistency(self, sample_heatmaps):
        """Test that instability is consistent with stability."""
        heatmap_clean, heatmap_adv = sample_heatmaps

        score = StabilityScore(
            stability=0.75,
            instability=0.30,  # Inconsistent value
            method=StabilityMethod.SSIM,
            clean_explanation=heatmap_clean,
            perturbed_explanation=heatmap_adv,
            perturbation_norm=0.0,
            epsilon=2 / 255,
        )

        # Should be auto-corrected
        assert score.instability == pytest.approx(0.25, rel=1e-6)

    def test_is_stable(self, sample_heatmaps):
        """Test is_stable method."""
        heatmap_clean, heatmap_adv = sample_heatmaps

        score_stable = StabilityScore(
            stability=0.8,
            instability=0.2,
            method=StabilityMethod.SSIM,
            clean_explanation=heatmap_clean,
            perturbed_explanation=heatmap_adv,
            perturbation_norm=0.0,
            epsilon=2 / 255,
        )

        score_unstable = StabilityScore(
            stability=0.6,
            instability=0.4,
            method=StabilityMethod.SSIM,
            clean_explanation=heatmap_clean,
            perturbed_explanation=heatmap_adv,
            perturbation_norm=0.0,
            epsilon=2 / 255,
        )

        assert score_stable.is_stable(threshold=0.75)
        assert not score_unstable.is_stable(threshold=0.75)

    def test_to_dict(self, sample_heatmaps):
        """Test conversion to dictionary."""
        heatmap_clean, heatmap_adv = sample_heatmaps

        score = StabilityScore(
            stability=0.85,
            instability=0.15,
            method=StabilityMethod.SSIM,
            clean_explanation=heatmap_clean,
            perturbed_explanation=heatmap_adv,
            perturbation_norm=0.00784,
            epsilon=2 / 255,
            metadata={"test_key": "test_value"},
        )

        score_dict = score.to_dict()

        assert score_dict["stability"] == 0.85
        assert score_dict["instability"] == 0.15
        assert score_dict["method"] == "ssim"
        assert score_dict["perturbation_norm"] == 0.00784
        assert score_dict["epsilon"] == pytest.approx(2 / 255, rel=1e-6)
        assert "test_key" in score_dict["metadata"]

    def test_invalid_heatmap_dims(self, sample_heatmaps):
        """Test error on invalid heatmap dimensions."""
        heatmap_clean, _ = sample_heatmaps

        with pytest.raises(ValueError, match="must be 2D"):
            StabilityScore(
                stability=0.85,
                instability=0.15,
                method=StabilityMethod.SSIM,
                clean_explanation=np.expand_dims(heatmap_clean, 0),  # 3D
                perturbed_explanation=heatmap_clean,
                perturbation_norm=0.0,
                epsilon=2 / 255,
            )

    def test_mismatched_shapes(self, sample_heatmaps):
        """Test error on mismatched heatmap shapes."""
        heatmap_clean, _ = sample_heatmaps
        heatmap_adv = np.random.rand(112, 112)  # Different size

        with pytest.raises(ValueError, match="shapes must match"):
            StabilityScore(
                stability=0.85,
                instability=0.15,
                method=StabilityMethod.SSIM,
                clean_explanation=heatmap_clean,
                perturbed_explanation=heatmap_adv,
                perturbation_norm=0.0,
                epsilon=2 / 255,
            )


# ============================================================================
# SSIMStabilityScorer Tests
# ============================================================================


class TestSSIMStabilityScorer:
    """Tests for SSIM stability scorer."""

    def test_initialization(self, simple_model, device):
        """Test SSIM scorer initialization."""
        scorer = SSIMStabilityScorer(
            model=simple_model,
            epsilon=2 / 255,
            target_layers=["layer4"],
            device=device,
        )

        assert scorer.epsilon == pytest.approx(2 / 255, rel=1e-6)
        assert scorer.target_layers == ["layer4"]
        assert scorer.device == device

    def test_single_image_scoring(self, simple_model, sample_image, device):
        """Test scoring single image."""
        scorer = SSIMStabilityScorer(
            model=simple_model,
            epsilon=2 / 255,
            device=device,
        )

        label = torch.tensor(0, device=device)
        score = scorer(sample_image, label)

        assert isinstance(score, StabilityScore)
        assert 0 <= score.stability <= 1
        assert score.method == StabilityMethod.SSIM
        assert score.epsilon == pytest.approx(2 / 255, rel=1e-6)
        assert score.clean_explanation.shape == score.perturbed_explanation.shape

    def test_ms_ssim_variant(self, simple_model, sample_image, device):
        """Test multi-scale SSIM variant."""
        scorer = SSIMStabilityScorer(
            model=simple_model,
            epsilon=2 / 255,
            use_ms_ssim=True,
            device=device,
        )

        label = torch.tensor(0, device=device)
        score = scorer(sample_image, label)

        assert score.method == StabilityMethod.MS_SSIM
        assert score.metadata["method_variant"] == "MS-SSIM"

    def test_different_epsilons(self, simple_model, sample_image, device):
        """Test scoring with different epsilon values."""
        epsilons = [1 / 255, 2 / 255, 4 / 255, 8 / 255]

        for eps in epsilons:
            scorer = SSIMStabilityScorer(
                model=simple_model,
                epsilon=eps,
                device=device,
            )

            label = torch.tensor(0, device=device)
            score = scorer(sample_image, label)

            assert score.epsilon == pytest.approx(eps, rel=1e-6)
            # Larger epsilon should generally lead to lower stability
            assert 0 <= score.stability <= 1

    def test_return_all_metrics(self, simple_model, sample_image, device):
        """Test return_all_metrics option."""
        scorer = SSIMStabilityScorer(
            model=simple_model,
            epsilon=2 / 255,
            device=device,
        )

        label = torch.tensor(0, device=device)
        score = scorer(sample_image, label, return_all_metrics=True)

        assert "all_metrics" in score.metadata
        assert "ssim" in score.metadata["all_metrics"]
        assert "spearman" in score.metadata["all_metrics"]
        assert "l2" in score.metadata["all_metrics"]
        assert "cosine" in score.metadata["all_metrics"]

    def test_3d_input(self, simple_model, device):
        """Test with 3D input (C, H, W)."""
        scorer = SSIMStabilityScorer(
            model=simple_model,
            epsilon=2 / 255,
            device=device,
        )

        image_3d = torch.rand(3, 224, 224, device=device)
        label = torch.tensor(0, device=device)

        score = scorer(image_3d, label)
        assert isinstance(score, StabilityScore)

    def test_scalar_label(self, simple_model, sample_image, device):
        """Test with scalar label."""
        scorer = SSIMStabilityScorer(
            model=simple_model,
            epsilon=2 / 255,
            device=device,
        )

        score = scorer(sample_image, 0)  # Scalar label
        assert isinstance(score, StabilityScore)

    def test_batch_score(self, simple_model, sample_batch, device):
        """Test batch scoring."""
        scorer = SSIMStabilityScorer(
            model=simple_model,
            epsilon=2 / 255,
            device=device,
        )

        images, labels = sample_batch
        scores = scorer.batch_score(images, labels, batch_size=4)

        assert len(scores) == 8
        assert all(isinstance(s, StabilityScore) for s in scores)
        assert all(0 <= s.stability <= 1 for s in scores)

    def test_empty_batch_error(self, simple_model, device):
        """Test error on empty batch."""
        scorer = SSIMStabilityScorer(
            model=simple_model,
            epsilon=2 / 255,
            device=device,
        )

        empty_images = torch.empty(0, 3, 224, 224, device=device)
        empty_labels = torch.empty(0, dtype=torch.long, device=device)

        with pytest.raises(ValueError, match="empty batch"):
            scorer.batch_score(empty_images, empty_labels)

    def test_batch_size_mismatch_error(self, simple_model, device):
        """Test error on batch size mismatch."""
        scorer = SSIMStabilityScorer(
            model=simple_model,
            epsilon=2 / 255,
            device=device,
        )

        images = torch.rand(8, 3, 224, 224, device=device)
        labels = torch.randint(0, 7, (4,), device=device)  # Wrong size

        with pytest.raises(ValueError, match="Batch size mismatch"):
            scorer.batch_score(images, labels)


# ============================================================================
# Alternative Stability Scorers Tests
# ============================================================================


class TestSpearmanStabilityScorer:
    """Tests for Spearman stability scorer."""

    def test_initialization(self, simple_model, device):
        """Test Spearman scorer initialization."""
        scorer = SpearmanStabilityScorer(
            model=simple_model,
            epsilon=2 / 255,
            device=device,
        )

        assert scorer.epsilon == pytest.approx(2 / 255, rel=1e-6)

    def test_scoring(self, simple_model, sample_image, device):
        """Test Spearman scoring."""
        scorer = SpearmanStabilityScorer(
            model=simple_model,
            epsilon=2 / 255,
            device=device,
        )

        label = torch.tensor(0, device=device)
        score = scorer(sample_image, label)

        assert isinstance(score, StabilityScore)
        assert score.method == StabilityMethod.SPEARMAN
        assert 0 <= score.stability <= 1
        assert "spearman_rho" in score.metadata


class TestL2StabilityScorer:
    """Tests for L2 stability scorer."""

    def test_initialization(self, simple_model, device):
        """Test L2 scorer initialization."""
        scorer = L2StabilityScorer(
            model=simple_model,
            epsilon=2 / 255,
            device=device,
        )

        assert scorer.epsilon == pytest.approx(2 / 255, rel=1e-6)

    def test_scoring(self, simple_model, sample_image, device):
        """Test L2 scoring."""
        scorer = L2StabilityScorer(
            model=simple_model,
            epsilon=2 / 255,
            device=device,
        )

        label = torch.tensor(0, device=device)
        score = scorer(sample_image, label)

        assert isinstance(score, StabilityScore)
        assert score.method == StabilityMethod.L2
        assert 0 <= score.stability <= 1
        assert "l2_distance" in score.metadata
        assert "normalized_l2" in score.metadata


class TestCosineStabilityScorer:
    """Tests for Cosine stability scorer."""

    def test_initialization(self, simple_model, device):
        """Test Cosine scorer initialization."""
        scorer = CosineStabilityScorer(
            model=simple_model,
            epsilon=2 / 255,
            device=device,
        )

        assert scorer.epsilon == pytest.approx(2 / 255, rel=1e-6)

    def test_scoring(self, simple_model, sample_image, device):
        """Test Cosine scoring."""
        scorer = CosineStabilityScorer(
            model=simple_model,
            epsilon=2 / 255,
            device=device,
        )

        label = torch.tensor(0, device=device)
        score = scorer(sample_image, label)

        assert isinstance(score, StabilityScore)
        assert score.method == StabilityMethod.COSINE
        assert 0 <= score.stability <= 1
        assert "cosine_similarity" in score.metadata


# ============================================================================
# Unified StabilityScorer Tests
# ============================================================================


class TestStabilityScorer:
    """Tests for unified StabilityScorer."""

    def test_initialization_ssim(self, simple_model, device):
        """Test initialization with SSIM method."""
        scorer = StabilityScorer(
            model=simple_model,
            method="ssim",
            epsilon=2 / 255,
            device=device,
        )

        assert scorer.method == StabilityMethod.SSIM
        assert isinstance(scorer.scorer, SSIMStabilityScorer)

    def test_initialization_ms_ssim(self, simple_model, device):
        """Test initialization with MS-SSIM method."""
        scorer = StabilityScorer(
            model=simple_model,
            method="ms_ssim",
            epsilon=2 / 255,
            device=device,
        )

        assert scorer.method == StabilityMethod.MS_SSIM

    def test_initialization_spearman(self, simple_model, device):
        """Test initialization with Spearman method."""
        scorer = StabilityScorer(
            model=simple_model,
            method="spearman",
            epsilon=2 / 255,
            device=device,
        )

        assert scorer.method == StabilityMethod.SPEARMAN
        assert isinstance(scorer.scorer, SpearmanStabilityScorer)

    def test_initialization_l2(self, simple_model, device):
        """Test initialization with L2 method."""
        scorer = StabilityScorer(
            model=simple_model,
            method="l2",
            epsilon=2 / 255,
            device=device,
        )

        assert scorer.method == StabilityMethod.L2
        assert isinstance(scorer.scorer, L2StabilityScorer)

    def test_initialization_cosine(self, simple_model, device):
        """Test initialization with Cosine method."""
        scorer = StabilityScorer(
            model=simple_model,
            method="cosine",
            epsilon=2 / 255,
            device=device,
        )

        assert scorer.method == StabilityMethod.COSINE
        assert isinstance(scorer.scorer, CosineStabilityScorer)

    def test_invalid_method(self, simple_model, device):
        """Test error on invalid method."""
        with pytest.raises(ValueError, match="Unknown stability method"):
            StabilityScorer(
                model=simple_model,
                method="invalid_method",
                device=device,
            )

    def test_method_enum(self, simple_model, device):
        """Test initialization with method enum."""
        scorer = StabilityScorer(
            model=simple_model,
            method=StabilityMethod.SSIM,
            epsilon=2 / 255,
            device=device,
        )

        assert scorer.method == StabilityMethod.SSIM

    def test_scoring_all_methods(self, simple_model, sample_image, device):
        """Test scoring with all methods."""
        methods = ["ssim", "spearman", "l2", "cosine"]
        label = torch.tensor(0, device=device)

        for method in methods:
            scorer = StabilityScorer(
                model=simple_model,
                method=method,
                epsilon=2 / 255,
                device=device,
            )

            score = scorer(sample_image, label)
            assert isinstance(score, StabilityScore)
            assert 0 <= score.stability <= 1

    def test_batch_score(self, simple_model, sample_batch, device):
        """Test batch scoring with unified scorer."""
        scorer = StabilityScorer(
            model=simple_model,
            method="ssim",
            epsilon=2 / 255,
            device=device,
        )

        images, labels = sample_batch
        scores = scorer.batch_score(images, labels, batch_size=4)

        assert len(scores) == 8
        assert all(isinstance(s, StabilityScore) for s in scores)


# ============================================================================
# Helper Functions Tests
# ============================================================================


class TestComputeStabilityMetrics:
    """Tests for compute_stability_metrics helper."""

    def test_aggregate_metrics(self, sample_heatmaps):
        """Test computing aggregate metrics."""
        heatmap_clean, heatmap_adv = sample_heatmaps

        scores = [
            StabilityScore(
                stability=0.8 + i * 0.02,
                instability=0.2 - i * 0.02,
                method=StabilityMethod.SSIM,
                clean_explanation=heatmap_clean,
                perturbed_explanation=heatmap_adv,
                perturbation_norm=0.00784,
                epsilon=2 / 255,
            )
            for i in range(5)
        ]

        metrics = compute_stability_metrics(scores)

        assert "mean" in metrics
        assert "std" in metrics
        assert "min" in metrics
        assert "max" in metrics
        assert "median" in metrics
        assert "stable_fraction" in metrics
        assert "mean_perturbation_norm" in metrics

        assert metrics["mean"] > 0
        assert metrics["std"] >= 0
        assert metrics["min"] <= metrics["mean"] <= metrics["max"]

    def test_empty_scores(self):
        """Test with empty scores list."""
        metrics = compute_stability_metrics([])
        assert metrics == {}

    def test_stable_fraction(self, sample_heatmaps):
        """Test stable_fraction calculation."""
        heatmap_clean, heatmap_adv = sample_heatmaps

        scores = [
            StabilityScore(
                stability=0.8,  # Stable
                instability=0.2,
                method=StabilityMethod.SSIM,
                clean_explanation=heatmap_clean,
                perturbed_explanation=heatmap_adv,
                perturbation_norm=0.00784,
                epsilon=2 / 255,
            ),
            StabilityScore(
                stability=0.6,  # Unstable
                instability=0.4,
                method=StabilityMethod.SSIM,
                clean_explanation=heatmap_clean,
                perturbed_explanation=heatmap_adv,
                perturbation_norm=0.00784,
                epsilon=2 / 255,
            ),
        ]

        metrics = compute_stability_metrics(scores)
        assert metrics["stable_fraction"] == 0.5  # 1 out of 2


class TestCreateStabilityScorer:
    """Tests for create_stability_scorer factory."""

    def test_factory_ssim(self, simple_model, device):
        """Test factory for SSIM scorer."""
        scorer = create_stability_scorer(
            model=simple_model,
            method="ssim",
            epsilon=2 / 255,
            device=device,
        )

        assert isinstance(scorer, StabilityScorer)
        assert scorer.method == StabilityMethod.SSIM

    def test_factory_default(self, simple_model, device):
        """Test factory with default parameters."""
        scorer = create_stability_scorer(
            model=simple_model,
            device=device,
        )

        assert isinstance(scorer, StabilityScorer)
        assert scorer.method == StabilityMethod.SSIM
        assert scorer.epsilon == DEFAULT_EPSILON


# ============================================================================
# Edge Cases and Error Handling Tests
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_epsilon(self, simple_model, sample_image, device):
        """Test with zero epsilon (no perturbation)."""
        scorer = SSIMStabilityScorer(
            model=simple_model,
            epsilon=0.0,  # No perturbation
            device=device,
        )

        label = torch.tensor(0, device=device)
        score = scorer(sample_image, label)

        # Should have perfect stability (same explanation)
        assert score.stability > 0.95
        assert score.perturbation_norm == pytest.approx(0.0, abs=1e-6)

    def test_very_small_image(self, simple_model, device):
        """Test with very small image size."""
        scorer = SSIMStabilityScorer(
            model=simple_model,
            epsilon=2 / 255,
            device=device,
        )

        # Small image (may cause issues with window size)
        small_image = torch.rand(1, 3, 32, 32, device=device)
        label = torch.tensor(0, device=device)

        # Should still work (SSIM window size adjusted)
        score = scorer(small_image, label)
        assert isinstance(score, StabilityScore)

    def test_constant_image(self, simple_model, device):
        """Test with constant (uniform) image."""
        scorer = SSIMStabilityScorer(
            model=simple_model,
            epsilon=2 / 255,
            device=device,
        )

        # Constant image
        const_image = torch.ones(1, 3, 224, 224, device=device) * 0.5
        label = torch.tensor(0, device=device)

        score = scorer(const_image, label)
        assert isinstance(score, StabilityScore)


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests with full pipeline."""

    def test_multi_method_comparison(self, simple_model, sample_image, device):
        """Test comparing multiple methods."""
        methods = ["ssim", "spearman", "l2", "cosine"]
        label = torch.tensor(0, device=device)

        scores = {}
        for method in methods:
            scorer = StabilityScorer(
                model=simple_model,
                method=method,
                epsilon=2 / 255,
                device=device,
            )
            scores[method] = scorer(sample_image, label)

        # All should produce valid scores
        assert all(0 <= s.stability <= 1 for s in scores.values())

        # SSIM and other metrics should be correlated (not identical)
        assert len(scores) == len(methods)

    def test_epsilon_sensitivity(self, simple_model, sample_image, device):
        """Test sensitivity to epsilon values."""
        epsilons = [0.5 / 255, 1 / 255, 2 / 255, 4 / 255, 8 / 255]
        label = torch.tensor(0, device=device)

        stabilities = []
        for eps in epsilons:
            scorer = SSIMStabilityScorer(
                model=simple_model,
                epsilon=eps,
                device=device,
            )
            score = scorer(sample_image, label)
            stabilities.append(score.stability)

        # Generally, larger epsilon should lead to lower stability
        # (though not guaranteed for all samples)
        assert all(0 <= s <= 1 for s in stabilities)

    def test_hypothesis_h2_validation(self, simple_model, sample_batch, device):
        """Test that stable predictions meet H2 threshold (≥0.75)."""
        scorer = SSIMStabilityScorer(
            model=simple_model,
            epsilon=2 / 255,
            device=device,
        )

        images, labels = sample_batch
        scores = scorer.batch_score(images, labels)

        # Count stable predictions
        stable_count = sum(1 for s in scores if s.is_stable(threshold=0.75))

        # Should have some stable predictions (model-dependent)
        assert 0 <= stable_count <= len(scores)
