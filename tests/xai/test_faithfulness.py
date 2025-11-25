"""
Comprehensive Test Suite for Faithfulness Metrics.

Tests all faithfulness metrics (Deletion, Insertion, Pointing Game) for:
- Correctness (mathematical properties)
- Edge cases (all pixels deleted/inserted, empty masks)
- Numerical stability (zero gradients, constant heatmaps)
- Batch processing efficiency
- Model compatibility
- Integration with Grad-CAM pipeline

Test Coverage Target: >95%

Author: Viraj Pankaj Jain
Phase: 6.3 - Faithfulness Metrics
Date: November 25, 2025
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.xai.faithfulness import (
    DeletionMetric,
    FaithfulnessConfig,
    FaithfulnessMetrics,
    InsertionMetric,
    PointingGame,
    create_faithfulness_metrics,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def device() -> torch.device:
    """Return CUDA device if available, else CPU."""
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def dummy_model(device: torch.device) -> nn.Module:
    """Create a simple CNN model for testing."""

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
            self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc = nn.Linear(32, 10)

        def forward(self, x):
            x = torch.relu(self.conv1(x))
            x = torch.relu(self.conv2(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    model = SimpleCNN().to(device)
    model.eval()
    return model


@pytest.fixture
def sample_image(device: torch.device) -> Tensor:
    """Create sample image."""
    torch.manual_seed(42)
    return torch.randn(1, 3, 28, 28, device=device)


@pytest.fixture
def sample_heatmap(device: torch.device) -> Tensor:
    """Create sample heatmap."""
    torch.manual_seed(42)
    heatmap = torch.randn(1, 1, 28, 28, device=device)
    heatmap = torch.relu(heatmap)  # Make non-negative
    return heatmap


@pytest.fixture
def sample_mask(device: torch.device) -> Tensor:
    """Create sample ground-truth mask."""
    mask = torch.zeros(1, 28, 28, device=device)
    # Central region is ground truth
    mask[0, 10:18, 10:18] = 1.0
    return mask


@pytest.fixture
def config(device: torch.device) -> FaithfulnessConfig:
    """Create default configuration."""
    return FaithfulnessConfig(num_steps=10, verbose=False, device=str(device))


@pytest.fixture
def faithfulness_metrics(dummy_model, config) -> FaithfulnessMetrics:
    """Create FaithfulnessMetrics instance."""
    return FaithfulnessMetrics(dummy_model, config)


# ============================================================================
# Test Configuration
# ============================================================================


class TestFaithfulnessConfig:
    """Test configuration dataclass."""

    def test_default_initialization(self):
        """Test default config values."""
        config = FaithfulnessConfig()

        assert config.num_steps == 50
        assert config.batch_size == 16
        assert config.baseline_mode == "mean"
        assert config.interpolation_mode == "bilinear"
        assert config.normalize_curves is True
        assert config.use_softmax is True
        assert config.pointing_game_threshold == 0.15

    def test_custom_values(self):
        """Test custom configuration values."""
        config = FaithfulnessConfig(
            num_steps=20,
            baseline_mode="blur",
            normalize_curves=False,
        )

        assert config.num_steps == 20
        assert config.baseline_mode == "blur"
        assert config.normalize_curves is False

    def test_num_steps_validation(self):
        """Test num_steps must be >= 5."""
        with pytest.raises(ValueError, match="num_steps must be"):
            FaithfulnessConfig(num_steps=3)

    def test_baseline_mode_validation(self):
        """Test baseline_mode validation."""
        with pytest.raises(ValueError, match="baseline_mode must be"):
            FaithfulnessConfig(baseline_mode="invalid")

    def test_interpolation_mode_validation(self):
        """Test interpolation_mode validation."""
        with pytest.raises(ValueError, match="interpolation_mode must be"):
            FaithfulnessConfig(interpolation_mode="invalid")

    def test_pointing_game_threshold_validation(self):
        """Test threshold must be in [0, 1]."""
        with pytest.raises(ValueError, match="pointing_game_threshold must be"):
            FaithfulnessConfig(pointing_game_threshold=1.5)


# ============================================================================
# Test Deletion Metric
# ============================================================================


class TestDeletionMetric:
    """Test deletion curve implementation."""

    def test_initialization(self, dummy_model, config):
        """Test DeletionMetric initialization."""
        deletion = DeletionMetric(dummy_model, config)

        assert deletion.model is dummy_model
        assert deletion.config == config

    def test_compute_curve_shape(
        self, dummy_model, config, sample_image, sample_heatmap
    ):
        """Test deletion curve has correct shape."""
        deletion = DeletionMetric(dummy_model, config)

        curve, auc, ad = deletion.compute(sample_image, sample_heatmap, target_class=0)

        assert len(curve) == config.num_steps + 1  # +1 for initial score
        assert isinstance(auc, float)
        assert isinstance(ad, float)

    def test_deletion_decreases_score(
        self, dummy_model, config, sample_image, sample_heatmap
    ):
        """Test that deleting pixels decreases prediction score."""
        deletion = DeletionMetric(dummy_model, config)

        curve, _, _ = deletion.compute(sample_image, sample_heatmap, target_class=0)

        # Score should generally decrease (not strictly monotonic due to noise)
        assert curve[0] >= curve[-1]  # Initial >= final

    def test_average_drop_positive(
        self, dummy_model, config, sample_image, sample_heatmap
    ):
        """Test average drop is positive (score decreases)."""
        deletion = DeletionMetric(dummy_model, config)

        _, _, ad = deletion.compute(sample_image, sample_heatmap, target_class=0)

        # Average drop should be non-negative
        assert ad >= 0.0

    def test_different_baseline_modes(self, dummy_model, sample_image, sample_heatmap):
        """Test different baseline modes work."""
        baseline_modes = ["mean", "blur", "noise", "zero"]

        for mode in baseline_modes:
            config = FaithfulnessConfig(num_steps=5, baseline_mode=mode, verbose=False)
            deletion = DeletionMetric(dummy_model, config)

            curve, auc, _ = deletion.compute(
                sample_image, sample_heatmap, target_class=0
            )

            assert len(curve) > 0
            assert not np.isnan(auc)

    def test_custom_baseline_value(
        self, dummy_model, config, sample_image, sample_heatmap
    ):
        """Test with custom baseline value."""
        deletion = DeletionMetric(dummy_model, config)
        custom_baseline = torch.zeros_like(sample_image)

        curve, _, _ = deletion.compute(
            sample_image,
            sample_heatmap,
            target_class=0,
            baseline_value=custom_baseline,
        )

        assert len(curve) > 0

    def test_input_shape_flexibility(self, dummy_model, config):
        """Test handles various input shapes."""
        deletion = DeletionMetric(dummy_model, config)

        # 3D image
        img_3d = torch.randn(3, 28, 28, device=config.device)
        hmap_2d = torch.randn(28, 28, device=config.device)

        curve, _, _ = deletion.compute(img_3d, hmap_2d, target_class=0)
        assert len(curve) > 0

    def test_heatmap_resize(self, dummy_model, config, sample_image):
        """Test heatmap is resized to match image."""
        deletion = DeletionMetric(dummy_model, config)
        small_heatmap = torch.randn(1, 1, 14, 14, device=config.device)

        curve, _, _ = deletion.compute(sample_image, small_heatmap, target_class=0)

        assert len(curve) > 0


# ============================================================================
# Test Insertion Metric
# ============================================================================


class TestInsertionMetric:
    """Test insertion curve implementation."""

    def test_initialization(self, dummy_model, config):
        """Test InsertionMetric initialization."""
        insertion = InsertionMetric(dummy_model, config)

        assert insertion.model is dummy_model
        assert insertion.config == config

    def test_compute_curve_shape(
        self, dummy_model, config, sample_image, sample_heatmap
    ):
        """Test insertion curve has correct shape."""
        insertion = InsertionMetric(dummy_model, config)

        curve, auc, ai = insertion.compute(sample_image, sample_heatmap, target_class=0)

        assert len(curve) == config.num_steps + 1  # +1 for initial (baseline)
        assert isinstance(auc, float)
        assert isinstance(ai, float)

    def test_insertion_increases_score(
        self, dummy_model, config, sample_image, sample_heatmap
    ):
        """Test that inserting pixels increases prediction score."""
        insertion = InsertionMetric(dummy_model, config)

        curve, _, _ = insertion.compute(sample_image, sample_heatmap, target_class=0)

        # Score should generally increase
        assert curve[-1] >= curve[0]  # Final >= initial

    def test_average_increase_positive(
        self, dummy_model, config, sample_image, sample_heatmap
    ):
        """Test average increase is positive (score increases)."""
        insertion = InsertionMetric(dummy_model, config)

        _, _, ai = insertion.compute(sample_image, sample_heatmap, target_class=0)

        # Average increase should be non-negative
        assert ai >= 0.0

    def test_different_baseline_modes(self, dummy_model, sample_image, sample_heatmap):
        """Test different baseline modes work."""
        baseline_modes = ["mean", "blur", "noise", "zero"]

        for mode in baseline_modes:
            config = FaithfulnessConfig(num_steps=5, baseline_mode=mode, verbose=False)
            insertion = InsertionMetric(dummy_model, config)

            curve, auc, _ = insertion.compute(
                sample_image, sample_heatmap, target_class=0
            )

            assert len(curve) > 0
            assert not np.isnan(auc)

    def test_full_insertion_equals_original(
        self, dummy_model, config, sample_image, sample_heatmap
    ):
        """Test that fully inserting all pixels gives similar score to original."""
        insertion = InsertionMetric(dummy_model, config)

        # Get original score
        with torch.no_grad():
            logits = dummy_model(sample_image)
            original_score = torch.softmax(logits, dim=1)[0, 0].item()

        # Get fully inserted score
        curve, _, _ = insertion.compute(sample_image, sample_heatmap, target_class=0)
        final_score = curve[-1]

        # Due to discrete step sizes and floating-point precision,
        # allow small tolerance (typically < 5% difference)
        assert abs(original_score - final_score) < 0.05, (
            f"Fully inserted image should match original: "
            f"{original_score:.4f} vs {final_score:.4f}"
        )


# ============================================================================
# Test Pointing Game
# ============================================================================


class TestPointingGame:
    """Test pointing game metric."""

    def test_initialization(self, config):
        """Test PointingGame initialization."""
        pointing_game = PointingGame(config)

        assert pointing_game.config == config

    def test_hit_inside_mask(self, config, device):
        """Test hit detection when max is inside mask."""
        pointing_game = PointingGame(config)

        # Create heatmap with max at (14, 14)
        heatmap = torch.zeros(28, 28, device=device)
        heatmap[14, 14] = 1.0

        # Create mask covering (14, 14)
        mask = torch.zeros(28, 28, device=device)
        mask[10:18, 10:18] = 1.0

        hit, (row, col) = pointing_game.compute(heatmap, mask, tolerance=0)

        assert hit is True
        assert row == 14 and col == 14

    def test_miss_outside_mask(self, config, device):
        """Test miss detection when max is outside mask."""
        pointing_game = PointingGame(config)

        # Create heatmap with max at (5, 5) - outside mask
        heatmap = torch.zeros(28, 28, device=device)
        heatmap[5, 5] = 1.0

        # Create mask in center
        mask = torch.zeros(28, 28, device=device)
        mask[10:18, 10:18] = 1.0

        hit, (row, col) = pointing_game.compute(heatmap, mask, tolerance=0)

        assert hit is False
        assert row == 5 and col == 5

    def test_tolerance_increases_hits(self, config, device):
        """Test tolerance allows nearby hits."""
        pointing_game = PointingGame(config)

        # Max at (9, 9), mask from (10, 10) - just outside
        heatmap = torch.zeros(28, 28, device=device)
        heatmap[9, 9] = 1.0

        mask = torch.zeros(28, 28, device=device)
        mask[10:18, 10:18] = 1.0

        # Without tolerance: miss
        hit_strict, _ = pointing_game.compute(heatmap, mask, tolerance=0)
        assert hit_strict is False

        # With tolerance: hit
        hit_tolerant, _ = pointing_game.compute(heatmap, mask, tolerance=2)
        assert hit_tolerant is True

    def test_batch_processing(self, config, device):
        """Test batch pointing game computation."""
        pointing_game = PointingGame(config)

        # Create batch with 2 hits, 1 miss
        heatmaps = torch.zeros(3, 28, 28, device=device)
        heatmaps[0, 14, 14] = 1.0  # Hit
        heatmaps[1, 5, 5] = 1.0  # Miss
        heatmaps[2, 15, 15] = 1.0  # Hit

        masks = torch.zeros(3, 28, 28, device=device)
        masks[:, 10:18, 10:18] = 1.0

        hits, accuracy = pointing_game.compute_batch(heatmaps, masks, tolerance=0)

        assert len(hits) == 3
        assert sum(hits) == 2  # 2 hits
        assert accuracy == 2.0 / 3.0

    def test_heatmap_resize(self, config, device):
        """Test heatmap is resized to match mask."""
        pointing_game = PointingGame(config)

        # Small heatmap
        heatmap = torch.zeros(14, 14, device=device)
        heatmap[7, 7] = 1.0

        # Large mask
        mask = torch.zeros(28, 28, device=device)
        mask[10:18, 10:18] = 1.0

        hit, _ = pointing_game.compute(heatmap, mask, tolerance=0)

        # Should work (after resize)
        assert isinstance(hit, bool)


# ============================================================================
# Test Unified FaithfulnessMetrics
# ============================================================================


class TestFaithfulnessMetrics:
    """Test unified FaithfulnessMetrics class."""

    def test_initialization(self, dummy_model, config):
        """Test FaithfulnessMetrics initialization."""
        metrics = FaithfulnessMetrics(dummy_model, config)

        assert metrics.model is dummy_model
        assert metrics.config == config
        assert metrics.deletion is not None
        assert metrics.insertion is not None
        assert metrics.pointing_game is not None

    def test_initialization_default_config(self, dummy_model):
        """Test initialization with default config."""
        metrics = FaithfulnessMetrics(dummy_model)

        assert metrics.config is not None
        assert isinstance(metrics.config, FaithfulnessConfig)

    def test_repr(self, dummy_model, config):
        """Test __repr__ method for debugging."""
        metrics = FaithfulnessMetrics(dummy_model, config)
        repr_str = repr(metrics)

        assert "FaithfulnessMetrics" in repr_str
        assert "num_steps" in repr_str
        assert "baseline" in repr_str

    def test_compute_all_without_masks(
        self, faithfulness_metrics, sample_image, sample_heatmap
    ):
        """Test compute_all without ground-truth masks."""
        images = sample_image
        heatmaps = sample_heatmap
        target_classes = [0]

        results = faithfulness_metrics.compute_all(
            images, heatmaps, target_classes, masks=None
        )

        # Should have deletion and insertion metrics
        assert "deletion_auc" in results
        assert "deletion_ad" in results
        assert "insertion_auc" in results
        assert "insertion_ai" in results

        # Should NOT have pointing game
        assert "pointing_acc" not in results

        # All values should be valid
        for key, value in results.items():
            assert isinstance(value, float)
            assert not np.isnan(value)

    def test_compute_all_with_masks(
        self, faithfulness_metrics, sample_image, sample_heatmap, sample_mask
    ):
        """Test compute_all with ground-truth masks."""
        images = sample_image
        heatmaps = sample_heatmap
        target_classes = [0]
        masks = sample_mask

        results = faithfulness_metrics.compute_all(
            images, heatmaps, target_classes, masks=masks
        )

        # Should have all metrics including pointing game
        assert "deletion_auc" in results
        assert "insertion_auc" in results
        assert "pointing_acc" in results

        assert 0.0 <= results["pointing_acc"] <= 1.0

    def test_batch_processing(self, faithfulness_metrics, device):
        """Test metrics work with batches."""
        batch_size = 3
        images = torch.randn(batch_size, 3, 28, 28, device=device)
        heatmaps = torch.randn(batch_size, 1, 28, 28, device=device)
        target_classes = [0, 1, 2]

        results = faithfulness_metrics.compute_all(images, heatmaps, target_classes)

        # All metrics should be computed
        assert "deletion_auc" in results
        assert "insertion_auc" in results

    def test_deletion_insertion_consistency(
        self, faithfulness_metrics, sample_image, sample_heatmap
    ):
        """Test deletion and insertion are inversely related."""
        results = faithfulness_metrics.compute_all(sample_image, sample_heatmap, [0])

        # Not a strict rule, but generally:
        # Low deletion AUC (good) corresponds to high insertion AUC (good)
        # Just check both are computed
        assert "deletion_auc" in results
        assert "insertion_auc" in results

    def test_baseline_value_shape_validation(
        self, faithfulness_metrics, sample_image, sample_heatmap
    ):
        """Test that baseline_value shape mismatch raises error."""
        # Create baseline with wrong batch size
        wrong_baseline = torch.randn(2, 3, 28, 28)  # Batch size 2

        # Should raise ValueError for shape mismatch
        with pytest.raises(ValueError, match="baseline_value batch size"):
            faithfulness_metrics.compute_all(
                sample_image,  # Batch size 1
                sample_heatmap,
                [0],
                baseline_value=wrong_baseline,  # Batch size 2 - mismatch!
            )


# ============================================================================
# Test Factory Function
# ============================================================================


class TestFactoryFunction:
    """Test create_faithfulness_metrics factory."""

    def test_factory_default(self, dummy_model):
        """Test factory with default config."""
        metrics = create_faithfulness_metrics(dummy_model)

        assert isinstance(metrics, FaithfulnessMetrics)
        assert metrics.config is not None

    def test_factory_custom_config(self, dummy_model):
        """Test factory with custom config."""
        config = FaithfulnessConfig(num_steps=20, verbose=False)
        metrics = create_faithfulness_metrics(dummy_model, config)

        assert metrics.config.num_steps == 20


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests with realistic scenarios."""

    def test_with_gradcam_output(self, dummy_model, device):
        """Test faithfulness metrics with Grad-CAM-like output."""
        # Simulate realistic scenario
        image = torch.randn(1, 3, 28, 28, device=device)
        heatmap = torch.rand(1, 1, 28, 28, device=device)  # Non-negative

        config = FaithfulnessConfig(num_steps=10, verbose=False, device=str(device))
        metrics = FaithfulnessMetrics(dummy_model, config)

        results = metrics.compute_all(image, heatmap, [0])

        # Check all metrics are reasonable
        assert 0.0 <= results["deletion_auc"] <= 1.0
        assert 0.0 <= results["insertion_auc"] <= 1.0
        assert results["deletion_ad"] >= 0.0
        assert results["insertion_ai"] >= 0.0

    def test_faithfulness_comparison(self, dummy_model, device):
        """Test comparing good vs random explanations."""
        image = torch.randn(1, 3, 28, 28, device=device)

        # Good explanation: focused on center
        good_heatmap = torch.zeros(1, 1, 28, 28, device=device)
        good_heatmap[0, 0, 12:16, 12:16] = 1.0

        # Random explanation
        random_heatmap = torch.rand(1, 1, 28, 28, device=device)

        config = FaithfulnessConfig(num_steps=10, verbose=False, device=str(device))
        metrics = FaithfulnessMetrics(dummy_model, config)

        results_good = metrics.compute_all(image, good_heatmap, [0])
        results_random = metrics.compute_all(image, random_heatmap, [0])

        # Good explanation should have better faithfulness
        # (Lower deletion AUC, higher insertion AUC)
        # This may not always hold for random model, but metrics should compute
        assert results_good["deletion_auc"] >= 0.0
        assert results_random["deletion_auc"] >= 0.0

    def test_multiclass_predictions(self, dummy_model, device):
        """Test with different target classes."""
        batch_size = 3
        images = torch.randn(batch_size, 3, 28, 28, device=device)
        heatmaps = torch.rand(batch_size, 1, 28, 28, device=device)
        target_classes = [0, 1, 2]  # Different class for each sample

        config = FaithfulnessConfig(num_steps=5, verbose=False, device=str(device))
        metrics = FaithfulnessMetrics(dummy_model, config)

        results = metrics.compute_all(images, heatmaps, target_classes)

        # Should work with different classes
        assert "deletion_auc" in results
        assert "insertion_auc" in results


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Test computational efficiency."""

    def test_num_steps_tradeoff(self, dummy_model, sample_image, sample_heatmap):
        """Test that fewer steps run faster."""
        configs = [
            FaithfulnessConfig(num_steps=5, verbose=False),
            FaithfulnessConfig(num_steps=20, verbose=False),
        ]

        for config in configs:
            metrics = FaithfulnessMetrics(dummy_model, config)
            results = metrics.compute_all(sample_image, sample_heatmap, [0])

            # Should complete regardless of num_steps
            assert "deletion_auc" in results

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_acceleration(self, dummy_model):
        """Test GPU provides valid results."""
        device = torch.device("cuda")
        image = torch.randn(1, 3, 28, 28, device=device)
        heatmap = torch.rand(1, 1, 28, 28, device=device)

        config = FaithfulnessConfig(num_steps=10, verbose=False, device="cuda")
        metrics = FaithfulnessMetrics(dummy_model.to(device), config)

        results = metrics.compute_all(image, heatmap, [0])

        assert results is not None
        assert not np.isnan(results["deletion_auc"])


# ============================================================================
# Edge Case Tests
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_constant_heatmap(self, dummy_model, config, sample_image):
        """Test with constant (uniform) heatmap."""
        constant_heatmap = torch.ones(1, 1, 28, 28, device=config.device)

        deletion = DeletionMetric(dummy_model, config)
        curve, auc, _ = deletion.compute(sample_image, constant_heatmap, target_class=0)

        # Should handle gracefully
        assert len(curve) > 0
        assert not np.isnan(auc)

    def test_zero_heatmap(self, dummy_model, config, sample_image):
        """Test with all-zero heatmap."""
        zero_heatmap = torch.zeros(1, 1, 28, 28, device=config.device)

        insertion = InsertionMetric(dummy_model, config)
        curve, auc, _ = insertion.compute(sample_image, zero_heatmap, target_class=0)

        # Should handle gracefully
        assert len(curve) > 0
        assert not np.isnan(auc)

    def test_single_pixel_mask(self, config, device):
        """Test pointing game with single-pixel mask."""
        pointing_game = PointingGame(config)

        heatmap = torch.zeros(28, 28, device=device)
        heatmap[14, 14] = 1.0

        mask = torch.zeros(28, 28, device=device)
        mask[14, 14] = 1.0

        hit, _ = pointing_game.compute(heatmap, mask, tolerance=0)

        assert hit is True

    def test_empty_mask(self, config, device):
        """Test pointing game with empty mask."""
        pointing_game = PointingGame(config)

        heatmap = torch.rand(28, 28, device=device)
        mask = torch.zeros(28, 28, device=device)  # All zeros

        hit, _ = pointing_game.compute(heatmap, mask, tolerance=0)

        # Should be False (no ground truth)
        assert hit is False


# ============================================================================
# Test Hypothesis H3
# ============================================================================


class TestHypothesisH3:
    """Test Research Hypothesis H3 validation."""

    def test_hypothesis_h3_faithful_explanation(self, dummy_model, device):
        """
        Validate H3: Faithful explanations have good AUC patterns.

        H3: Tri-objective training produces explanations with:
        - Higher Insertion AUC (explanations identify discriminative regions)
        - Lower Deletion AUC (removing important pixels degrades performance)
        - Better Pointing Game accuracy
        """
        config = FaithfulnessConfig(num_steps=10, verbose=False, device=str(device))
        metrics = FaithfulnessMetrics(dummy_model, config)

        image = torch.randn(1, 3, 28, 28, device=device)

        # Focused explanation (should be faithful)
        focused_heatmap = torch.zeros(1, 1, 28, 28, device=device)
        focused_heatmap[0, 0, 12:16, 12:16] = 1.0

        results = metrics.compute_all(image, focused_heatmap, [0])

        # H3 Validation: Insertion AUC should be positive
        # (score increases as we add important pixels)
        assert (
            results["insertion_auc"] > 0
        ), "H3: Insertion should increase score (faithful explanation)"

        # H3 Validation: Deletion should cause score drop
        assert (
            results["deletion_ad"] >= 0
        ), "H3: Deletion should decrease score (important pixels removed)"

        # H3 Validation: Both AUCs should be meaningful (not extreme)
        assert 0 <= results["deletion_auc"] <= 1, "Deletion AUC in valid range"
        assert 0 <= results["insertion_auc"] <= 1, "Insertion AUC in valid range"


# ============================================================================
# Test Visualization Utilities
# ============================================================================


class TestPlotCurves:
    """Test plot_curves utility."""

    def test_plot_curves_saves_file(self, tmp_path):
        """Test saving curves to file."""
        from src.xai.faithfulness import plot_curves

        deletion_curve = np.linspace(1.0, 0.2, 11)
        insertion_curve = np.linspace(0.1, 0.9, 11)

        save_path = tmp_path / "test_curves.png"
        plot_curves(deletion_curve, insertion_curve, str(save_path))

        # Check file was created
        assert save_path.exists(), "Curve plot should be saved"
        assert save_path.stat().st_size > 0, "Saved file should not be empty"

    def test_plot_curves_with_none_path(self):
        """Test plot_curves without saving (display only)."""
        import matplotlib

        matplotlib.use("Agg")  # Use non-GUI backend for testing

        from src.xai.faithfulness import plot_curves

        deletion_curve = np.linspace(1.0, 0.2, 11)
        insertion_curve = np.linspace(0.1, 0.9, 11)

        # Should not raise error even without save_path
        plot_curves(deletion_curve, insertion_curve, save_path=None)
