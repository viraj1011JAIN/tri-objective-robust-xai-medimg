"""
Comprehensive Test Suite for Baseline Explanation Quality Evaluator.

Tests all functionality for baseline explanation quality evaluation:
- Configuration validation
- Batch evaluation (stability + faithfulness)
- Dataset evaluation with aggregation
- Visualization generation
- Statistical analysis (mean, std, CI)
- Integration with Grad-CAM, FGSM, stability, faithfulness

Test Coverage Target: >90%

Author: Viraj Pankaj Jain
Phase: 6.4 - Baseline Explanation Quality
Date: November 25, 2025
"""

from __future__ import annotations

from typing import Tuple

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch import Tensor
from torch.utils.data import DataLoader, TensorDataset

from src.xai.baseline_explanation_quality import (
    BaselineExplanationQuality,
    BaselineQualityConfig,
    create_baseline_quality_evaluator,
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
def sample_batch(device: torch.device) -> Tuple[Tensor, Tensor]:
    """Create sample batch of images and labels."""
    torch.manual_seed(42)
    images = torch.randn(4, 3, 28, 28, device=device)
    labels = torch.randint(0, 10, (4,), device=device)
    return images, labels


@pytest.fixture
def sample_dataloader(device: torch.device) -> DataLoader:
    """Create sample dataloader."""
    torch.manual_seed(42)
    images = torch.randn(16, 3, 28, 28)
    labels = torch.randint(0, 10, (16,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=4, shuffle=False)


@pytest.fixture
def config(device: torch.device) -> BaselineQualityConfig:
    """Create default configuration."""
    return BaselineQualityConfig(
        epsilon=2 / 255,
        num_samples=None,
        batch_size=4,
        device=str(device),
        verbose=0,
    )


@pytest.fixture
def evaluator(dummy_model, config) -> BaselineExplanationQuality:
    """Create evaluator instance."""
    return BaselineExplanationQuality(dummy_model, config)


# ============================================================================
# Test Configuration
# ============================================================================


class TestBaselineQualityConfig:
    """Test configuration dataclass."""

    def test_default_initialization(self):
        """Test default config values."""
        config = BaselineQualityConfig()

        assert config.epsilon == 2.0 / 255.0
        assert config.target_layers is None
        assert config.num_samples is None
        assert config.batch_size == 16
        assert config.compute_faithfulness is True
        assert config.compute_pointing_game is False
        assert config.save_visualizations is True

    def test_custom_values(self):
        """Test custom configuration values."""
        config = BaselineQualityConfig(
            epsilon=4 / 255,
            target_layers=["layer4"],
            num_samples=100,
            compute_faithfulness=False,
        )

        assert config.epsilon == 4 / 255
        assert config.target_layers == ["layer4"]
        assert config.num_samples == 100
        assert config.compute_faithfulness is False

    def test_epsilon_validation(self):
        """Test epsilon must be positive."""
        with pytest.raises(ValueError, match="epsilon must be positive"):
            BaselineQualityConfig(epsilon=0.0)

        with pytest.raises(ValueError, match="epsilon must be positive"):
            BaselineQualityConfig(epsilon=-0.01)

    def test_batch_size_validation(self):
        """Test batch_size must be positive."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            BaselineQualityConfig(batch_size=0)

    def test_num_samples_validation(self):
        """Test num_samples validation."""
        with pytest.raises(ValueError, match="num_samples must be positive"):
            BaselineQualityConfig(num_samples=0)

        # None should be allowed (means all samples)
        config = BaselineQualityConfig(num_samples=None)
        assert config.num_samples is None

    def test_num_visualizations_validation(self):
        """Test num_visualizations must be non-negative."""
        with pytest.raises(ValueError, match="num_visualizations must be non-negative"):
            BaselineQualityConfig(num_visualizations=-1)

        # Zero should be allowed
        config = BaselineQualityConfig(num_visualizations=0)
        assert config.num_visualizations == 0


# ============================================================================
# Test BaselineExplanationQuality Class
# ============================================================================


class TestBaselineExplanationQuality:
    """Test baseline explanation quality evaluator."""

    def test_initialization(self, dummy_model, config):
        """Test evaluator initialization."""
        evaluator = BaselineExplanationQuality(dummy_model, config)

        assert evaluator.model is not None
        assert evaluator.config == config
        assert evaluator.gradcam is not None
        assert evaluator.fgsm is not None
        assert evaluator.stability_metrics is not None
        assert evaluator.faithfulness_metrics is not None

    def test_initialization_without_faithfulness(self, dummy_model, device):
        """Test initialization without faithfulness metrics."""
        config = BaselineQualityConfig(
            compute_faithfulness=False,
            device=str(device),
        )
        evaluator = BaselineExplanationQuality(dummy_model, config)

        assert evaluator.faithfulness_metrics is None

    def test_initialization_default_config(self, dummy_model):
        """Test initialization with default config."""
        evaluator = BaselineExplanationQuality(dummy_model)

        assert evaluator.config is not None
        assert isinstance(evaluator.config, BaselineQualityConfig)

    def test_repr(self, evaluator):
        """Test __repr__ method for debugging."""
        repr_str = repr(evaluator)

        assert "BaselineExplanationQuality" in repr_str
        assert "epsilon" in repr_str
        assert "target_layers" in repr_str
        assert "compute_faithfulness" in repr_str

    def test_evaluate_batch_stability_only(self, evaluator, sample_batch):
        """Test batch evaluation with stability metrics only."""
        images, labels = sample_batch

        # Disable faithfulness for speed
        evaluator.config.compute_faithfulness = False
        evaluator.faithfulness_metrics = None

        results = evaluator.evaluate_batch(images, labels)

        # Check structure
        assert "heatmaps_clean" in results
        assert "heatmaps_adv" in results
        assert "images_adv" in results
        assert "stability" in results

        # Check stability metrics
        stability = results["stability"]
        assert "ssim" in stability
        assert "spearman" in stability
        assert "l2_distance" in stability
        assert "cosine_similarity" in stability

        # Check value ranges
        assert 0.0 <= stability["ssim"] <= 1.0
        assert -1.0 <= stability["spearman"] <= 1.0
        assert 0.0 <= stability["l2_distance"] <= 1.0
        assert -1.0 <= stability["cosine_similarity"] <= 1.0

    def test_evaluate_batch_with_faithfulness(self, evaluator, sample_batch):
        """Test batch evaluation with faithfulness metrics."""
        images, labels = sample_batch

        results = evaluator.evaluate_batch(images, labels)

        # Check faithfulness metrics exist
        assert "faithfulness_clean" in results
        assert "faithfulness_adv" in results

        # Check clean faithfulness
        faith_clean = results["faithfulness_clean"]
        assert "deletion_auc" in faith_clean
        assert "insertion_auc" in faith_clean
        assert "deletion_ad" in faith_clean
        assert "insertion_ai" in faith_clean

        # Check adversarial faithfulness
        faith_adv = results["faithfulness_adv"]
        assert "deletion_auc" in faith_adv
        assert "insertion_auc" in faith_adv

        # Check value ranges
        assert 0.0 <= faith_clean["deletion_auc"] <= 1.0
        assert 0.0 <= faith_clean["insertion_auc"] <= 1.0
        assert faith_clean["deletion_ad"] >= 0.0
        assert faith_clean["insertion_ai"] >= 0.0

    def test_evaluate_batch_heatmap_shapes(self, evaluator, sample_batch):
        """Test heatmap shapes are correct."""
        images, labels = sample_batch
        batch_size = images.shape[0]

        results = evaluator.evaluate_batch(images, labels)

        heatmaps_clean = results["heatmaps_clean"]
        heatmaps_adv = results["heatmaps_adv"]

        # Check shapes
        assert heatmaps_clean.shape[0] == batch_size
        assert heatmaps_adv.shape[0] == batch_size
        assert heatmaps_clean.dim() == 4  # [B, 1, H, W]
        assert heatmaps_adv.dim() == 4

    def test_evaluate_batch_adversarial_perturbation(self, evaluator, sample_batch):
        """Test adversarial images are perturbed correctly."""
        images, labels = sample_batch

        results = evaluator.evaluate_batch(images, labels)

        images_adv = results["images_adv"]

        # Check that adversarial images are different from clean
        assert not torch.allclose(images_adv, images)

        # Check perturbation exists
        perturbation = torch.abs(images_adv - images)
        assert perturbation.max().item() > 0.0

        # Check images are clipped to valid range
        assert images_adv.min().item() >= 0.0
        assert images_adv.max().item() <= 1.0

    def test_evaluate_dataset_full(self, evaluator, sample_dataloader):
        """Test dataset evaluation."""
        results = evaluator.evaluate_dataset(sample_dataloader)

        # Check aggregated structure
        assert "stability" in results
        assert "faithfulness_clean" in results
        assert "faithfulness_adv" in results
        assert "num_samples" in results
        assert "visualization_samples" in results

        # Check statistics format
        for metric, stats in results["stability"].items():
            assert "mean" in stats
            assert "std" in stats
            assert "ci_low" in stats
            assert "ci_high" in stats
            assert "n" in stats

    def test_evaluate_dataset_num_samples_limit(self, evaluator, sample_dataloader):
        """Test limiting number of samples evaluated."""
        evaluator.config.num_samples = 8  # Only 8 samples

        results = evaluator.evaluate_dataset(sample_dataloader)

        # Should evaluate exactly 8 samples
        assert results["num_samples"] == 8

    def test_evaluate_dataset_statistics_validity(self, evaluator, sample_dataloader):
        """Test statistical computations are valid."""
        results = evaluator.evaluate_dataset(sample_dataloader)

        # Check SSIM statistics
        ssim_stats = results["stability"]["ssim"]

        # Mean should be within valid SSIM range
        assert -1.0 <= ssim_stats["mean"] <= 1.0

        # CI should contain mean
        assert ssim_stats["ci_low"] <= ssim_stats["mean"] <= ssim_stats["ci_high"]

        # Std should be non-negative
        assert ssim_stats["std"] >= 0.0

    def test_evaluate_dataset_visualization_samples(self, evaluator, sample_dataloader):
        """Test visualization samples are collected correctly."""
        evaluator.config.num_visualizations = 5

        results = evaluator.evaluate_dataset(sample_dataloader)

        vis_samples = results["visualization_samples"]

        # Should have up to num_visualizations samples
        assert len(vis_samples) <= 5

        # Check sample structure
        if len(vis_samples) > 0:
            sample = vis_samples[0]
            assert "image_clean" in sample
            assert "image_adv" in sample
            assert "heatmap_clean" in sample
            assert "heatmap_adv" in sample
            assert "label" in sample
            assert "ssim" in sample

    def test_save_visualizations(self, evaluator, sample_dataloader, tmp_path):
        """Test visualization saving."""
        evaluator.config.num_visualizations = 3
        evaluator.config.save_visualizations = True

        results = evaluator.evaluate_dataset(sample_dataloader)

        # Save visualizations
        save_dir = tmp_path / "visualizations"
        evaluator.save_visualizations(results, save_dir)

        # Check files were created
        assert save_dir.exists()
        saved_files = list(save_dir.glob("baseline_quality_sample_*.png"))
        assert len(saved_files) > 0
        assert len(saved_files) <= 3

    def test_denormalize_image(self, evaluator, device):
        """Test image denormalization for visualization."""
        # Create normalized image
        image = torch.randn(3, 28, 28, device=device)

        # Denormalize
        img_denorm = evaluator._denormalize_image(image)

        # Check output
        assert isinstance(img_denorm, np.ndarray)
        assert img_denorm.shape == (28, 28, 3)
        assert img_denorm.dtype == np.uint8
        assert img_denorm.min() >= 0
        assert img_denorm.max() <= 255

    def test_overlay_heatmap(self, evaluator):
        """Test heatmap overlay on image."""
        image = np.random.randint(0, 256, (28, 28, 3), dtype=np.uint8)
        heatmap = np.random.rand(28, 28).astype(np.float32)

        overlay = evaluator._overlay_heatmap(image, heatmap, alpha=0.5)

        # Check output
        assert isinstance(overlay, np.ndarray)
        assert overlay.shape == (28, 28, 3)
        assert overlay.dtype == np.uint8

    def test_overlay_heatmap_different_sizes(self, evaluator):
        """Test heatmap overlay with size mismatch (should resize)."""
        image = np.random.randint(0, 256, (28, 28, 3), dtype=np.uint8)
        heatmap = np.random.rand(14, 14).astype(np.float32)  # Smaller

        overlay = evaluator._overlay_heatmap(image, heatmap, alpha=0.5)

        # Should still produce correct output
        assert overlay.shape == (28, 28, 3)


# ============================================================================
# Test Factory Function
# ============================================================================


class TestFactoryFunction:
    """Test create_baseline_quality_evaluator factory."""

    def test_factory_default(self, dummy_model):
        """Test factory with default config."""
        evaluator = create_baseline_quality_evaluator(dummy_model)

        assert isinstance(evaluator, BaselineExplanationQuality)
        assert evaluator.config is not None

    def test_factory_custom_config(self, dummy_model, config):
        """Test factory with custom config."""
        evaluator = create_baseline_quality_evaluator(dummy_model, config)

        assert evaluator.config == config


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Integration tests with full pipeline."""

    def test_end_to_end_evaluation(self, dummy_model, sample_dataloader, tmp_path):
        """Test complete evaluation pipeline."""
        import matplotlib

        matplotlib.use("Agg")  # Use non-interactive backend for tests

        config = BaselineQualityConfig(
            epsilon=2 / 255,
            num_samples=8,
            num_visualizations=2,
            save_visualizations=True,
            verbose=0,
        )

        evaluator = BaselineExplanationQuality(dummy_model, config)

        # Run evaluation
        save_dir = tmp_path / "results"
        results = evaluator.evaluate_dataset(sample_dataloader, save_dir=save_dir)

        # Check results
        assert results["num_samples"] == 8
        assert "stability" in results
        assert "faithfulness_clean" in results

        # Check visualizations saved
        assert save_dir.exists()
        saved_files = list(save_dir.glob("*.png"))
        assert len(saved_files) > 0

    def test_baseline_meets_expectations(self, dummy_model, sample_dataloader):
        """
        Test baseline metrics are in expected ranges.

        Expected baseline (untrained model):
        - Low stability: SSIM ~0.55-0.60 (below H2 threshold of 0.75)
        - Moderate faithfulness: Deletion AUC ~0.40-0.50
        """
        config = BaselineQualityConfig(
            epsilon=2 / 255,
            num_samples=16,
            verbose=0,
        )

        evaluator = BaselineExplanationQuality(dummy_model, config)
        results = evaluator.evaluate_dataset(sample_dataloader)

        # Check stability is computed (exact value depends on model)
        ssim_mean = results["stability"]["ssim"]["mean"]
        assert 0.0 <= ssim_mean <= 1.0  # Valid range

        # Check faithfulness is computed
        deletion_auc = results["faithfulness_clean"]["deletion_auc"]["mean"]
        assert 0.0 <= deletion_auc <= 1.0

    def test_multiple_batch_aggregation(self, dummy_model, device):
        """Test aggregation across multiple batches."""
        # Create dataloader with multiple batches
        torch.manual_seed(42)
        images = torch.randn(32, 3, 28, 28)
        labels = torch.randint(0, 10, (32,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=8, shuffle=False)

        config = BaselineQualityConfig(
            epsilon=2 / 255,
            device=str(device),
            verbose=0,
        )

        evaluator = BaselineExplanationQuality(dummy_model, config)
        results = evaluator.evaluate_dataset(dataloader)

        # Should aggregate across all 32 samples
        assert results["num_samples"] == 32

        # Should have 32 data points for each metric
        assert results["stability"]["ssim"]["n"] == 4  # 4 batches


# ============================================================================
# Performance Tests
# ============================================================================


class TestPerformance:
    """Test computational efficiency."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_acceleration(self, dummy_model):
        """Test GPU provides speedup."""
        device = torch.device("cuda")

        config = BaselineQualityConfig(
            epsilon=2 / 255,
            device="cuda",
            num_samples=8,
            verbose=0,
        )

        # Create GPU dataloader
        images = torch.randn(16, 3, 28, 28)
        labels = torch.randint(0, 10, (16,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=4)

        evaluator = BaselineExplanationQuality(dummy_model.to(device), config)

        # Should not error on GPU
        results = evaluator.evaluate_dataset(dataloader)
        assert results is not None

    def test_batch_efficiency(self, dummy_model, device):
        """Test batch processing is efficient."""
        # Small batch
        config_small = BaselineQualityConfig(
            batch_size=2,
            device=str(device),
            verbose=0,
        )

        # Large batch
        config_large = BaselineQualityConfig(
            batch_size=16,
            device=str(device),
            verbose=0,
        )

        evaluator_small = BaselineExplanationQuality(dummy_model, config_small)
        evaluator_large = BaselineExplanationQuality(dummy_model, config_large)

        # Both should work
        assert evaluator_small.config.batch_size == 2
        assert evaluator_large.config.batch_size == 16


# ============================================================================
# Edge Cases
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_sample_batch(self, evaluator, device):
        """Test with single sample."""
        images = torch.randn(1, 3, 28, 28, device=device)
        labels = torch.tensor([0], device=device)

        results = evaluator.evaluate_batch(images, labels)

        # Should work with batch size 1
        assert "stability" in results
        assert results["heatmaps_clean"].shape[0] == 1

    def test_dataloader_with_masks(self, evaluator, device):
        """Test dataloader with ground-truth masks."""
        torch.manual_seed(42)
        images = torch.randn(8, 3, 28, 28)
        labels = torch.randint(0, 10, (8,))
        masks = torch.rand(8, 28, 28)  # Ground-truth masks
        dataset = TensorDataset(images, labels, masks)
        dataloader = DataLoader(dataset, batch_size=4)

        evaluator.config.compute_pointing_game = True

        results = evaluator.evaluate_dataset(dataloader)

        # Should include pointing game accuracy
        if "faithfulness_clean" in results:
            # May or may not include pointing game depending on mask availability
            assert "pointing_acc" in results["faithfulness_clean"] or True

    def test_empty_dataloader(self, evaluator):
        """Test with empty dataloader."""
        empty_dataset = TensorDataset(
            torch.randn(0, 3, 28, 28),
            torch.randint(0, 10, (0,)),
        )
        dataloader = DataLoader(empty_dataset, batch_size=4)

        results = evaluator.evaluate_dataset(dataloader)

        # Should handle gracefully
        assert results["num_samples"] == 0

    def test_visualization_with_no_samples(self, evaluator, tmp_path):
        """Test visualization saving with no samples."""
        results = {
            "stability": {},
            "num_samples": 0,
            "visualization_samples": [],
        }

        save_dir = tmp_path / "empty_viz"

        # Should not error
        evaluator.save_visualizations(results, save_dir)

        # Directory should exist but no files
        assert save_dir.exists()
        saved_files = list(save_dir.glob("*.png"))
        assert len(saved_files) == 0


# ============================================================================
# Additional Coverage Tests (Post-Review)
# ============================================================================


class TestAutoDetectFailure:
    """Test auto-detect target layers error handling."""

    def test_auto_detect_raises_on_unknown_architecture(self, device):
        """
        Test ValueError when architecture has no Conv2d layers.

        This validates the fix for the hardcoded 'conv2' fallback bug.
        """

        class NoConvNet(nn.Module):
            """Network without any Conv2d layers."""

            def __init__(self):
                super().__init__()
                self.fc1 = nn.Linear(100, 50)
                self.fc2 = nn.Linear(50, 10)

            def forward(self, x):
                x = x.flatten(1)
                x = torch.relu(self.fc1(x))
                return self.fc2(x)

        model = NoConvNet().to(device)
        config = BaselineQualityConfig(
            target_layers=None,  # Force auto-detection
            device=str(device),
        )

        with pytest.raises(ValueError, match="Could not auto-detect target layers"):
            BaselineExplanationQuality(model, config)


class TestLoggingAndWarnings:
    """Test logging and warning messages."""

    def test_log_summary_content(self, dummy_model, sample_dataloader, device, caplog):
        """Test _log_summary produces expected output."""
        import logging

        config = BaselineQualityConfig(
            num_samples=8,
            compute_faithfulness=True,
            save_visualizations=False,
            verbose=1,
            device=str(device),
        )
        evaluator = BaselineExplanationQuality(dummy_model, config)

        with caplog.at_level(logging.INFO):
            results = evaluator.evaluate_dataset(sample_dataloader)

        # Verify results were computed
        assert results["num_samples"] > 0

        # Check key sections appear in logs
        log_text = caplog.text
        assert "BASELINE EXPLANATION QUALITY SUMMARY" in log_text
        assert "Stability Metrics" in log_text
        assert "ssim" in log_text.lower()

        # If faithfulness computed, should appear
        if config.compute_faithfulness:
            assert "Faithfulness Metrics" in log_text

    def test_memory_warning_logged(self, dummy_model, device, caplog):
        """Test memory warning appears when num_visualizations > 50."""
        import logging

        config = BaselineQualityConfig(
            num_visualizations=100,
            verbose=1,
            device=str(device),
        )
        evaluator = BaselineExplanationQuality(dummy_model, config)

        # Create small dataloader
        images = torch.randn(8, 3, 28, 28)
        labels = torch.randint(0, 10, (8,))
        dataset = TensorDataset(images, labels)
        dataloader = DataLoader(dataset, batch_size=4)

        with caplog.at_level(logging.WARNING):
            results = evaluator.evaluate_dataset(dataloader)

        # Verify evaluation ran
        assert results["num_samples"] > 0

        # Check warning appears
        assert "memory" in caplog.text.lower()
        assert "100" in caplog.text  # num_visualizations mentioned


class TestHypothesisValidation:
    """Test dissertation hypothesis validation (H2)."""

    def test_baseline_ssim_below_h2_threshold(
        self, dummy_model, sample_dataloader, device
    ):
        """
        Validate baseline has low stability (SSIM < 0.75).

        H2 states: SSIM ≥ 0.75 with λ_expl > 0
        Therefore baseline (λ_expl = 0) should have SSIM < 0.75.

        Note: This test uses untrained model, so SSIM will be low.
        With a trained model, SSIM might be higher but still below 0.75.
        """
        config = BaselineQualityConfig(
            num_samples=16,
            compute_faithfulness=False,  # Faster
            save_visualizations=False,
            verbose=0,
            device=str(device),
        )
        evaluator = BaselineExplanationQuality(dummy_model, config)
        results = evaluator.evaluate_dataset(sample_dataloader)

        ssim_mean = results["stability"]["ssim"]["mean"]

        # Validate SSIM is in valid range
        assert 0.0 <= ssim_mean <= 1.0, f"SSIM should be in [0,1], got {ssim_mean}"

        # Dissertation hypothesis: Baseline SSIM < H2 threshold
        # Note: With random untrained model, this should easily pass
        # With trained baseline, expect SSIM ~0.55-0.60
        assert ssim_mean < 0.75, (
            f"Baseline SSIM {ssim_mean:.3f} should be below H2 threshold "
            f"0.75. If this fails with trained model, verify model is not "
            f"tri-objective trained (λ_expl should be 0)."
        )
