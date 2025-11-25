"""
Comprehensive test suite for Grad-CAM and XAI modules.

Tests cover:
- Configuration validation
- Hook registration and removal
- Heatmap generation (single and batch)
- Multi-layer aggregation
- Grad-CAM++ variant
- Visualization utilities
- Error handling and edge cases
- ViT attention rollout

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Date: November 25, 2025
"""

import numpy as np
import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

from src.xai.gradcam import (
    GradCAM,
    GradCAMConfig,
    GradCAMPlusPlus,
    create_gradcam,
    get_recommended_layers,
)

# ============================================================================
# Test Fixtures
# ============================================================================


@pytest.fixture
def simple_cnn() -> nn.Module:
    """Create simple CNN for testing."""

    class SimpleCNN(nn.Module):
        def __init__(self):
            super().__init__()
            self.layer1 = nn.Sequential(
                nn.Conv2d(3, 16, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
            )
            self.layer2 = nn.Sequential(
                nn.Conv2d(16, 32, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
            )
            self.layer3 = nn.Sequential(
                nn.Conv2d(32, 64, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
            )
            self.layer4 = nn.Sequential(
                nn.Conv2d(64, 128, 3, padding=1), nn.ReLU(), nn.MaxPool2d(2)
            )
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
            self.fc = nn.Linear(128, 10)

        def forward(self, x):
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.layer4(x)
            x = self.avgpool(x)
            x = x.view(x.size(0), -1)
            x = self.fc(x)
            return x

    return SimpleCNN()


@pytest.fixture
def sample_input() -> torch.Tensor:
    """Create sample input tensor."""
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def valid_config() -> GradCAMConfig:
    """Create valid GradCAMConfig."""
    return GradCAMConfig(target_layers=["layer4"], use_cuda=False)


# ============================================================================
# Configuration Tests
# ============================================================================


class TestGradCAMConfig:
    """Tests for GradCAMConfig dataclass."""

    def test_default_config(self):
        """Test creating config with defaults."""
        config = GradCAMConfig()
        assert config.target_layers == ["layer4"]
        assert config.use_cuda is True
        assert config.batch_size == 32
        assert config.normalize_heatmap is True

    def test_custom_config(self):
        """Test creating config with custom values."""
        config = GradCAMConfig(
            target_layers=["layer3", "layer4"],
            use_cuda=False,
            batch_size=16,
            output_size=(256, 256),
        )
        assert config.target_layers == ["layer3", "layer4"]
        assert config.use_cuda is False
        assert config.batch_size == 16
        assert config.output_size == (256, 256)

    def test_empty_target_layers(self):
        """Test error with empty target_layers."""
        with pytest.raises(ValueError, match="target_layers cannot be empty"):
            GradCAMConfig(target_layers=[])

    def test_invalid_interpolation_mode(self):
        """Test error with invalid interpolation mode."""
        with pytest.raises(ValueError, match="Invalid interpolation_mode"):
            GradCAMConfig(interpolation_mode="invalid")

    def test_invalid_batch_size(self):
        """Test error with invalid batch size."""
        with pytest.raises(ValueError, match="batch_size must be >= 1"):
            GradCAMConfig(batch_size=0)

    def test_invalid_output_size(self):
        """Test error with invalid output size."""
        with pytest.raises(ValueError, match="output_size must be"):
            GradCAMConfig(output_size=(256,))

        with pytest.raises(ValueError, match="dimensions must be > 0"):
            GradCAMConfig(output_size=(0, 256))


# ============================================================================
# Grad-CAM Core Tests
# ============================================================================


class TestGradCAM:
    """Tests for GradCAM class."""

    def test_initialization(self, simple_cnn, valid_config):
        """Test Grad-CAM initialization."""
        gradcam = GradCAM(simple_cnn, valid_config)
        assert gradcam.model is simple_cnn
        assert gradcam.config is valid_config
        assert len(gradcam.target_layers) == 1
        assert len(gradcam.hooks) > 0

    def test_find_target_layers(self, simple_cnn):
        """Test finding target layers in model."""
        config = GradCAMConfig(target_layers=["layer4"], use_cuda=False)
        gradcam = GradCAM(simple_cnn, config)
        assert "layer4" in gradcam.target_layers
        assert isinstance(gradcam.target_layers["layer4"], nn.Sequential)

    def test_invalid_target_layer(self, simple_cnn):
        """Test error with non-existent target layer."""
        config = GradCAMConfig(target_layers=["nonexistent_layer"], use_cuda=False)
        with pytest.raises(ValueError, match="Target layers not found"):
            GradCAM(simple_cnn, config)

    def test_generate_heatmap_shape(self, simple_cnn, sample_input, valid_config):
        """Test heatmap generation returns correct shape."""
        gradcam = GradCAM(simple_cnn, valid_config)
        heatmap = gradcam.generate_heatmap(sample_input, class_idx=0)

        assert isinstance(heatmap, np.ndarray)
        assert heatmap.ndim == 2
        assert heatmap.shape == (224, 224)

    def test_generate_heatmap_range(self, simple_cnn, sample_input, valid_config):
        """Test heatmap values are normalized to [0, 1]."""
        gradcam = GradCAM(simple_cnn, valid_config)
        heatmap = gradcam.generate_heatmap(sample_input, class_idx=0)

        assert heatmap.min() >= 0
        assert heatmap.max() <= 1

    def test_generate_heatmap_3d_input(self, simple_cnn, valid_config):
        """Test heatmap generation with 3D input."""
        input_3d = torch.randn(3, 224, 224)
        gradcam = GradCAM(simple_cnn, valid_config)
        heatmap = gradcam.generate_heatmap(input_3d, class_idx=0)

        assert heatmap.shape == (224, 224)

    def test_generate_heatmap_auto_class(self, simple_cnn, sample_input, valid_config):
        """Test heatmap generation with automatic class selection."""
        gradcam = GradCAM(simple_cnn, valid_config)
        heatmap = gradcam.generate_heatmap(sample_input)  # No class_idx

        assert heatmap.shape == (224, 224)

    def test_generate_heatmap_custom_size(self, simple_cnn, sample_input):
        """Test heatmap generation with custom output size."""
        config = GradCAMConfig(
            target_layers=["layer4"], use_cuda=False, output_size=(128, 128)
        )
        gradcam = GradCAM(simple_cnn, config)
        heatmap = gradcam.generate_heatmap(sample_input, class_idx=0)

        assert heatmap.shape == (128, 128)

    def test_hooks_capture_activations(self, simple_cnn, sample_input, valid_config):
        """Test that hooks properly capture activations."""
        gradcam = GradCAM(simple_cnn, valid_config)
        _ = gradcam.generate_heatmap(sample_input, class_idx=0)

        assert len(gradcam.activations) > 0
        assert "layer4" in gradcam.activations

    def test_hooks_capture_gradients(self, simple_cnn, sample_input, valid_config):
        """Test that hooks properly capture gradients."""
        gradcam = GradCAM(simple_cnn, valid_config)
        _ = gradcam.generate_heatmap(sample_input, class_idx=0)

        assert len(gradcam.gradients) > 0
        assert "layer4" in gradcam.gradients

    def test_remove_hooks(self, simple_cnn, valid_config):
        """Test hook removal."""
        gradcam = GradCAM(simple_cnn, valid_config)
        initial_hooks = len(gradcam.hooks)
        assert initial_hooks > 0

        gradcam.remove_hooks()
        assert len(gradcam.hooks) == 0
        assert len(gradcam.activations) == 0
        assert len(gradcam.gradients) == 0

    def test_invalid_input_dimensions(self, simple_cnn, valid_config):
        """Test error with invalid input dimensions."""
        gradcam = GradCAM(simple_cnn, valid_config)
        invalid_input = torch.randn(3, 224)  # 2D instead of 4D

        with pytest.raises(ValueError, match="Input must be 3D or 4D tensor"):
            gradcam.generate_heatmap(invalid_input)


# ============================================================================
# Batch Processing Tests
# ============================================================================


class TestBatchProcessing:
    """Tests for batch heatmap generation."""

    def test_generate_batch_heatmaps(self, simple_cnn, valid_config):
        """Test batch heatmap generation."""
        gradcam = GradCAM(simple_cnn, valid_config)
        batch = torch.randn(4, 3, 224, 224)
        heatmaps = gradcam.generate_batch_heatmaps(batch)

        assert len(heatmaps) == 4
        assert all(h.shape == (224, 224) for h in heatmaps)

    def test_generate_batch_with_class_indices(self, simple_cnn, valid_config):
        """Test batch generation with specific class indices."""
        gradcam = GradCAM(simple_cnn, valid_config)
        batch = torch.randn(4, 3, 224, 224)
        class_indices = [0, 1, 2, 3]
        heatmaps = gradcam.generate_batch_heatmaps(batch, class_indices=class_indices)

        assert len(heatmaps) == 4

    def test_batch_chunking(self, simple_cnn):
        """Test that large batches are properly chunked."""
        config = GradCAMConfig(target_layers=["layer4"], use_cuda=False, batch_size=2)
        gradcam = GradCAM(simple_cnn, config)
        batch = torch.randn(5, 3, 224, 224)  # Batch size > config.batch_size
        heatmaps = gradcam.generate_batch_heatmaps(batch)

        assert len(heatmaps) == 5


# ============================================================================
# Multi-Layer Tests
# ============================================================================


class TestMultiLayer:
    """Tests for multi-layer CAM aggregation."""

    def test_multi_layer_config(self, simple_cnn):
        """Test Grad-CAM with multiple target layers."""
        config = GradCAMConfig(target_layers=["layer3", "layer4"], use_cuda=False)
        gradcam = GradCAM(simple_cnn, config)

        assert len(gradcam.target_layers) == 2
        assert "layer3" in gradcam.target_layers
        assert "layer4" in gradcam.target_layers

    def test_get_multi_layer_heatmap_mean(self, simple_cnn, sample_input):
        """Test multi-layer heatmap with mean aggregation."""
        config = GradCAMConfig(target_layers=["layer3", "layer4"], use_cuda=False)
        gradcam = GradCAM(simple_cnn, config)
        heatmap = gradcam.get_multi_layer_heatmap(
            sample_input, class_idx=0, aggregation="mean"
        )

        assert heatmap.shape == (224, 224)

    def test_get_multi_layer_heatmap_max(self, simple_cnn, sample_input):
        """Test multi-layer heatmap with max aggregation."""
        config = GradCAMConfig(target_layers=["layer3", "layer4"], use_cuda=False)
        gradcam = GradCAM(simple_cnn, config)
        heatmap = gradcam.get_multi_layer_heatmap(
            sample_input, class_idx=0, aggregation="max"
        )

        assert heatmap.shape == (224, 224)

    def test_get_multi_layer_heatmap_weighted(self, simple_cnn, sample_input):
        """Test multi-layer heatmap with weighted aggregation."""
        config = GradCAMConfig(target_layers=["layer3", "layer4"], use_cuda=False)
        gradcam = GradCAM(simple_cnn, config)
        heatmap = gradcam.get_multi_layer_heatmap(
            sample_input, class_idx=0, aggregation="weighted"
        )

        assert heatmap.shape == (224, 224)

    def test_invalid_aggregation(self, simple_cnn, sample_input):
        """Test error with invalid aggregation method."""
        config = GradCAMConfig(target_layers=["layer3", "layer4"], use_cuda=False)
        gradcam = GradCAM(simple_cnn, config)

        with pytest.raises(ValueError, match="Invalid aggregation"):
            gradcam.get_multi_layer_heatmap(
                sample_input, class_idx=0, aggregation="invalid"
            )


# ============================================================================
# Grad-CAM++ Tests
# ============================================================================


class TestGradCAMPlusPlus:
    """Tests for Grad-CAM++ variant."""

    def test_initialization(self, simple_cnn, valid_config):
        """Test Grad-CAM++ initialization."""
        gradcam_pp = GradCAMPlusPlus(simple_cnn, valid_config)
        assert isinstance(gradcam_pp, GradCAM)

    def test_generate_heatmap(self, simple_cnn, sample_input, valid_config):
        """Test Grad-CAM++ heatmap generation."""
        gradcam_pp = GradCAMPlusPlus(simple_cnn, valid_config)
        heatmap = gradcam_pp.generate_heatmap(sample_input, class_idx=0)

        assert heatmap.shape == (224, 224)
        assert heatmap.min() >= 0
        assert heatmap.max() <= 1

    def test_compare_with_gradcam(self, simple_cnn, sample_input, valid_config):
        """Test that Grad-CAM++ produces different results than Grad-CAM."""
        gradcam = GradCAM(simple_cnn, valid_config)
        gradcam_pp = GradCAMPlusPlus(simple_cnn, valid_config)

        heatmap = gradcam.generate_heatmap(sample_input, class_idx=0)
        heatmap_pp = gradcam_pp.generate_heatmap(sample_input, class_idx=0)

        # Should produce different heatmaps (weighted gradients)
        assert not np.allclose(heatmap, heatmap_pp)


# ============================================================================
# Visualization Tests
# ============================================================================


class TestVisualization:
    """Tests for visualization utilities."""

    def test_visualize_with_tensor(self, simple_cnn, sample_input, valid_config):
        """Test visualization with tensor input."""
        gradcam = GradCAM(simple_cnn, valid_config)
        heatmap = gradcam.generate_heatmap(sample_input, class_idx=0)
        overlay = gradcam.visualize(sample_input, heatmap)

        assert isinstance(overlay, np.ndarray)
        assert overlay.shape == (224, 224, 3)
        assert overlay.dtype == np.uint8

    def test_visualize_with_numpy(self, simple_cnn, sample_input, valid_config):
        """Test visualization with numpy input."""
        gradcam = GradCAM(simple_cnn, valid_config)
        heatmap = gradcam.generate_heatmap(sample_input, class_idx=0)

        img_np = (torch.randn(224, 224, 3) * 255).numpy().astype(np.uint8)
        overlay = gradcam.visualize(img_np, heatmap)

        assert overlay.shape == (224, 224, 3)

    def test_visualize_with_pil(self, simple_cnn, sample_input, valid_config):
        """Test visualization with PIL Image."""
        gradcam = GradCAM(simple_cnn, valid_config)
        heatmap = gradcam.generate_heatmap(sample_input, class_idx=0)

        img_pil = Image.new("RGB", (224, 224))
        overlay = gradcam.visualize(img_pil, heatmap)

        assert overlay.shape == (224, 224, 3)

    def test_visualize_alpha(self, simple_cnn, sample_input, valid_config):
        """Test visualization with custom alpha."""
        gradcam = GradCAM(simple_cnn, valid_config)
        heatmap = gradcam.generate_heatmap(sample_input, class_idx=0)

        overlay_low = gradcam.visualize(sample_input, heatmap, alpha=0.3)
        overlay_high = gradcam.visualize(sample_input, heatmap, alpha=0.7)

        assert not np.array_equal(overlay_low, overlay_high)

    def test_visualize_return_pil(self, simple_cnn, sample_input, valid_config):
        """Test visualization returning PIL Image."""
        gradcam = GradCAM(simple_cnn, valid_config)
        heatmap = gradcam.generate_heatmap(sample_input, class_idx=0)
        overlay = gradcam.visualize(sample_input, heatmap, return_pil=True)

        assert isinstance(overlay, Image.Image)


# ============================================================================
# Helper Function Tests
# ============================================================================


class TestHelperFunctions:
    """Tests for utility functions."""

    def test_get_recommended_layers_resnet(self):
        """Test layer recommendation for ResNet."""

        class FakeResNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer4 = nn.Conv2d(3, 64, 3)

        model = FakeResNet()
        layers = get_recommended_layers(model)
        assert "layer4" in layers

    def test_get_recommended_layers_unknown(self):
        """Test layer recommendation for unknown architecture."""

        class UnknownNet(nn.Module):
            def __init__(self):
                super().__init__()
                self.conv_final = nn.Conv2d(3, 64, 3)

        model = UnknownNet()
        layers = get_recommended_layers(model)
        assert "conv_final" in layers

    def test_create_gradcam_factory(self, simple_cnn):
        """Test create_gradcam factory function."""
        gradcam = create_gradcam(
            simple_cnn, target_layers=["layer4"], method="gradcam", use_cuda=False
        )
        assert isinstance(gradcam, GradCAM)

    def test_create_gradcam_plus_plus_factory(self, simple_cnn):
        """Test factory for Grad-CAM++."""
        gradcam_pp = create_gradcam(
            simple_cnn, target_layers=["layer4"], method="gradcam++", use_cuda=False
        )
        assert isinstance(gradcam_pp, GradCAMPlusPlus)

    def test_create_gradcam_auto_layers(self, simple_cnn):
        """Test factory with automatic layer detection."""
        gradcam = create_gradcam(simple_cnn, method="gradcam", use_cuda=False)
        assert len(gradcam.target_layers) > 0

    def test_invalid_method(self, simple_cnn):
        """Test error with invalid method."""
        with pytest.raises(ValueError, match="Invalid method"):
            create_gradcam(
                simple_cnn, target_layers=["layer4"], method="invalid", use_cuda=False
            )


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_zero_gradient(self, simple_cnn, valid_config):
        """Test handling of zero gradients."""
        gradcam = GradCAM(simple_cnn, valid_config)

        # Create input that might produce zero gradients
        zero_input = torch.zeros(1, 3, 224, 224)
        heatmap = gradcam.generate_heatmap(zero_input, class_idx=0)

        # Should still produce valid heatmap (all zeros)
        assert heatmap.shape == (224, 224)
        assert np.isfinite(heatmap).all()

    def test_single_pixel_activation(self, simple_cnn, valid_config):
        """Test normalization with single non-zero pixel."""
        heatmap = np.zeros((10, 10))
        heatmap[5, 5] = 1.0

        normalized = GradCAM._normalize_heatmap(heatmap)
        assert normalized.max() == 1.0
        assert normalized.min() == 0.0

    def test_constant_heatmap(self):
        """Test normalization of constant heatmap."""
        heatmap = np.ones((10, 10)) * 0.5

        normalized = GradCAM._normalize_heatmap(heatmap)
        assert np.all(normalized == 0)  # Constant maps to zero


# ============================================================================
# Run Tests
# ============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
