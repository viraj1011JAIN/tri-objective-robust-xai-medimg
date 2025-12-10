"""Tests for Grad-CAM production implementation."""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.xai.gradcam_production import GradCAM


class SimpleTestModel(nn.Module):
    """Simple CNN model for testing Grad-CAM."""

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
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        self.fc = nn.Linear(64, 10)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@pytest.fixture
def device():
    """Get test device."""
    return torch.device("cpu")


@pytest.fixture
def simple_model():
    """Create simple test model."""
    model = SimpleTestModel()
    model.eval()
    return model


@pytest.fixture
def sample_images():
    """Create sample input images."""
    # Batch of 4 images, 3 channels, 32x32
    return torch.randn(4, 3, 32, 32)


class TestGradCAMInitialization:
    """Test GradCAM initialization."""

    def test_init_registers_hooks(self, simple_model, device):
        """Test that initialization registers forward and backward hooks."""
        gradcam = GradCAM(simple_model, "layer2", device)

        assert gradcam.model is simple_model
        assert gradcam.target_layer == "layer2"
        assert gradcam.device == device
        assert gradcam.gradients is None
        assert gradcam.activations is None
        assert hasattr(gradcam, "forward_handle")
        assert hasattr(gradcam, "backward_handle")

    def test_init_different_layers(self, simple_model, device):
        """Test initialization with different target layers."""
        for layer_name in ["layer1", "layer2", "layer3"]:
            gradcam = GradCAM(simple_model, layer_name, device)
            assert gradcam.target_layer == layer_name
            gradcam.cleanup()


class TestGradCAMHeatmapGeneration:
    """Test Grad-CAM heatmap generation."""

    def test_generate_heatmap_basic(self, simple_model, device, sample_images):
        """Test basic heatmap generation."""
        gradcam = GradCAM(simple_model, "layer2", device)

        heatmaps = gradcam.generate_heatmap(sample_images)

        # Check shape
        assert heatmaps.shape == (4, 32, 32)  # (B, H, W)

        # Check range [0, 1]
        assert torch.all(heatmaps >= 0.0)
        assert torch.all(heatmaps <= 1.0)

        gradcam.cleanup()

    def test_generate_heatmap_with_target_class(
        self, simple_model, device, sample_images
    ):
        """Test heatmap generation with specified target class."""
        gradcam = GradCAM(simple_model, "layer2", device)

        target_class = torch.tensor([0, 1, 2, 3])
        heatmaps = gradcam.generate_heatmap(sample_images, target_class=target_class)

        # Check shape
        assert heatmaps.shape == (4, 32, 32)

        # Check range
        assert torch.all(heatmaps >= 0.0)
        assert torch.all(heatmaps <= 1.0)

        gradcam.cleanup()

    def test_generate_heatmap_without_relu(self, simple_model, device, sample_images):
        """Test heatmap generation without ReLU (allows negative values)."""
        gradcam = GradCAM(simple_model, "layer2", device)

        heatmaps = gradcam.generate_heatmap(sample_images, use_relu=False)

        # Still normalized to [0, 1] after min-max normalization
        assert heatmaps.shape == (4, 32, 32)
        assert torch.all(heatmaps >= 0.0)
        assert torch.all(heatmaps <= 1.0)

        gradcam.cleanup()

    def test_generate_heatmap_single_image(self, simple_model, device):
        """Test heatmap generation with single image."""
        gradcam = GradCAM(simple_model, "layer2", device)

        single_image = torch.randn(1, 3, 32, 32)
        heatmaps = gradcam.generate_heatmap(single_image)

        assert heatmaps.shape == (1, 32, 32)
        assert torch.all(heatmaps >= 0.0)
        assert torch.all(heatmaps <= 1.0)

        gradcam.cleanup()

    def test_generate_heatmap_different_layers(
        self, simple_model, device, sample_images
    ):
        """Test heatmap generation from different layers."""
        for layer_name in ["layer1", "layer2", "layer3"]:
            gradcam = GradCAM(simple_model, layer_name, device)

            heatmaps = gradcam.generate_heatmap(sample_images)

            # All should produce valid heatmaps
            assert heatmaps.shape == (4, 32, 32)
            assert torch.all(heatmaps >= 0.0)
            assert torch.all(heatmaps <= 1.0)

            gradcam.cleanup()

    def test_generate_heatmap_gradient_flow(self, simple_model, device, sample_images):
        """Test that gradients flow correctly during heatmap generation."""
        gradcam = GradCAM(simple_model, "layer2", device)

        # Generate heatmaps
        heatmaps = gradcam.generate_heatmap(sample_images)

        # Check that gradients and activations were captured
        assert gradcam.gradients is not None
        assert gradcam.activations is not None

        # Check shapes match
        assert gradcam.gradients.shape == gradcam.activations.shape

        gradcam.cleanup()

    def test_heatmap_upsampling(self, simple_model, device):
        """Test that heatmaps are upsampled to input resolution."""
        gradcam = GradCAM(simple_model, "layer2", device)

        # Input image 64x64
        large_image = torch.randn(1, 3, 64, 64)
        heatmaps = gradcam.generate_heatmap(large_image)

        # Heatmap should match input resolution
        assert heatmaps.shape == (1, 64, 64)

        gradcam.cleanup()

    def test_heatmap_normalization_edge_cases(self, simple_model, device):
        """Test heatmap normalization with edge cases."""
        gradcam = GradCAM(simple_model, "layer2", device)

        # Run on multiple images to potentially get different normalization cases
        for _ in range(5):
            images = torch.randn(2, 3, 32, 32)
            heatmaps = gradcam.generate_heatmap(images)

            # All heatmaps should be normalized
            assert torch.all(heatmaps >= 0.0)
            assert torch.all(heatmaps <= 1.0)

        gradcam.cleanup()


class TestGradCAMCleanup:
    """Test GradCAM cleanup functionality."""

    def test_cleanup_removes_hooks(self, simple_model, device):
        """Test that cleanup removes forward and backward hooks."""
        gradcam = GradCAM(simple_model, "layer2", device)

        # Get initial hook count
        target_layer = dict(simple_model.named_modules())["layer2"]
        initial_forward_hooks = len(target_layer._forward_hooks)
        initial_backward_hooks = len(target_layer._backward_hooks)

        # Cleanup
        gradcam.cleanup()

        # Check hooks are removed
        final_forward_hooks = len(target_layer._forward_hooks)
        final_backward_hooks = len(target_layer._backward_hooks)

        assert final_forward_hooks < initial_forward_hooks
        assert final_backward_hooks < initial_backward_hooks

    def test_multiple_cleanup_calls(self, simple_model, device):
        """Test that multiple cleanup calls don't cause errors."""
        gradcam = GradCAM(simple_model, "layer2", device)

        # Multiple cleanup calls should not raise errors
        gradcam.cleanup()
        gradcam.cleanup()


class TestGradCAMIntegration:
    """Integration tests for Grad-CAM."""

    def test_multiple_forward_passes(self, simple_model, device, sample_images):
        """Test multiple forward passes with same GradCAM instance."""
        gradcam = GradCAM(simple_model, "layer2", device)

        # First forward pass
        heatmaps1 = gradcam.generate_heatmap(sample_images)

        # Second forward pass with same images
        heatmaps2 = gradcam.generate_heatmap(sample_images)

        # Results should be consistent
        assert torch.allclose(heatmaps1, heatmaps2, atol=1e-5)

        gradcam.cleanup()

    def test_batch_consistency(self, simple_model, device):
        """Test that batch processing gives consistent results."""
        gradcam = GradCAM(simple_model, "layer2", device)

        # Create fixed images
        torch.manual_seed(42)
        image1 = torch.randn(1, 3, 32, 32)
        image2 = torch.randn(1, 3, 32, 32)

        # Process individually
        torch.manual_seed(42)
        heatmap1_single = gradcam.generate_heatmap(image1)
        torch.manual_seed(42)
        heatmap2_single = gradcam.generate_heatmap(image2)

        # Process as batch
        torch.manual_seed(42)
        batch = torch.cat([image1, image2], dim=0)
        heatmaps_batch = gradcam.generate_heatmap(batch)

        # Check individual vs batch consistency
        assert torch.allclose(heatmap1_single, heatmaps_batch[0:1], atol=1e-5)
        assert torch.allclose(heatmap2_single, heatmaps_batch[1:2], atol=1e-5)

        gradcam.cleanup()

    def test_different_target_classes(self, simple_model, device):
        """Test GradCAM with different target classes."""
        gradcam = GradCAM(simple_model, "layer2", device)

        images = torch.randn(3, 3, 32, 32)

        # Different target classes
        target_class = torch.tensor([0, 5, 9])
        heatmaps = gradcam.generate_heatmap(images, target_class=target_class)

        # All heatmaps should be valid
        assert heatmaps.shape == (3, 32, 32)
        assert torch.all(heatmaps >= 0.0)
        assert torch.all(heatmaps <= 1.0)

        gradcam.cleanup()

    def test_predicted_class_fallback(self, simple_model, device, sample_images):
        """Test that GradCAM uses predicted class when target_class is None."""
        gradcam = GradCAM(simple_model, "layer2", device)

        # Without specifying target class (uses prediction)
        heatmaps = gradcam.generate_heatmap(sample_images, target_class=None)

        # Should still produce valid heatmaps
        assert heatmaps.shape == (4, 32, 32)
        assert torch.all(heatmaps >= 0.0)
        assert torch.all(heatmaps <= 1.0)

        gradcam.cleanup()


class TestGradCAMEdgeCases:
    """Test edge cases for Grad-CAM."""

    def test_zero_gradients_case(self, simple_model, device):
        """Test behavior when gradients might be very small."""
        gradcam = GradCAM(simple_model, "layer2", device)

        # Create images that might produce small gradients
        images = torch.zeros(2, 3, 32, 32)
        heatmaps = gradcam.generate_heatmap(images)

        # Should still produce valid heatmaps (all zeros or normalized)
        assert heatmaps.shape == (2, 32, 32)
        assert torch.all(heatmaps >= 0.0)
        assert torch.all(heatmaps <= 1.0)

        gradcam.cleanup()

    def test_large_batch_size(self, simple_model, device):
        """Test Grad-CAM with larger batch size."""
        gradcam = GradCAM(simple_model, "layer2", device)

        # Large batch
        large_batch = torch.randn(16, 3, 32, 32)
        heatmaps = gradcam.generate_heatmap(large_batch)

        assert heatmaps.shape == (16, 32, 32)
        assert torch.all(heatmaps >= 0.0)
        assert torch.all(heatmaps <= 1.0)

        gradcam.cleanup()

    def test_constant_activation_normalization(self, simple_model, device):
        """Test normalization when all activations have same value."""
        gradcam = GradCAM(simple_model, "layer2", device)

        # Run and check that normalization handles edge cases
        images = torch.ones(2, 3, 32, 32) * 0.5
        heatmaps = gradcam.generate_heatmap(images)

        # Should handle constant activation case (min == max)
        assert heatmaps.shape == (2, 32, 32)
        assert torch.all(heatmaps >= 0.0)
        assert torch.all(heatmaps <= 1.0)

        gradcam.cleanup()
