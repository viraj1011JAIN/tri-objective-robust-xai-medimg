"""
Comprehensive unit tests for model architectures.

Tests cover:
- Forward pass functionality
- Output shapes for different batch sizes
- Feature extraction
- Multi-class and multi-label outputs
- Edge cases (batch size 1, large batches)
- Gradient flow
- Device compatibility (CPU/CUDA)

Run with: pytest tests/test_models_comprehensive.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn

from src.models.base_model import BaseModel
from src.models.build import build_model


class TestModelForwardPass:
    """Test forward pass functionality for all architectures."""

    @pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0"])
    @pytest.mark.parametrize("num_classes", [7, 14, 2])
    def test_forward_pass_basic(self, arch: str, num_classes: int):
        """Test basic forward pass for all architectures."""
        model = build_model(arch, num_classes=num_classes, config={"pretrained": False})
        model.eval()

        # Create dummy input
        x = torch.randn(4, 3, 224, 224)

        # Forward pass
        with torch.no_grad():
            output = model(x)

        # Assertions
        assert output.shape == (4, num_classes), f"Expected shape (4, {num_classes}), got {output.shape}"
        assert torch.isfinite(output).all(), "Output contains NaN or Inf"
        assert output.dtype == torch.float32, f"Expected float32, got {output.dtype}"

    @pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0"])
    def test_forward_pass_training_mode(self, arch: str):
        """Test forward pass in training mode (with gradients)."""
        model = build_model(arch, num_classes=7, config={"pretrained": False})
        model.train()

        x = torch.randn(4, 3, 224, 224, requires_grad=True)
        output = model(x)

        # Test backward pass
        loss = output.sum()
        loss.backward()

        # Check gradients
        assert x.grad is not None, "Gradients not computed for input"
        assert torch.isfinite(x.grad).all(), "Gradients contain NaN or Inf"

    @pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0"])
    @pytest.mark.parametrize("in_channels", [1, 3, 4])
    def test_forward_pass_custom_channels(self, arch: str, in_channels: int):
        """Test forward pass with custom input channels."""
        model = build_model(
            arch,
            num_classes=7,
            config={"pretrained": False, "in_channels": in_channels}
        )
        model.eval()

        x = torch.randn(2, in_channels, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 7)
        assert torch.isfinite(output).all()


class TestModelOutputShapes:
    """Test output shapes for different batch sizes and configurations."""

    @pytest.mark.parametrize("batch_size", [1, 4, 8, 16, 32])
    @pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0"])
    def test_output_shape_various_batch_sizes(self, batch_size: int, arch: str):
        """Test output shapes for various batch sizes."""
        model = build_model(arch, num_classes=14, config={"pretrained": False})
        model.eval()

        x = torch.randn(batch_size, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (batch_size, 14), \
            f"Expected shape ({batch_size}, 14), got {output.shape}"

    @pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0"])
    @pytest.mark.parametrize("img_size", [224, 256])
    def test_output_shape_various_image_sizes(self, arch: str, img_size: int):
        """Test output shapes for various image sizes."""
        model = build_model(arch, num_classes=7, config={"pretrained": False})
        model.eval()

        x = torch.randn(2, 3, img_size, img_size)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (2, 7)

    def test_output_shape_edge_case_batch_1(self):
        """Test output shape for edge case: batch size 1."""
        model = build_model("resnet50", num_classes=7, config={"pretrained": False})
        model.eval()

        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 7)
        assert output.dim() == 2, "Output should be 2D even for batch size 1"


class TestModelFeatureExtraction:
    """Test feature extraction functionality."""

    @pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0"])
    def test_feature_extraction_intermediate(self, arch: str):
        """Test extracting intermediate features."""
        model = build_model(arch, num_classes=7, config={"pretrained": False})
        model.eval()

        # Access backbone features
        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            # Get features before classification head
            if hasattr(model, 'backbone'):
                features = model.backbone(x)
                assert features.dim() >= 2, "Features should be at least 2D"
                assert features.shape[0] == 2, "Batch dimension should be preserved"

    @pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0"])
    def test_feature_dimension(self, arch: str):
        """Test feature dimension before classification head."""
        model = build_model(arch, num_classes=7, config={"pretrained": False})

        # Check classifier input dimension
        if hasattr(model, 'classifier'):
            if isinstance(model.classifier, nn.Linear):
                feature_dim = model.classifier.in_features
                assert feature_dim > 0, "Feature dimension should be positive"
            elif isinstance(model.classifier, nn.Sequential):
                for layer in model.classifier:
                    if isinstance(layer, nn.Linear):
                        feature_dim = layer.in_features
                        assert feature_dim > 0
                        break


class TestModelGradientFlow:
    """Test gradient flow through models."""

    @pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0"])
    def test_gradient_flow(self, arch: str):
        """Test that gradients flow through all layers."""
        model = build_model(arch, num_classes=7, config={"pretrained": False})
        model.train()

        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        output = model(x)

        # Create dummy loss
        loss = output.sum()
        loss.backward()

        # Check that at least some parameters have gradients
        params_with_grad = sum(1 for p in model.parameters() if p.grad is not None)
        total_params = sum(1 for _ in model.parameters())

        assert params_with_grad > 0, "No parameters received gradients"
        assert params_with_grad == total_params, \
            f"Only {params_with_grad}/{total_params} parameters received gradients"

    @pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0"])
    def test_gradient_magnitude(self, arch: str):
        """Test that gradients are within reasonable magnitude."""
        model = build_model(arch, num_classes=7, config={"pretrained": False})
        model.train()

        x = torch.randn(4, 3, 224, 224)
        output = model(x)

        loss = output.mean()
        loss.backward()

        # Check gradient magnitudes
        for name, param in model.named_parameters():
            if param.grad is not None:
                grad_norm = param.grad.norm().item()
                assert torch.isfinite(param.grad).all(), \
                    f"Gradients for {name} contain NaN or Inf"
                assert grad_norm < 1e6, \
                    f"Gradient norm too large for {name}: {grad_norm}"


class TestModelDeviceCompatibility:
    """Test model compatibility with different devices."""

    @pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0"])
    def test_cpu_device(self, arch: str):
        """Test model on CPU device."""
        model = build_model(arch, num_classes=7, config={"pretrained": False})
        model = model.to('cpu')
        model.eval()

        x = torch.randn(2, 3, 224, 224, device='cpu')

        with torch.no_grad():
            output = model(x)

        assert output.device.type == 'cpu'
        assert output.shape == (2, 7)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0"])
    def test_cuda_device(self, arch: str):
        """Test model on CUDA device."""
        model = build_model(arch, num_classes=7, config={"pretrained": False})
        model = model.to('cuda')
        model.eval()

        x = torch.randn(2, 3, 224, 224, device='cuda')

        with torch.no_grad():
            output = model(x)

        assert output.device.type == 'cuda'
        assert output.shape == (2, 7)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0"])
    def test_device_transfer(self, arch: str):
        """Test transferring model between devices."""
        model = build_model(arch, num_classes=7, config={"pretrained": False})

        # Start on CPU
        x_cpu = torch.randn(2, 3, 224, 224)
        model.eval()

        with torch.no_grad():
            output_cpu = model(x_cpu)

        # Move to CUDA
        model = model.to('cuda')
        x_cuda = x_cpu.to('cuda')

        with torch.no_grad():
            output_cuda = model(x_cuda)

        # Move back to CPU
        model = model.to('cpu')

        with torch.no_grad():
            output_cpu2 = model(x_cpu)

        assert output_cpu.shape == output_cuda.shape == output_cpu2.shape


class TestModelEdgeCases:
    """Test edge cases and error handling."""

    def test_zero_batch_size_error(self):
        """Test that zero batch size raises appropriate error or returns empty."""
        model = build_model("resnet50", num_classes=7, config={"pretrained": False})
        model.eval()

        x = torch.randn(0, 3, 224, 224)

        # Zero batch may raise error or return empty tensor
        try:
            with torch.no_grad():
                output = model(x)
            # If no error, check it's empty
            assert output.shape[0] == 0, "Output should have 0 batch size"
        except (RuntimeError, ValueError):
            # Expected - some layers don't handle zero batch
            pass

    @pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0"])
    def test_single_sample_batch(self, arch: str):
        """Test model with single sample batch."""
        model = build_model(arch, num_classes=7, config={"pretrained": False})
        model.eval()

        x = torch.randn(1, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (1, 7)

    @pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0"])
    def test_large_batch_size(self, arch: str):
        """Test model with large batch size (memory permitting)."""
        model = build_model(arch, num_classes=7, config={"pretrained": False})
        model.eval()

        # Use smaller batch if GPU memory is limited
        batch_size = 64
        x = torch.randn(batch_size, 3, 224, 224)

        try:
            with torch.no_grad():
                output = model(x)
            assert output.shape == (batch_size, 7)
        except RuntimeError as e:
            if "out of memory" in str(e).lower():
                pytest.skip("Insufficient memory for large batch test")
            else:
                raise

    def test_invalid_architecture_name(self):
        """Test that invalid architecture name raises error."""
        with pytest.raises((ValueError, KeyError)):
            build_model("invalid_arch_name", num_classes=7)

    def test_zero_classes_error(self):
        """Test that zero classes raises error."""
        with pytest.raises((ValueError, AssertionError)):
            build_model("resnet50", num_classes=0)

    def test_negative_classes_error(self):
        """Test that negative classes raises error."""
        with pytest.raises((ValueError, AssertionError)):
            build_model("resnet50", num_classes=-5)


class TestModelPersistence:
    """Test model saving and loading."""

    @pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0"])
    def test_state_dict_save_load(self, arch: str, tmp_path):
        """Test saving and loading model state dict."""
        model1 = build_model(arch, num_classes=7, config={"pretrained": False})

        # Save state dict
        state_dict_path = tmp_path / "model_state.pt"
        torch.save(model1.state_dict(), state_dict_path)

        # Load into new model
        model2 = build_model(arch, num_classes=7, config={"pretrained": False})
        model2.load_state_dict(torch.load(state_dict_path))

        # Test that outputs match
        model1.eval()
        model2.eval()

        x = torch.randn(2, 3, 224, 224)

        with torch.no_grad():
            output1 = model1(x)
            output2 = model2(x)

        assert torch.allclose(output1, output2, atol=1e-6), \
            "Outputs differ after loading state dict"


class TestModelMultiLabelOutput:
    """Test models for multi-label classification."""

    @pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0"])
    def test_multilabel_output_shape(self, arch: str):
        """Test multi-label output shape (14 diseases for chest X-ray)."""
        model = build_model(arch, num_classes=14, config={"pretrained": False})
        model.eval()

        x = torch.randn(4, 3, 224, 224)

        with torch.no_grad():
            output = model(x)

        assert output.shape == (4, 14)

        # After sigmoid, values should be in [0, 1]
        probs = torch.sigmoid(output)
        assert (probs >= 0).all() and (probs <= 1).all()

    @pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0"])
    def test_multilabel_all_positive(self, arch: str):
        """Test multi-label case where all labels are positive."""
        model = build_model(arch, num_classes=14, config={"pretrained": False})
        model.train()

        x = torch.randn(2, 3, 224, 224)
        output = model(x)

        # Create all-positive targets
        targets = torch.ones(2, 14)

        # Compute BCE loss
        loss = nn.BCEWithLogitsLoss()(output, targets)

        assert torch.isfinite(loss), "Loss is not finite"
        assert loss.item() >= 0, "BCE loss should be non-negative"


class TestModelMemoryUsage:
    """Test model memory usage patterns."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    @pytest.mark.parametrize("arch", ["resnet50", "efficientnet_b0"])
    def test_memory_cleanup_after_forward(self, arch: str):
        """Test that memory is properly cleaned up after forward pass."""
        model = build_model(arch, num_classes=7, config={"pretrained": False})
        model = model.to('cuda')
        model.eval()

        # Get initial memory
        torch.cuda.empty_cache()
        initial_memory = torch.cuda.memory_allocated()

        # Forward pass
        x = torch.randn(8, 3, 224, 224, device='cuda')
        with torch.no_grad():
            _ = model(x)

        # Clear cache
        del x, _
        torch.cuda.empty_cache()

        final_memory = torch.cuda.memory_allocated()

        # Memory should return close to initial (within model parameters)
        assert final_memory <= initial_memory * 2, \
            "Memory not properly cleaned up after forward pass"


# Pytest configuration for this module
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU (deselect with '-m \"not gpu\"')"
    )
