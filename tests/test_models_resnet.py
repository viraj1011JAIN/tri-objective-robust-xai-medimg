"""
Unit tests for ResNet50Classifier.

Tests cover:
- Forward pass with different input channels (RGB, grayscale)
- Feature map extraction for XAI (Grad-CAM compatibility)
- Embedding extraction for representation analysis
- Backbone freezing/unfreezing for transfer learning
- Model metadata and parameter counting
- Edge cases and error handling

Run with: pytest tests/test_models_resnet.py -v
"""

import pytest
import torch
import torch.nn as nn

from src.models.resnet import ResNet50Classifier


class TestResNet50ClassifierBasics:
    """Test basic initialization and forward pass."""

    def test_initialization_default(self):
        """Test default initialization with standard parameters."""
        model = ResNet50Classifier(num_classes=7, pretrained=False)

        assert model.num_classes == 7
        assert model.in_channels == 3
        assert model.embedding_dim == 2048
        assert model.pretrained is False
        assert model.feature_layers == ["layer1", "layer2", "layer3", "layer4"]

    def test_initialization_custom_channels(self):
        """Test initialization with custom input channels (grayscale)."""
        model = ResNet50Classifier(num_classes=2, pretrained=False, in_channels=1)

        assert model.in_channels == 1
        assert model.num_classes == 2
        # Verify first conv was adapted
        assert model.backbone.conv1.in_channels == 1

    def test_initialization_with_dropout(self):
        """Test classifier head includes dropout when specified."""
        model = ResNet50Classifier(num_classes=5, pretrained=False, dropout=0.3)

        # Classifier should be Sequential with Dropout + Linear
        assert isinstance(model.fc, nn.Sequential)
        assert isinstance(model.fc[0], nn.Dropout)
        assert model.fc[0].p == 0.3
        assert isinstance(model.fc[1], nn.Linear)

    def test_initialization_invalid_num_classes(self):
        """Test that invalid num_classes raises ValueError."""
        with pytest.raises(ValueError, match="num_classes must be"):
            ResNet50Classifier(num_classes=0, pretrained=False)

        with pytest.raises(ValueError, match="num_classes must be"):
            ResNet50Classifier(num_classes=-5, pretrained=False)

    def test_initialization_invalid_channels(self):
        """Test that invalid in_channels raises ValueError."""
        with pytest.raises(ValueError, match="in_channels must be positive"):
            ResNet50Classifier(num_classes=7, in_channels=0)

        with pytest.raises(ValueError, match="in_channels must be positive"):
            ResNet50Classifier(num_classes=7, in_channels=-1)

    def test_initialization_invalid_dropout(self):
        """Test that invalid dropout raises ValueError."""
        with pytest.raises(ValueError, match="dropout must be"):
            ResNet50Classifier(num_classes=7, dropout=1.5)

        with pytest.raises(ValueError, match="dropout must be"):
            ResNet50Classifier(num_classes=7, dropout=-0.1)


class TestResNet50ForwardPass:
    """Test forward pass with various configurations."""

    @pytest.fixture
    def model_rgb(self):
        """RGB model for testing."""
        return ResNet50Classifier(num_classes=7, pretrained=False, in_channels=3)

    @pytest.fixture
    def model_gray(self):
        """Grayscale model for testing."""
        return ResNet50Classifier(num_classes=2, pretrained=False, in_channels=1)

    def test_forward_rgb_standard_resolution(self, model_rgb):
        """Test forward pass with RGB images at 224x224."""
        x = torch.randn(4, 3, 224, 224)
        logits = model_rgb(x)

        assert logits.shape == (4, 7)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_forward_grayscale_standard_resolution(self, model_gray):
        """Test forward pass with grayscale images at 224x224."""
        x = torch.randn(8, 1, 224, 224)
        logits = model_gray(x)

        assert logits.shape == (8, 2)
        assert not torch.isnan(logits).any()

    def test_forward_high_resolution(self, model_rgb):
        """Test forward pass with higher resolution (512x512) medical images."""
        x = torch.randn(2, 3, 512, 512)
        logits = model_rgb(x)

        assert logits.shape == (2, 7)
        assert not torch.isnan(logits).any()

    def test_forward_single_sample(self, model_rgb):
        """Test forward pass with batch size 1."""
        x = torch.randn(1, 3, 224, 224)
        logits = model_rgb(x)

        assert logits.shape == (1, 7)

    def test_forward_with_features_flag(self, model_rgb):
        """Test return_features=True returns both logits and features."""
        x = torch.randn(4, 3, 224, 224)
        result = model_rgb(x, return_features=True)

        assert isinstance(result, tuple)
        assert len(result) == 2

        logits, features = result
        assert logits.shape == (4, 7)
        assert isinstance(features, dict)
        assert set(features.keys()) == {"layer1", "layer2", "layer3", "layer4"}

    def test_forward_without_features_returns_tensor(self, model_rgb):
        """Test return_features=False returns only logits tensor."""
        x = torch.randn(4, 3, 224, 224)
        result = model_rgb(x, return_features=False)

        assert isinstance(result, torch.Tensor)
        assert result.shape == (4, 7)


class TestResNet50FeatureExtraction:
    """Test feature map extraction for XAI methods."""

    @pytest.fixture
    def model(self):
        """Model for feature extraction testing."""
        return ResNet50Classifier(num_classes=7, pretrained=False, in_channels=3)

    def test_feature_maps_default_layer4(self, model):
        """Test default behavior returns layer4 only."""
        x = torch.randn(2, 3, 224, 224)
        features = model.get_feature_maps(x)

        assert len(features) == 1
        assert "layer4" in features
        assert features["layer4"].shape == (2, 2048, 7, 7)

    def test_feature_maps_single_layer(self, model):
        """Test extraction of a single specified layer."""
        x = torch.randn(4, 3, 224, 224)
        features = model.get_feature_maps(x, layer_names=["layer3"])

        assert len(features) == 1
        assert "layer3" in features
        assert features["layer3"].shape == (4, 1024, 14, 14)

    def test_feature_maps_multiple_layers(self, model):
        """Test extraction of multiple layers simultaneously."""
        x = torch.randn(2, 3, 224, 224)
        features = model.get_feature_maps(x, layer_names=["layer2", "layer3", "layer4"])

        assert len(features) == 3
        assert features["layer2"].shape == (2, 512, 28, 28)
        assert features["layer3"].shape == (2, 1024, 14, 14)
        assert features["layer4"].shape == (2, 2048, 7, 7)

    def test_feature_maps_all_layers(self, model):
        """Test extraction of all available layers."""
        x = torch.randn(2, 3, 224, 224)
        features = model.get_feature_maps(
            x, layer_names=["layer1", "layer2", "layer3", "layer4"]
        )

        assert len(features) == 4
        assert features["layer1"].shape == (2, 256, 56, 56)
        assert features["layer2"].shape == (2, 512, 28, 28)
        assert features["layer3"].shape == (2, 1024, 14, 14)
        assert features["layer4"].shape == (2, 2048, 7, 7)

    def test_feature_maps_high_resolution(self, model):
        """Test feature extraction with 512x512 input."""
        x = torch.randn(1, 3, 512, 512)
        features = model.get_feature_maps(x, layer_names=["layer4"])

        # 512 / 32 (downsampling factor) = 16
        assert features["layer4"].shape == (1, 2048, 16, 16)

    def test_feature_maps_invalid_layer_raises_error(self, model):
        """Test that requesting invalid layer raises ValueError."""
        x = torch.randn(2, 3, 224, 224)

        with pytest.raises(ValueError, match="Invalid layer name"):
            model.get_feature_maps(x, layer_names=["invalid_layer"])

        with pytest.raises(ValueError, match="Invalid layer name"):
            model.get_feature_maps(x, layer_names=["layer4", "nonexistent"])

    def test_feature_maps_preserve_gradients(self, model):
        """Test that feature maps preserve gradients for Grad-CAM."""
        x = torch.randn(2, 3, 224, 224, requires_grad=True)
        features = model.get_feature_maps(x, layer_names=["layer4"])

        # Check gradients can flow back
        loss = features["layer4"].sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()


class TestResNet50Embedding:
    """Test embedding extraction for representation analysis."""

    @pytest.fixture
    def model(self):
        """Model for embedding testing."""
        return ResNet50Classifier(num_classes=5, pretrained=False)

    def test_embedding_shape(self, model):
        """Test embedding has correct shape (B, 2048)."""
        x = torch.randn(10, 3, 224, 224)
        embeddings = model.get_embedding(x)

        assert embeddings.shape == (10, 2048)
        assert not torch.isnan(embeddings).any()

    def test_embedding_different_batch_sizes(self, model):
        """Test embedding works with various batch sizes."""
        for batch_size in [1, 4, 16, 32]:
            x = torch.randn(batch_size, 3, 224, 224)
            embeddings = model.get_embedding(x)
            assert embeddings.shape == (batch_size, 2048)

    def test_embedding_preserves_gradients(self, model):
        """Test embeddings preserve gradients."""
        x = torch.randn(4, 3, 224, 224, requires_grad=True)
        embeddings = model.get_embedding(x)

        loss = embeddings.sum()
        loss.backward()

        assert x.grad is not None


class TestResNet50FreezingMechanisms:
    """Test backbone freezing for transfer learning."""

    @pytest.fixture
    def model(self):
        """Model for freezing tests."""
        return ResNet50Classifier(num_classes=7, pretrained=False)

    @pytest.mark.skip(reason="freeze_backbone not implemented in Phase 1")
    def test_freeze_backbone(self, model):
        """Test freeze_backbone freezes backbone but not classifier."""
        model.freeze_backbone()

        # Check backbone is frozen
        for name, param in model.backbone.named_parameters():
            assert not param.requires_grad, f"Backbone param {name} should be frozen"

        # Check classifier is trainable
        for param in model.fc.parameters():
            assert param.requires_grad, "Classifier should remain trainable"

    @pytest.mark.skip(reason="freeze/unfreeze not implemented in Phase 1")
    def test_unfreeze_backbone(self, model):
        """Test unfreeze_backbone makes all parameters trainable."""
        model.freeze_backbone()  # First freeze
        model.unfreeze_backbone()  # Then unfreeze

        # All parameters should be trainable
        for param in model.parameters():
            assert param.requires_grad, "All params should be trainable after unfreeze"

    @pytest.mark.skip(reason="freeze_backbone not implemented in Phase 1")
    def test_freeze_backbone_custom_trainable_modules(self, model):
        """Test freeze_backbone with custom trainable module names."""
        # Create a custom trainable prefix
        model.freeze_backbone(trainable_module_names=["fc", "backbone.layer4"])

        # fc should be trainable
        for param in model.fc.parameters():
            assert param.requires_grad

        # layer4 should be trainable
        for param in model.backbone.layer4.parameters():
            assert param.requires_grad

        # Earlier layers should be frozen
        for param in model.backbone.layer1.parameters():
            assert not param.requires_grad

    @pytest.mark.skip(reason="freeze_backbone_except_bn not implemented in Phase 1")
    def test_freeze_backbone_except_bn(self, model):
        """Test selective freezing that keeps BatchNorm trainable."""
        model.freeze_backbone_except_bn()

        # Check BatchNorm layers are trainable
        for name, module in model.backbone.named_modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    assert param.requires_grad, f"BN {name} should be trainable"

        # Check Conv layers are frozen
        for name, module in model.backbone.named_modules():
            if isinstance(module, nn.Conv2d):
                for param in module.parameters():
                    assert not param.requires_grad, f"Conv {name} should be frozen"

    @pytest.mark.skip(reason="Freezing methods not implemented in Phase 1")
    def test_parameter_count_after_freezing(self, model):
        """Test num_parameters correctly counts trainable params."""
        total_before = model.num_parameters(trainable_only=False)
        trainable_before = model.get_num_parameters(trainable_only=True)

        model.freeze_backbone()

        total_after = model.get_num_parameters(trainable_only=False)
        trainable_after = model.get_num_parameters(trainable_only=True)

        # Total params unchanged
        assert total_before == total_after
        # Trainable params reduced (only classifier remains)
        assert trainable_after < trainable_before
        assert trainable_after > 0  # Classifier is still trainable


@pytest.mark.skip(reason="Metadata methods not in Phase 1")
class TestResNet50Metadata:
    """Test model metadata and introspection."""

    @pytest.fixture
    def model(self):
        """Model for metadata testing."""
        return ResNet50Classifier(
            num_classes=7, pretrained=False, dropout=0.2, in_channels=1
        )

    def test_get_model_info_structure(self, model):
        """Test get_model_info returns correct structure."""
        info = model.get_model_info()

        assert isinstance(info, dict)
        assert "architecture" in info
        assert "num_classes" in info
        assert "pretrained" in info
        assert "total_params" in info
        assert "trainable_params" in info
        assert "feature_layers" in info

    def test_get_model_info_values(self, model):
        """Test get_model_info returns correct values."""
        info = model.get_model_info()

        assert info["architecture"] == "ResNet50Classifier"
        assert info["num_classes"] == 7
        assert info["pretrained"] is False
        assert info["feature_layers"] == ["layer1", "layer2", "layer3", "layer4"]

        # Check parameter counts are reasonable
        assert info["total_params"] > 20_000_000  # ResNet-50 is ~25M params
        assert info["trainable_params"] == info["total_params"]  # Nothing frozen

    def test_get_model_info_after_freezing(self, model):
        """Test get_model_info reflects freezing changes."""
        info_before = model.get_model_info()
        trainable_before = info_before["trainable_params"]

        model.freeze_backbone()

        info_after = model.get_model_info()
        trainable_after = info_after["trainable_params"]

        assert trainable_after < trainable_before
        assert info_after["total_params"] == info_before["total_params"]

    def test_get_classifier_returns_correct_module(self, model):
        """Test get_classifier returns the actual classifier module."""
        classifier = model.get_classifier()

        # Should be either Linear or Sequential (if dropout is used)
        assert isinstance(classifier, (nn.Linear, nn.Sequential))

        # For this model (dropout=0.2), should be Sequential
        assert isinstance(classifier, nn.Sequential)
        assert isinstance(classifier[-1], nn.Linear)

    def test_device_property(self, model):
        """Test device property returns correct device."""
        assert model.device.type == "cpu"

        if torch.cuda.is_available():
            model = model.to("cuda")
            assert model.device.type == "cuda"

    def test_get_layer_output_shapes(self, model):
        """Test get_layer_output_shapes returns expected shapes."""
        shapes = model.get_layer_output_shapes((224, 224))

        assert shapes == {
            "layer1": (256, 56, 56),
            "layer2": (512, 28, 28),
            "layer3": (1024, 14, 14),
            "layer4": (2048, 7, 7),
        }

        # Test with different resolution
        shapes_512 = model.get_layer_output_shapes((512, 512))
        assert shapes_512["layer4"] == (2048, 16, 16)


class TestResNet50EdgeCases:
    """Test edge cases and error handling."""

    def test_very_small_batch_size(self):
        """Test model handles batch size of 1."""
        model = ResNet50Classifier(num_classes=2, pretrained=False)
        x = torch.randn(1, 3, 224, 224)

        logits = model(x)
        assert logits.shape == (1, 2)

        embeddings = model.get_embedding(x)
        assert embeddings.shape == (1, 2048)

    def test_large_batch_size(self):
        """Test model handles large batch sizes (memory permitting)."""
        model = ResNet50Classifier(num_classes=2, pretrained=False)
        x = torch.randn(64, 3, 224, 224)

        logits = model(x)
        assert logits.shape == (64, 2)

    def test_non_square_input(self):
        """Test model handles non-square inputs."""
        model = ResNet50Classifier(num_classes=2, pretrained=False)
        x = torch.randn(4, 3, 224, 448)  # Non-square

        logits = model(x)
        assert logits.shape == (4, 2)

    def test_two_channel_input(self):
        """Test model with 2-channel input (edge case)."""
        model = ResNet50Classifier(num_classes=5, pretrained=False, in_channels=2)
        x = torch.randn(4, 2, 224, 224)

        logits = model(x)
        assert logits.shape == (4, 5)

    def test_max_pooling_option(self):
        """Test model with max pooling instead of avg pooling."""
        model = ResNet50Classifier(num_classes=3, pretrained=False, global_pool="max")
        x = torch.randn(4, 3, 224, 224)

        logits = model(x)
        assert logits.shape == (4, 3)

    def test_invalid_global_pool_raises_error(self):
        """Test invalid global_pool parameter raises error."""
        with pytest.raises(ValueError, match="Unsupported global_pool"):
            ResNet50Classifier(num_classes=5, global_pool="invalid")


class TestResNet50Integration:
    """Integration tests for realistic workflows."""

    @pytest.mark.skip(reason="freeze_backbone not in Phase 1")
    def test_complete_training_workflow_simulation(self):
        """Simulate a complete training workflow."""
        # 1. Initialize with pretrained weights (in real use)
        model = ResNet50Classifier(num_classes=5, pretrained=False)

        # 2. Freeze backbone for initial training
        model.freeze_backbone()
        assert model.get_num_parameters(trainable_only=True) < 1_000_000

        # 3. Forward pass
        x = torch.randn(8, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (8, 5)

        # 4. Unfreeze for fine-tuning
        model.unfreeze_backbone()
        assert model.get_num_parameters(trainable_only=True) > 20_000_000

        # 5. Extract features for XAI
        features = model.get_feature_maps(x, layer_names=["layer4"])
        assert "layer4" in features

    def test_gradcam_workflow_simulation(self):
        """Simulate Grad-CAM workflow."""
        model = ResNet50Classifier(num_classes=7, pretrained=False)
        model.eval()

        x = torch.randn(1, 3, 224, 224, requires_grad=True)

        # Forward with features
        logits, features = model(x, return_features=True)

        # Select target class
        target_class = logits.argmax(dim=1)

        # Backward from target class
        logits[0, target_class].backward()

        # Verify gradients exist
        assert x.grad is not None
        assert features["layer4"].requires_grad

    def test_embedding_clustering_workflow(self):
        """Simulate embedding extraction for clustering."""
        model = ResNet50Classifier(num_classes=10, pretrained=False)
        model.eval()

        # Generate batch of samples
        x = torch.randn(20, 3, 224, 224)

        # Extract embeddings
        with torch.no_grad():
            embeddings = model.get_embedding(x)

        assert embeddings.shape == (20, 2048)

        # Verify embeddings are usable (compute pairwise distances)
        from torch.nn.functional import cosine_similarity

        sim = cosine_similarity(embeddings[0:1], embeddings[1:2])
        assert not torch.isnan(sim).any()


# Pytest configuration
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
