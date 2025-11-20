"""
Unit tests for EfficientNetB0Classifier.

Covers:
- Initialization and parameter validation
- Forward pass (with and without features)
- Feature map extraction API
- Embedding extraction API
- Backbone freezing (freeze_backbone_except_bn)
- Layer output shape utilities

Run with:
    pytest tests/test_models_efficientnet.py -v
"""

import pytest
import torch
import torch.nn as nn

import src.models.efficientnet as eff_mod
from src.models.efficientnet import EfficientNetB0Classifier

# Skip the whole module if torchvision is not available
pytestmark = pytest.mark.skipif(
    not getattr(eff_mod, "_HAS_TORCHVISION", False),
    reason="torchvision is required for EfficientNetB0Classifier tests",
)


class TestEfficientNetB0Basics:
    """Basic initialization and validation tests."""

    def test_initialization_default(self):
        """Default init: RGB, pretrained=False, dropout=0.2, avg pool."""
        model = EfficientNetB0Classifier(num_classes=7, pretrained=False)

        assert model.num_classes == 7
        assert model.in_channels == 3
        assert model.embedding_dim == 1280
        assert model.dropout_prob == pytest.approx(0.2)
        assert model.global_pool_type == "avg"
        assert set(model.feature_layers) == {
            "block3",
            "block5",
            "block7",
            "features",
        }

    def test_initialization_custom_channels_and_dropout(self):
        """Custom in_channels, dropout, and global_pool."""
        model = EfficientNetB0Classifier(
            num_classes=5,
            pretrained=False,
            in_channels=1,
            dropout=0.5,
            global_pool="max",
        )

        assert model.num_classes == 5
        assert model.in_channels == 1
        assert model.dropout_prob == pytest.approx(0.5)
        assert model.global_pool_type == "max"
        # Classifier should be Sequential when dropout > 0
        assert isinstance(model.fc, nn.Sequential)
        assert isinstance(model.fc[0], nn.Dropout)
        assert isinstance(model.fc[1], nn.Linear)

    def test_initialization_invalid_in_channels(self):
        """in_channels <= 0 should raise ValueError."""
        with pytest.raises(ValueError, match="in_channels must be positive"):
            EfficientNetB0Classifier(num_classes=3, in_channels=0)

        with pytest.raises(ValueError, match="in_channels must be positive"):
            EfficientNetB0Classifier(num_classes=3, in_channels=-1)

    def test_initialization_invalid_dropout(self):
        """Dropout outside [0, 1) should raise ValueError."""
        with pytest.raises(ValueError, match="dropout must be in"):
            EfficientNetB0Classifier(num_classes=3, dropout=1.0)

        with pytest.raises(ValueError, match="dropout must be in"):
            EfficientNetB0Classifier(num_classes=3, dropout=-0.1)

    def test_initialization_invalid_global_pool(self):
        """Unsupported global_pool should raise ValueError."""
        with pytest.raises(ValueError, match="Unsupported global_pool"):
            EfficientNetB0Classifier(num_classes=3, global_pool="median")


class TestEfficientNetB0Forward:
    """Forward pass tests."""

    @pytest.fixture
    def model_rgb(self):
        """RGB model for forward tests."""
        return EfficientNetB0Classifier(
            num_classes=5,
            pretrained=False,
            in_channels=3,
        )

    @pytest.fixture
    def model_gray(self):
        """Grayscale model for forward tests."""
        return EfficientNetB0Classifier(
            num_classes=2,
            pretrained=False,
            in_channels=1,
        )

    def test_forward_basic_logits_shape(self, model_rgb):
        """Forward pass returns logits with correct shape."""
        x = torch.randn(4, 3, 224, 224)
        logits = model_rgb(x)

        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (4, 5)
        assert not torch.isnan(logits).any()
        assert not torch.isinf(logits).any()

    def test_forward_grayscale(self, model_gray):
        """Forward pass with 1-channel input."""
        x = torch.randn(8, 1, 224, 224)
        logits = model_gray(x)

        assert logits.shape == (8, 2)
        assert not torch.isnan(logits).any()

    def test_forward_with_features_flag(self, model_rgb):
        """return_features=True returns (logits, features_dict)."""
        x = torch.randn(2, 3, 224, 224)
        logits, features = model_rgb(x, return_features=True)

        assert isinstance(logits, torch.Tensor)
        assert logits.shape == (2, 5)
        assert isinstance(features, dict)
        # Should expose only declared feature_layers
        assert set(features.keys()) == set(model_rgb.feature_layers)
        for name, feat in features.items():
            assert isinstance(feat, torch.Tensor)
            assert feat.shape[0] == 2  # batch dimension

    def test_forward_invalid_input_dim_raises(self, model_rgb):
        """3D input should raise ValueError."""
        x = torch.randn(3, 224, 224)  # Missing batch dim
        with pytest.raises(ValueError, match="Expected 4D input"):
            _ = model_rgb(x)


class TestEfficientNetB0FeatureMaps:
    """Feature map extraction tests."""

    @pytest.fixture
    def model(self):
        return EfficientNetB0Classifier(num_classes=7, pretrained=False)

    def test_feature_maps_default_deepest_layer(self, model):
        """Default get_feature_maps returns only deepest 'features' layer."""
        x = torch.randn(2, 3, 224, 224)
        feats = model.get_feature_maps(x)

        assert isinstance(feats, dict)
        assert list(feats.keys()) == ["features"]
        assert feats["features"].shape[0] == 2
        assert feats["features"].dim() == 4  # (B, C, H, W)

    def test_feature_maps_multiple_layers(self, model):
        """Request multiple layers and check downsampling ordering."""
        x = torch.randn(2, 3, 224, 224)
        layer_names = ["block3", "block5", "block7", "features"]
        feats = model.get_feature_maps(x, layer_names=layer_names)

        assert set(feats.keys()) == set(layer_names)

        # Spatial sizes should monotonically decrease
        h3, w3 = feats["block3"].shape[-2:]
        h5, w5 = feats["block5"].shape[-2:]
        h7, w7 = feats["block7"].shape[-2:]
        hf, wf = feats["features"].shape[-2:]

        assert h3 >= h5 >= h7 >= hf
        assert w3 >= w5 >= w7 >= wf

    def test_feature_maps_invalid_layer_raises(self, model):
        """Invalid layer names should raise ValueError."""
        x = torch.randn(2, 3, 224, 224)

        with pytest.raises(ValueError, match="Invalid layer name"):
            model.get_feature_maps(x, layer_names=["invalid_layer"])

        with pytest.raises(ValueError, match="Invalid layer name"):
            model.get_feature_maps(x, layer_names=["block3", "nope"])


class TestEfficientNetB0Embedding:
    """Embedding extraction tests."""

    @pytest.fixture
    def model(self):
        return EfficientNetB0Classifier(num_classes=4, pretrained=False)

    def test_embedding_shape(self, model):
        """get_embedding returns (B, 1280)."""
        x = torch.randn(10, 3, 224, 224)
        embeddings = model.get_embedding(x)

        assert isinstance(embeddings, torch.Tensor)
        assert embeddings.shape == (10, 1280)
        assert not torch.isnan(embeddings).any()

    def test_embedding_preserves_gradients(self, model):
        """Gradients can flow through embedding."""
        x = torch.randn(4, 3, 224, 224, requires_grad=True)
        embeddings = model.get_embedding(x)

        loss = embeddings.sum()
        loss.backward()

        assert x.grad is not None
        assert not torch.isnan(x.grad).any()

    def test_embedding_invalid_input_dim_raises(self, model):
        """Non-4D input should raise ValueError."""
        x = torch.randn(3, 224, 224)
        with pytest.raises(ValueError, match="Expected 4D input"):
            _ = model.get_embedding(x)


class TestEfficientNetB0Freezing:
    """Backbone freezing behaviour tests."""

    def test_freeze_backbone_except_bn(self):
        """Conv/Linear in backbone frozen; BN trainable; fc trainable."""
        model = EfficientNetB0Classifier(num_classes=7, pretrained=False)
        model.freeze_backbone_except_bn()

        # Backbone: conv/linear frozen, BN/GroupNorm trainable
        for module in model.backbone.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.GroupNorm)):
                for param in module.parameters():
                    assert param.requires_grad, "BN/GroupNorm should be trainable"
            elif isinstance(module, (nn.Conv2d, nn.Linear)):
                for param in module.parameters():
                    assert not param.requires_grad, "Conv/Linear should be frozen"

        # Classifier head must remain trainable
        for param in model.fc.parameters():
            assert param.requires_grad, "Classifier head should remain trainable"


class TestEfficientNetB0Shapes:
    """Output shape utility tests."""

    def test_get_layer_output_shapes_224(self):
        """Shapes for canonical 224x224 input."""
        model = EfficientNetB0Classifier(num_classes=3, pretrained=False)

        shapes = model.get_layer_output_shapes((224, 224))
        # Exact values are based on EfficientNet-B0 design and our implementation
        assert shapes == {
            "block3": (40, 28, 28),
            "block5": (112, 14, 14),
            "block7": (320, 7, 7),
            "features": (1280, 7, 7),
        }

    def test_get_layer_output_shapes_512(self):
        """Shapes scale correctly with larger input size."""
        model = EfficientNetB0Classifier(num_classes=3, pretrained=False)

        shapes = model.get_layer_output_shapes((512, 512))

        assert shapes["block3"][1:] == (512 // 8, 512 // 8)
        assert shapes["block5"][1:] == (512 // 16, 512 // 16)
        assert shapes["block7"][1:] == (512 // 32, 512 // 32)
        assert shapes["features"][1:] == (512 // 32, 512 // 32)
        assert shapes["features"][0] == 1280


# Allow running this test file directly
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
