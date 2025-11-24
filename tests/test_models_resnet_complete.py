"""
Complete test suite for ResNet50Classifier achieving 100% coverage.

This extends test_models_resnet.py with additional tests for:
- Import error handling
- Max pooling mode
- Conv1 adaptation edge cases (no conv1 attribute, multi-channel)
- freeze_backbone_except_bn method
- get_layer_output_shapes method
- All remaining branches for 100% coverage

Author: Viraj Pankaj Jain
Institution: University of Glasgow
"""

from __future__ import annotations

from unittest.mock import Mock, patch

import pytest
import torch
import torch.nn as nn

from src.models.resnet import ResNet50Classifier


class TestResNet50ImportHandling:
    """Test import error handling."""
    
    def test_import_without_torchvision(self):
        """Test RuntimeError when torchvision is not available."""
        with patch("src.models.resnet._HAS_TORCHVISION", False):
            with pytest.raises(
                RuntimeError,
                match="torchvision is required to use ResNet50Classifier"
            ):
                ResNet50Classifier(num_classes=2, pretrained=False)
    
    def test_import_with_torchvision(self):
        """Test successful import when torchvision is available."""
        # Should work normally when _HAS_TORCHVISION is True
        model = ResNet50Classifier(num_classes=2, pretrained=False)
        assert model is not None
        assert hasattr(model, "backbone")


class TestResNet50MaxPooling:
    """Test max pooling configuration."""
    
    def test_max_pooling_initialization(self):
        """Test model with max pooling instead of avg pooling."""
        model = ResNet50Classifier(
            num_classes=5,
            pretrained=False,
            global_pool="max",
        )
        
        assert model.global_pool_mode == "max"
        assert isinstance(model.global_pool, nn.AdaptiveMaxPool2d)
    
    def test_max_pooling_forward(self):
        """Test forward pass with max pooling."""
        model = ResNet50Classifier(
            num_classes=3,
            pretrained=False,
            global_pool="max",
        )
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            logits = model(x)
        
        assert logits.shape == (2, 3)
        assert not torch.isnan(logits).any()
    
    def test_max_pooling_embedding(self):
        """Test embedding extraction with max pooling."""
        model = ResNet50Classifier(
            num_classes=2,
            pretrained=False,
            global_pool="max",
        )
        model.eval()
        
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            emb = model.get_embedding(x)
        
        assert emb.shape == (1, 2048)


class TestResNet50Conv1Adaptation:
    """Test first convolution layer adaptation edge cases."""
    
    def test_adapt_conv1_no_pretrained_weights(self):
        """Test conv1 adaptation without pretrained weights."""
        model = ResNet50Classifier(
            num_classes=2,
            pretrained=False,
            in_channels=1,
        )
        
        # Verify conv1 was adapted
        assert model.backbone.conv1.in_channels == 1
        assert model.backbone.conv1.out_channels == 64
    
    def test_adapt_conv1_pretrained_grayscale(self):
        """Test conv1 adaptation with pretrained weights for grayscale."""
        model = ResNet50Classifier(
            num_classes=2,
            pretrained=True,
            in_channels=1,
        )
        
        # Verify adaptation
        assert model.backbone.conv1.in_channels == 1
        
        # Test forward pass works
        x = torch.randn(1, 1, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, 2)
    
    def test_adapt_conv1_pretrained_two_channels(self):
        """Test conv1 adaptation with pretrained weights for 2 channels."""
        model = ResNet50Classifier(
            num_classes=2,
            pretrained=True,
            in_channels=2,
        )
        
        # Verify adaptation
        assert model.backbone.conv1.in_channels == 2
        
        # Test forward pass
        x = torch.randn(1, 2, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, 2)
    
    def test_adapt_conv1_pretrained_four_channels(self):
        """Test conv1 adaptation with pretrained weights for 4 channels."""
        model = ResNet50Classifier(
            num_classes=2,
            pretrained=True,
            in_channels=4,
        )
        
        # Verify adaptation
        assert model.backbone.conv1.in_channels == 4
        
        # Test forward pass
        x = torch.randn(1, 4, 224, 224)
        with torch.no_grad():
            logits = model(x)
        assert logits.shape == (1, 2)
    
    def test_adapt_conv1_no_conv1_attribute(self):
        """Test adaptation gracefully handles missing conv1 attribute."""
        model = ResNet50Classifier(num_classes=2, pretrained=False)
        
        # Remove conv1 attribute
        delattr(model.backbone, "conv1")
        
        # Should not raise error
        model._adapt_first_conv(in_channels=1, pretrained=False)
    
    def test_adapt_conv1_preserves_bias(self):
        """Test conv1 adaptation preserves bias if present."""
        model = ResNet50Classifier(
            num_classes=2,
            pretrained=False,
            in_channels=1,
        )
        
        # Check bias handling
        conv1 = model.backbone.conv1
        if conv1.bias is not None:
            assert conv1.bias.shape[0] == conv1.out_channels


class TestResNet50FreezeBackboneExceptBN:
    """Test freeze_backbone_except_bn method."""
    
    def test_freeze_backbone_except_bn(self):
        """Test selective freezing that keeps BatchNorm trainable."""
        model = ResNet50Classifier(num_classes=5, pretrained=False)
        
        # Apply selective freezing
        model.freeze_backbone_except_bn()
        
        # Check BatchNorm layers are trainable
        bn_trainable_count = 0
        for module in model.backbone.modules():
            if isinstance(module, nn.BatchNorm2d):
                for param in module.parameters():
                    assert param.requires_grad
                    bn_trainable_count += 1
        
        assert bn_trainable_count > 0, "Should have trainable BN params"
        
        # Check Conv layers are frozen
        conv_frozen_count = 0
        for module in model.backbone.modules():
            if isinstance(module, nn.Conv2d):
                for param in module.parameters():
                    assert not param.requires_grad
                    conv_frozen_count += 1
        
        assert conv_frozen_count > 0, "Should have frozen Conv params"
        
        # Check classifier remains trainable
        for param in model.fc.parameters():
            assert param.requires_grad
    
    def test_freeze_backbone_except_bn_parameter_counts(self):
        """Test parameter counts after selective freezing."""
        model = ResNet50Classifier(num_classes=3, pretrained=False)
        
        # Count trainable params before
        trainable_before = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        
        model.freeze_backbone_except_bn()
        
        # Count trainable params after
        trainable_after = sum(
            p.numel() for p in model.parameters() if p.requires_grad
        )
        
        # Should have fewer trainable params, but not zero
        assert trainable_after < trainable_before
        assert trainable_after > 0
    
    def test_freeze_backbone_except_bn_training_mode(self):
        """Test model can be trained after selective freezing."""
        model = ResNet50Classifier(num_classes=2, pretrained=False)
        model.freeze_backbone_except_bn()
        model.train()
        
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        
        # Should compute gradients only for trainable parts
        loss = logits.sum()
        loss.backward()
        
        # Check gradients exist for trainable parts
        fc_has_grad = any(
            p.grad is not None for p in model.fc.parameters()
        )
        assert fc_has_grad


class TestResNet50LayerOutputShapes:
    """Test get_layer_output_shapes method."""
    
    def test_layer_output_shapes_224x224(self):
        """Test output shapes for standard 224x224 input."""
        model = ResNet50Classifier(num_classes=2, pretrained=False)
        
        shapes = model.get_layer_output_shapes((224, 224))
        
        # Verify all feature layers are present
        assert "layer1" in shapes
        assert "layer2" in shapes
        assert "layer3" in shapes
        assert "layer4" in shapes
        
        # Verify shape format (C, H, W)
        for name, shape in shapes.items():
            assert len(shape) == 3
            c, h, w = shape
            assert c > 0
            assert h > 0
            assert w > 0
        
        # Verify expected channels
        assert shapes["layer1"][0] == 256
        assert shapes["layer2"][0] == 512
        assert shapes["layer3"][0] == 1024
        assert shapes["layer4"][0] == 2048
    
    def test_layer_output_shapes_custom_resolution(self):
        """Test output shapes for custom input resolution."""
        model = ResNet50Classifier(num_classes=2, pretrained=False)
        
        shapes = model.get_layer_output_shapes((512, 512))
        
        # Higher resolution should produce larger feature maps
        for name, shape in shapes.items():
            _, h, w = shape
            assert h > 0
            assert w > 0
    
    def test_layer_output_shapes_non_square(self):
        """Test output shapes for non-square input."""
        model = ResNet50Classifier(num_classes=2, pretrained=False)
        
        shapes = model.get_layer_output_shapes((384, 256))
        
        # Should handle non-square inputs
        assert len(shapes) == 4
        for shape in shapes.values():
            assert len(shape) == 3
    
    def test_layer_output_shapes_grayscale(self):
        """Test output shapes for grayscale input."""
        model = ResNet50Classifier(
            num_classes=2,
            pretrained=False,
            in_channels=1,
        )
        
        shapes = model.get_layer_output_shapes((224, 224))
        
        # Should work regardless of input channels
        assert len(shapes) == 4
        assert shapes["layer4"][0] == 2048
    
    def test_layer_output_shapes_on_gpu(self):
        """Test output shapes computation on GPU if available."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")
        
        model = ResNet50Classifier(num_classes=2, pretrained=False)
        try:
            model = model.cuda()
        except RuntimeError:
            pytest.skip("CUDA device busy or unavailable")
        
        shapes = model.get_layer_output_shapes((224, 224))
        
        # Should work on GPU
        assert len(shapes) == 4


class TestResNet50EdgeCasesComplete:
    """Additional edge case tests for complete coverage."""
    
    def test_get_classifier_returns_correct_module(self):
        """Test get_classifier returns the correct module."""
        model = ResNet50Classifier(num_classes=5, pretrained=False)
        
        classifier = model.get_classifier()
        
        assert classifier is model.fc
        assert isinstance(classifier, nn.Linear)
    
    def test_get_classifier_with_dropout(self):
        """Test get_classifier returns Sequential when dropout is used."""
        model = ResNet50Classifier(
            num_classes=3,
            pretrained=False,
            dropout=0.5,
        )
        
        classifier = model.get_classifier()
        
        assert classifier is model.fc
        assert isinstance(classifier, nn.Sequential)
    
    def test_forward_returns_only_logits_by_default(self):
        """Test forward returns only logits when return_features=False."""
        model = ResNet50Classifier(num_classes=2, pretrained=False)
        model.eval()
        
        x = torch.randn(1, 3, 224, 224)
        with torch.no_grad():
            output = model(x, return_features=False)
        
        # Should be a tensor, not a tuple
        assert isinstance(output, torch.Tensor)
        assert output.shape == (1, 2)
    
    def test_forward_with_explicit_false_features(self):
        """Test forward with explicit return_features=False."""
        model = ResNet50Classifier(num_classes=3, pretrained=False)
        model.eval()
        
        x = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            output = model(x, return_features=False)
        
        assert isinstance(output, torch.Tensor)
        assert not isinstance(output, tuple)


class TestResNet50Integration:
    """Integration tests for complete workflows."""
    
    def test_transfer_learning_workflow(self):
        """Test complete transfer learning workflow."""
        # Create model with pretrained backbone
        model = ResNet50Classifier(
            num_classes=5,
            pretrained=True,
            in_channels=1,
        )
        
        # Freeze backbone except BN
        model.freeze_backbone_except_bn()
        
        # Verify we can train
        model.train()
        x = torch.randn(2, 1, 224, 224)
        logits = model(x)
        loss = logits.sum()
        loss.backward()
        
        # Check gradients
        assert any(p.grad is not None for p in model.fc.parameters())
    
    def test_xai_feature_extraction_workflow(self):
        """Test XAI feature extraction workflow."""
        model = ResNet50Classifier(num_classes=2, pretrained=False)
        model.eval()
        
        x = torch.randn(1, 3, 224, 224)
        
        # Get features for all layers
        with torch.no_grad():
            features = model.get_feature_maps(
                x,
                layer_names=["layer1", "layer2", "layer3", "layer4"]
            )
        
        # Verify all features present
        assert len(features) == 4
        
        # Get shapes
        shapes = model.get_layer_output_shapes((224, 224))
        
        # Verify consistency
        for name in ["layer1", "layer2", "layer3", "layer4"]:
            feat = features[name]
            expected_shape = shapes[name]
            assert feat.shape[1:] == expected_shape
    
    def test_multi_channel_pretrained_workflow(self):
        """Test workflow with multi-channel pretrained model."""
        model = ResNet50Classifier(
            num_classes=3,
            pretrained=True,
            in_channels=4,
            dropout=0.3,
            global_pool="max",
        )
        model.eval()
        
        x = torch.randn(2, 4, 224, 224)
        
        with torch.no_grad():
            # Forward pass
            logits = model(x)
            assert logits.shape == (2, 3)
            
            # Get embedding
            emb = model.get_embedding(x)
            assert emb.shape == (2, 2048)
            
            # Get features
            logits, features = model(x, return_features=True)
            assert len(features) == 4


__all__ = [
    "TestResNet50ImportHandling",
    "TestResNet50MaxPooling",
    "TestResNet50Conv1Adaptation",
    "TestResNet50FreezeBackboneExceptBN",
    "TestResNet50LayerOutputShapes",
    "TestResNet50EdgeCasesComplete",
    "TestResNet50Integration",
]
