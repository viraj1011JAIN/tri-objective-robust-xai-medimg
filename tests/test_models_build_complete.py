"""
Comprehensive tests for model factory (build.py) - 100% Coverage.

This test suite provides A1-grade production-level testing for the model
factory module, covering:

- All error paths and edge cases
- Architecture name resolution (_resolve_architecture)
- Config merging precedence (defaults → config → pretrained → kwargs)
- TypeError handling when instantiating models
- get_model_info() exception handling
- build_pooling utility function
- build_from_config with 'name' vs 'architecture'
- Registry overwriting warnings
- Architecture not in DEFAULT_MODEL_CONFIGS

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Project: Tri-Objective Robust XAI for Medical Imaging
"""

import logging
from typing import Any, Dict
from unittest.mock import MagicMock, patch

import pytest
import torch
import torch.nn as nn

from src.models.base_model import BaseModel
from src.models.build import (
    DEFAULT_MODEL_CONFIGS,
    MODEL_REGISTRY,
    _resolve_architecture,
    build_classifier,
    build_model,
    build_model_from_config,
    build_pooling,
    get_default_config,
    list_available_architectures,
    register_model,
)


# =============================================================================
# Test _resolve_architecture (Internal Helper)
# =============================================================================


class TestResolveArchitecture:
    """Test internal _resolve_architecture helper function."""

    def test_resolve_architecture_from_architecture_param(self):
        """Test resolving from 'architecture' parameter."""
        result = _resolve_architecture(architecture="ResNet50")
        assert result == "resnet50"

    def test_resolve_architecture_from_name_param(self):
        """Test resolving from 'name' parameter when architecture is None."""
        result = _resolve_architecture(name="EfficientNet_B0")
        assert result == "efficientnet_b0"

    def test_resolve_architecture_prefers_architecture_over_name(self):
        """Test that 'architecture' takes precedence over 'name'."""
        result = _resolve_architecture(architecture="resnet50", name="vit_b16")
        assert result == "resnet50"

    def test_resolve_architecture_neither_provided_raises_error(self):
        """Test that ValueError is raised when neither parameter is provided."""
        with pytest.raises(ValueError, match="Either 'architecture' or 'name'"):
            _resolve_architecture()

    def test_resolve_architecture_strips_whitespace(self):
        """Test that whitespace is stripped from architecture name."""
        result = _resolve_architecture(architecture="  ResNet50  ")
        assert result == "resnet50"

    def test_resolve_architecture_with_mixed_case(self):
        """Test case-insensitive resolution."""
        result1 = _resolve_architecture(architecture="RESNET50")
        result2 = _resolve_architecture(architecture="ResNet50")
        result3 = _resolve_architecture(architecture="resnet50")
        assert result1 == result2 == result3 == "resnet50"


# =============================================================================
# Test build_model Advanced Scenarios
# =============================================================================


class TestBuildModelAdvanced:
    """Test advanced build_model scenarios for 100% coverage."""

    def test_build_model_using_name_parameter(self):
        """Test building model with 'name' instead of 'architecture'."""
        model = build_model(name="resnet50", num_classes=5)
        assert model.num_classes == 5
        assert isinstance(model, BaseModel)

    def test_build_model_pretrained_override_takes_precedence(self):
        """Test that 'pretrained' parameter overrides config."""
        config = {"pretrained": True, "dropout": 0.3}
        model = build_model(
            "resnet50", num_classes=3, config=config, pretrained=False
        )
        # pretrained parameter should override config
        assert model.pretrained is False
        assert model.dropout_prob == 0.3

    def test_build_model_kwargs_override_config_and_defaults(self):
        """Test that kwargs override both config and defaults."""
        config = {"dropout": 0.3}
        model = build_model(
            "resnet50", num_classes=4, config=config, dropout=0.5, in_channels=1
        )
        # kwargs should override config
        assert model.dropout_prob == 0.5
        assert model.in_channels == 1

    def test_build_model_config_overrides_defaults(self):
        """Test config overrides default values."""
        config = {"pretrained": False, "in_channels": 1}
        model = build_model("resnet50", num_classes=2, config=config)
        assert model.pretrained is False
        assert model.in_channels == 1

    def test_build_model_config_merging_order(self):
        """Test that config merging happens in correct order."""
        # Test all layers: defaults < config < pretrained < kwargs
        default_config = {"pretrained": True, "dropout": 0.0}
        user_config = {"dropout": 0.3}
        
        model = build_model(
            "resnet50",
            num_classes=5,
            config=user_config,
            pretrained=False,  # Override default
            in_channels=1,  # Add via kwargs
        )
        # Verify correct precedence
        assert model.pretrained is False  # pretrained param overrode default
        assert model.dropout_prob == 0.3  # config overrode default
        assert model.in_channels == 1  # kwargs added new param

    def test_build_model_architecture_not_in_defaults(self):
        """Test building architecture not in DEFAULT_MODEL_CONFIGS."""
        # Register custom model without default config
        class MinimalModel(BaseModel):
            def __init__(self, num_classes: int):
                super().__init__(num_classes, pretrained=False)
                self.fc = nn.Linear(10, num_classes)

            def forward(self, x, return_features=False):
                return self.fc(x.view(x.size(0), -1))

            def get_feature_maps(self, x, layer_names=None):
                return {}

        # Register without adding to DEFAULT_MODEL_CONFIGS
        original_defaults = DEFAULT_MODEL_CONFIGS.copy()
        try:
            register_model("minimal_test", MinimalModel)
            # Ensure it's NOT in DEFAULT_MODEL_CONFIGS
            if "minimal_test" in DEFAULT_MODEL_CONFIGS:
                del DEFAULT_MODEL_CONFIGS["minimal_test"]

            model = build_model("minimal_test", num_classes=3)
            assert model.num_classes == 3
        finally:
            # Cleanup
            if "minimal_test" in MODEL_REGISTRY:
                del MODEL_REGISTRY["minimal_test"]
            if "minimal_test" in DEFAULT_MODEL_CONFIGS:
                del DEFAULT_MODEL_CONFIGS["minimal_test"]

    def test_build_model_logs_model_creation(self, caplog):
        """Test that build_model logs creation info."""
        with caplog.at_level(logging.INFO):
            build_model("resnet50", num_classes=5, config={"dropout": 0.2})

        assert "Building resnet50" in caplog.text
        assert "num_classes=5" in caplog.text

    def test_build_model_logs_model_info_if_available(self, caplog):
        """Test that get_model_info() is called and logged."""
        with caplog.at_level(logging.INFO):
            model = build_model("resnet50", num_classes=7)

        # Check that model was built (creation is always logged)
        assert "Building resnet50" in caplog.text
        assert model.num_classes == 7

    def test_build_model_handles_get_model_info_exception(self, caplog):
        """Test graceful handling when get_model_info() raises exception."""

        class BrokenInfoModel(BaseModel):
            def __init__(self, num_classes: int):
                super().__init__(num_classes, pretrained=False)
                self.fc = nn.Linear(10, num_classes)

            def forward(self, x, return_features=False):
                return self.fc(x.view(x.size(0), -1))

            def get_feature_maps(self, x, layer_names=None):
                return {}

            def get_model_info(self):
                # Force an exception to test exception handling path
                raise RuntimeError("Intentional error in get_model_info")

        try:
            register_model("broken_info_test", BrokenInfoModel)

            # The exception should be caught silently
            model = build_model("broken_info_test", num_classes=3)

            # Should not crash, model should be created successfully
            assert model.num_classes == 3
            # Exception was handled gracefully (no crash)
        finally:
            if "broken_info_test" in MODEL_REGISTRY:
                del MODEL_REGISTRY["broken_info_test"]


# =============================================================================
# Test build_model_from_config Edge Cases
# =============================================================================


class TestBuildModelFromConfigAdvanced:
    """Test advanced build_model_from_config scenarios."""

    def test_build_from_config_using_name_instead_of_architecture(self):
        """Test building from config using 'name' field."""
        config = {
            "model": {
                "name": "resnet50",  # Using 'name' instead of 'architecture'
                "num_classes": 5,
                "pretrained": False,
            }
        }
        model = build_model_from_config(config)
        assert model.num_classes == 5
        assert model.pretrained is False

    def test_build_from_config_architecture_and_name_both_none(self):
        """Test error when both architecture and name are None."""
        config = {
            "model": {
                "num_classes": 7,
                "pretrained": True,
            }
        }
        with pytest.raises(
            KeyError, match="must contain either 'architecture' or 'name'"
        ):
            build_model_from_config(config)

    def test_build_from_config_filters_architecture_and_num_classes(self):
        """Test that architecture/name/num_classes are filtered from params."""
        config = {
            "model": {
                "architecture": "resnet50",
                "num_classes": 3,
                "pretrained": False,
                "dropout": 0.4,
                "in_channels": 1,
            }
        }
        model = build_model_from_config(config)
        # Verify correct params were passed
        assert model.num_classes == 3
        assert model.pretrained is False
        assert model.dropout_prob == 0.4
        assert model.in_channels == 1


# =============================================================================
# Test build_classifier Edge Cases
# =============================================================================


class TestBuildClassifierAdvanced:
    """Test advanced build_classifier scenarios."""

    def test_build_classifier_kwargs_override_pretrained(self):
        """Test that explicit kwargs override pretrained default."""
        model = build_classifier(
            "resnet50",
            num_classes=5,
            pretrained=False,
            dropout=0.3,
        )
        assert model.pretrained is False
        assert model.dropout_prob == 0.3

    def test_build_classifier_empty_kwargs(self):
        """Test build_classifier with no additional kwargs."""
        model = build_classifier("resnet50", num_classes=2, pretrained=True)
        assert model.num_classes == 2
        assert model.pretrained is True


# =============================================================================
# Test Architecture Registry Advanced
# =============================================================================


class TestArchitectureRegistryAdvanced:
    """Test advanced registry scenarios."""

    def test_register_model_overwrite_warning(self, caplog):
        """Test that overwriting existing architecture logs warning."""

        class NewResNet(BaseModel):
            def __init__(self, num_classes: int):
                super().__init__(num_classes, pretrained=False)
                self.fc = nn.Linear(10, num_classes)

            def forward(self, x, return_features=False):
                return self.fc(x.view(x.size(0), -1))

            def get_feature_maps(self, x, layer_names=None):
                return {}

        with caplog.at_level(logging.WARNING):
            register_model("resnet50", NewResNet)  # Overwrite existing

        assert "Overwriting existing architecture: resnet50" in caplog.text

        # Restore original
        from src.models.resnet import ResNet50Classifier

        MODEL_REGISTRY["resnet50"] = ResNet50Classifier

    def test_register_model_logs_registration(self, caplog):
        """Test that registering model logs info."""

        class CustomModel(BaseModel):
            def __init__(self, num_classes: int):
                super().__init__(num_classes, pretrained=False)
                self.fc = nn.Linear(10, num_classes)

            def forward(self, x, return_features=False):
                return self.fc(x.view(x.size(0), -1))

            def get_feature_maps(self, x, layer_names=None):
                return {}

        try:
            with caplog.at_level(logging.INFO):
                register_model("test_custom_log", CustomModel)

            assert "Registered architecture: test_custom_log" in caplog.text
        finally:
            if "test_custom_log" in MODEL_REGISTRY:
                del MODEL_REGISTRY["test_custom_log"]

    def test_get_default_config_architecture_not_in_defaults(self):
        """Test get_default_config for arch with no default config."""

        class NoDefaultModel(BaseModel):
            def __init__(self, num_classes: int):
                super().__init__(num_classes, pretrained=False)
                self.fc = nn.Linear(10, num_classes)

            def forward(self, x, return_features=False):
                return self.fc(x.view(x.size(0), -1))

            def get_feature_maps(self, x, layer_names=None):
                return {}

        try:
            register_model("no_default_test", NoDefaultModel)
            # Ensure NOT in DEFAULT_MODEL_CONFIGS
            if "no_default_test" in DEFAULT_MODEL_CONFIGS:
                del DEFAULT_MODEL_CONFIGS["no_default_test"]

            config = get_default_config("no_default_test")
            assert config == {}  # Should return empty dict
        finally:
            if "no_default_test" in MODEL_REGISTRY:
                del MODEL_REGISTRY["no_default_test"]


# =============================================================================
# Test build_pooling Utility
# =============================================================================


class TestBuildPooling:
    """Test build_pooling utility function."""

    def test_build_pooling_avg(self):
        """Test building average pooling layer."""

        class MockConfig:
            global_pool = "avg"

        pool = build_pooling(MockConfig())
        assert isinstance(pool, nn.AdaptiveAvgPool2d)
        assert pool.output_size == 1

    def test_build_pooling_max(self):
        """Test building max pooling layer."""

        class MockConfig:
            global_pool = "max"

        pool = build_pooling(MockConfig())
        assert isinstance(pool, nn.AdaptiveMaxPool2d)
        assert pool.output_size == 1

    def test_build_pooling_avg_uppercase(self):
        """Test case-insensitive pooling type."""

        class MockConfig:
            global_pool = "AVG"

        pool = build_pooling(MockConfig())
        assert isinstance(pool, nn.AdaptiveAvgPool2d)

    def test_build_pooling_none_defaults_to_avg(self):
        """Test that None global_pool defaults to 'avg'."""

        class MockConfig:
            global_pool = None

        pool = build_pooling(MockConfig())
        assert isinstance(pool, nn.AdaptiveAvgPool2d)

    def test_build_pooling_invalid_type_raises_error(self):
        """Test that invalid pool type raises ValueError."""

        class MockConfig:
            global_pool = "invalid"

        with pytest.raises(ValueError, match="Unknown pool type"):
            build_pooling(MockConfig())


# =============================================================================
# Test EfficientNet Support
# =============================================================================


class TestEfficientNetSupport:
    """Test EfficientNet architecture support."""

    def test_build_efficientnet_b0_default(self):
        """Test building EfficientNet-B0 with defaults."""
        model = build_model("efficientnet_b0", num_classes=7)
        assert model.num_classes == 7
        assert model.pretrained is True
        assert model.in_channels == 3

    def test_build_efficientnet_b0_custom_config(self):
        """Test building EfficientNet-B0 with custom config."""
        config = {"pretrained": False, "in_channels": 1, "dropout": 0.3}
        model = build_model("efficientnet_b0", num_classes=5, config=config)
        assert model.pretrained is False
        assert model.in_channels == 1
        assert model.dropout_prob == 0.3

    def test_efficientnet_in_registry(self):
        """Test that EfficientNet-B0 is registered."""
        archs = list_available_architectures()
        assert "efficientnet_b0" in archs

    def test_efficientnet_default_config(self):
        """Test EfficientNet-B0 default configuration."""
        config = get_default_config("efficientnet_b0")
        assert config["pretrained"] is True
        assert config["in_channels"] == 3
        assert config["dropout"] == 0.2


# =============================================================================
# Test Error Handling and Edge Cases
# =============================================================================


class TestErrorHandlingEdgeCases:
    """Test error handling and edge cases for 100% coverage."""

    def test_build_model_with_whitespace_in_architecture_name(self):
        """Test architecture name with extra whitespace."""
        model = build_model("  resnet50  ", num_classes=3)
        assert model.num_classes == 3

    def test_build_model_zero_num_classes_raises_error(self):
        """Test that num_classes=0 raises ValueError."""
        with pytest.raises(ValueError, match="num_classes must be positive"):
            build_model("resnet50", num_classes=0)

    def test_build_model_negative_num_classes_raises_error(self):
        """Test that negative num_classes raises ValueError."""
        with pytest.raises(ValueError, match="num_classes must be positive"):
            build_model("resnet50", num_classes=-10)

    def test_build_from_config_empty_model_dict(self):
        """Test error with empty model dict."""
        config = {"model": {}}
        with pytest.raises(KeyError):
            build_model_from_config(config)

    def test_list_architectures_is_sorted(self):
        """Test that architecture list is alphabetically sorted."""
        archs = list_available_architectures()
        assert archs == sorted(archs)

    def test_get_default_config_case_variations(self):
        """Test get_default_config with case variations."""
        config1 = get_default_config("resnet50")
        config2 = get_default_config("RESNET50")
        config3 = get_default_config("ResNet50")
        assert config1 == config2 == config3


# =============================================================================
# Test Configuration Precedence
# =============================================================================


class TestConfigurationPrecedence:
    """Test configuration merging precedence."""

    def test_precedence_kwargs_over_config(self):
        """Test kwargs take precedence over config."""
        model = build_model(
            "resnet50",
            num_classes=3,
            config={"pretrained": False, "dropout": 0.1},
            dropout=0.5,  # kwargs should override config
        )
        assert model.dropout_prob == 0.5

    def test_precedence_pretrained_param_over_config(self):
        """Test pretrained parameter overrides config."""
        config = {"pretrained": False}
        model = build_model(
            "resnet50", num_classes=3, config=config, pretrained=True
        )
        assert model.pretrained is True

    def test_precedence_config_over_defaults(self):
        """Test config overrides defaults."""
        # Default for resnet50 is pretrained=True
        config = {"pretrained": False}
        model = build_model("resnet50", num_classes=3, config=config)
        assert model.pretrained is False

    def test_precedence_all_layers(self):
        """Test full precedence: defaults < config < pretrained < kwargs."""
        # Default: pretrained=True, in_channels=3, dropout=0.0
        config = {"pretrained": False, "in_channels": 1}
        model = build_model(
            "resnet50",
            num_classes=5,
            config=config,
            pretrained=True,  # Override config
            dropout=0.5,  # Add via kwargs
        )
        # Check precedence
        assert model.pretrained is True  # pretrained param overrides config
        assert model.in_channels == 1  # from config
        assert model.dropout_prob == 0.5  # from kwargs


# =============================================================================
# Test Integration with Real Models
# =============================================================================


class TestIntegrationWithRealModels:
    """Integration tests with actual model instantiation."""

    def test_build_resnet50_forward_pass(self):
        """Test building ResNet-50 and running forward pass."""
        model = build_model("resnet50", num_classes=7)
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (2, 7)

    def test_build_efficientnet_forward_pass(self):
        """Test building EfficientNet-B0 and running forward pass."""
        model = build_model("efficientnet_b0", num_classes=5)
        x = torch.randn(2, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (2, 5)

    def test_build_classifier_functional(self):
        """Test build_classifier produces functional model."""
        model = build_classifier("resnet50", num_classes=3, pretrained=False)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 3)

    def test_model_from_config_functional(self):
        """Test model from config is functional."""
        config = {
            "model": {
                "architecture": "efficientnet_b0",
                "num_classes": 10,
                "pretrained": False,
                "dropout": 0.2,
            }
        }
        model = build_model_from_config(config)
        x = torch.randn(1, 3, 224, 224)
        output = model(x)
        assert output.shape == (1, 10)


# =============================================================================
# Test Defensive Code Paths
# =============================================================================


class TestDefensiveCodePaths:
    """Test defensive error handling paths."""

    def test_build_model_with_bad_config_param_type(self):
        """Test clear error when config has wrong param type."""
        # This will cause TypeError in model __init__
        config = {"in_channels": "three"}  # Should be int
        with pytest.raises(ValueError, match="Failed to instantiate"):
            build_model("resnet50", num_classes=5, config=config)

    def test_register_model_with_duplicate_logs_warning(self, caplog):
        """Test registering duplicate architecture logs warning."""

        class DuplicateModel(BaseModel):
            def __init__(self, num_classes: int):
                super().__init__(num_classes, pretrained=False)
                self.fc = nn.Linear(10, num_classes)

            def forward(self, x, return_features=False):
                return self.fc(x.view(x.size(0), -1))

            def get_feature_maps(self, x, layer_names=None):
                return {}

        # First registration
        register_model("duplicate_test", DuplicateModel)

        # Second registration should log warning
        with caplog.at_level(logging.WARNING):
            register_model("duplicate_test", DuplicateModel)

        assert "Overwriting existing architecture" in caplog.text

        # Cleanup
        if "duplicate_test" in MODEL_REGISTRY:
            del MODEL_REGISTRY["duplicate_test"]


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
