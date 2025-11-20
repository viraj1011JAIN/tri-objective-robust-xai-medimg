"""
Tests for model factory (build.py).

Tests cover:
- Basic model building with valid architectures
- Configuration merging (defaults + user config)
- Error handling for invalid inputs
- Registry manipulation
- Config-based instantiation

Run with: pytest tests/test_models_build.py -v
"""

import pytest
import torch

from src.models.base_model import BaseModel
from src.models.build import (
    build_classifier,
    build_model,
    build_model_from_config,
    get_default_config,
    list_available_architectures,
    register_model,
)


class TestBuildModel:
    """Test basic model building functionality."""

    def test_build_resnet50_default_config(self):
        """Test building ResNet-50 with default configuration."""
        model = build_model("resnet50", num_classes=7)

        assert model.num_classes == 7
        assert model.pretrained is True  # Default
        assert model.in_channels == 3  # Default
        assert isinstance(model, BaseModel)

    def test_build_resnet50_custom_config(self):
        """Test building ResNet-50 with custom configuration."""
        config = {
            "pretrained": False,
            "in_channels": 1,
            "dropout": 0.3,
        }
        model = build_model("resnet50", num_classes=2, config=config)

        assert model.num_classes == 2
        assert model.pretrained is False
        assert model.in_channels == 1
        assert model.dropout_prob == 0.3

    def test_build_model_case_insensitive(self):
        """Test that architecture names are case-insensitive."""
        model1 = build_model("ResNet50", num_classes=5)
        model2 = build_model("RESNET50", num_classes=5)
        model3 = build_model("resnet50", num_classes=5)

        assert all(m.num_classes == 5 for m in [model1, model2, model3])

    def test_build_model_invalid_architecture(self):
        """Test that invalid architecture raises ValueError."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            build_model("invalid_model", num_classes=7)

    def test_build_model_invalid_num_classes(self):
        """Test that invalid num_classes raises ValueError."""
        with pytest.raises(ValueError, match="num_classes must be positive"):
            build_model("resnet50", num_classes=0)

        with pytest.raises(ValueError, match="num_classes must be positive"):
            build_model("resnet50", num_classes=-5)

    def test_build_model_partial_config_uses_defaults(self):
        """Test that partial config merges with defaults."""
        config = {"dropout": 0.5}  # Only override dropout
        model = build_model("resnet50", num_classes=3, config=config)

        # Custom value
        assert model.dropout_prob == 0.5
        # Default values
        assert model.pretrained is True
        assert model.in_channels == 3

    def test_build_model_empty_config_uses_all_defaults(self):
        """Test that empty config uses all defaults."""
        model = build_model("resnet50", num_classes=10, config={})

        assert model.pretrained is True
        assert model.in_channels == 3
        assert model.dropout_prob == 0.0

    def test_build_model_none_config_uses_all_defaults(self):
        """Test that None config uses all defaults."""
        model = build_model("resnet50", num_classes=10, config=None)

        assert model.pretrained is True
        assert model.in_channels == 3


class TestBuildModelFromConfig:
    """Test config-based model building."""

    def test_build_from_valid_config(self):
        """Test building model from complete config."""
        config = {
            "model": {
                "architecture": "resnet50",
                "num_classes": 7,
                "pretrained": True,
                "dropout": 0.2,
            }
        }
        model = build_model_from_config(config)

        assert model.num_classes == 7
        assert model.pretrained is True
        assert model.dropout_prob == 0.2

    def test_build_from_config_minimal(self):
        """Test building with minimal config (only required fields)."""
        config = {
            "model": {
                "architecture": "resnet50",
                "num_classes": 5,
            }
        }
        model = build_model_from_config(config)

        assert model.num_classes == 5
        # Should use defaults for other params
        assert model.pretrained is True

    def test_build_from_config_missing_model_key(self):
        """Test that missing 'model' key raises KeyError."""
        config = {"data": {"batch_size": 32}}  # Wrong top-level key

        with pytest.raises(KeyError, match="Config must contain a 'model' key"):
            build_model_from_config(config)

    def test_build_from_config_missing_architecture(self):
        """Test that missing 'architecture' raises KeyError."""
        config = {
            "model": {
                "num_classes": 7,
                "pretrained": True,
            }
        }

        with pytest.raises(KeyError, match="must contain 'architecture' key"):
            build_model_from_config(config)

    def test_build_from_config_missing_num_classes(self):
        """Test that missing 'num_classes' raises KeyError."""
        config = {
            "model": {
                "architecture": "resnet50",
                "pretrained": True,
            }
        }

        with pytest.raises(KeyError, match="must contain 'num_classes' key"):
            build_model_from_config(config)

    def test_build_from_config_with_nested_structure(self):
        """Test building from realistic nested config."""
        config = {
            "experiment": {
                "name": "baseline_resnet50",
                "seed": 42,
            },
            "model": {
                "architecture": "resnet50",
                "num_classes": 7,
                "pretrained": True,
                "in_channels": 1,
                "dropout": 0.3,
            },
            "training": {
                "epochs": 100,
                "batch_size": 32,
            },
        }
        model = build_model_from_config(config)

        assert model.num_classes == 7
        assert model.in_channels == 1
        assert model.dropout_prob == 0.3


class TestBuildClassifier:
    """Test convenience builder function."""

    def test_build_classifier_basic(self):
        """Test basic classifier building."""
        model = build_classifier("resnet50", num_classes=5)

        assert model.num_classes == 5
        assert model.pretrained is True

    def test_build_classifier_with_pretrained_false(self):
        """Test building with pretrained=False."""
        model = build_classifier("resnet50", num_classes=3, pretrained=False)

        assert model.pretrained is False

    def test_build_classifier_with_kwargs(self):
        """Test passing additional kwargs."""
        model = build_classifier(
            "resnet50",
            num_classes=7,
            pretrained=True,
            in_channels=1,
            dropout=0.4,
        )

        assert model.num_classes == 7
        assert model.in_channels == 1
        assert model.dropout_prob == 0.4


class TestArchitectureRegistry:
    """Test architecture listing and registration."""

    def test_list_available_architectures(self):
        """Test listing available architectures."""
        architectures = list_available_architectures()

        assert isinstance(architectures, list)
        assert "resnet50" in architectures
        # List should be sorted
        assert architectures == sorted(architectures)

    def test_get_default_config_resnet50(self):
        """Test getting default config for ResNet-50."""
        config = get_default_config("resnet50")

        assert isinstance(config, dict)
        assert "pretrained" in config
        assert "in_channels" in config
        assert config["pretrained"] is True
        assert config["in_channels"] == 3

    def test_get_default_config_case_insensitive(self):
        """Test get_default_config is case-insensitive."""
        config1 = get_default_config("resnet50")
        config2 = get_default_config("ResNet50")

        assert config1 == config2

    def test_get_default_config_invalid_architecture(self):
        """Test that invalid architecture raises ValueError."""
        with pytest.raises(ValueError, match="Unknown architecture"):
            get_default_config("nonexistent_model")

    def test_get_default_config_returns_copy(self):
        """Test that get_default_config returns a copy (not reference)."""
        config1 = get_default_config("resnet50")
        config2 = get_default_config("resnet50")

        config1["pretrained"] = False
        # config2 should still have original value
        assert config2["pretrained"] is True

    def test_register_custom_model(self):
        """Test registering a custom model architecture."""

        # Create a simple custom model
        class CustomModel(BaseModel):
            def __init__(self, num_classes: int, pretrained: bool = False):
                super().__init__(num_classes, pretrained)
                self.fc = torch.nn.Linear(128, num_classes)

            def forward(self, x, return_features=False):
                return self.fc(x.view(x.size(0), -1))

            def get_feature_maps(self, x, layer_names=None):
                return {}

        # Register it
        register_model("custom_test", CustomModel)

        # Verify it's registered
        assert "custom_test" in list_available_architectures()

        # Build it
        model = build_model("custom_test", num_classes=10)
        assert isinstance(model, CustomModel)
        assert model.num_classes == 10

    def test_register_non_basemodel_raises_error(self):
        """Test that registering non-BaseModel class raises error."""

        class NotAModel:
            pass

        with pytest.raises(ValueError, match="must inherit from BaseModel"):
            register_model("invalid", NotAModel)


class TestIntegration:
    """Integration tests for realistic workflows."""

    def test_yaml_style_config_workflow(self):
        """Test workflow that mimics loading from YAML config."""
        # Simulate loaded YAML config
        yaml_config = {
            "experiment": {
                "name": "baseline_exp",
                "seed": 42,
            },
            "model": {
                "architecture": "resnet50",
                "num_classes": 7,
                "pretrained": True,
                "in_channels": 3,
                "dropout": 0.2,
            },
            "data": {
                "dataset": "isic2019",
                "batch_size": 32,
            },
        }

        # Build model from config
        model = build_model_from_config(yaml_config)

        # Verify model is correctly configured
        assert model.num_classes == 7
        assert model.in_channels == 3

        # Verify model is functional
        x = torch.randn(4, 3, 224, 224)
        logits = model(x)
        assert logits.shape == (4, 7)

    def test_hyperparameter_sweep_workflow(self):
        """Test workflow for hyperparameter sweeps."""
        # Sweep over dropout values
        dropout_values = [0.0, 0.2, 0.4, 0.6]
        models = []

        for dropout in dropout_values:
            config = {"pretrained": True, "dropout": dropout}
            model = build_model(architecture="resnet50", num_classes=5, config=config)
            models.append(model)

        assert len(models) == 4
        assert all(m.num_classes == 5 for m in models)

        # Verify dropout values are correct
        for i, model in enumerate(models):
            expected_dropout = dropout_values[i]
            if expected_dropout > 0:
                assert model.dropout_prob == expected_dropout

    def test_model_info_after_building(self):
        """Test that built models have correct metadata."""
        model = build_model("resnet50", num_classes=7)
        info = model.get_model_info()

        assert info["architecture"] == "ResNet50Classifier"
        assert info["num_classes"] == 7
        assert info["pretrained"] is True
        assert info["total_params"] > 20_000_000  # ResNet-50 size


# Run tests
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
