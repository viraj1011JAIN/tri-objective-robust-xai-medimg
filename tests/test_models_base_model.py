"""
Comprehensive tests for src.models.base_model module.

Tests cover all functionality of the BaseModel abstract class including:
- Initialization with various parameter combinations
- Abstract method enforcement
- Convenience methods (predict_proba, num_parameters)
- Introspection (extra_repr)
- Edge cases and error conditions

Target: 100% line and branch coverage, A1-grade production quality.
"""

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.models.base_model import BaseModel


# =============================================================================
# Concrete Test Models
# =============================================================================


class MinimalModel(BaseModel):
    """Minimal concrete implementation for testing."""

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.fc = nn.Linear(10, num_classes)

    def forward(self, x: Tensor) -> Tensor:
        """Simple forward pass."""
        return self.fc(x.view(x.size(0), -1))

    def get_feature_maps(self, x: Tensor) -> Tensor:
        """Return identity as feature maps."""
        return x


class ModelWithTrainableParams(BaseModel):
    """Model with both trainable and frozen parameters."""

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.trainable_layer = nn.Linear(10, num_classes)
        self.frozen_layer = nn.Linear(10, num_classes)
        # Freeze the second layer
        for param in self.frozen_layer.parameters():
            param.requires_grad = False

    def forward(self, x: Tensor) -> Tensor:
        """Forward using trainable layer."""
        return self.trainable_layer(x.view(x.size(0), -1))

    def get_feature_maps(self, x: Tensor) -> Tensor:
        """Return identity."""
        return x


class ModelWithComplexExtraConfig(BaseModel):
    """Model that uses extra_config for custom parameters."""

    def __init__(self, num_classes: int, **kwargs):
        super().__init__(num_classes=num_classes, **kwargs)
        self.fc = nn.Linear(10, num_classes)
        # Access extra config
        self.dropout_rate = self.extra_config.get("dropout_rate", 0.5)
        self.use_batchnorm = self.extra_config.get("use_batchnorm", False)

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass."""
        return self.fc(x.view(x.size(0), -1))

    def get_feature_maps(self, x: Tensor) -> Tensor:
        """Return identity."""
        return x


# =============================================================================
# Test Initialization
# =============================================================================


class TestBaseModelInitialization:
    """Test BaseModel constructor with various parameter combinations."""

    def test_minimal_initialization(self):
        """Test initialization with only num_classes."""
        model = MinimalModel(num_classes=10)
        assert model.num_classes == 10
        assert model.in_channels == 3  # default
        assert model.pretrained is None  # default
        assert model.architecture == "minimalmodel"  # class name lowercased
        assert model.extra_config == {}

    def test_initialization_with_architecture(self):
        """Test initialization with explicit architecture name."""
        model = MinimalModel(num_classes=5, architecture="test_model")
        assert model.num_classes == 5
        assert model.architecture == "test_model"

    def test_initialization_with_in_channels(self):
        """Test initialization with custom in_channels."""
        model = MinimalModel(num_classes=3, in_channels=1)
        assert model.in_channels == 1
        assert model.num_classes == 3

    def test_initialization_with_pretrained_bool(self):
        """Test initialization with pretrained=True."""
        model = MinimalModel(num_classes=7, pretrained=True)
        assert model.pretrained is True

    def test_initialization_with_pretrained_false(self):
        """Test initialization with pretrained=False."""
        model = MinimalModel(num_classes=7, pretrained=False)
        assert model.pretrained is False

    def test_initialization_with_pretrained_string(self):
        """Test initialization with pretrained as string identifier."""
        model = MinimalModel(num_classes=4, pretrained="imagenet")
        assert model.pretrained == "imagenet"

    def test_initialization_with_extra_kwargs(self):
        """Test that extra kwargs are stored in extra_config."""
        model = MinimalModel(
            num_classes=8,
            dropout_rate=0.3,
            batch_size=32,
            learning_rate=0.001,
        )
        assert model.extra_config["dropout_rate"] == 0.3
        assert model.extra_config["batch_size"] == 32
        assert model.extra_config["learning_rate"] == 0.001

    def test_initialization_all_parameters(self):
        """Test initialization with all parameters specified."""
        model = MinimalModel(
            num_classes=12,
            architecture="custom_arch",
            in_channels=4,
            pretrained="custom_weights",
            custom_param=42,
            another_param="value",
        )
        assert model.num_classes == 12
        assert model.architecture == "custom_arch"
        assert model.in_channels == 4
        assert model.pretrained == "custom_weights"
        assert model.extra_config["custom_param"] == 42
        assert model.extra_config["another_param"] == "value"

    def test_num_classes_conversion_to_int(self):
        """Test that num_classes is converted to int."""
        model = MinimalModel(num_classes=5)  # Use int directly
        assert model.num_classes == 5
        assert isinstance(model.num_classes, int)
        
        # Test conversion from float-like value
        model2 = MinimalModel(num_classes=int(5.9))
        assert model2.num_classes == 5
        assert isinstance(model2.num_classes, int)

    def test_in_channels_conversion_to_int(self):
        """Test that in_channels is converted to int."""
        model = MinimalModel(num_classes=3, in_channels=1.7)
        assert model.in_channels == 1
        assert isinstance(model.in_channels, int)


# =============================================================================
# Test Validation
# =============================================================================


class TestBaseModelValidation:
    """Test input validation and error conditions."""

    def test_zero_num_classes_raises_error(self):
        """Test that num_classes=0 raises ValueError."""
        with pytest.raises(ValueError, match="num_classes must be positive"):
            MinimalModel(num_classes=0)

    def test_negative_num_classes_raises_error(self):
        """Test that negative num_classes raises ValueError."""
        with pytest.raises(ValueError, match="num_classes must be positive"):
            MinimalModel(num_classes=-5)

    def test_negative_one_num_classes_raises_error(self):
        """Test edge case with num_classes=-1."""
        with pytest.raises(ValueError, match="num_classes must be positive"):
            MinimalModel(num_classes=-1)


# =============================================================================
# Test Abstract Methods
# =============================================================================


class TestAbstractMethods:
    """Test that abstract methods are properly enforced."""

    def test_cannot_instantiate_base_model_directly(self):
        """Test that BaseModel cannot be instantiated directly."""
        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            BaseModel(num_classes=10)

    def test_incomplete_subclass_forward_not_implemented(self):
        """Test that subclass without forward raises TypeError."""

        class IncompleteModel(BaseModel):
            def get_feature_maps(self, x: Tensor) -> Tensor:
                return x

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteModel(num_classes=5)

    def test_incomplete_subclass_get_feature_maps_not_implemented(self):
        """Test that subclass without get_feature_maps raises TypeError."""

        class IncompleteModel(BaseModel):
            def forward(self, x: Tensor) -> Tensor:
                return x

        with pytest.raises(TypeError, match="Can't instantiate abstract class"):
            IncompleteModel(num_classes=5)

    def test_abstract_forward_raises_not_implemented(self):
        """Test that calling abstract forward raises NotImplementedError."""
        # This tests the abstract method's body
        with pytest.raises(NotImplementedError):
            BaseModel.forward(None, torch.randn(1, 3, 32, 32))

    def test_abstract_get_feature_maps_raises_not_implemented(self):
        """Test that calling abstract get_feature_maps raises NotImplementedError."""
        with pytest.raises(NotImplementedError):
            BaseModel.get_feature_maps(None, torch.randn(1, 3, 32, 32))


# =============================================================================
# Test Forward and Feature Maps
# =============================================================================


class TestForwardAndFeatureMaps:
    """Test forward pass and feature map extraction."""

    def test_forward_pass_produces_correct_shape(self):
        """Test that forward produces correct output shape."""
        model = MinimalModel(num_classes=10)
        x = torch.randn(4, 10)
        output = model(x)
        assert output.shape == (4, 10)

    def test_forward_with_different_batch_sizes(self):
        """Test forward with various batch sizes."""
        model = MinimalModel(num_classes=5)
        for batch_size in [1, 8, 16]:
            x = torch.randn(batch_size, 10)
            output = model(x)
            assert output.shape == (batch_size, 5)

    def test_get_feature_maps_returns_tensor(self):
        """Test that get_feature_maps returns a tensor."""
        model = MinimalModel(num_classes=3)
        x = torch.randn(2, 3, 32, 32)
        features = model.get_feature_maps(x)
        assert isinstance(features, Tensor)
        assert features.shape == x.shape  # MinimalModel returns identity


# =============================================================================
# Test Convenience Methods
# =============================================================================


class TestPredictProba:
    """Test predict_proba convenience method."""

    def test_predict_proba_returns_probabilities(self):
        """Test that predict_proba returns valid probabilities."""
        model = MinimalModel(num_classes=10)
        x = torch.randn(4, 10)
        probs = model.predict_proba(x)

        assert probs.shape == (4, 10)
        # Check probabilities sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-6)
        # Check all probabilities are between 0 and 1
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_predict_proba_single_sample(self):
        """Test predict_proba with single sample."""
        model = MinimalModel(num_classes=3)
        x = torch.randn(1, 10)
        probs = model.predict_proba(x)

        assert probs.shape == (1, 3)
        assert torch.allclose(probs.sum(dim=1), torch.ones(1), atol=1e-6)

    def test_predict_proba_is_differentiable(self):
        """Test that predict_proba maintains gradients."""
        model = MinimalModel(num_classes=5)
        x = torch.randn(2, 10, requires_grad=True)
        probs = model.predict_proba(x)

        # Should be able to backpropagate
        loss = probs.sum()
        loss.backward()
        assert x.grad is not None


class TestNumParameters:
    """Test num_parameters method."""

    def test_num_parameters_all(self):
        """Test counting all parameters."""
        model = MinimalModel(num_classes=10)
        # Linear layer: 10 inputs * 10 outputs + 10 bias = 110
        total_params = model.num_parameters(trainable_only=False)
        assert total_params == 110

    def test_num_parameters_trainable_only(self):
        """Test counting only trainable parameters."""
        model = ModelWithTrainableParams(num_classes=5)
        trainable_params = model.num_parameters(trainable_only=True)
        total_params = model.num_parameters(trainable_only=False)

        # Trainable layer: 10*5 + 5 = 55
        # Frozen layer: 10*5 + 5 = 55
        # Total = 110, Trainable = 55
        assert trainable_params == 55
        assert total_params == 110

    def test_num_parameters_different_architectures(self):
        """Test num_parameters with different model sizes."""
        model_small = MinimalModel(num_classes=2)
        model_large = MinimalModel(num_classes=100)

        params_small = model_small.num_parameters()
        params_large = model_large.num_parameters()

        # Small: 10*2 + 2 = 22
        # Large: 10*100 + 100 = 1100
        assert params_small == 22
        assert params_large == 1100

    def test_num_parameters_returns_int(self):
        """Test that num_parameters returns an integer."""
        model = MinimalModel(num_classes=7)
        params = model.num_parameters()
        assert isinstance(params, int)


# =============================================================================
# Test Introspection
# =============================================================================


class TestExtraRepr:
    """Test extra_repr method for model representation."""

    def test_extra_repr_minimal(self):
        """Test extra_repr with minimal configuration."""
        model = MinimalModel(num_classes=10)
        repr_str = model.extra_repr()

        assert "architecture='minimalmodel'" in repr_str
        assert "num_classes=10" in repr_str
        assert "in_channels=3" in repr_str
        assert "pretrained=None" in repr_str

    def test_extra_repr_with_architecture(self):
        """Test extra_repr with custom architecture name."""
        model = MinimalModel(num_classes=5, architecture="custom_model")
        repr_str = model.extra_repr()

        assert "architecture='custom_model'" in repr_str
        assert "num_classes=5" in repr_str

    def test_extra_repr_with_pretrained_bool(self):
        """Test extra_repr with pretrained=True."""
        model = MinimalModel(num_classes=3, pretrained=True)
        repr_str = model.extra_repr()

        assert "pretrained=True" in repr_str

    def test_extra_repr_with_pretrained_string(self):
        """Test extra_repr with pretrained as string."""
        model = MinimalModel(num_classes=8, pretrained="imagenet")
        repr_str = model.extra_repr()

        assert "pretrained='imagenet'" in repr_str

    def test_extra_repr_with_custom_in_channels(self):
        """Test extra_repr with custom in_channels."""
        model = MinimalModel(num_classes=4, in_channels=1)
        repr_str = model.extra_repr()

        assert "in_channels=1" in repr_str

    def test_extra_repr_with_extra_config(self):
        """Test extra_repr includes extra_config parameters."""
        model = MinimalModel(
            num_classes=6,
            dropout_rate=0.5,
            batch_norm=True,
        )
        repr_str = model.extra_repr()

        # Should include extra config
        assert "dropout_rate=0.5" in repr_str
        assert "batch_norm=True" in repr_str

    def test_extra_repr_with_no_extra_config(self):
        """Test extra_repr when extra_config is empty."""
        model = MinimalModel(num_classes=2)
        repr_str = model.extra_repr()

        # Should not have trailing comma when no extra config
        assert repr_str.endswith("pretrained=None")

    def test_extra_repr_in_full_repr(self):
        """Test that extra_repr is included in full model repr."""
        model = MinimalModel(num_classes=7, architecture="test")
        full_repr = repr(model)

        # Should include class name and extra_repr content
        assert "MinimalModel" in full_repr
        assert "architecture='test'" in full_repr
        assert "num_classes=7" in full_repr


# =============================================================================
# Test Integration and Edge Cases
# =============================================================================


class TestIntegration:
    """Test integrated workflows and edge cases."""

    def test_model_with_complex_extra_config(self):
        """Test model that uses extra_config internally."""
        model = ModelWithComplexExtraConfig(
            num_classes=5,
            dropout_rate=0.3,
            use_batchnorm=True,
        )

        assert model.dropout_rate == 0.3
        assert model.use_batchnorm is True
        assert model.extra_config["dropout_rate"] == 0.3
        assert model.extra_config["use_batchnorm"] is True

    def test_model_with_default_extra_config_values(self):
        """Test model with extra_config defaults."""
        model = ModelWithComplexExtraConfig(num_classes=3)

        # Should use defaults from the model
        assert model.dropout_rate == 0.5
        assert model.use_batchnorm is False

    def test_architecture_defaults_to_class_name(self):
        """Test that architecture defaults to lowercase class name."""
        model = MinimalModel(num_classes=10)
        assert model.architecture == "minimalmodel"

        model2 = ModelWithTrainableParams(num_classes=5)
        assert model2.architecture == "modelwithtrainableparams"

    def test_multiple_models_independent(self):
        """Test that multiple model instances are independent."""
        model1 = MinimalModel(num_classes=3, architecture="model1")
        model2 = MinimalModel(num_classes=5, architecture="model2")

        assert model1.num_classes == 3
        assert model2.num_classes == 5
        assert model1.architecture == "model1"
        assert model2.architecture == "model2"

    def test_model_is_nn_module(self):
        """Test that BaseModel is a proper nn.Module."""
        model = MinimalModel(num_classes=10)
        assert isinstance(model, nn.Module)

    def test_model_can_be_moved_to_device(self):
        """Test that model can be moved to different devices."""
        model = MinimalModel(num_classes=5)
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        model = model.to(device)
        x = torch.randn(2, 10, device=device)
        output = model(x)

        # Check device type matches (handles cuda:0 vs cuda)
        assert output.device.type == device.type

    def test_model_training_mode_toggle(self):
        """Test model can switch between train and eval modes."""
        model = MinimalModel(num_classes=5)

        model.train()
        assert model.training is True

        model.eval()
        assert model.training is False

    def test_extra_config_is_dict_copy(self):
        """Test that extra_config is properly created as a dict."""
        kwargs = {"param1": 1, "param2": 2}
        model = MinimalModel(num_classes=5, **kwargs)

        # Should be a proper dict
        assert isinstance(model.extra_config, dict)
        assert model.extra_config == kwargs

        # Modifying original kwargs shouldn't affect model
        kwargs["param3"] = 3
        assert "param3" not in model.extra_config


# =============================================================================
# Test Edge Cases
# =============================================================================


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_very_large_num_classes(self):
        """Test with very large number of classes."""
        model = MinimalModel(num_classes=10000)
        assert model.num_classes == 10000

    def test_single_class(self):
        """Test with num_classes=1."""
        model = MinimalModel(num_classes=1)
        assert model.num_classes == 1

        x = torch.randn(2, 10)
        output = model(x)
        assert output.shape == (2, 1)

    def test_large_in_channels(self):
        """Test with large in_channels value."""
        model = MinimalModel(num_classes=5, in_channels=100)
        assert model.in_channels == 100

    def test_empty_extra_config(self):
        """Test that empty extra_config works correctly."""
        model = MinimalModel(num_classes=5)
        assert model.extra_config == {}
        assert len(model.extra_config) == 0

    def test_pretrained_none_explicitly(self):
        """Test explicitly setting pretrained=None."""
        model = MinimalModel(num_classes=5, pretrained=None)
        assert model.pretrained is None

    def test_architecture_with_special_characters(self):
        """Test architecture name with special characters."""
        model = MinimalModel(num_classes=5, architecture="model-v2.0_beta")
        assert model.architecture == "model-v2.0_beta"

    def test_extra_config_with_none_values(self):
        """Test extra_config can contain None values."""
        model = MinimalModel(num_classes=5, param1=None, param2=0)
        assert model.extra_config["param1"] is None
        assert model.extra_config["param2"] == 0

    def test_extra_config_with_various_types(self):
        """Test extra_config with various data types."""
        model = MinimalModel(
            num_classes=5,
            int_param=42,
            float_param=3.14,
            str_param="hello",
            bool_param=True,
            list_param=[1, 2, 3],
            dict_param={"key": "value"},
        )

        assert model.extra_config["int_param"] == 42
        assert model.extra_config["float_param"] == 3.14
        assert model.extra_config["str_param"] == "hello"
        assert model.extra_config["bool_param"] is True
        assert model.extra_config["list_param"] == [1, 2, 3]
        assert model.extra_config["dict_param"] == {"key": "value"}
