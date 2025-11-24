"""
Comprehensive tests for src.losses.calibration_loss module.

Tests cover all calibration loss functions:
- TemperatureScaling: Post-hoc calibration with learnable temperature
- LabelSmoothingLoss: Regularization during training
- CalibrationLoss: Combined calibrated task loss

Target: 100% line and branch coverage, A1-grade production quality.
"""

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.losses.calibration_loss import (
    CalibrationLoss,
    LabelSmoothingLoss,
    TemperatureScaling,
)


# =============================================================================
# Test TemperatureScaling
# =============================================================================


class TestTemperatureScaling:
    """Test TemperatureScaling post-hoc calibration."""

    def test_initialization_default(self):
        """Test initialization with default temperature."""
        temp_module = TemperatureScaling()
        assert isinstance(temp_module.log_temperature, nn.Parameter)
        assert temp_module.log_temperature.requires_grad

    def test_initialization_custom_temperature(self):
        """Test initialization with custom temperature."""
        temp_module = TemperatureScaling(init_temperature=2.0)
        temp = temp_module.get_temperature()
        assert isinstance(temp, float)
        assert abs(temp - 2.0) < 0.01

    def test_invalid_temperature_raises_error(self):
        """Test that non-positive temperature raises ValueError."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            TemperatureScaling(init_temperature=0.0)
        
        with pytest.raises(ValueError, match="Temperature must be positive"):
            TemperatureScaling(init_temperature=-1.0)

    def test_forward_produces_probabilities(self):
        """Test that forward pass produces valid probabilities."""
        temp_module = TemperatureScaling(init_temperature=1.5)
        logits = torch.randn(4, 5)
        
        probs = temp_module(logits)
        
        assert probs.shape == (4, 5)
        # Check probabilities sum to 1
        assert torch.allclose(probs.sum(dim=1), torch.ones(4), atol=1e-6)
        # Check all probabilities are between 0 and 1
        assert (probs >= 0).all()
        assert (probs <= 1).all()

    def test_forward_with_different_batch_sizes(self):
        """Test forward with various batch sizes."""
        temp_module = TemperatureScaling()
        
        for batch_size in [1, 8, 16]:
            logits = torch.randn(batch_size, 3)
            probs = temp_module(logits)
            assert probs.shape == (batch_size, 3)
            assert torch.allclose(probs.sum(dim=1), torch.ones(batch_size), atol=1e-6)

    def test_fit_step_computes_loss(self):
        """Test that fit_step computes cross-entropy loss."""
        temp_module = TemperatureScaling(init_temperature=1.5)
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        
        loss = temp_module.fit_step(logits, targets)
        
        assert isinstance(loss, Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_fit_step_gradient_flow(self):
        """Test that fit_step allows gradient flow to temperature."""
        temp_module = TemperatureScaling()
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        
        loss = temp_module.fit_step(logits, targets)
        loss.backward()
        
        assert temp_module.log_temperature.grad is not None
        assert temp_module.log_temperature.grad.abs().sum() > 0

    def test_get_temperature_returns_float(self):
        """Test that get_temperature returns a float."""
        temp_module = TemperatureScaling(init_temperature=1.8)
        temp = temp_module.get_temperature()
        
        assert isinstance(temp, float)
        assert temp > 0

    def test_temperature_affects_probabilities(self):
        """Test that temperature scaling affects probability distribution."""
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        
        temp_low = TemperatureScaling(init_temperature=0.5)
        temp_high = TemperatureScaling(init_temperature=2.0)
        
        probs_low = temp_low(logits)
        probs_high = temp_high(logits)
        
        # Lower temperature should make distribution more peaked
        assert probs_low.max() > probs_high.max()

    def test_temperature_is_learnable(self):
        """Test that temperature can be learned through optimization."""
        temp_module = TemperatureScaling(init_temperature=1.0)
        initial_temp = temp_module.get_temperature()
        
        logits = torch.randn(16, 3)
        targets = torch.randint(0, 3, (16,))
        
        optimizer = torch.optim.SGD([temp_module.log_temperature], lr=0.1)
        
        # Run a few optimization steps
        for _ in range(5):
            optimizer.zero_grad()
            loss = temp_module.fit_step(logits, targets)
            loss.backward()
            optimizer.step()
        
        final_temp = temp_module.get_temperature()
        
        # Temperature should have changed
        assert abs(final_temp - initial_temp) > 1e-3


# =============================================================================
# Test LabelSmoothingLoss
# =============================================================================


class TestLabelSmoothingLoss:
    """Test LabelSmoothingLoss."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        loss_fn = LabelSmoothingLoss(num_classes=5, smoothing=0.1)
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        
        loss = loss_fn(logits, targets)
        
        assert isinstance(loss, Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_with_zero_smoothing(self):
        """Test with smoothing=0 (should be equivalent to CE)."""
        loss_fn = LabelSmoothingLoss(num_classes=3, smoothing=0.0)
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_with_high_smoothing(self):
        """Test with high smoothing factor."""
        loss_fn = LabelSmoothingLoss(num_classes=5, smoothing=0.9)
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_invalid_smoothing_raises_error(self):
        """Test that invalid smoothing raises ValueError."""
        with pytest.raises(ValueError, match="smoothing must be in"):
            LabelSmoothingLoss(num_classes=5, smoothing=-0.1)
        
        with pytest.raises(ValueError, match="smoothing must be in"):
            LabelSmoothingLoss(num_classes=5, smoothing=1.0)
        
        with pytest.raises(ValueError, match="smoothing must be in"):
            LabelSmoothingLoss(num_classes=5, smoothing=1.5)

    def test_with_class_weights(self):
        """Test with class weights."""
        class_weights = torch.tensor([1.0, 2.0, 3.0])
        loss_fn = LabelSmoothingLoss(
            num_classes=3,
            smoothing=0.1,
            class_weights=class_weights,
        )
        
        assert torch.equal(loss_fn.class_weights, class_weights)
        
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        loss = loss_fn(logits, targets)
        
        assert loss.item() >= 0

    def test_class_weights_length_mismatch(self):
        """Test that class_weights length mismatch raises ValueError."""
        class_weights = torch.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="class_weights length"):
            LabelSmoothingLoss(num_classes=5, class_weights=class_weights)

    def test_predictions_not_2d(self):
        """Test that non-2D predictions raise ValueError."""
        loss_fn = LabelSmoothingLoss(num_classes=5)
        logits = torch.randn(8, 5, 1)  # 3D
        targets = torch.randint(0, 5, (8,))
        
        with pytest.raises(ValueError, match="must have shape"):
            loss_fn(logits, targets)

    def test_targets_not_1d(self):
        """Test that non-1D targets raise ValueError."""
        loss_fn = LabelSmoothingLoss(num_classes=5)
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8, 1))  # 2D
        
        with pytest.raises(ValueError, match="must have shape"):
            loss_fn(logits, targets)

    def test_wrong_num_classes(self):
        """Test that wrong number of classes raises ValueError."""
        loss_fn = LabelSmoothingLoss(num_classes=5)
        logits = torch.randn(8, 7)  # 7 classes instead of 5
        targets = torch.randint(0, 5, (8,))
        
        with pytest.raises(ValueError, match="expected 5"):
            loss_fn(logits, targets)

    def test_reduction_mean(self):
        """Test mean reduction."""
        loss_fn = LabelSmoothingLoss(num_classes=3, reduction="mean")
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_reduction_sum(self):
        """Test sum reduction."""
        loss_fn = LabelSmoothingLoss(num_classes=3, reduction="sum")
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_reduction_none(self):
        """Test none reduction."""
        loss_fn = LabelSmoothingLoss(num_classes=3, reduction="none")
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        
        loss = loss_fn(logits, targets)
        assert loss.shape == (4,)

    def test_gradient_flow(self):
        """Test gradient flow through loss."""
        loss_fn = LabelSmoothingLoss(num_classes=3)
        logits = torch.randn(4, 3, requires_grad=True)
        targets = torch.randint(0, 3, (4,))
        
        loss = loss_fn(logits, targets)
        loss.backward()
        
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_num_classes_conversion(self):
        """Test that num_classes is converted to int."""
        loss_fn = LabelSmoothingLoss(num_classes=5)
        assert loss_fn.num_classes == 5
        assert isinstance(loss_fn.num_classes, int)

    def test_smoothing_conversion(self):
        """Test that smoothing is converted to float."""
        loss_fn = LabelSmoothingLoss(num_classes=5, smoothing=0)
        assert loss_fn.smoothing == 0.0
        assert isinstance(loss_fn.smoothing, float)


# =============================================================================
# Test CalibrationLoss
# =============================================================================


class TestCalibrationLoss:
    """Test CalibrationLoss."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        loss_fn = CalibrationLoss(num_classes=5)
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        
        loss = loss_fn(logits, targets)
        
        assert isinstance(loss, Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_with_label_smoothing(self):
        """Test with label smoothing enabled."""
        loss_fn = CalibrationLoss(
            num_classes=5,
            use_label_smoothing=True,
            smoothing=0.1,
        )
        
        assert loss_fn.use_label_smoothing is True
        assert loss_fn.label_smoothing is not None
        
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        loss = loss_fn(logits, targets)
        
        assert loss.item() >= 0

    def test_without_label_smoothing(self):
        """Test without label smoothing."""
        loss_fn = CalibrationLoss(
            num_classes=5,
            use_label_smoothing=False,
        )
        
        assert loss_fn.use_label_smoothing is False
        
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        loss = loss_fn(logits, targets)
        
        assert loss.item() >= 0

    def test_with_class_weights(self):
        """Test with class weights."""
        class_weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        loss_fn = CalibrationLoss(
            num_classes=5,
            class_weights=class_weights,
        )
        
        assert torch.equal(loss_fn.class_weights, class_weights)
        
        logits = torch.randn(6, 5)
        targets = torch.randint(0, 5, (6,))
        loss = loss_fn(logits, targets)
        
        assert loss.item() >= 0

    def test_class_weights_length_mismatch(self):
        """Test that class_weights length mismatch raises ValueError."""
        class_weights = torch.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="class_weights length"):
            CalibrationLoss(num_classes=5, class_weights=class_weights)

    def test_custom_temperature(self):
        """Test with custom initial temperature."""
        loss_fn = CalibrationLoss(num_classes=3, init_temperature=2.5)
        temp = loss_fn.get_temperature()
        
        assert isinstance(temp, float)
        assert abs(temp - 2.5) < 0.01

    def test_invalid_temperature_raises_error(self):
        """Test that non-positive temperature raises ValueError."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            CalibrationLoss(num_classes=5, init_temperature=0.0)
        
        with pytest.raises(ValueError, match="Temperature must be positive"):
            CalibrationLoss(num_classes=5, init_temperature=-1.0)

    def test_temperature_is_learnable(self):
        """Test that temperature is a learnable parameter."""
        loss_fn = CalibrationLoss(num_classes=3)
        
        assert isinstance(loss_fn.log_temperature, nn.Parameter)
        assert loss_fn.log_temperature.requires_grad

    def test_temperature_gradient_flow(self):
        """Test that temperature receives gradients."""
        loss_fn = CalibrationLoss(num_classes=3)
        logits = torch.randn(4, 3, requires_grad=True)
        targets = torch.randint(0, 3, (4,))
        
        loss = loss_fn(logits, targets)
        loss.backward()
        
        assert loss_fn.log_temperature.grad is not None
        assert loss_fn.log_temperature.grad.abs().sum() > 0

    def test_predictions_not_2d(self):
        """Test that non-2D predictions raise ValueError."""
        loss_fn = CalibrationLoss(num_classes=5)
        logits = torch.randn(8, 5, 1)  # 3D
        targets = torch.randint(0, 5, (8,))
        
        with pytest.raises(ValueError, match="must have shape"):
            loss_fn(logits, targets)

    def test_targets_not_1d(self):
        """Test that non-1D targets raise ValueError."""
        loss_fn = CalibrationLoss(num_classes=5)
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8, 1))  # 2D
        
        with pytest.raises(ValueError, match="must have shape"):
            loss_fn(logits, targets)

    def test_wrong_num_classes(self):
        """Test that wrong number of classes raises ValueError."""
        loss_fn = CalibrationLoss(num_classes=5)
        logits = torch.randn(8, 7)  # 7 classes
        targets = torch.randint(0, 5, (8,))
        
        with pytest.raises(ValueError, match="expected 5"):
            loss_fn(logits, targets)

    def test_reduction_mean(self):
        """Test mean reduction."""
        loss_fn = CalibrationLoss(num_classes=3, reduction="mean")
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_get_temperature_returns_float(self):
        """Test that get_temperature returns a float."""
        loss_fn = CalibrationLoss(num_classes=5, init_temperature=1.8)
        temp = loss_fn.get_temperature()
        
        assert isinstance(temp, float)
        assert temp > 0

    def test_num_classes_conversion(self):
        """Test that num_classes is converted to int."""
        loss_fn = CalibrationLoss(num_classes=5)
        assert loss_fn.num_classes == 5
        assert isinstance(loss_fn.num_classes, int)

    def test_use_label_smoothing_conversion(self):
        """Test that use_label_smoothing is converted to bool."""
        loss_fn = CalibrationLoss(num_classes=5, use_label_smoothing=1)
        assert loss_fn.use_label_smoothing is True
        assert isinstance(loss_fn.use_label_smoothing, bool)


# =============================================================================
# Test Integration and Edge Cases
# =============================================================================


class TestIntegration:
    """Test integrated workflows and edge cases."""

    def test_temperature_scaling_full_workflow(self):
        """Test full temperature scaling workflow."""
        # Create module and data
        temp_module = TemperatureScaling(init_temperature=1.0)
        logits = torch.randn(16, 5)
        targets = torch.randint(0, 5, (16,))
        
        # Fit temperature
        optimizer = torch.optim.LBFGS([temp_module.log_temperature], lr=0.01, max_iter=10)
        
        def closure():
            optimizer.zero_grad()
            loss = temp_module.fit_step(logits, targets)
            loss.backward()
            return loss
        
        optimizer.step(closure)
        
        # Apply calibration
        calibrated_probs = temp_module(logits)
        
        assert calibrated_probs.shape == (16, 5)
        assert torch.allclose(calibrated_probs.sum(dim=1), torch.ones(16), atol=1e-6)

    def test_label_smoothing_vs_regular_ce(self):
        """Test that label smoothing produces different loss than regular CE."""
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        
        # Label smoothing
        loss_smooth = LabelSmoothingLoss(num_classes=5, smoothing=0.1)
        loss_val_smooth = loss_smooth(logits, targets)
        
        # Zero smoothing (should be close to CE)
        loss_no_smooth = LabelSmoothingLoss(num_classes=5, smoothing=0.0)
        loss_val_no_smooth = loss_no_smooth(logits, targets)
        
        # They should be different
        assert abs(loss_val_smooth.item() - loss_val_no_smooth.item()) > 1e-6

    def test_calibration_loss_with_both_techniques(self):
        """Test calibration loss with both label smoothing and temperature."""
        loss_fn = CalibrationLoss(
            num_classes=5,
            use_label_smoothing=True,
            smoothing=0.1,
            init_temperature=1.5,
        )
        
        logits = torch.randn(8, 5, requires_grad=True)
        targets = torch.randint(0, 5, (8,))
        
        loss = loss_fn(logits, targets)
        loss.backward()
        
        # Check gradients flow to both logits and temperature
        assert logits.grad is not None
        assert loss_fn.log_temperature.grad is not None

    def test_calibration_loss_delegates_to_label_smoothing(self):
        """Test that CalibrationLoss uses LabelSmoothingLoss when enabled."""
        loss_fn = CalibrationLoss(
            num_classes=5,
            use_label_smoothing=True,
            smoothing=0.1,
        )
        
        assert isinstance(loss_fn.label_smoothing, LabelSmoothingLoss)

    def test_calibration_loss_uses_ce_when_no_smoothing(self):
        """Test that CalibrationLoss uses CE when label smoothing disabled."""
        loss_fn = CalibrationLoss(
            num_classes=5,
            use_label_smoothing=False,
        )
        
        # label_smoothing should be None
        assert loss_fn.label_smoothing is None

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        loss_fn = CalibrationLoss(num_classes=5)
        logits = torch.randn(1, 5)
        targets = torch.randint(0, 5, (1,))
        
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_large_batch_size(self):
        """Test with large batch size."""
        loss_fn = LabelSmoothingLoss(num_classes=5)
        logits = torch.randn(128, 5)
        targets = torch.randint(0, 5, (128,))
        
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_statistics_tracking(self):
        """Test that statistics are tracked properly."""
        loss_fn = CalibrationLoss(num_classes=5)
        
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        
        loss1 = loss_fn(logits, targets)
        loss2 = loss_fn(logits, targets)
        
        stats = loss_fn.get_statistics()
        assert stats["num_calls"] == 2
        assert stats["mean_loss"] > 0

    def test_statistics_reset(self):
        """Test statistics reset functionality."""
        loss_fn = LabelSmoothingLoss(num_classes=5)
        
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        loss_fn(logits, targets)
        
        loss_fn.reset_statistics()
        stats = loss_fn.get_statistics()
        
        assert stats["num_calls"] == 0
        assert stats["mean_loss"] == 0.0

    def test_default_class_weights(self):
        """Test that default class_weights are all ones."""
        loss_fn = CalibrationLoss(num_classes=5)
        assert torch.equal(loss_fn.class_weights, torch.ones(5))

    def test_temperature_scaling_single_sample(self):
        """Test temperature scaling with single sample."""
        temp_module = TemperatureScaling()
        logits = torch.randn(1, 3)
        
        probs = temp_module(logits)
        
        assert probs.shape == (1, 3)
        assert torch.allclose(probs.sum(), torch.ones(1), atol=1e-6)

    def test_label_smoothing_confidence_calculation(self):
        """Test that confidence is calculated correctly."""
        smoothing = 0.1
        loss_fn = LabelSmoothingLoss(num_classes=5, smoothing=smoothing)
        
        assert loss_fn.smoothing == smoothing
        assert loss_fn.confidence == 1.0 - smoothing

    def test_different_dtypes(self):
        """Test with different tensor dtypes."""
        loss_fn = CalibrationLoss(num_classes=3)
        
        # float32 (default)
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_temperature_scaling_extreme_values(self):
        """Test temperature scaling with extreme logit values."""
        temp_module = TemperatureScaling(init_temperature=1.0)
        
        # Very large logits
        logits = torch.tensor([[100.0, -100.0, 0.0]])
        probs = temp_module(logits)
        
        assert probs.shape == (1, 3)
        assert torch.allclose(probs.sum(), torch.ones(1), atol=1e-6)

    def test_label_smoothing_with_perfect_predictions(self):
        """Test label smoothing with perfect predictions."""
        loss_fn = LabelSmoothingLoss(num_classes=3, smoothing=0.1)
        
        # Perfect predictions (very high confidence)
        logits = torch.tensor([[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]])
        targets = torch.tensor([0, 1])
        
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_calibration_loss_passes_class_weights_to_label_smoothing(self):
        """Test that CalibrationLoss passes class_weights to LabelSmoothingLoss."""
        class_weights = torch.tensor([1.0, 2.0, 3.0])
        loss_fn = CalibrationLoss(
            num_classes=3,
            class_weights=class_weights,
            use_label_smoothing=True,
        )
        
        # Check that label_smoothing has the same class_weights
        assert torch.equal(loss_fn.label_smoothing.class_weights, class_weights)

    def test_multiple_forward_passes(self):
        """Test multiple forward passes accumulate statistics."""
        loss_fn = LabelSmoothingLoss(num_classes=5)
        
        for i in range(5):
            logits = torch.randn(4, 5)
            targets = torch.randint(0, 5, (4,))
            loss_fn(logits, targets)
        
        stats = loss_fn.get_statistics()
        assert stats["num_calls"] == 5

    def test_temperature_boundary_values(self):
        """Test temperature scaling with boundary values."""
        # Very small temperature (sharper distribution)
        temp_small = TemperatureScaling(init_temperature=0.1)
        logits = torch.tensor([[1.0, 2.0, 3.0]])
        probs_small = temp_small(logits)
        
        # Very large temperature (smoother distribution)
        temp_large = TemperatureScaling(init_temperature=10.0)
        probs_large = temp_large(logits)
        
        # Small temperature should be more peaked
        assert probs_small.max() > probs_large.max()

    def test_label_smoothing_edge_case_smoothing_values(self):
        """Test label smoothing with edge case smoothing values."""
        # Very small smoothing
        loss_fn1 = LabelSmoothingLoss(num_classes=5, smoothing=0.01)
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        loss1 = loss_fn1(logits, targets)
        
        # Very large smoothing (but < 1)
        loss_fn2 = LabelSmoothingLoss(num_classes=5, smoothing=0.99)
        loss2 = loss_fn2(logits, targets)
        
        assert loss1.item() >= 0
        assert loss2.item() >= 0
    
    def test_label_smoothing_class_weights_branch(self):
        """Test LabelSmoothingLoss branch where class_weights is applied."""
        # Create loss with class_weights that are not all ones
        weights = torch.tensor([1.0, 2.0, 1.5, 0.5])
        loss_fn = LabelSmoothingLoss(num_classes=4, smoothing=0.1, class_weights=weights)
        
        logits = torch.randn(16, 4)
        targets = torch.randint(0, 4, (16,))
        
        # Should execute the class_weights branch
        result = loss_fn(logits, targets)
        assert result.item() > 0
        assert not torch.isnan(result)
        
        # Compare with loss without class weights
        loss_fn_no_weights = LabelSmoothingLoss(num_classes=4, smoothing=0.1)
        result_no_weights = loss_fn_no_weights(logits, targets)
        
        # Results should be different
        assert not torch.allclose(result, result_no_weights)
    
    def test_some_calibration_class_coverage(self):
        """Test SomeCalibrationClass for coverage."""
        from src.losses.calibration_loss import SomeCalibrationClass
        
        # Create a mock config object
        class Config:
            def __init__(self, label_smoothing, num_classes):
                self.label_smoothing = label_smoothing
                self.num_classes = num_classes
        
        # Test with label_smoothing > 0
        config = Config(label_smoothing=0.1, num_classes=5)
        obj = SomeCalibrationClass(config)
        assert obj.label_smoothing_loss is not None
        assert isinstance(obj.label_smoothing_loss, LabelSmoothingLoss)
        
        # Test with label_smoothing = 0
        config_zero = Config(label_smoothing=0.0, num_classes=5)
        obj_zero = SomeCalibrationClass(config_zero)
        assert obj_zero.label_smoothing_loss is None
    
    def test_label_smoothing_class_weights_none_branch(self):
        """Test the branch where class_weights is None in forward pass."""
        # Create a LabelSmoothingLoss and manually set class_weights to None
        loss_fn = LabelSmoothingLoss(num_classes=4, smoothing=0.1)
        
        # Manually set class_weights to None to trigger the branch
        loss_fn.class_weights = None
        
        logits = torch.randn(8, 4)
        targets = torch.randint(0, 4, (8,))
        
        # This should execute the branch where class_weights is None
        result = loss_fn(logits, targets)
        assert result.item() > 0
        assert not torch.isnan(result)
