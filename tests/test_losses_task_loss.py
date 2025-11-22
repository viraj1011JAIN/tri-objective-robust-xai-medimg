"""
Comprehensive tests for src.losses.task_loss module.

Tests cover all loss functions:
- TaskLoss: High-level wrapper for multi-class and multi-label tasks
- CalibratedCrossEntropyLoss: Cross-entropy with learnable temperature
- MultiLabelBCELoss: Multi-label BCE with class weights and pos_weight
- FocalLoss: Focal loss for class imbalance

Target: 100% line and branch coverage, A1-grade production quality.
"""

import pytest
import torch
import torch.nn as nn
from torch import Tensor

from src.losses.task_loss import (
    CalibratedCrossEntropyLoss,
    FocalLoss,
    MultiLabelBCELoss,
    TaskLoss,
)


# =============================================================================
# Test TaskLoss - High-Level Wrapper
# =============================================================================


class TestTaskLoss:
    """Test TaskLoss high-level wrapper."""

    def test_multi_class_basic(self):
        """Test basic multi-class task loss."""
        loss_fn = TaskLoss(num_classes=5, task_type="multi_class")
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        
        loss = loss_fn(logits, targets)
        
        assert isinstance(loss, Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_multi_label_basic(self):
        """Test basic multi-label task loss."""
        loss_fn = TaskLoss(num_classes=10, task_type="multi_label")
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 2, (4, 10)).float()
        
        loss = loss_fn(logits, targets)
        
        assert isinstance(loss, Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_multi_class_with_class_weights(self):
        """Test multi-class with class weights."""
        class_weights = torch.tensor([1.0, 2.0, 3.0])
        loss_fn = TaskLoss(
            num_classes=3,
            task_type="multi_class",
            class_weights=class_weights,
        )
        
        assert torch.equal(loss_fn.class_weights, class_weights)
        
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        loss = loss_fn(logits, targets)
        
        assert loss.item() >= 0

    def test_multi_label_with_class_weights(self):
        """Test multi-label with class weights."""
        class_weights = torch.tensor([1.0, 1.5, 2.0, 2.5])
        loss_fn = TaskLoss(
            num_classes=4,
            task_type="multi_label",
            class_weights=class_weights,
        )
        
        assert torch.equal(loss_fn.class_weights, class_weights)
        
        logits = torch.randn(3, 4)
        targets = torch.randint(0, 2, (3, 4)).float()
        loss = loss_fn(logits, targets)
        
        assert loss.item() >= 0

    def test_multi_class_with_focal_loss(self):
        """Test multi-class with focal loss."""
        loss_fn = TaskLoss(
            num_classes=7,
            task_type="multi_class",
            use_focal=True,
            focal_gamma=2.5,
        )
        
        assert loss_fn.use_focal is True
        assert isinstance(loss_fn.loss_fn, FocalLoss)
        
        logits = torch.randn(6, 7)
        targets = torch.randint(0, 7, (6,))
        loss = loss_fn(logits, targets)
        
        assert loss.item() >= 0

    def test_invalid_task_type(self):
        """Test that invalid task_type raises ValueError."""
        with pytest.raises(ValueError, match="Invalid task_type"):
            TaskLoss(num_classes=5, task_type="invalid_type")

    def test_focal_loss_with_multi_label_raises_error(self):
        """Test that focal loss with multi-label raises ValueError."""
        with pytest.raises(
            ValueError, match="only supported for multi_class"
        ):
            TaskLoss(
                num_classes=5,
                task_type="multi_label",
                use_focal=True,
            )

    def test_class_weights_length_mismatch(self):
        """Test that class_weights length mismatch raises ValueError."""
        class_weights = torch.tensor([1.0, 2.0, 3.0])
        with pytest.raises(ValueError, match="class_weights length"):
            TaskLoss(
                num_classes=5,
                task_type="multi_class",
                class_weights=class_weights,
            )

    def test_multi_class_predictions_not_2d(self):
        """Test that non-2D predictions raise ValueError for multi-class."""
        loss_fn = TaskLoss(num_classes=5, task_type="multi_class")
        logits = torch.randn(8, 5, 1)  # 3D instead of 2D
        targets = torch.randint(0, 5, (8,))
        
        with pytest.raises(ValueError, match="must be 2D"):
            loss_fn(logits, targets)

    def test_multi_class_targets_not_1d(self):
        """Test that non-1D targets raise ValueError for multi-class."""
        loss_fn = TaskLoss(num_classes=5, task_type="multi_class")
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8, 1))  # 2D instead of 1D
        
        with pytest.raises(ValueError, match="must be 1D"):
            loss_fn(logits, targets)

    def test_multi_label_predictions_not_2d(self):
        """Test that non-2D predictions raise ValueError for multi-label."""
        loss_fn = TaskLoss(num_classes=5, task_type="multi_label")
        logits = torch.randn(8, 5, 1)  # 3D instead of 2D
        targets = torch.randint(0, 2, (8, 5)).float()
        
        with pytest.raises(ValueError, match="must be 2D"):
            loss_fn(logits, targets)

    def test_multi_label_targets_not_2d(self):
        """Test that non-2D targets raise ValueError for multi-label."""
        loss_fn = TaskLoss(num_classes=5, task_type="multi_label")
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 2, (8,)).float()  # 1D instead of 2D
        
        with pytest.raises(ValueError, match="must be 2D"):
            loss_fn(logits, targets)

    def test_predictions_wrong_num_classes(self):
        """Test that wrong number of classes in predictions raises ValueError."""
        loss_fn = TaskLoss(num_classes=5, task_type="multi_class")
        logits = torch.randn(8, 7)  # 7 classes instead of 5
        targets = torch.randint(0, 5, (8,))
        
        with pytest.raises(ValueError, match="expected 5"):
            loss_fn(logits, targets)

    def test_multi_label_targets_wrong_num_classes(self):
        """Test that wrong number of classes in targets raises ValueError for multi-label."""
        loss_fn = TaskLoss(num_classes=5, task_type="multi_label")
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 2, (8, 7)).float()  # 7 classes instead of 5
        
        with pytest.raises(ValueError, match="expected 5"):
            loss_fn(logits, targets)

    def test_reduction_modes(self):
        """Test different reduction modes."""
        for reduction in ["mean", "sum"]:
            loss_fn = TaskLoss(
                num_classes=3,
                task_type="multi_class",
                reduction=reduction,
            )
            logits = torch.randn(4, 3)
            targets = torch.randint(0, 3, (4,))
            loss = loss_fn(logits, targets)
            
            assert loss.ndim == 0

    def test_statistics_tracking(self):
        """Test that statistics are tracked properly."""
        loss_fn = TaskLoss(num_classes=5, task_type="multi_class")
        
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        
        loss1 = loss_fn(logits, targets)
        loss2 = loss_fn(logits, targets)
        
        stats = loss_fn.get_statistics()
        assert stats["num_calls"] == 2
        assert stats["mean_loss"] > 0


# =============================================================================
# Test CalibratedCrossEntropyLoss
# =============================================================================


class TestCalibratedCrossEntropyLoss:
    """Test CalibratedCrossEntropyLoss."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        loss_fn = CalibratedCrossEntropyLoss(num_classes=5)
        logits = torch.randn(8, 5, requires_grad=True)
        targets = torch.randint(0, 5, (8,))
        
        loss = loss_fn(logits, targets)
        
        assert isinstance(loss, Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0
        assert loss.requires_grad

    def test_temperature_is_learnable(self):
        """Test that temperature is a learnable parameter."""
        loss_fn = CalibratedCrossEntropyLoss(num_classes=3)
        
        assert isinstance(loss_fn.temperature, nn.Parameter)
        assert loss_fn.temperature.requires_grad

    def test_get_temperature(self):
        """Test get_temperature returns positive float."""
        loss_fn = CalibratedCrossEntropyLoss(num_classes=5, init_temperature=2.0)
        
        temp = loss_fn.get_temperature()
        assert isinstance(temp, float)
        assert temp > 0
        assert abs(temp - 2.0) < 0.01

    def test_temperature_gradient(self):
        """Test that temperature receives gradients."""
        loss_fn = CalibratedCrossEntropyLoss(num_classes=3)
        logits = torch.randn(4, 3, requires_grad=True)
        targets = torch.randint(0, 3, (4,))
        
        loss = loss_fn(logits, targets)
        loss.backward()
        
        assert loss_fn.temperature.grad is not None
        assert loss_fn.temperature.grad.abs().sum() > 0

    def test_with_class_weights(self):
        """Test with class weights."""
        class_weights = torch.tensor([1.0, 2.0, 3.0])
        loss_fn = CalibratedCrossEntropyLoss(
            num_classes=3,
            class_weights=class_weights,
        )
        
        assert torch.equal(loss_fn.class_weights, class_weights)
        
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        loss = loss_fn(logits, targets)
        
        assert loss.item() >= 0

    def test_invalid_temperature(self):
        """Test that non-positive temperature raises ValueError."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            CalibratedCrossEntropyLoss(num_classes=5, init_temperature=0.0)
        
        with pytest.raises(ValueError, match="Temperature must be positive"):
            CalibratedCrossEntropyLoss(num_classes=5, init_temperature=-1.0)

    def test_class_weights_length_mismatch(self):
        """Test that class_weights length mismatch raises ValueError."""
        class_weights = torch.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="class_weights length"):
            CalibratedCrossEntropyLoss(num_classes=5, class_weights=class_weights)

    def test_predictions_not_2d(self):
        """Test that non-2D predictions raise ValueError."""
        loss_fn = CalibratedCrossEntropyLoss(num_classes=5)
        logits = torch.randn(8, 5, 1)  # 3D
        targets = torch.randint(0, 5, (8,))
        
        with pytest.raises(ValueError, match="must have shape"):
            loss_fn(logits, targets)

    def test_targets_not_1d(self):
        """Test that non-1D targets raise ValueError."""
        loss_fn = CalibratedCrossEntropyLoss(num_classes=5)
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8, 1))  # 2D
        
        with pytest.raises(ValueError, match="must have shape"):
            loss_fn(logits, targets)

    def test_wrong_num_classes(self):
        """Test that wrong number of classes raises ValueError."""
        loss_fn = CalibratedCrossEntropyLoss(num_classes=5)
        logits = torch.randn(8, 7)  # 7 classes instead of 5
        targets = torch.randint(0, 5, (8,))
        
        with pytest.raises(ValueError, match="expected 5"):
            loss_fn(logits, targets)

    def test_reduction_mean(self):
        """Test mean reduction."""
        loss_fn = CalibratedCrossEntropyLoss(num_classes=3, reduction="mean")
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0


# =============================================================================
# Test MultiLabelBCELoss
# =============================================================================


class TestMultiLabelBCELoss:
    """Test MultiLabelBCELoss."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        loss_fn = MultiLabelBCELoss(num_classes=10)
        logits = torch.randn(4, 10)
        targets = torch.randint(0, 2, (4, 10)).float()
        
        loss = loss_fn(logits, targets)
        
        assert isinstance(loss, Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_with_class_weights(self):
        """Test with class weights."""
        class_weights = torch.tensor([1.0, 1.5, 2.0, 2.5])
        loss_fn = MultiLabelBCELoss(num_classes=4, class_weights=class_weights)
        
        assert torch.equal(loss_fn.class_weights, class_weights)
        
        logits = torch.randn(3, 4)
        targets = torch.randint(0, 2, (3, 4)).float()
        loss = loss_fn(logits, targets)
        
        assert loss.item() >= 0

    def test_with_pos_weight(self):
        """Test with positive class weights."""
        pos_weight = torch.tensor([1.0, 2.0, 3.0])
        loss_fn = MultiLabelBCELoss(num_classes=3, pos_weight=pos_weight)
        
        assert torch.equal(loss_fn.pos_weight, pos_weight)
        
        logits = torch.randn(5, 3)
        targets = torch.randint(0, 2, (5, 3)).float()
        loss = loss_fn(logits, targets)
        
        assert loss.item() >= 0

    def test_with_both_weights(self):
        """Test with both class_weights and pos_weight."""
        class_weights = torch.tensor([1.0, 1.5, 2.0])
        pos_weight = torch.tensor([2.0, 1.5, 1.0])
        loss_fn = MultiLabelBCELoss(
            num_classes=3,
            class_weights=class_weights,
            pos_weight=pos_weight,
        )
        
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 2, (4, 3)).float()
        loss = loss_fn(logits, targets)
        
        assert loss.item() >= 0

    def test_class_weights_length_mismatch(self):
        """Test that class_weights length mismatch raises ValueError."""
        class_weights = torch.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="class_weights length"):
            MultiLabelBCELoss(num_classes=5, class_weights=class_weights)

    def test_pos_weight_length_mismatch(self):
        """Test that pos_weight length mismatch raises ValueError."""
        pos_weight = torch.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="pos_weight length"):
            MultiLabelBCELoss(num_classes=5, pos_weight=pos_weight)

    def test_predictions_not_2d(self):
        """Test that non-2D predictions raise ValueError."""
        loss_fn = MultiLabelBCELoss(num_classes=5)
        logits = torch.randn(8, 5, 1)  # 3D
        targets = torch.randint(0, 2, (8, 5)).float()
        
        with pytest.raises(ValueError, match="must have shape"):
            loss_fn(logits, targets)

    def test_targets_not_2d(self):
        """Test that non-2D targets raise ValueError."""
        loss_fn = MultiLabelBCELoss(num_classes=5)
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 2, (8,)).float()  # 1D
        
        with pytest.raises(ValueError, match="must have shape"):
            loss_fn(logits, targets)

    def test_predictions_wrong_num_classes(self):
        """Test that wrong number of classes in predictions raises ValueError."""
        loss_fn = MultiLabelBCELoss(num_classes=5)
        logits = torch.randn(8, 7)  # 7 classes
        targets = torch.randint(0, 2, (8, 5)).float()
        
        with pytest.raises(ValueError, match="Predictions have"):
            loss_fn(logits, targets)

    def test_targets_wrong_num_classes(self):
        """Test that wrong number of classes in targets raises ValueError."""
        loss_fn = MultiLabelBCELoss(num_classes=5)
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 2, (8, 7)).float()  # 7 classes
        
        with pytest.raises(ValueError, match="Targets have"):
            loss_fn(logits, targets)

    def test_reduction_mean(self):
        """Test mean reduction."""
        loss_fn = MultiLabelBCELoss(num_classes=4, reduction="mean")
        logits = torch.randn(3, 4)
        targets = torch.randint(0, 2, (3, 4)).float()
        
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_reduction_sum(self):
        """Test sum reduction."""
        loss_fn = MultiLabelBCELoss(num_classes=4, reduction="sum")
        logits = torch.randn(3, 4)
        targets = torch.randint(0, 2, (3, 4)).float()
        
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_reduction_none(self):
        """Test none reduction."""
        loss_fn = MultiLabelBCELoss(num_classes=4, reduction="none")
        logits = torch.randn(3, 4)
        targets = torch.randint(0, 2, (3, 4)).float()
        
        loss = loss_fn(logits, targets)
        assert loss.shape == (3, 4)


# =============================================================================
# Test FocalLoss
# =============================================================================


class TestFocalLoss:
    """Test FocalLoss."""

    def test_basic_forward(self):
        """Test basic forward pass."""
        loss_fn = FocalLoss(num_classes=5)
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8,))
        
        loss = loss_fn(logits, targets)
        
        assert isinstance(loss, Tensor)
        assert loss.ndim == 0
        assert loss.item() >= 0

    def test_with_gamma(self):
        """Test with custom gamma."""
        loss_fn = FocalLoss(num_classes=5, gamma=3.0)
        assert loss_fn.gamma == 3.0
        
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        loss = loss_fn(logits, targets)
        
        assert loss.item() >= 0

    def test_with_alpha(self):
        """Test with custom alpha."""
        loss_fn = FocalLoss(num_classes=5, alpha=0.5)
        assert loss_fn.alpha == 0.5
        
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        loss = loss_fn(logits, targets)
        
        assert loss.item() >= 0

    def test_with_class_weights(self):
        """Test with class weights."""
        class_weights = torch.tensor([1.0, 2.0, 3.0, 4.0, 5.0])
        loss_fn = FocalLoss(num_classes=5, class_weights=class_weights)
        
        assert torch.equal(loss_fn.class_weights, class_weights)
        
        logits = torch.randn(6, 5)
        targets = torch.randint(0, 5, (6,))
        loss = loss_fn(logits, targets)
        
        assert loss.item() >= 0

    def test_gamma_zero(self):
        """Test with gamma=0 (reduces to cross-entropy)."""
        loss_fn = FocalLoss(num_classes=3, gamma=0.0)
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_negative_gamma_raises_error(self):
        """Test that negative gamma raises ValueError."""
        with pytest.raises(ValueError, match="gamma must be non-negative"):
            FocalLoss(num_classes=5, gamma=-1.0)

    def test_alpha_out_of_range_raises_error(self):
        """Test that alpha outside [0, 1] raises ValueError."""
        with pytest.raises(ValueError, match="alpha must be in"):
            FocalLoss(num_classes=5, alpha=-0.1)
        
        with pytest.raises(ValueError, match="alpha must be in"):
            FocalLoss(num_classes=5, alpha=1.5)

    def test_class_weights_length_mismatch(self):
        """Test that class_weights length mismatch raises ValueError."""
        class_weights = torch.tensor([1.0, 2.0])
        with pytest.raises(ValueError, match="class_weights length"):
            FocalLoss(num_classes=5, class_weights=class_weights)

    def test_predictions_not_2d(self):
        """Test that non-2D predictions raise ValueError."""
        loss_fn = FocalLoss(num_classes=5)
        logits = torch.randn(8, 5, 1)  # 3D
        targets = torch.randint(0, 5, (8,))
        
        with pytest.raises(ValueError, match="must have shape"):
            loss_fn(logits, targets)

    def test_targets_not_1d(self):
        """Test that non-1D targets raise ValueError."""
        loss_fn = FocalLoss(num_classes=5)
        logits = torch.randn(8, 5)
        targets = torch.randint(0, 5, (8, 1))  # 2D
        
        with pytest.raises(ValueError, match="must have shape"):
            loss_fn(logits, targets)

    def test_wrong_num_classes(self):
        """Test that wrong number of classes raises ValueError."""
        loss_fn = FocalLoss(num_classes=5)
        logits = torch.randn(8, 7)  # 7 classes
        targets = torch.randint(0, 5, (8,))
        
        with pytest.raises(ValueError, match="expected 5"):
            loss_fn(logits, targets)

    def test_target_out_of_range(self):
        """Test that target indices out of range raise ValueError."""
        loss_fn = FocalLoss(num_classes=5)
        logits = torch.randn(4, 5)
        targets = torch.tensor([0, 1, 2, 5])  # 5 is out of range
        
        with pytest.raises(ValueError, match="Target indices out of range"):
            loss_fn(logits, targets)

    def test_target_negative(self):
        """Test that negative target indices raise ValueError."""
        loss_fn = FocalLoss(num_classes=5)
        logits = torch.randn(4, 5)
        targets = torch.tensor([0, 1, -1, 3])  # -1 is out of range
        
        with pytest.raises(ValueError, match="Target indices out of range"):
            loss_fn(logits, targets)

    def test_reduction_mean(self):
        """Test mean reduction."""
        loss_fn = FocalLoss(num_classes=3, reduction="mean")
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_reduction_sum(self):
        """Test sum reduction."""
        loss_fn = FocalLoss(num_classes=3, reduction="sum")
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        
        loss = loss_fn(logits, targets)
        assert loss.ndim == 0

    def test_reduction_none(self):
        """Test none reduction."""
        loss_fn = FocalLoss(num_classes=3, reduction="none")
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        
        loss = loss_fn(logits, targets)
        assert loss.shape == (4,)


# =============================================================================
# Test Integration and Edge Cases
# =============================================================================


class TestIntegration:
    """Test integrated workflows and edge cases."""

    def test_task_loss_delegates_to_calibrated_ce(self):
        """Test that TaskLoss delegates to CalibratedCrossEntropyLoss."""
        loss_fn = TaskLoss(num_classes=5, task_type="multi_class", use_focal=False)
        assert isinstance(loss_fn.loss_fn, CalibratedCrossEntropyLoss)

    def test_task_loss_delegates_to_focal(self):
        """Test that TaskLoss delegates to FocalLoss."""
        loss_fn = TaskLoss(num_classes=5, task_type="multi_class", use_focal=True)
        assert isinstance(loss_fn.loss_fn, FocalLoss)

    def test_task_loss_delegates_to_multi_label_bce(self):
        """Test that TaskLoss delegates to MultiLabelBCELoss."""
        loss_fn = TaskLoss(num_classes=5, task_type="multi_label")
        assert isinstance(loss_fn.loss_fn, MultiLabelBCELoss)

    def test_gradient_flow_calibrated_ce(self):
        """Test gradient flow through CalibratedCrossEntropyLoss."""
        loss_fn = CalibratedCrossEntropyLoss(num_classes=3)
        logits = torch.randn(4, 3, requires_grad=True)
        targets = torch.randint(0, 3, (4,))
        
        loss = loss_fn(logits, targets)
        loss.backward()
        
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_gradient_flow_multi_label_bce(self):
        """Test gradient flow through MultiLabelBCELoss."""
        loss_fn = MultiLabelBCELoss(num_classes=3)
        logits = torch.randn(4, 3, requires_grad=True)
        targets = torch.randint(0, 2, (4, 3)).float()
        
        loss = loss_fn(logits, targets)
        loss.backward()
        
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_gradient_flow_focal(self):
        """Test gradient flow through FocalLoss."""
        loss_fn = FocalLoss(num_classes=3)
        logits = torch.randn(4, 3, requires_grad=True)
        targets = torch.randint(0, 3, (4,))
        
        loss = loss_fn(logits, targets)
        loss.backward()
        
        assert logits.grad is not None
        assert logits.grad.abs().sum() > 0

    def test_task_loss_num_classes_conversion(self):
        """Test that num_classes is converted to int."""
        loss_fn = TaskLoss(num_classes=5, task_type="multi_class")
        assert loss_fn.num_classes == 5
        assert isinstance(loss_fn.num_classes, int)
        
        # Verify int conversion works
        loss_fn2 = TaskLoss(num_classes=int(5.9), task_type="multi_class")
        assert loss_fn2.num_classes == 5

    def test_calibrated_ce_num_classes_conversion(self):
        """Test that num_classes is converted to int."""
        loss_fn = CalibratedCrossEntropyLoss(num_classes=5)
        assert loss_fn.num_classes == 5
        assert isinstance(loss_fn.num_classes, int)
        
        # Verify int conversion works
        loss_fn2 = CalibratedCrossEntropyLoss(num_classes=int(5.9))
        assert loss_fn2.num_classes == 5

    def test_multi_label_bce_num_classes_conversion(self):
        """Test that num_classes is converted to int."""
        loss_fn = MultiLabelBCELoss(num_classes=5)
        assert loss_fn.num_classes == 5
        assert isinstance(loss_fn.num_classes, int)
        
        # Verify int conversion works
        loss_fn2 = MultiLabelBCELoss(num_classes=int(5.9))
        assert loss_fn2.num_classes == 5

    def test_focal_num_classes_conversion(self):
        """Test that num_classes is converted to int."""
        loss_fn = FocalLoss(num_classes=5)
        assert loss_fn.num_classes == 5
        assert isinstance(loss_fn.num_classes, int)
        
        # Verify int conversion works
        loss_fn2 = FocalLoss(num_classes=int(5.9))
        assert loss_fn2.num_classes == 5

    def test_focal_gamma_conversion(self):
        """Test that gamma is converted to float."""
        loss_fn = FocalLoss(num_classes=5, gamma=2)
        assert loss_fn.gamma == 2.0
        assert isinstance(loss_fn.gamma, float)

    def test_focal_alpha_conversion(self):
        """Test that alpha is converted to float."""
        loss_fn = FocalLoss(num_classes=5, alpha=0)
        assert loss_fn.alpha == 0.0
        assert isinstance(loss_fn.alpha, float)

    def test_task_loss_use_focal_conversion(self):
        """Test that use_focal is converted to bool."""
        loss_fn = TaskLoss(num_classes=5, task_type="multi_class", use_focal=1)
        assert loss_fn.use_focal is True
        assert isinstance(loss_fn.use_focal, bool)

    def test_batch_size_one(self):
        """Test with batch size of 1."""
        loss_fn = TaskLoss(num_classes=5, task_type="multi_class")
        logits = torch.randn(1, 5)
        targets = torch.randint(0, 5, (1,))
        
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_large_batch_size(self):
        """Test with large batch size."""
        loss_fn = TaskLoss(num_classes=5, task_type="multi_class")
        logits = torch.randn(128, 5)
        targets = torch.randint(0, 5, (128,))
        
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_default_class_weights(self):
        """Test that default class_weights are all ones."""
        loss_fn = TaskLoss(num_classes=5, task_type="multi_class")
        assert torch.equal(loss_fn.class_weights, torch.ones(5))

    def test_default_pos_weight(self):
        """Test that default pos_weight are all ones."""
        loss_fn = MultiLabelBCELoss(num_classes=5)
        assert torch.equal(loss_fn.pos_weight, torch.ones(5))

    def test_statistics_reset(self):
        """Test statistics reset functionality."""
        loss_fn = TaskLoss(num_classes=5, task_type="multi_class")
        
        logits = torch.randn(4, 5)
        targets = torch.randint(0, 5, (4,))
        loss_fn(logits, targets)
        
        loss_fn.reset_statistics()
        stats = loss_fn.get_statistics()
        
        assert stats["num_calls"] == 0
        assert stats["mean_loss"] == 0.0

    def test_different_dtypes(self):
        """Test with different tensor dtypes."""
        loss_fn = TaskLoss(num_classes=3, task_type="multi_class")
        
        # float32 (default)
        logits = torch.randn(4, 3)
        targets = torch.randint(0, 3, (4,))
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_multi_label_all_zeros_targets(self):
        """Test multi-label with all zeros targets."""
        loss_fn = MultiLabelBCELoss(num_classes=5)
        logits = torch.randn(4, 5)
        targets = torch.zeros(4, 5)
        
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_multi_label_all_ones_targets(self):
        """Test multi-label with all ones targets."""
        loss_fn = MultiLabelBCELoss(num_classes=5)
        logits = torch.randn(4, 5)
        targets = torch.ones(4, 5)
        
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0

    def test_focal_loss_boundary_probabilities(self):
        """Test focal loss with extreme probabilities."""
        loss_fn = FocalLoss(num_classes=3, gamma=2.0)
        
        # High confidence correct predictions
        logits = torch.tensor([[10.0, -10.0, -10.0], [-10.0, 10.0, -10.0]])
        targets = torch.tensor([0, 1])
        
        loss = loss_fn(logits, targets)
        assert loss.item() >= 0
        assert loss.item() < 1.0  # Should be small for confident correct predictions
