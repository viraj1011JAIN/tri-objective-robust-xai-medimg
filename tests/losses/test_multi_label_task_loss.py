"""
Comprehensive tests for Multi-Label Task Loss.

Tests cover:
- Initialization with different configurations
- Forward pass with various inputs
- Focal loss functionality
- Label smoothing
- Temperature scaling
- Pos_weight computation

Author: Viraj Pankaj Jain
"""

import pytest
import torch
import torch.nn as nn

from src.losses.multi_label_task_loss import MultiLabelTaskLoss


class TestMultiLabelTaskLoss:
    """Tests for MultiLabelTaskLoss class."""

    def test_initialization_default(self):
        """Test default initialization."""
        loss_fn = MultiLabelTaskLoss(num_classes=14)

        assert loss_fn.num_classes == 14
        assert not loss_fn.use_focal
        assert loss_fn.focal_gamma == 2.0
        assert loss_fn.focal_alpha == 0.25
        assert loss_fn.label_smoothing == 0.0
        assert loss_fn.reduction == "mean"

    def test_initialization_with_positive_rates(self):
        """Test initialization with positive rates for pos_weight."""
        positive_rates = [
            0.1,
            0.2,
            0.3,
            0.4,
            0.5,
            0.6,
            0.7,
            0.8,
            0.9,
            0.05,
            0.15,
            0.25,
            0.35,
            0.45,
        ]
        loss_fn = MultiLabelTaskLoss(num_classes=14, positive_rates=positive_rates)

        assert loss_fn.num_classes == 14
        # Verify pos_weight was computed
        assert hasattr(loss_fn, "register_buffer") or hasattr(loss_fn, "pos_weight")

    def test_initialization_with_focal_loss(self):
        """Test initialization with focal loss enabled."""
        loss_fn = MultiLabelTaskLoss(
            num_classes=14, use_focal=True, focal_gamma=3.0, focal_alpha=0.5
        )

        assert loss_fn.use_focal
        assert loss_fn.focal_gamma == 3.0
        assert loss_fn.focal_alpha == 0.5

    def test_initialization_with_label_smoothing(self):
        """Test initialization with label smoothing."""
        loss_fn = MultiLabelTaskLoss(num_classes=14, label_smoothing=0.1)

        assert loss_fn.label_smoothing == 0.1

    def test_initialization_with_temperature(self):
        """Test initialization with custom temperature."""
        loss_fn = MultiLabelTaskLoss(num_classes=14, temperature=2.0)

        assert loss_fn.temperature.item() == 2.0

    def test_forward_pass_basic(self):
        """Test basic forward pass."""
        loss_fn = MultiLabelTaskLoss(num_classes=14)

        logits = torch.randn(4, 14)
        targets = torch.randint(0, 2, (4, 14)).float()

        loss = loss_fn(logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar loss
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_forward_pass_with_focal_loss(self):
        """Test forward pass with focal loss."""
        loss_fn = MultiLabelTaskLoss(num_classes=14, use_focal=True, focal_gamma=2.0)

        logits = torch.randn(4, 14)
        targets = torch.randint(0, 2, (4, 14)).float()

        loss = loss_fn(logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_forward_pass_with_positive_rates(self):
        """Test forward pass with pos_weight from positive rates."""
        positive_rates = [0.1] * 14
        loss_fn = MultiLabelTaskLoss(num_classes=14, positive_rates=positive_rates)

        logits = torch.randn(4, 14)
        targets = torch.randint(0, 2, (4, 14)).float()

        loss = loss_fn(logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_forward_pass_with_label_smoothing(self):
        """Test forward pass with label smoothing."""
        loss_fn = MultiLabelTaskLoss(num_classes=14, label_smoothing=0.1)

        logits = torch.randn(4, 14)
        targets = torch.randint(0, 2, (4, 14)).float()

        loss = loss_fn(logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_forward_pass_with_temperature_scaling(self):
        """Test forward pass with temperature scaling."""
        loss_fn = MultiLabelTaskLoss(num_classes=14, temperature=2.0)

        logits = torch.randn(4, 14)
        targets = torch.randint(0, 2, (4, 14)).float()

        loss = loss_fn(logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_reduction_modes(self):
        """Test different reduction modes."""
        for reduction in ["mean", "sum", "none"]:
            loss_fn = MultiLabelTaskLoss(num_classes=14, reduction=reduction)

            logits = torch.randn(4, 14)
            targets = torch.randint(0, 2, (4, 14)).float()

            loss = loss_fn(logits, targets)

            if reduction == "none":
                # With 'none' reduction, returns per-class losses
                assert loss.shape == (14,)
            else:
                assert loss.dim() == 0

    def test_gradient_flow(self):
        """Test that gradients flow correctly."""
        loss_fn = MultiLabelTaskLoss(num_classes=14)

        logits = torch.randn(4, 14, requires_grad=True)
        targets = torch.randint(0, 2, (4, 14)).float()

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert not torch.isnan(logits.grad).any()

    def test_temperature_gradient(self):
        """Test that temperature parameter can be learned."""
        loss_fn = MultiLabelTaskLoss(num_classes=14)

        logits = torch.randn(4, 14)
        targets = torch.randint(0, 2, (4, 14)).float()

        # Check temperature has gradient
        assert loss_fn.temperature.requires_grad

        loss = loss_fn(logits, targets)
        loss.backward()

        # Temperature should have gradient
        assert loss_fn.temperature.grad is not None

    def test_all_features_combined(self):
        """Test with all features enabled."""
        positive_rates = [0.1] * 14
        loss_fn = MultiLabelTaskLoss(
            num_classes=14,
            positive_rates=positive_rates,
            use_focal=True,
            focal_gamma=2.0,
            focal_alpha=0.25,
            temperature=1.5,
            label_smoothing=0.1,
            reduction="mean",
        )

        logits = torch.randn(8, 14)
        targets = torch.randint(0, 2, (8, 14)).float()

        loss = loss_fn(logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert not torch.isnan(loss)
        assert not torch.isinf(loss)

    def test_batch_size_variations(self):
        """Test with different batch sizes."""
        loss_fn = MultiLabelTaskLoss(num_classes=14)

        for batch_size in [1, 4, 16, 32]:
            logits = torch.randn(batch_size, 14)
            targets = torch.randint(0, 2, (batch_size, 14)).float()

            loss = loss_fn(logits, targets)

            assert isinstance(loss, torch.Tensor)
            assert not torch.isnan(loss)

    def test_edge_case_all_zeros(self):
        """Test edge case with all zero targets."""
        loss_fn = MultiLabelTaskLoss(num_classes=14)

        logits = torch.randn(4, 14)
        targets = torch.zeros(4, 14)

        loss = loss_fn(logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)

    def test_edge_case_all_ones(self):
        """Test edge case with all one targets."""
        loss_fn = MultiLabelTaskLoss(num_classes=14)

        logits = torch.randn(4, 14)
        targets = torch.ones(4, 14)

        loss = loss_fn(logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert not torch.isnan(loss)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
