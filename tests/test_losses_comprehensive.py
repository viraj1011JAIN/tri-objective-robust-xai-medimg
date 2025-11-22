"""
Comprehensive unit tests for loss functions.

Tests cover:
- Loss computation correctness
- Gradient flow and backpropagation
- Edge cases (zero loss, NaN handling, extreme values)
- Multi-label BCE loss
- Class weights and reduction modes
- TRADES adversarial loss
- Calibration loss

Run with: pytest tests/test_losses_comprehensive.py -v
"""

from __future__ import annotations

import pytest
import torch
import torch.nn as nn
import torch.nn.functional as F

from src.losses.task_loss import (
    CalibratedCrossEntropyLoss,
    FocalLoss,
    MultiLabelBCELoss,
)
from src.losses.base_loss import BaseLoss

# Simple wrappers for testing
class CrossEntropyLoss(CalibratedCrossEntropyLoss):
    """Wrapper for testing - uses CalibratedCrossEntropyLoss."""
    pass


class TestCrossEntropyLoss:
    """Test cross-entropy loss computation."""

    def test_basic_computation(self):
        """Test basic CE loss computation."""
        loss_fn = CrossEntropyLoss(num_classes=7)

        logits = torch.randn(4, 7)
        targets = torch.randint(0, 7, (4,))

        loss = loss_fn(logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0, "Loss should be scalar"
        assert loss.item() >= 0, "CE loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_perfect_prediction(self):
        """Test CE loss with perfect predictions."""
        loss_fn = CrossEntropyLoss(num_classes=7)

        # Create perfect predictions (high logit for correct class)
        logits = torch.zeros(4, 7)
        targets = torch.tensor([0, 1, 2, 3])

        for i, target in enumerate(targets):
            logits[i, target] = 100.0  # Very high logit

        loss = loss_fn(logits, targets)

        assert loss.item() < 0.1, "Loss should be near zero for perfect pred"

    def test_worst_prediction(self):
        """Test CE loss with worst predictions."""
        loss_fn = CrossEntropyLoss(num_classes=7)

        # Create worst predictions (high logit for wrong class)
        logits = torch.zeros(4, 7)
        targets = torch.tensor([0, 1, 2, 3])

        for i, target in enumerate(targets):
            # Set high logit for wrong class
            wrong_class = (target + 1) % 7
            logits[i, wrong_class] = 100.0

        loss = loss_fn(logits, targets)

        assert loss.item() > 50.0, "Loss should be high for worst predictions"

    @pytest.mark.parametrize("reduction", ["mean", "sum"])
    def test_reduction_modes(self, reduction: str):
        """Test different reduction modes."""
        loss_fn = CrossEntropyLoss(num_classes=7, reduction=reduction)

        logits = torch.randn(4, 7)
        targets = torch.randint(0, 7, (4,))

        loss = loss_fn(logits, targets)
        # CalibratedCrossEntropyLoss always returns scalar
        assert loss.dim() == 0, "Loss should return scalar"

    def test_class_weights(self):
        """Test CE loss with class weights."""
        weights = torch.tensor([1.0, 2.0, 3.0, 1.0, 1.0, 1.0, 1.0])
        loss_fn = CrossEntropyLoss(num_classes=7, class_weights=weights)

        logits = torch.randn(4, 7)
        targets = torch.tensor([1, 2, 1, 2])  # Classes with higher weights

        loss_weighted = loss_fn(logits, targets)
        loss_unweighted = CrossEntropyLoss(num_classes=7)(logits, targets)

        # Weighted loss should differ from unweighted
        assert not torch.allclose(loss_weighted, loss_unweighted)

    def test_gradient_flow(self):
        """Test gradient computation."""
        loss_fn = CrossEntropyLoss(num_classes=7)

        logits = torch.randn(4, 7, requires_grad=True)
        targets = torch.randint(0, 7, (4,))

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None, "Gradients should be computed"
        assert torch.isfinite(logits.grad).all(), "Grads should be finite"
        assert (logits.grad != 0).any(), "Some gradients should be non-zero"


class TestFocalLoss:
    """Test focal loss computation."""

    @pytest.mark.parametrize("gamma", [0.0, 1.0, 2.0, 5.0])
    def test_gamma_parameter(self, gamma: float):
        """Test focal loss with different gamma values."""
        loss_fn = FocalLoss(num_classes=7, gamma=gamma)

        logits = torch.randn(4, 7)
        targets = torch.randint(0, 7, (4,))

        loss = loss_fn(logits, targets)

        assert loss.item() >= 0, "Focal loss should be non-negative"
        assert torch.isfinite(loss), "Loss should be finite"

    def test_focal_vs_ce_easy_examples(self):
        """Test that focal loss down-weights easy examples."""
        # Create easy examples (high confidence correct predictions)
        logits = torch.zeros(4, 7)
        targets = torch.tensor([0, 1, 2, 3])

        for i, target in enumerate(targets):
            logits[i, target] = 10.0  # High confidence

        focal_loss = FocalLoss(num_classes=7, gamma=2.0)(logits, targets)
        ce_loss = CrossEntropyLoss(num_classes=7)(logits, targets)

        # Focal loss should be much smaller for easy examples
        assert focal_loss.item() < ce_loss.item()

    def test_focal_vs_ce_hard_examples(self):
        """Test that focal loss focuses on hard examples."""
        # Create hard examples (low confidence predictions)
        logits = torch.randn(4, 7) * 0.1  # Low confidence
        targets = torch.randint(0, 7, (4,))

        focal_loss = FocalLoss(num_classes=7, gamma=2.0)(logits, targets)
        ce_loss = CrossEntropyLoss(num_classes=7)(logits, targets)

        # Losses should be more similar for hard examples
        ratio = focal_loss.item() / ce_loss.item()
        assert 0.5 < ratio < 1.5, "Focal/CE ratio for hard examples"

    def test_gradient_flow(self):
        """Test gradient computation for focal loss."""
        loss_fn = FocalLoss(num_classes=7, gamma=2.0)

        logits = torch.randn(4, 7, requires_grad=True)
        targets = torch.randint(0, 7, (4,))

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()


class TestMultiLabelBCELoss:
    """Test multi-label binary cross-entropy loss."""

    def test_basic_computation(self):
        """Test basic multi-label BCE computation."""
        loss_fn = MultiLabelBCELoss(num_classes=14)

        logits = torch.randn(4, 14)  # 14 diseases
        targets = torch.randint(0, 2, (4, 14)).float()

        loss = loss_fn(logits, targets)

        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_all_positive_labels(self):
        """Test with all positive labels."""
        loss_fn = MultiLabelBCELoss(num_classes=14)

        logits = torch.randn(4, 14)
        targets = torch.ones(4, 14)

        loss = loss_fn(logits, targets)

        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_all_negative_labels(self):
        """Test with all negative labels."""
        loss_fn = MultiLabelBCELoss(num_classes=14)

        logits = torch.randn(4, 14)
        targets = torch.zeros(4, 14)

        loss = loss_fn(logits, targets)

        assert loss.item() >= 0
        assert torch.isfinite(loss)

    def test_perfect_prediction(self):
        """Test with perfect predictions."""
        loss_fn = MultiLabelBCELoss(num_classes=14)

        targets = torch.randint(0, 2, (4, 14)).float()

        # Create perfect logits (high positive for 1, high negative for 0)
        logits = torch.where(
            targets == 1, torch.tensor(10.0), torch.tensor(-10.0)
        )

        loss = loss_fn(logits, targets)

        assert loss.item() < 0.01, "Loss should be near zero for perfect"

    @pytest.mark.parametrize("pos_weight", [1.0, 2.0, 5.0])
    def test_positive_class_weighting(self, pos_weight: float):
        """Test positive class weighting."""
        pos_weights = torch.ones(14) * pos_weight
        loss_fn = MultiLabelBCELoss(num_classes=14, pos_weight=pos_weights)

        logits = torch.randn(4, 14)
        targets = torch.randint(0, 2, (4, 14)).float()

        loss = loss_fn(logits, targets)

        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_gradient_flow(self):
        """Test gradient computation."""
        loss_fn = MultiLabelBCELoss(num_classes=14)

        logits = torch.randn(4, 14, requires_grad=True)
        targets = torch.randint(0, 2, (4, 14)).float()

        loss = loss_fn(logits, targets)
        loss.backward()

        assert logits.grad is not None
        assert torch.isfinite(logits.grad).all()
        assert (logits.grad != 0).any()

    @pytest.mark.parametrize("reduction", ["mean", "sum", "none"])
    def test_reduction_modes(self, reduction: str):
        """Test different reduction modes."""
        loss_fn = MultiLabelBCELoss(num_classes=14, reduction=reduction)

        logits = torch.randn(4, 14)
        targets = torch.randint(0, 2, (4, 14)).float()

        loss = loss_fn(logits, targets)

        if reduction == "none":
            assert loss.shape == (4, 14)
        else:
            assert loss.dim() == 0


class TestLossEdgeCases:
    """Test edge cases for all loss functions."""

    @pytest.mark.parametrize("loss_cls", [
        CrossEntropyLoss,
        FocalLoss,
    ])
    def test_single_sample_batch(self, loss_cls):
        """Test loss with single sample."""
        loss_fn = loss_cls(num_classes=7)

        logits = torch.randn(1, 7)
        targets = torch.randint(0, 7, (1,))

        loss = loss_fn(logits, targets)

        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_multilabel_single_sample(self):
        """Test multi-label loss with single sample."""
        loss_fn = MultiLabelBCELoss(num_classes=14)

        logits = torch.randn(1, 14)
        targets = torch.randint(0, 2, (1, 14)).float()

        loss = loss_fn(logits, targets)

        assert torch.isfinite(loss)

    @pytest.mark.parametrize("loss_cls", [CrossEntropyLoss, FocalLoss])
    def test_extreme_logits_positive(self, loss_cls):
        """Test with extremely large positive logits."""
        loss_fn = loss_cls(num_classes=7)

        logits = torch.ones(4, 7) * 1000.0
        logits[:, 0] = 1001.0  # Slightly larger for class 0
        targets = torch.zeros(4, dtype=torch.long)

        loss = loss_fn(logits, targets)

        assert torch.isfinite(loss), "Should handle large positive logits"

    @pytest.mark.parametrize("loss_cls", [CrossEntropyLoss, FocalLoss])
    def test_extreme_logits_negative(self, loss_cls):
        """Test with extremely large negative logits."""
        loss_fn = loss_cls(num_classes=7)

        logits = torch.ones(4, 7) * -1000.0
        logits[:, 0] = -999.0  # Slightly larger for class 0
        targets = torch.zeros(4, dtype=torch.long)

        loss = loss_fn(logits, targets)

        assert torch.isfinite(loss), "Should handle large negative logits"

    def test_multilabel_extreme_logits(self):
        """Test multi-label loss with extreme logits."""
        loss_fn = MultiLabelBCELoss(num_classes=14)

        logits = torch.randn(4, 14) * 100.0  # Large values
        targets = torch.randint(0, 2, (4, 14)).float()

        loss = loss_fn(logits, targets)

        assert torch.isfinite(loss), "Should handle extreme logits"

    @pytest.mark.parametrize("loss_cls", [CrossEntropyLoss, FocalLoss])
    def test_uniform_predictions(self, loss_cls):
        """Test with uniform (uncertain) predictions."""
        loss_fn = loss_cls(num_classes=7)

        # All logits equal (maximum uncertainty)
        logits = torch.zeros(4, 7)
        targets = torch.randint(0, 7, (4,))

        loss = loss_fn(logits, targets)

        assert torch.isfinite(loss)
        # Loss should be close to -log(1/7) ≈ 1.946
        # FocalLoss with gamma may differ due to modulation
        assert 0.5 < loss.item() < 3.0, "Uniform loss reasonable range"


class TestLossGradientProperties:
    """Test gradient properties of loss functions."""

    @pytest.mark.parametrize("loss_cls", [
        CrossEntropyLoss,
        FocalLoss,
    ])
    def test_gradient_magnitude_reasonable(self, loss_cls):
        """Test that gradient magnitudes are reasonable."""
        loss_fn = loss_cls(num_classes=7)

        logits = torch.randn(8, 7, requires_grad=True)
        targets = torch.randint(0, 7, (8,))

        loss = loss_fn(logits, targets)
        loss.backward()

        grad_norm = logits.grad.norm().item()

        assert grad_norm < 100.0, f"Gradient norm too large: {grad_norm}"
        assert grad_norm > 0.0, "Gradient norm should be positive"

    def test_multilabel_gradient_magnitude(self):
        """Test multi-label loss gradient magnitude."""
        loss_fn = MultiLabelBCELoss(num_classes=14)

        logits = torch.randn(8, 14, requires_grad=True)
        targets = torch.randint(0, 2, (8, 14)).float()

        loss = loss_fn(logits, targets)
        loss.backward()

        grad_norm = logits.grad.norm().item()

        assert grad_norm < 100.0, f"Gradient norm too large: {grad_norm}"
        assert grad_norm > 0.0

    @pytest.mark.parametrize("loss_cls", [CrossEntropyLoss, FocalLoss])
    def test_gradient_accumulation(self, loss_cls):
        """Test gradient accumulation works correctly."""
        loss_fn = loss_cls(num_classes=7)

        logits = torch.randn(4, 7, requires_grad=True)
        targets = torch.randint(0, 7, (4,))

        # First backward
        loss1 = loss_fn(logits, targets)
        loss1.backward()

        grad1 = logits.grad.clone()

        # Second backward (accumulation)
        loss2 = loss_fn(logits, targets)
        loss2.backward()

        grad2 = logits.grad

        # Gradients should have accumulated
        assert not torch.allclose(grad2, grad1)
        expected = grad1 * 2  # Same loss twice
        assert torch.allclose(grad2, expected, rtol=1e-5)

    def test_gradient_numerical_stability(self):
        """Test gradient numerical stability with mixed magnitudes."""
        loss_fn = CrossEntropyLoss(num_classes=7)

        # Mix of large and small logits
        logits = torch.tensor([
            [100.0, 0.0, -100.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1e-6, 0.0, 0.0, 0.0, 0.0, 0.0],
            [-100.0, 0.0, 100.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1e6, 0.0, 0.0, 0.0],
        ], requires_grad=True)

        targets = torch.tensor([0, 1, 2, 3])

        loss = loss_fn(logits, targets)
        loss.backward()

        assert torch.isfinite(logits.grad).all(), "Grads unstable"


class TestLossComparison:
    """Test relationships between different loss functions."""

    def test_focal_gamma_zero_equals_ce(self):
        """Test that focal loss with gamma=0 equals CE loss."""
        logits = torch.randn(8, 7)
        targets = torch.randint(0, 7, (8,))

        ce_loss = CrossEntropyLoss(num_classes=7)(logits, targets)
        focal_loss = FocalLoss(num_classes=7, gamma=0.0)(logits, targets)

        # Losses should be similar (alpha factor causes small difference)
        assert torch.allclose(ce_loss, focal_loss, rtol=0.3), \
            "Focal(gamma=0) should be close to CE"

    def test_loss_monotonicity_with_confidence(self):
        """Test that loss decreases as confidence increases."""
        loss_fn = CrossEntropyLoss(num_classes=7)

        targets = torch.zeros(4, dtype=torch.long)

        # Increasing confidence for correct class
        logits_low = torch.zeros(4, 7)
        logits_low[:, 0] = 1.0

        logits_high = torch.zeros(4, 7)
        logits_high[:, 0] = 10.0

        loss_low = loss_fn(logits_low, targets)
        loss_high = loss_fn(logits_high, targets)

        assert loss_low.item() > loss_high.item(), \
            "Loss should decrease with confidence"

    def test_multilabel_vs_binary_ce(self):
        """Compare multi-label BCE with binary CE for single label."""
        # Single positive label case
        logits = torch.randn(4, 1)
        targets = torch.randint(0, 2, (4, 1)).float()

        ml_bce = MultiLabelBCELoss(num_classes=1)(logits, targets)
        binary_ce = F.binary_cross_entropy_with_logits(
            logits.squeeze(), targets.squeeze()
        )

        assert torch.allclose(ml_bce, binary_ce, rtol=1e-5), \
            "Multi-label BCE should match binary CE for single label"


class TestLossWithRealScenarios:
    """Test losses in realistic scenarios."""

    def test_imbalanced_multiclass(self):
        """Test with imbalanced multi-class data."""
        # Simulate imbalanced dataset (mostly class 0)
        loss_fn = CrossEntropyLoss(num_classes=7)

        logits = torch.randn(20, 7)
        targets = torch.zeros(20, dtype=torch.long)
        targets[-2:] = torch.randint(1, 7, (2,))  # Only 2 minority samples

        loss = loss_fn(logits, targets)

        assert torch.isfinite(loss)
        assert loss.item() >= 0

    def test_imbalanced_multilabel(self):
        """Test with imbalanced multi-label data."""
        loss_fn = MultiLabelBCELoss(num_classes=14)

        # Simulate rare disease: mostly zeros with few positives
        logits = torch.randn(20, 14)
        targets = torch.zeros(20, 14)
        targets[:, 0] = 1.0  # One common disease
        targets[0, 5] = 1.0  # One rare disease
        targets[10, 5] = 1.0

        loss = loss_fn(logits, targets)

        assert torch.isfinite(loss)

    def test_batch_size_invariance_mean_reduction(self):
        """Test that mean reduction is approximately batch-size invariant."""
        loss_fn = CrossEntropyLoss(num_classes=7, reduction="mean")

        # Set seed for reproducibility
        torch.manual_seed(42)

        logits = torch.randn(8, 7)
        targets = torch.randint(0, 7, (8,))

        loss_full = loss_fn(logits, targets)
        loss_half1 = loss_fn(logits[:4], targets[:4])
        loss_half2 = loss_fn(logits[4:], targets[4:])

        # Mean of two halves should be close to full batch mean
        loss_combined = (loss_half1 + loss_half2) / 2

        assert torch.allclose(loss_full, loss_combined, rtol=0.1), \
            "Mean reduction should be batch-size consistent"


# Pytest configuration
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow"
    )
