"""
Comprehensive tests for confidence scoring module (Phase 8.1).

Tests cover all confidence scoring methods:
- Softmax Maximum
- Predictive Entropy
- MC Dropout
- Temperature Scaling

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
"""

import numpy as np
import pytest
import torch
import torch.nn as nn

from src.losses.calibration_loss import TemperatureScaling
from src.validation.confidence_scorer import (
    ConfidenceMethod,
    ConfidenceScore,
    ConfidenceScorer,
    EntropyScorer,
    MCDropoutScorer,
    SoftmaxMaxScorer,
    TemperatureScaledScorer,
    compute_confidence_metrics,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def sample_logits():
    """Generate sample logits for testing."""
    return torch.tensor([[2.0, 1.0, 0.5, 0.2, 0.1, 0.05, 0.02]])


@pytest.fixture
def batch_logits():
    """Generate batch of logits."""
    torch.manual_seed(42)
    return torch.randn(8, 7)  # 8 samples, 7 classes


@pytest.fixture
def simple_model():
    """Create simple model for testing."""

    class SimpleModel(nn.Module):
        def __init__(self):
            super().__init__()
            self.conv = nn.Conv2d(3, 16, 3, padding=1)
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.dropout = nn.Dropout(0.5)
            self.fc = nn.Linear(16, 7)

        def forward(self, x):
            x = self.conv(x)
            x = self.pool(x)
            x = x.flatten(1)
            x = self.dropout(x)
            x = self.fc(x)
            return x

    return SimpleModel()


@pytest.fixture
def sample_input():
    """Generate sample input image."""
    torch.manual_seed(42)
    return torch.randn(1, 3, 224, 224)


@pytest.fixture
def batch_input():
    """Generate batch of input images."""
    torch.manual_seed(42)
    return torch.randn(4, 3, 224, 224)


# ============================================================================
# Test ConfidenceScore Dataclass
# ============================================================================


class TestConfidenceScore:
    """Test ConfidenceScore dataclass."""

    def test_valid_confidence_score(self):
        """Test creation of valid confidence score."""
        score = ConfidenceScore(
            confidence=0.85,
            uncertainty=0.15,
            method=ConfidenceMethod.SOFTMAX_MAX,
            prediction=0,
            probabilities=np.array([0.85, 0.10, 0.05]),
            metadata={"temperature": 1.0},
        )

        assert score.confidence == 0.85
        assert score.uncertainty == 0.15
        assert score.method == ConfidenceMethod.SOFTMAX_MAX
        assert score.prediction == 0
        assert score.metadata["temperature"] == 1.0

    def test_invalid_confidence_raises_error(self):
        """Test that out-of-range confidence values are clipped for numerical stability."""
        # Confidence > 1 should be clipped to 1.0
        score = ConfidenceScore(
            confidence=1.5,  # Will be clipped to 1.0
            uncertainty=0.15,
            method=ConfidenceMethod.SOFTMAX_MAX,
            prediction=0,
            probabilities=np.array([0.85, 0.10, 0.05]),
            metadata={},
        )
        assert score.confidence == 1.0  # Clipped to valid range

        # Confidence < 0 should be clipped to 0.0
        score = ConfidenceScore(
            confidence=-0.1,  # Will be clipped to 0.0
            uncertainty=0.15,
            method=ConfidenceMethod.SOFTMAX_MAX,
            prediction=0,
            probabilities=np.array([0.85, 0.10, 0.05]),
            metadata={},
        )
        assert score.confidence == 0.0  # Clipped to valid range

    def test_invalid_uncertainty_raises_error(self):
        """Test that negative uncertainty raises error."""
        with pytest.raises(ValueError, match="Uncertainty must be >= 0"):
            ConfidenceScore(
                confidence=0.85,
                uncertainty=-0.1,  # Invalid: < 0
                method=ConfidenceMethod.SOFTMAX_MAX,
                prediction=0,
                probabilities=np.array([0.85, 0.10, 0.05]),
                metadata={},
            )


# ============================================================================
# Test SoftmaxMaxScorer
# ============================================================================


class TestSoftmaxMaxScorer:
    """Test SoftmaxMaxScorer."""

    def test_initialization_default(self):
        """Test default initialization."""
        scorer = SoftmaxMaxScorer()
        assert scorer.temperature == 1.0

    def test_initialization_custom_temperature(self):
        """Test initialization with custom temperature."""
        scorer = SoftmaxMaxScorer(temperature=2.0)
        assert scorer.temperature == 2.0

    def test_invalid_temperature_raises_error(self):
        """Test that invalid temperature raises error."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            SoftmaxMaxScorer(temperature=0.0)

        with pytest.raises(ValueError, match="Temperature must be positive"):
            SoftmaxMaxScorer(temperature=-1.0)

    def test_score_single_sample(self, sample_logits):
        """Test scoring single sample."""
        scorer = SoftmaxMaxScorer(temperature=1.0)
        score = scorer(sample_logits)

        assert isinstance(score, ConfidenceScore)
        assert 0 <= score.confidence <= 1
        assert score.uncertainty >= 0
        assert score.method == ConfidenceMethod.SOFTMAX_MAX
        assert score.prediction == 0  # Highest logit is at index 0
        assert len(score.probabilities) == 7

    def test_score_batch(self, batch_logits):
        """Test scoring batch returns first sample score."""
        scorer = SoftmaxMaxScorer(temperature=1.0)
        score = scorer(batch_logits)

        assert isinstance(score, ConfidenceScore)
        assert len(score.probabilities) == 7

    def test_temperature_affects_confidence(self, sample_logits):
        """Test that temperature affects confidence scores."""
        scorer_low = SoftmaxMaxScorer(temperature=0.5)
        scorer_high = SoftmaxMaxScorer(temperature=2.0)

        score_low = scorer_low(sample_logits)
        score_high = scorer_high(sample_logits)

        # Lower temperature should produce higher confidence (sharper distribution)
        assert score_low.confidence > score_high.confidence

    def test_probabilities_sum_to_one(self, sample_logits):
        """Test that probabilities sum to 1."""
        scorer = SoftmaxMaxScorer(temperature=1.0)
        score = scorer(sample_logits)

        assert np.allclose(np.sum(score.probabilities), 1.0, atol=1e-6)

    def test_metadata_contains_temperature(self, sample_logits):
        """Test that metadata contains temperature."""
        scorer = SoftmaxMaxScorer(temperature=1.5)
        score = scorer(sample_logits)

        assert "temperature" in score.metadata
        assert score.metadata["temperature"] == 1.5


# ============================================================================
# Test EntropyScorer
# ============================================================================


class TestEntropyScorer:
    """Test EntropyScorer."""

    def test_initialization_default(self):
        """Test default initialization."""
        scorer = EntropyScorer()
        assert scorer.temperature == 1.0
        assert scorer.epsilon > 0

    def test_initialization_custom_parameters(self):
        """Test initialization with custom parameters."""
        scorer = EntropyScorer(temperature=2.0, epsilon=1e-8)
        assert scorer.temperature == 2.0
        assert scorer.epsilon == 1e-8

    def test_invalid_parameters_raise_errors(self):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="Temperature must be positive"):
            EntropyScorer(temperature=0.0)

        with pytest.raises(ValueError, match="Epsilon must be positive"):
            EntropyScorer(epsilon=0.0)

    def test_score_single_sample(self, sample_logits):
        """Test scoring single sample."""
        scorer = EntropyScorer(temperature=1.0)
        score = scorer(sample_logits)

        assert isinstance(score, ConfidenceScore)
        assert 0 <= score.confidence <= 1
        assert score.uncertainty >= 0
        assert score.method == ConfidenceMethod.ENTROPY
        assert "entropy" in score.metadata
        assert "normalized_entropy" in score.metadata
        assert "max_entropy" in score.metadata

    def test_entropy_increases_with_uncertainty(self):
        """Test that entropy increases with uncertainty."""
        scorer = EntropyScorer()

        # Confident prediction (peaked distribution)
        confident_logits = torch.tensor([[10.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        score_confident = scorer(confident_logits)

        # Uncertain prediction (uniform distribution)
        uncertain_logits = torch.tensor([[1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0]])
        score_uncertain = scorer(uncertain_logits)

        # Uncertain prediction should have higher entropy
        assert score_uncertain.metadata["entropy"] > score_confident.metadata["entropy"]
        # Uncertain prediction should have lower confidence
        assert score_uncertain.confidence < score_confident.confidence

    def test_normalized_entropy_in_valid_range(self, sample_logits):
        """Test that normalized entropy is in [0, 1]."""
        scorer = EntropyScorer()
        score = scorer(sample_logits)

        normalized_entropy = score.metadata["normalized_entropy"]
        assert 0 <= normalized_entropy <= 1

    def test_max_entropy_calculation(self, sample_logits):
        """Test max entropy calculation."""
        scorer = EntropyScorer()
        score = scorer(sample_logits)

        num_classes = 7
        expected_max_entropy = np.log(num_classes)

        assert np.allclose(
            score.metadata["max_entropy"], expected_max_entropy, rtol=1e-5
        )


# ============================================================================
# Test MCDropoutScorer
# ============================================================================


class TestMCDropoutScorer:
    """Test MCDropoutScorer."""

    def test_initialization_default(self, simple_model):
        """Test default initialization."""
        scorer = MCDropoutScorer(simple_model, device="cpu")
        assert scorer.num_samples == 20
        assert scorer.temperature == 1.0

    def test_initialization_custom_parameters(self, simple_model):
        """Test initialization with custom parameters."""
        scorer = MCDropoutScorer(
            simple_model, num_samples=10, temperature=1.5, device="cpu"
        )
        assert scorer.num_samples == 10
        assert scorer.temperature == 1.5

    def test_invalid_parameters_raise_errors(self, simple_model):
        """Test that invalid parameters raise errors."""
        with pytest.raises(ValueError, match="num_samples must be >= 1"):
            MCDropoutScorer(simple_model, num_samples=0, device="cpu")

        with pytest.raises(ValueError, match="Temperature must be positive"):
            MCDropoutScorer(simple_model, temperature=0.0, device="cpu")

    def test_score_single_sample(self, simple_model, sample_input):
        """Test scoring single sample."""
        scorer = MCDropoutScorer(simple_model, num_samples=5, device="cpu")
        score = scorer(sample_input)

        assert isinstance(score, ConfidenceScore)
        assert 0 <= score.confidence <= 1
        assert score.uncertainty >= 0
        assert score.method == ConfidenceMethod.MC_DROPOUT
        assert "variance" in score.metadata
        assert "num_samples" in score.metadata
        assert score.metadata["num_samples"] == 5

    def test_variance_computation(self, simple_model, sample_input):
        """Test that variance is computed correctly."""
        scorer = MCDropoutScorer(simple_model, num_samples=10, device="cpu")
        score = scorer(sample_input)

        # Variance should be non-negative
        assert score.metadata["variance"] >= 0
        assert "normalized_variance" in score.metadata
        assert 0 <= score.metadata["normalized_variance"] <= 1

    def test_multiple_forward_passes(self, simple_model, sample_input):
        """Test that multiple forward passes produce different results."""
        scorer = MCDropoutScorer(simple_model, num_samples=10, device="cpu")
        score = scorer(sample_input)

        # Should have individual predictions stored
        assert "individual_predictions" in score.metadata
        predictions = score.metadata["individual_predictions"]
        assert len(predictions) == 10

        # Predictions should vary (due to dropout)
        predictions_array = np.array(predictions)
        # Check that not all predictions are identical
        variances = np.var(predictions_array, axis=0)
        assert np.any(variances > 1e-6)

    def test_warning_no_dropout(self, sample_input):
        """Test warning when model has no dropout layers."""

        class ModelNoDropout(nn.Module):
            def __init__(self):
                super().__init__()
                self.fc = nn.Linear(3 * 224 * 224, 7)

            def forward(self, x):
                return self.fc(x.flatten(1))

        model = ModelNoDropout()

        with pytest.warns(UserWarning, match="Model has no Dropout layers"):
            scorer = MCDropoutScorer(model, device="cpu")

    def test_batch_warning(self, simple_model, batch_input):
        """Test warning when processing batch."""
        scorer = MCDropoutScorer(simple_model, num_samples=5, device="cpu")

        with pytest.warns(UserWarning, match="Batch size > 1 detected"):
            score = scorer(batch_input)

        # Should still return a valid score for first sample
        assert isinstance(score, ConfidenceScore)


# ============================================================================
# Test TemperatureScaledScorer
# ============================================================================


class TestTemperatureScaledScorer:
    """Test TemperatureScaledScorer."""

    def test_initialization_default(self, simple_model):
        """Test default initialization without temperature module."""
        with pytest.warns(UserWarning, match="No temperature module provided"):
            scorer = TemperatureScaledScorer(simple_model, device="cpu")

        assert scorer.temperature_module is not None

    def test_initialization_with_temperature_module(self, simple_model):
        """Test initialization with custom temperature module."""
        temp_module = TemperatureScaling(init_temperature=2.0)
        scorer = TemperatureScaledScorer(
            simple_model, temperature_module=temp_module, device="cpu"
        )

        assert scorer.temperature_module is temp_module
        assert np.allclose(scorer.temperature_module.get_temperature(), 2.0, atol=0.01)

    def test_score_single_sample(self, simple_model, sample_input):
        """Test scoring single sample."""
        temp_module = TemperatureScaling(init_temperature=1.5)
        scorer = TemperatureScaledScorer(
            simple_model, temperature_module=temp_module, device="cpu"
        )
        score = scorer(sample_input)

        assert isinstance(score, ConfidenceScore)
        assert 0 <= score.confidence <= 1
        assert score.method == ConfidenceMethod.TEMPERATURE_SCALED
        assert "temperature" in score.metadata

    def test_temperature_affects_confidence(self, simple_model, sample_input):
        """Test that different temperatures produce different confidences."""
        temp_module_low = TemperatureScaling(init_temperature=0.5)
        temp_module_high = TemperatureScaling(init_temperature=2.0)

        scorer_low = TemperatureScaledScorer(
            simple_model, temperature_module=temp_module_low, device="cpu"
        )
        scorer_high = TemperatureScaledScorer(
            simple_model, temperature_module=temp_module_high, device="cpu"
        )

        score_low = scorer_low(sample_input)
        score_high = scorer_high(sample_input)

        # Lower temperature should produce higher confidence
        assert score_low.confidence > score_high.confidence

    def test_metadata_contains_fitted_flag(self, simple_model, sample_input):
        """Test that metadata contains fitted flag."""
        temp_module = TemperatureScaling(init_temperature=1.5)
        scorer = TemperatureScaledScorer(
            simple_model, temperature_module=temp_module, device="cpu"
        )
        score = scorer(sample_input)

        assert "is_fitted" in score.metadata
        assert isinstance(score.metadata["is_fitted"], bool)


# ============================================================================
# Test ConfidenceScorer (Unified Interface)
# ============================================================================


class TestConfidenceScorer:
    """Test ConfidenceScorer unified interface."""

    def test_initialization_softmax_max(self, simple_model):
        """Test initialization with softmax_max method."""
        scorer = ConfidenceScorer(
            simple_model, method="softmax_max", temperature=1.0, device="cpu"
        )
        assert scorer.method == ConfidenceMethod.SOFTMAX_MAX

    def test_initialization_entropy(self, simple_model):
        """Test initialization with entropy method."""
        scorer = ConfidenceScorer(
            simple_model, method="entropy", temperature=1.0, device="cpu"
        )
        assert scorer.method == ConfidenceMethod.ENTROPY

    def test_initialization_mc_dropout(self, simple_model):
        """Test initialization with MC dropout method."""
        scorer = ConfidenceScorer(
            simple_model, method="mc_dropout", num_mc_samples=10, device="cpu"
        )
        assert scorer.method == ConfidenceMethod.MC_DROPOUT

    def test_initialization_temperature_scaled(self, simple_model):
        """Test initialization with temperature scaling method."""
        with pytest.warns(UserWarning, match="No temperature module provided"):
            scorer = ConfidenceScorer(
                simple_model, method="temperature_scaled", device="cpu"
            )
        assert scorer.method == ConfidenceMethod.TEMPERATURE_SCALED

    def test_invalid_method_raises_error(self, simple_model):
        """Test that invalid method raises error."""
        with pytest.raises(ValueError, match="Invalid method"):
            ConfidenceScorer(simple_model, method="invalid_method", device="cpu")

    def test_score_with_logits_softmax_max(self, simple_model, sample_logits):
        """Test scoring with logits for softmax_max."""
        scorer = ConfidenceScorer(simple_model, method="softmax_max", device="cpu")
        score = scorer(sample_logits)

        assert isinstance(score, ConfidenceScore)
        assert score.method == ConfidenceMethod.SOFTMAX_MAX

    def test_score_with_logits_entropy(self, simple_model, sample_logits):
        """Test scoring with logits for entropy."""
        scorer = ConfidenceScorer(simple_model, method="entropy", device="cpu")
        score = scorer(sample_logits)

        assert isinstance(score, ConfidenceScore)
        assert score.method == ConfidenceMethod.ENTROPY

    def test_score_with_input_tensor(self, simple_model, sample_input):
        """Test scoring with input tensor."""
        scorer = ConfidenceScorer(simple_model, method="softmax_max", device="cpu")
        score = scorer(sample_input)

        assert isinstance(score, ConfidenceScore)

    def test_batch_score(self, simple_model, batch_input):
        """Test batch scoring."""
        scorer = ConfidenceScorer(simple_model, method="softmax_max", device="cpu")
        scores = scorer.batch_score(batch_input, batch_size=2)

        assert len(scores) == 4
        assert all(isinstance(s, ConfidenceScore) for s in scores)

    def test_enum_method_initialization(self, simple_model):
        """Test initialization with ConfidenceMethod enum."""
        scorer = ConfidenceScorer(
            simple_model, method=ConfidenceMethod.SOFTMAX_MAX, device="cpu"
        )
        assert scorer.method == ConfidenceMethod.SOFTMAX_MAX


# ============================================================================
# Test compute_confidence_metrics
# ============================================================================


class TestComputeConfidenceMetrics:
    """Test compute_confidence_metrics function."""

    def test_metrics_without_labels(self, simple_model, batch_input):
        """Test metrics computation without labels."""
        scorer = ConfidenceScorer(simple_model, method="softmax_max", device="cpu")
        scores = scorer.batch_score(batch_input)

        metrics = compute_confidence_metrics(scores)

        assert "mean_confidence" in metrics
        assert "std_confidence" in metrics
        assert "mean_uncertainty" in metrics
        assert "method" in metrics
        assert "accuracy" not in metrics

    def test_metrics_with_labels(self, simple_model, batch_input):
        """Test metrics computation with labels."""
        scorer = ConfidenceScorer(simple_model, method="softmax_max", device="cpu")
        scores = scorer.batch_score(batch_input)

        labels = np.array([0, 1, 2, 0])  # Mock labels

        metrics = compute_confidence_metrics(scores, labels=labels)

        assert "accuracy" in metrics
        assert "correct_confidence" in metrics
        assert "incorrect_confidence" in metrics
        assert 0 <= metrics["accuracy"] <= 1

    def test_metrics_values_in_valid_range(self, simple_model, batch_input):
        """Test that metric values are in valid ranges."""
        scorer = ConfidenceScorer(simple_model, method="entropy", device="cpu")
        scores = scorer.batch_score(batch_input)

        metrics = compute_confidence_metrics(scores)

        assert 0 <= metrics["mean_confidence"] <= 1
        assert metrics["std_confidence"] >= 0
        assert metrics["mean_uncertainty"] >= 0

    def test_method_recorded_correctly(self, simple_model, batch_input):
        """Test that method is recorded correctly."""
        scorer = ConfidenceScorer(simple_model, method="entropy", device="cpu")
        scores = scorer.batch_score(batch_input)

        metrics = compute_confidence_metrics(scores)

        assert metrics["method"] == "entropy"


# ============================================================================
# Integration Tests
# ============================================================================


class TestIntegration:
    """Test integration scenarios."""

    def test_all_methods_produce_consistent_predictions(
        self, simple_model, sample_input
    ):
        """Test that all methods produce the same predictions."""
        methods = ["softmax_max", "entropy", "mc_dropout", "temperature_scaled"]
        predictions = []

        for method in methods:
            if method == "temperature_scaled":
                with pytest.warns(UserWarning):
                    scorer = ConfidenceScorer(simple_model, method=method, device="cpu")
            else:
                scorer = ConfidenceScorer(
                    simple_model, method=method, num_mc_samples=5, device="cpu"
                )

            score = scorer(sample_input)
            predictions.append(score.prediction)

        # All methods should agree on prediction (they use same model)
        assert len(set(predictions)) == 1

    def test_mc_dropout_vs_softmax_uncertainty(self, simple_model, sample_input):
        """Test that MC Dropout captures more uncertainty than softmax."""
        scorer_softmax = ConfidenceScorer(
            simple_model, method="softmax_max", device="cpu"
        )
        scorer_mc = ConfidenceScorer(
            simple_model, method="mc_dropout", num_mc_samples=10, device="cpu"
        )

        score_softmax = scorer_softmax(sample_input)
        score_mc = scorer_mc(sample_input)

        # MC Dropout should provide variance information
        assert "variance" in score_mc.metadata
        assert "variance" not in score_softmax.metadata

    def test_temperature_scaling_calibration_effect(self, simple_model, sample_input):
        """Test that temperature scaling affects confidence calibration."""
        # Create two temperature modules with different temperatures
        temp_low = TemperatureScaling(init_temperature=0.5)
        temp_high = TemperatureScaling(init_temperature=2.0)

        scorer_low = ConfidenceScorer(
            simple_model,
            method="temperature_scaled",
            temperature_module=temp_low,
            device="cpu",
        )
        scorer_high = ConfidenceScorer(
            simple_model,
            method="temperature_scaled",
            temperature_module=temp_high,
            device="cpu",
        )

        score_low = scorer_low(sample_input)
        score_high = scorer_high(sample_input)

        # Lower temperature should produce higher confidence
        assert score_low.confidence > score_high.confidence

    def test_entropy_vs_softmax_max_comparison(self, sample_logits):
        """Test that entropy and softmax max provide complementary information."""
        scorer_softmax = SoftmaxMaxScorer(temperature=1.0)
        scorer_entropy = EntropyScorer(temperature=1.0)

        score_softmax = scorer_softmax(sample_logits)
        score_entropy = scorer_entropy(sample_logits)

        # Both should agree on prediction
        assert score_softmax.prediction == score_entropy.prediction

        # Entropy provides additional uncertainty information
        assert "entropy" in score_entropy.metadata
        assert "normalized_entropy" in score_entropy.metadata


# ============================================================================
# Edge Cases and Error Handling
# ============================================================================


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_uniform_distribution_max_entropy(self):
        """Test that uniform distribution produces maximum entropy."""
        scorer = EntropyScorer()

        # Uniform logits
        uniform_logits = torch.ones(1, 7)
        score = scorer(uniform_logits)

        # Should have normalized entropy close to 1
        assert score.metadata["normalized_entropy"] > 0.99

    def test_deterministic_distribution_min_entropy(self):
        """Test that deterministic distribution produces minimum entropy."""
        scorer = EntropyScorer()

        # Very peaked distribution
        peaked_logits = torch.tensor([[100.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        score = scorer(peaked_logits)

        # Should have very low entropy
        assert score.metadata["entropy"] < 0.01

    def test_single_class_edge_case(self):
        """Test edge case with single class."""
        scorer = SoftmaxMaxScorer()

        # Single class logits
        single_class_logits = torch.tensor([[5.0]])
        score = scorer(single_class_logits)

        # Should have confidence 1.0 (only one choice)
        assert np.allclose(score.confidence, 1.0, atol=1e-6)

    def test_numerical_stability_extreme_logits(self):
        """Test numerical stability with extreme logit values."""
        scorer_softmax = SoftmaxMaxScorer()
        scorer_entropy = EntropyScorer()

        # Very large logits
        large_logits = torch.tensor([[1000.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])

        score_softmax = scorer_softmax(large_logits)
        score_entropy = scorer_entropy(large_logits)

        # Should not produce NaN or Inf
        assert not np.isnan(score_softmax.confidence)
        assert not np.isinf(score_softmax.confidence)
        assert not np.isnan(score_entropy.metadata["entropy"])
        assert not np.isinf(score_entropy.metadata["entropy"])

    def test_empty_batch_handling(self, simple_model):
        """Test handling of empty batches."""
        scorer = ConfidenceScorer(simple_model, method="softmax_max", device="cpu")

        # Empty input
        empty_input = torch.zeros(0, 3, 224, 224)

        with pytest.raises(Exception):
            # Should raise an error or handle gracefully
            scorer.batch_score(empty_input)
