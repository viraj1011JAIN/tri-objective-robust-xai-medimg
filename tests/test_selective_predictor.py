"""
Property-Based Tests for Selective Predictor (Phase 8.3)

This module implements comprehensive property-based testing using Hypothesis
to ensure correctness of the selective predictor under all conditions.

Property-based testing generates hundreds of random test cases to verify
that invariants hold across the entire input space, not just handcrafted examples.

Test Categories
---------------
1. **Gating Logic Properties**: Monotonicity, boundary conditions
2. **Statistical Properties**: Coverage bounds, consistency
3. **Numerical Stability**: NaN/Inf handling, floating point edge cases
4. **Performance Properties**: Cascading optimization correctness
5. **Integration Properties**: Scorer compatibility

Author: Viraj Pankaj Jain
Version: 8.3.1
"""

from unittest.mock import MagicMock, Mock

import numpy as np
import pytest
import torch
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

from src.selection import (
    GatingStrategy,
    SelectionResult,
    SelectivePredictor,
    SelectivePredictorConfig,
)

# ============================================================================
# TEST HELPERS (Not fixtures - for use with Hypothesis)
# ============================================================================


def create_mock_confidence_scorer():
    """Create mock confidence scorer for testing."""
    scorer = Mock()
    scorer.model = Mock()
    scorer.compute_confidence = Mock()
    return scorer


def create_mock_stability_scorer():
    """Create mock stability scorer for testing."""
    scorer = Mock()
    scorer.compute_single_stability = Mock()
    return scorer


def create_predictor(
    confidence_threshold: float = 0.85,
    stability_threshold: float = 0.75,
    strategy: GatingStrategy = GatingStrategy.COMBINED,
    enable_cascading: bool = False,
):
    """Helper to create a predictor for testing."""
    return SelectivePredictor(
        confidence_scorer=create_mock_confidence_scorer(),
        stability_scorer=create_mock_stability_scorer(),
        confidence_threshold=confidence_threshold,
        stability_threshold=stability_threshold,
        strategy=strategy,
        device="cpu",
        verbose=False,
        enable_cascading=enable_cascading,
    )


# ============================================================================
# PROPERTY 1: GATING LOGIC MONOTONICITY
# ============================================================================


class TestGatingLogicProperties:
    """Test properties of gating logic."""

    @given(
        confidence=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        stability=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        conf_threshold=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        stab_threshold=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(
        max_examples=500,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_combined_gating_monotonicity(
        self, confidence, stability, conf_threshold, stab_threshold
    ):
        """
        Property: Combined gating is monotonic in both confidence and stability.

        If both thresholds are met, prediction must be accepted.
        If either threshold is not met, prediction must be rejected.
        """
        predictor = create_predictor(
            confidence_threshold=conf_threshold,
            stability_threshold=stab_threshold,
            strategy=GatingStrategy.COMBINED,
            enable_cascading=False,  # Disable for pure logic testing
        )

        is_accepted, reason, strategy = predictor._apply_gating_logic(
            confidence, stability
        )

        # Property 1: If both thresholds met, must accept
        if confidence >= conf_threshold and stability >= stab_threshold:
            assert is_accepted, (
                f"Should accept when both thresholds met: "
                f"conf={confidence:.3f} >= {conf_threshold:.3f}, "
                f"stab={stability:.3f} >= {stab_threshold:.3f}"
            )
            assert reason is None

        # Property 2: If either threshold not met, must reject
        if confidence < conf_threshold or stability < stab_threshold:
            assert not is_accepted, (
                f"Should reject when threshold not met: "
                f"conf={confidence:.3f} < {conf_threshold:.3f} OR "
                f"stab={stability:.3f} < {stab_threshold:.3f}"
            )
            assert reason is not None

    @given(
        confidence=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        stability=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        threshold=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(
        max_examples=300,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_confidence_only_strategy(self, confidence, stability, threshold):
        """
        Property: Confidence-only strategy ignores stability.

        Decision should depend only on confidence vs threshold.
        """
        predictor = create_predictor(
            confidence_threshold=threshold,
            stability_threshold=0.5,  # Should be ignored
            strategy=GatingStrategy.CONFIDENCE_ONLY,
            enable_cascading=False,
        )

        is_accepted, reason, strategy_used = predictor._apply_gating_logic(
            confidence, stability
        )

        if confidence >= threshold:
            assert is_accepted
            assert reason is None
        else:
            assert not is_accepted
            assert reason == "low_confidence"

    @given(
        confidence=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        stability=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        threshold=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(
        max_examples=300,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_stability_only_strategy(self, confidence, stability, threshold):
        """
        Property: Stability-only strategy ignores confidence.

        Decision should depend only on stability vs threshold.
        """
        predictor = create_predictor(
            confidence_threshold=0.5,  # Should be ignored
            stability_threshold=threshold,
            strategy=GatingStrategy.STABILITY_ONLY,
            enable_cascading=False,
        )

        is_accepted, reason, strategy_used = predictor._apply_gating_logic(
            confidence, stability
        )

        if stability >= threshold:
            assert is_accepted
            assert reason is None
        else:
            assert not is_accepted
            assert reason == "low_stability"


# ============================================================================
# PROPERTY 2: NUMERICAL STABILITY
# ============================================================================


class TestNumericalStability:
    """Test handling of edge cases and numerical issues."""

    @given(
        confidence=st.one_of(
            st.just(float("nan")), st.just(float("inf")), st.just(float("-inf"))
        ),
        stability=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_nan_inf_confidence_handling(self, confidence, stability):
        """
        Property: NaN/Inf confidence values should be rejected with 'invalid_scores'.
        """
        predictor = create_predictor(
            confidence_threshold=0.85,
            stability_threshold=0.75,
            strategy=GatingStrategy.COMBINED,
            enable_cascading=False,
        )

        is_accepted, reason, strategy = predictor._apply_gating_logic(
            confidence, stability
        )

        assert not is_accepted, "NaN/Inf confidence must be rejected"
        assert reason == "invalid_scores"

    @given(
        confidence=st.floats(min_value=0.0, max_value=1.0),
        stability=st.one_of(
            st.just(float("nan")), st.just(float("inf")), st.just(float("-inf"))
        ),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_nan_inf_stability_handling(self, confidence, stability):
        """
        Property: NaN/Inf stability values should be rejected with 'invalid_scores'.
        """
        predictor = create_predictor(
            confidence_threshold=0.85,
            stability_threshold=0.75,
            strategy=GatingStrategy.COMBINED,
            enable_cascading=False,
        )

        is_accepted, reason, strategy = predictor._apply_gating_logic(
            confidence, stability
        )

        assert not is_accepted, "NaN/Inf stability must be rejected"
        assert reason == "invalid_scores"

    @given(
        confidence=st.floats(
            min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
        stability=st.floats(
            min_value=-10.0, max_value=10.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_out_of_range_clipping(self, confidence, stability):
        """
        Property: Out-of-range values should be clipped to [0, 1].

        Values < 0 should be treated as 0.
        Values > 1 should be treated as 1.
        """
        predictor = create_predictor(
            confidence_threshold=0.5,
            stability_threshold=0.5,
            strategy=GatingStrategy.COMBINED,
            enable_cascading=False,
        )

        is_accepted, reason, strategy = predictor._apply_gating_logic(
            confidence, stability
        )

        # After clipping, values should behave as if in [0, 1]
        clipped_conf = np.clip(confidence, 0.0, 1.0)
        clipped_stab = np.clip(stability, 0.0, 1.0)

        expected_accept = clipped_conf >= 0.5 and clipped_stab >= 0.5

        assert is_accepted == expected_accept, (
            f"Clipping failed: conf={confidence:.3f}->{clipped_conf:.3f}, "
            f"stab={stability:.3f}->{clipped_stab:.3f}, "
            f"expected={expected_accept}, got={is_accepted}"
        )


# ============================================================================
# PROPERTY 3: STATISTICAL CONSISTENCY
# ============================================================================


class TestStatisticalProperties:
    """Test statistical properties of predictions."""

    def test_coverage_bounds(self):
        """
        Property: Coverage must always be in [0, 1].
        """
        predictor = create_predictor()
        # Make some mock predictions
        predictor.total_predictions = 100
        predictor.total_accepted = 75
        predictor.total_rejected = 25

        coverage = predictor.coverage

        assert 0.0 <= coverage <= 1.0, f"Invalid coverage: {coverage}"
        assert coverage == 0.75

    def test_accepted_rejected_sum(self):
        """
        Property: Accepted + Rejected = Total.
        """
        predictor = create_predictor()
        predictor.total_predictions = 150
        predictor.total_accepted = 90
        predictor.total_rejected = 60

        assert (
            predictor.total_accepted + predictor.total_rejected
            == predictor.total_predictions
        )

    @given(
        num_predictions=st.integers(min_value=0, max_value=1000),
        acceptance_rate=st.floats(min_value=0.0, max_value=1.0),
    )
    @settings(
        max_examples=200,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_statistics_consistency(self, num_predictions, acceptance_rate):
        """
        Property: Statistics must be internally consistent.
        """
        predictor = create_predictor()
        num_accepted = int(num_predictions * acceptance_rate)
        num_rejected = num_predictions - num_accepted

        predictor.total_predictions = num_predictions
        predictor.total_accepted = num_accepted
        predictor.total_rejected = num_rejected

        # Coverage calculation
        if num_predictions == 0:
            assert predictor.coverage == 0.0
        else:
            expected_coverage = num_accepted / num_predictions
            assert abs(predictor.coverage - expected_coverage) < 1e-9

        # Sum consistency
        assert (
            predictor.total_accepted + predictor.total_rejected
            == predictor.total_predictions
        )


# ============================================================================
# PROPERTY 4: CASCADING OPTIMIZATION
# ============================================================================


class TestCascadingProperties:
    """Test cascading gate optimization."""

    @given(
        confidence=st.floats(
            min_value=0.98, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
        stability=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_fast_accept_path(self, confidence, stability):
        """
        Property: Ultra-confident samples should use fast accept path.

        When confidence >= fast_accept_threshold, should accept regardless of stability.
        """
        predictor = create_predictor(
            confidence_threshold=0.85,
            stability_threshold=0.75,
            strategy=GatingStrategy.COMBINED,
            enable_cascading=True,
        )
        predictor.fast_accept_threshold = 0.98
        predictor.fast_reject_threshold = 0.50

        is_accepted, reason, strategy = predictor._apply_gating_logic(
            confidence, stability
        )

        assert is_accepted, f"Should fast accept for conf={confidence:.3f} >= 0.98"
        assert strategy == "FAST_ACCEPT"
        assert reason is None

    @given(
        confidence=st.floats(
            min_value=0.0, max_value=0.50, allow_nan=False, allow_infinity=False
        ),
        stability=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_fast_reject_path(self, confidence, stability):
        """
        Property: Very uncertain samples should use fast reject path.

        When confidence <= fast_reject_threshold, should reject regardless of stability.
        """
        predictor = create_predictor(
            confidence_threshold=0.85,
            stability_threshold=0.75,
            strategy=GatingStrategy.COMBINED,
            enable_cascading=True,
        )
        predictor.fast_accept_threshold = 0.98
        predictor.fast_reject_threshold = 0.50

        is_accepted, reason, strategy = predictor._apply_gating_logic(
            confidence, stability
        )

        assert not is_accepted, f"Should fast reject for conf={confidence:.3f} <= 0.50"
        assert strategy == "FAST_REJECT"
        assert reason == "low_confidence"

    @given(
        confidence=st.floats(
            min_value=0.51, max_value=0.97, allow_nan=False, allow_infinity=False
        ),
        stability=st.floats(
            min_value=0.0, max_value=1.0, allow_nan=False, allow_infinity=False
        ),
    )
    @settings(
        max_examples=100,
        deadline=None,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    def test_grey_zone_robust_path(self, confidence, stability):
        """
        Property: Samples in grey zone should use robust gating.

        When fast_reject < confidence < fast_accept, should check both signals.
        """
        predictor = create_predictor(
            confidence_threshold=0.85,
            stability_threshold=0.75,
            strategy=GatingStrategy.COMBINED,
            enable_cascading=True,
        )
        predictor.fast_accept_threshold = 0.98
        predictor.fast_reject_threshold = 0.50

        is_accepted, reason, strategy = predictor._apply_gating_logic(
            confidence, stability
        )

        # Should use robust path (either ROBUST_ACCEPT or ROBUST_REJECT)
        assert strategy in ["ROBUST_ACCEPT", "ROBUST_REJECT"]

        # Check correctness of robust gating
        if confidence >= 0.85 and stability >= 0.75:
            assert is_accepted
            assert strategy == "ROBUST_ACCEPT"
        else:
            assert not is_accepted
            assert strategy == "ROBUST_REJECT"


# ============================================================================
# PROPERTY 5: CONFIGURATION VALIDATION
# ============================================================================


class TestConfigurationValidation:
    """Test Pydantic configuration validation."""

    def test_valid_thresholds(self):
        """Property: Valid thresholds should be accepted."""
        try:
            from src.selection import SelectivePredictorConfig

            config = SelectivePredictorConfig(
                confidence_threshold=0.85,
                stability_threshold=0.75,
                fast_accept_threshold=0.98,
                fast_reject_threshold=0.50,
            )

            assert config.confidence_threshold == 0.85
            assert config.stability_threshold == 0.75
        except ImportError:
            pytest.skip("Pydantic not available")

    def test_invalid_threshold_range(self):
        """Property: Thresholds outside [0, 1] should be rejected."""
        try:
            from pydantic import ValidationError

            from src.selection import SelectivePredictorConfig

            with pytest.raises(ValidationError):
                SelectivePredictorConfig(confidence_threshold=1.5)

            with pytest.raises(ValidationError):
                SelectivePredictorConfig(stability_threshold=-0.1)
        except ImportError:
            pytest.skip("Pydantic not available")

    def test_cascading_threshold_ordering(self):
        """Property: Cascading thresholds must satisfy ordering constraint."""
        try:
            from pydantic import ValidationError

            from src.selection import SelectivePredictorConfig

            # Should fail: fast_accept <= confidence_threshold
            with pytest.raises(ValidationError):
                SelectivePredictorConfig(
                    confidence_threshold=0.85,
                    fast_accept_threshold=0.80,  # Should be > confidence_threshold
                )

            # Should fail: fast_reject >= confidence_threshold
            with pytest.raises(ValidationError):
                SelectivePredictorConfig(
                    confidence_threshold=0.85,
                    fast_reject_threshold=0.90,  # Should be < confidence_threshold
                )
        except ImportError:
            pytest.skip("Pydantic not available")


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegrationProperties:
    """Test integration with other components."""

    def test_selection_result_serialization(self):
        """Property: SelectionResult should be serializable to dict."""
        result = SelectionResult(
            prediction=1,
            confidence=0.92,
            stability=0.81,
            is_accepted=True,
            rejection_reason=None,
            decision_strategy="ROBUST_ACCEPT",
            true_label=1,
            sample_id="test_001",
        )

        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["prediction"] == 1
        assert result_dict["confidence"] == 0.92
        assert result_dict["stability"] == 0.81
        assert result_dict["is_accepted"] is True
        assert result_dict["decision_strategy"] == "ROBUST_ACCEPT"

    def test_statistics_reset(self):
        """Property: Statistics should be zeroed after reset."""
        predictor = create_predictor()
        # Set some statistics
        predictor.total_predictions = 100
        predictor.total_accepted = 75
        predictor.total_rejected = 25
        predictor.fast_accepts = 10
        predictor.fast_rejects = 5

        # Reset
        predictor.reset_statistics()

        # All should be zero
        assert predictor.total_predictions == 0
        assert predictor.total_accepted == 0
        assert predictor.total_rejected == 0
        assert predictor.fast_accepts == 0
        assert predictor.fast_rejects == 0
        assert predictor.coverage == 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
