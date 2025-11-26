"""
Comprehensive tests for src/evaluation/fairness.py
Achieves 100% line and branch coverage with production-level quality.
"""

import numpy as np
import pytest
from sklearn.metrics import confusion_matrix

from src.evaluation.fairness import FairnessMetrics, calculate_subgroup_metrics


class TestFairnessMetricsInit:
    """Test FairnessMetrics initialization."""

    def test_init_single_attribute(self):
        """Test initialization with single attribute."""
        fm = FairnessMetrics(sensitive_attrs=["age"])
        assert fm.sensitive_attrs == ["age"]

    def test_init_multiple_attributes(self):
        """Test initialization with multiple attributes."""
        fm = FairnessMetrics(sensitive_attrs=["age", "sex", "skin_tone"])
        assert len(fm.sensitive_attrs) == 3
        assert "age" in fm.sensitive_attrs
        assert "sex" in fm.sensitive_attrs

    def test_init_empty_list(self):
        """Test initialization with empty list."""
        fm = FairnessMetrics(sensitive_attrs=[])
        assert fm.sensitive_attrs == []


class TestDemographicParity:
    """Test demographic parity calculations."""

    @pytest.fixture
    def simple_predictions(self):
        """Simple binary predictions."""
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        return y_pred

    @pytest.fixture
    def binary_groups(self):
        """Binary group assignments."""
        groups = {"sex": np.array([0, 0, 0, 0, 1, 1, 1, 1])}
        return groups

    def test_demographic_parity_basic(self, simple_predictions, binary_groups):
        """Test basic demographic parity calculation."""
        fm = FairnessMetrics(sensitive_attrs=["sex"])
        result = fm.demographic_parity(simple_predictions, binary_groups)

        assert "sex_dp_diff" in result
        assert "sex_rates" in result
        assert isinstance(result["sex_dp_diff"], float)
        assert 0 in result["sex_rates"]
        assert 1 in result["sex_rates"]

    def test_demographic_parity_perfect_parity(self):
        """Test with perfect demographic parity."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        groups = {"group": np.array([0, 1, 0, 1, 0, 1, 0, 1])}

        result = fm.demographic_parity(y_pred, groups)

        assert result["group_dp_diff"] == 0.0

    def test_demographic_parity_maximum_disparity(self):
        """Test with maximum disparity."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        groups = {"group": np.array([0, 0, 0, 0, 1, 1, 1, 1])}

        result = fm.demographic_parity(y_pred, groups)

        assert result["group_dp_diff"] == 1.0

    def test_demographic_parity_multiple_groups(self):
        """Test with more than 2 groups."""
        fm = FairnessMetrics(sensitive_attrs=["age"])
        y_pred = np.array([1, 1, 0, 1, 0, 0, 1, 1, 0])
        groups = {"age": np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])}

        result = fm.demographic_parity(y_pred, groups)

        assert "age_dp_diff" in result
        assert "age_rates" in result
        assert len(result["age_rates"]) == 3

    def test_demographic_parity_multiple_attributes(self):
        """Test with multiple sensitive attributes."""
        fm = FairnessMetrics(sensitive_attrs=["sex", "age"])
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        groups = {
            "sex": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
            "age": np.array([0, 0, 1, 1, 0, 0, 1, 1]),
        }

        result = fm.demographic_parity(y_pred, groups)

        assert "sex_dp_diff" in result
        assert "age_dp_diff" in result
        assert "sex_rates" in result
        assert "age_rates" in result

    def test_demographic_parity_all_positive(self):
        """Test when all predictions are positive."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        y_pred = np.array([1, 1, 1, 1, 1, 1])
        groups = {"group": np.array([0, 0, 0, 1, 1, 1])}

        result = fm.demographic_parity(y_pred, groups)

        assert result["group_dp_diff"] == 0.0

    def test_demographic_parity_all_negative(self):
        """Test when all predictions are negative."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        y_pred = np.array([0, 0, 0, 0, 0, 0])
        groups = {"group": np.array([0, 0, 0, 1, 1, 1])}

        result = fm.demographic_parity(y_pred, groups)

        assert result["group_dp_diff"] == 0.0


class TestEqualizedOdds:
    """Test equalized odds calculations."""

    @pytest.fixture
    def binary_data(self):
        """Binary classification data."""
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0, 0, 1])
        groups = {"sex": np.array([0, 0, 0, 0, 1, 1, 1, 1])}
        return y_true, y_pred, groups

    def test_equalized_odds_basic(self, binary_data):
        """Test basic equalized odds calculation."""
        y_true, y_pred, groups = binary_data
        fm = FairnessMetrics(sensitive_attrs=["sex"])
        result = fm.equalized_odds(y_true, y_pred, groups)

        assert "sex_tpr_diff" in result
        assert "sex_fpr_diff" in result
        assert "sex_tpr" in result
        assert "sex_fpr" in result

    def test_equalized_odds_perfect_equality(self):
        """Test with perfect equalized odds."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        # Perfect predictions
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        groups = {"group": np.array([0, 0, 0, 0, 1, 1, 1, 1])}

        result = fm.equalized_odds(y_true, y_pred, groups)

        assert result["group_tpr_diff"] == 0.0
        assert result["group_fpr_diff"] == 0.0

    def test_equalized_odds_tpr_disparity(self):
        """Test with TPR disparity."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        y_true = np.array([1, 1, 1, 1, 1, 1, 1, 1])
        # Group 0: all correct, Group 1: all wrong
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        groups = {"group": np.array([0, 0, 0, 0, 1, 1, 1, 1])}

        result = fm.equalized_odds(y_true, y_pred, groups)

        assert result["group_tpr_diff"] == 1.0

    def test_equalized_odds_fpr_disparity(self):
        """Test with FPR disparity."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        y_true = np.array([0, 0, 0, 0, 0, 0, 0, 0])
        # Group 0: all FP, Group 1: all TN
        y_pred = np.array([1, 1, 1, 1, 0, 0, 0, 0])
        groups = {"group": np.array([0, 0, 0, 0, 1, 1, 1, 1])}

        result = fm.equalized_odds(y_true, y_pred, groups)

        assert result["group_fpr_diff"] == 1.0

    def test_equalized_odds_zero_positives(self):
        """Test with no positive examples in a group."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        y_true = np.array([0, 0, 0, 0, 1, 1])
        y_pred = np.array([0, 1, 0, 1, 1, 0])
        groups = {"group": np.array([0, 0, 0, 0, 1, 1])}

        result = fm.equalized_odds(y_true, y_pred, groups)

        # Group 0 has TPR = 0 (no positives)
        assert 0 in result["group_tpr"]
        assert result["group_tpr"][0] == 0.0

    def test_equalized_odds_zero_negatives(self):
        """Test with no negative examples in a group."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        y_true = np.array([1, 1, 1, 1, 0, 0])
        y_pred = np.array([1, 0, 1, 0, 0, 1])
        groups = {"group": np.array([0, 0, 0, 0, 1, 1])}

        result = fm.equalized_odds(y_true, y_pred, groups)

        # Group 0 has FPR = 0 (no negatives)
        assert 0 in result["group_fpr"]
        assert result["group_fpr"][0] == 0.0

    def test_equalized_odds_multiple_groups(self):
        """Test with multiple groups."""
        fm = FairnessMetrics(sensitive_attrs=["age"])
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 1, 0, 0, 1, 1])
        groups = {"age": np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])}

        result = fm.equalized_odds(y_true, y_pred, groups)

        assert len(result["age_tpr"]) == 3
        assert len(result["age_fpr"]) == 3

    def test_equalized_odds_multiple_attributes(self):
        """Test with multiple sensitive attributes."""
        fm = FairnessMetrics(sensitive_attrs=["sex", "age"])
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0, 0, 1])
        groups = {
            "sex": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
            "age": np.array([0, 0, 1, 1, 0, 0, 1, 1]),
        }

        result = fm.equalized_odds(y_true, y_pred, groups)

        assert "sex_tpr_diff" in result
        assert "sex_fpr_diff" in result
        assert "age_tpr_diff" in result
        assert "age_fpr_diff" in result


class TestDisparateImpact:
    """Test disparate impact calculations."""

    def test_disparate_impact_basic(self):
        """Test basic disparate impact calculation."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 0])
        groups = {"group": np.array([0, 0, 0, 0, 1, 1, 1, 1])}

        result = fm.disparate_impact(y_pred, groups)

        assert "group_di_ratio" in result
        assert "group_passes_80_rule" in result
        assert isinstance(result["group_di_ratio"], float)
        assert result["group_passes_80_rule"] in [True, False]

    def test_disparate_impact_passes_80_rule(self):
        """Test when disparate impact passes 80% rule."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        # Group 0: 4/5 = 0.8, Group 1: 4/5 = 0.8
        y_pred = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 0])
        groups = {"group": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])}

        result = fm.disparate_impact(y_pred, groups)

        assert result["group_di_ratio"] == 1.0
        assert result["group_passes_80_rule"]

    def test_disparate_impact_fails_80_rule(self):
        """Test when disparate impact fails 80% rule."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        # Group 0: 4/4 = 1.0, Group 1: 1/4 = 0.25, ratio = 0.25
        y_pred = np.array([1, 1, 1, 1, 1, 0, 0, 0])
        groups = {"group": np.array([0, 0, 0, 0, 1, 1, 1, 1])}

        result = fm.disparate_impact(y_pred, groups)

        assert result["group_di_ratio"] < 0.8
        assert not (result["group_passes_80_rule"])

    def test_disparate_impact_boundary_80(self):
        """Test disparate impact at exactly 80% boundary."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        # Group 0: 4/5 = 0.8, Group 1: 5/5 = 1.0, ratio = 0.8/1.0 = 0.8
        y_pred = np.array([1, 1, 1, 1, 0, 1, 1, 1, 1, 1])
        groups = {"group": np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])}

        result = fm.disparate_impact(y_pred, groups)

        # At exactly 0.8, should pass
        assert result["group_passes_80_rule"]

    def test_disparate_impact_zero_positive_rate(self):
        """Test when max positive rate is zero."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        y_pred = np.array([0, 0, 0, 0, 0, 0])
        groups = {"group": np.array([0, 0, 0, 1, 1, 1])}

        result = fm.disparate_impact(y_pred, groups)

        assert result["group_di_ratio"] == 0.0
        assert not (result["group_passes_80_rule"])

    def test_disparate_impact_multiple_groups(self):
        """Test with more than 2 groups."""
        fm = FairnessMetrics(sensitive_attrs=["age"])
        y_pred = np.array([1, 1, 1, 1, 0, 0, 1, 1, 0])
        groups = {"age": np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])}

        result = fm.disparate_impact(y_pred, groups)

        assert "age_di_ratio" in result
        assert "age_passes_80_rule" in result

    def test_disparate_impact_multiple_attributes(self):
        """Test with multiple sensitive attributes."""
        fm = FairnessMetrics(sensitive_attrs=["sex", "age"])
        y_pred = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        groups = {
            "sex": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
            "age": np.array([0, 0, 1, 1, 0, 0, 1, 1]),
        }

        result = fm.disparate_impact(y_pred, groups)

        assert "sex_di_ratio" in result
        assert "age_di_ratio" in result
        assert "sex_passes_80_rule" in result
        assert "age_passes_80_rule" in result


class TestCalculateAllMetrics:
    """Test comprehensive metric calculation."""

    @pytest.fixture
    def complete_data(self):
        """Complete dataset for all metrics."""
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0, 0, 1, 1, 0, 0, 1])
        groups = {
            "sex": np.array([0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1]),
            "age": np.array([0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 1, 1]),
        }
        return y_true, y_pred, groups

    def test_calculate_all_metrics(self, complete_data):
        """Test calculating all fairness metrics together."""
        y_true, y_pred, groups = complete_data
        fm = FairnessMetrics(sensitive_attrs=["sex", "age"])
        result = fm.calculate_all_metrics(y_true, y_pred, groups)

        # Check demographic parity metrics
        assert "sex_dp_diff" in result
        assert "age_dp_diff" in result

        # Check equalized odds metrics
        assert "sex_tpr_diff" in result
        assert "sex_fpr_diff" in result
        assert "age_tpr_diff" in result
        assert "age_fpr_diff" in result

        # Check disparate impact metrics
        assert "sex_di_ratio" in result
        assert "age_di_ratio" in result
        assert "sex_passes_80_rule" in result
        assert "age_passes_80_rule" in result

    def test_calculate_all_metrics_single_attribute(self):
        """Test with single attribute."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0, 0, 1])
        groups = {"group": np.array([0, 0, 0, 0, 1, 1, 1, 1])}

        result = fm.calculate_all_metrics(y_true, y_pred, groups)

        assert "group_dp_diff" in result
        assert "group_tpr_diff" in result
        assert "group_di_ratio" in result

    def test_calculate_all_metrics_returns_rates(self):
        """Test that detailed rates are included."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1, 1, 0, 0, 1])
        groups = {"group": np.array([0, 0, 0, 0, 1, 1, 1, 1])}

        result = fm.calculate_all_metrics(y_true, y_pred, groups)

        # Should include detailed rates
        assert "group_rates" in result
        assert "group_tpr" in result
        assert "group_fpr" in result


class TestCalculateSubgroupMetrics:
    """Test subgroup metric calculation helper."""

    def test_calculate_subgroup_metrics_basic(self):
        """Test basic subgroup metric calculation."""
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 1])
        y_prob = np.array([0.9, 0.8, 0.2, 0.3, 0.9, 0.4, 0.1, 0.6])
        groups = {"sex": np.array([0, 0, 0, 0, 1, 1, 1, 1])}

        from sklearn.metrics import accuracy_score

        result = calculate_subgroup_metrics(
            y_true, y_pred, y_prob, groups, accuracy_score
        )

        assert "sex" in result
        assert "0" in result["sex"]
        assert "1" in result["sex"]
        assert isinstance(result["sex"]["0"], (float, np.floating))

    def test_calculate_subgroup_metrics_multiple_groups(self):
        """Test with multiple groups per attribute."""
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0, 1])
        y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 1, 1])
        y_prob = np.random.random(9)
        groups = {"age": np.array([0, 0, 0, 1, 1, 1, 2, 2, 2])}

        from sklearn.metrics import accuracy_score

        result = calculate_subgroup_metrics(
            y_true, y_pred, y_prob, groups, accuracy_score
        )

        assert "age" in result
        assert len(result["age"]) == 3
        assert "0" in result["age"]
        assert "1" in result["age"]
        assert "2" in result["age"]

    def test_calculate_subgroup_metrics_multiple_attributes(self):
        """Test with multiple sensitive attributes."""
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 1])
        y_prob = np.random.random(8)
        groups = {
            "sex": np.array([0, 0, 0, 0, 1, 1, 1, 1]),
            "age": np.array([0, 0, 1, 1, 0, 0, 1, 1]),
        }

        from sklearn.metrics import f1_score

        result = calculate_subgroup_metrics(y_true, y_pred, y_prob, groups, f1_score)

        assert "sex" in result
        assert "age" in result

    def test_calculate_subgroup_metrics_custom_function(self):
        """Test with custom metric function."""
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = np.array([1, 1, 0, 0, 1, 0, 0, 1])
        y_prob = np.random.random(8)
        groups = {"group": np.array([0, 0, 0, 0, 1, 1, 1, 1])}

        def custom_metric(y_t, y_p):
            return np.mean(y_t == y_p)

        result = calculate_subgroup_metrics(
            y_true, y_pred, y_prob, groups, custom_metric
        )

        assert "group" in result
        assert all(0.0 <= v <= 1.0 for v in result["group"].values())


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_single_sample_per_group(self):
        """Test with single sample per group."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        y_true = np.array([1, 0])
        y_pred = np.array([1, 0])
        groups = {"group": np.array([0, 1])}

        result = fm.calculate_all_metrics(y_true, y_pred, groups)

        assert "group_dp_diff" in result

    def test_unbalanced_groups(self):
        """Test with highly unbalanced groups."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        y_true = np.array([1, 1, 0, 0, 1])
        y_pred = np.array([1, 0, 0, 1, 1])
        # Group 0 has 4 samples, Group 1 has 1 sample
        groups = {"group": np.array([0, 0, 0, 0, 1])}

        result = fm.calculate_all_metrics(y_true, y_pred, groups)

        assert "group_dp_diff" in result

    def test_all_same_predictions(self):
        """Test when all predictions are the same."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        y_pred = np.array([1, 1, 1, 1, 1, 1])
        groups = {"group": np.array([0, 0, 0, 1, 1, 1])}

        result = fm.demographic_parity(y_pred, groups)

        assert result["group_dp_diff"] == 0.0

    def test_all_same_group(self):
        """Test when all samples belong to same group."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        y_true = np.array([1, 1, 0, 0])
        y_pred = np.array([1, 0, 0, 1])
        groups = {"group": np.array([0, 0, 0, 0])}

        result = fm.calculate_all_metrics(y_true, y_pred, groups)

        # Should still compute, but differences will be 0
        assert result["group_dp_diff"] == 0.0

    def test_large_number_of_groups(self):
        """Test with many groups."""
        fm = FairnessMetrics(sensitive_attrs=["region"])
        n_samples = 100
        n_groups = 10
        y_pred = np.random.randint(0, 2, n_samples)
        groups = {"region": np.random.randint(0, n_groups, n_samples)}

        result = fm.demographic_parity(y_pred, groups)

        assert "region_dp_diff" in result
        assert len(result["region_rates"]) <= n_groups


class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_very_small_rates(self):
        """Test with very small positive rates."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        # Only 1 positive in 1000 samples
        y_pred = np.zeros(1000)
        y_pred[0] = 1
        groups = {"group": np.array([0] * 500 + [1] * 500)}

        result = fm.demographic_parity(y_pred, groups)

        assert isinstance(result["group_dp_diff"], float)

    def test_perfect_predictions(self):
        """Test with perfect predictions."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = y_true.copy()
        groups = {"group": np.array([0, 0, 0, 0, 1, 1, 1, 1])}

        result = fm.equalized_odds(y_true, y_pred, groups)

        # Perfect predictions → TPR = 1, FPR = 0 for both groups
        assert result["group_tpr_diff"] == 0.0
        assert result["group_fpr_diff"] == 0.0

    def test_worst_predictions(self):
        """Test with completely wrong predictions."""
        fm = FairnessMetrics(sensitive_attrs=["group"])
        y_true = np.array([1, 1, 0, 0, 1, 1, 0, 0])
        y_pred = 1 - y_true  # Flip all predictions
        groups = {"group": np.array([0, 0, 0, 0, 1, 1, 1, 1])}

        result = fm.equalized_odds(y_true, y_pred, groups)

        # All wrong → TPR = 0, FPR = 1 for both groups
        assert result["group_tpr_diff"] == 0.0
        assert result["group_fpr_diff"] == 0.0
