"""
Fairness Metrics for Medical Imaging Model Evaluation

This module implements fairness metrics to evaluate demographic parity,
equalized odds, and disparate impact across sensitive attributes
(e.g., age, sex, skin_tone).
"""

import numpy as np
from typing import Dict, List, Optional, Tuple
from sklearn.metrics import confusion_matrix


class FairnessMetrics:
    """Calculate fairness metrics for model predictions across demographic groups."""

    def __init__(self, sensitive_attrs: List[str]):
        """
        Initialize fairness metrics calculator.

        Args:
            sensitive_attrs: List of sensitive attribute names (e.g., ['age', 'sex', 'skin_tone'])
        """
        self.sensitive_attrs = sensitive_attrs

    def demographic_parity(
        self,
        y_pred: np.ndarray,
        sensitive_groups: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate demographic parity: P(Y_pred=1 | A=a) should be equal across groups.

        Args:
            y_pred: Predicted labels (binary or multi-class)
            sensitive_groups: Dict mapping attribute name to group indicators

        Returns:
            Dict with demographic parity differences for each attribute
        """
        results = {}

        for attr_name, groups in sensitive_groups.items():
            unique_groups = np.unique(groups)
            positive_rates = {}

            for group in unique_groups:
                group_mask = groups == group
                positive_rate = np.mean(y_pred[group_mask])
                positive_rates[group] = positive_rate

            # Calculate max difference between groups
            rates = list(positive_rates.values())
            dp_diff = max(rates) - min(rates)
            results[f"{attr_name}_dp_diff"] = dp_diff
            results[f"{attr_name}_rates"] = positive_rates

        return results

    def equalized_odds(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_groups: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate equalized odds: TPR and FPR should be equal across groups.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_groups: Dict mapping attribute name to group indicators

        Returns:
            Dict with TPR/FPR differences for each attribute
        """
        results = {}

        for attr_name, groups in sensitive_groups.items():
            unique_groups = np.unique(groups)
            tpr_by_group = {}
            fpr_by_group = {}

            for group in unique_groups:
                group_mask = groups == group
                y_true_group = y_true[group_mask]
                y_pred_group = y_pred[group_mask]

                tn, fp, fn, tp = confusion_matrix(
                    y_true_group, y_pred_group, labels=[0, 1]
                ).ravel()

                tpr = tp / (tp + fn) if (tp + fn) > 0 else 0
                fpr = fp / (fp + tn) if (fp + tn) > 0 else 0

                tpr_by_group[group] = tpr
                fpr_by_group[group] = fpr

            # Calculate max differences
            tpr_values = list(tpr_by_group.values())
            fpr_values = list(fpr_by_group.values())

            results[f"{attr_name}_tpr_diff"] = max(tpr_values) - min(tpr_values)
            results[f"{attr_name}_fpr_diff"] = max(fpr_values) - min(fpr_values)
            results[f"{attr_name}_tpr"] = tpr_by_group
            results[f"{attr_name}_fpr"] = fpr_by_group

        return results

    def disparate_impact(
        self,
        y_pred: np.ndarray,
        sensitive_groups: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate disparate impact ratio: min(P(Y=1|A=a)) / max(P(Y=1|A=a)).
        A ratio < 0.8 is considered problematic (80% rule).

        Args:
            y_pred: Predicted labels
            sensitive_groups: Dict mapping attribute name to group indicators

        Returns:
            Dict with disparate impact ratios for each attribute
        """
        results = {}

        for attr_name, groups in sensitive_groups.items():
            unique_groups = np.unique(groups)
            positive_rates = []

            for group in unique_groups:
                group_mask = groups == group
                positive_rate = np.mean(y_pred[group_mask])
                positive_rates.append(positive_rate)

            # Calculate ratio
            di_ratio = min(positive_rates) / max(positive_rates) if max(positive_rates) > 0 else 0
            results[f"{attr_name}_di_ratio"] = di_ratio
            results[f"{attr_name}_passes_80_rule"] = di_ratio >= 0.8

        return results

    def calculate_all_metrics(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        sensitive_groups: Dict[str, np.ndarray]
    ) -> Dict[str, float]:
        """
        Calculate all fairness metrics.

        Args:
            y_true: True labels
            y_pred: Predicted labels
            sensitive_groups: Dict mapping attribute name to group indicators

        Returns:
            Dict with all fairness metrics
        """
        metrics = {}

        # Demographic parity
        dp_metrics = self.demographic_parity(y_pred, sensitive_groups)
        metrics.update(dp_metrics)

        # Equalized odds
        eo_metrics = self.equalized_odds(y_true, y_pred, sensitive_groups)
        metrics.update(eo_metrics)

        # Disparate impact
        di_metrics = self.disparate_impact(y_pred, sensitive_groups)
        metrics.update(di_metrics)

        return metrics


def calculate_subgroup_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    groups: Dict[str, np.ndarray],
    metric_fn: callable
) -> Dict[str, Dict[str, float]]:
    """
    Calculate a given metric for each subgroup.

    Args:
        y_true: True labels
        y_pred: Predicted labels
        y_prob: Prediction probabilities
        groups: Dict mapping attribute name to group indicators
        metric_fn: Metric function to apply (e.g., accuracy_score, f1_score)

    Returns:
        Dict mapping attribute -> group -> metric value
    """
    results = {}

    for attr_name, group_ids in groups.items():
        unique_groups = np.unique(group_ids)
        group_metrics = {}

        for group in unique_groups:
            group_mask = group_ids == group
            metric_value = metric_fn(y_true[group_mask], y_pred[group_mask])
            group_metrics[str(group)] = metric_value

        results[attr_name] = group_metrics

    return results
