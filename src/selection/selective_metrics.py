"""
Selective Prediction Evaluation Metrics for Phase 8.5.

This module implements comprehensive evaluation metrics for selective prediction,
enabling rigorous assessment of model abstention strategies in medical imaging.
These metrics directly support Research Question 3 (RQ3) validation.

Mathematical Foundation
-----------------------

1. **Coverage** (φ):
   φ = |{x : g(x) = accept}| / |X|
   The fraction of samples for which the model makes predictions.

2. **Selective Accuracy** (acc_s):
   acc_s = Σ[𝟙(ŷ = y) · 𝟙(g(x) = accept)] / Σ[𝟙(g(x) = accept)]
   Accuracy computed only on accepted (non-abstained) samples.

3. **Selective Risk** (R_s):
   R_s = 1 - acc_s = E[𝟙(ŷ ≠ y) | g(x) = accept]
   Error rate on accepted samples (lower is better).

4. **Risk on Rejected** (R_r):
   R_r = E[𝟙(ŷ ≠ y) | g(x) = reject]
   Error rate on samples the model chose to reject.
   High R_r indicates good rejection quality (model rejects hard cases).

5. **AURC (Area Under Risk-Coverage Curve)**:
   AURC = ∫₀¹ risk(c) dc
   Where risk(c) = 1 - accuracy at coverage c.
   Lower AURC = better selective prediction (Geifman & El-Yaniv, 2017).

6. **E-AURC (Excess AURC)**:
   E-AURC = AURC - AURC_optimal
   Where AURC_optimal is achieved by oracle that only rejects mistakes.
   Measures how much worse than optimal the selector is.

7. **Selective Improvement** (Δ):
   Δ = acc_s - acc_overall
   Improvement in accuracy from selective prediction at given coverage.
   RQ3 Target: Δ ≥ 4pp at 90% coverage.

8. **Calibration Post-Rejection** (ECE_s):
   ECE on accepted samples only.
   Measures whether confidence is well-calibrated after rejection.

9. **Rejection Precision/Recall**:
   - Precision: P(mistake | rejected) - How many rejected are mistakes
   - Recall: P(rejected | mistake) - How many mistakes are rejected

Research Integration
--------------------
This module directly addresses RQ3:
    "Can multi-signal gating enable safe selective prediction?"

Hypothesis Validation:
- H3a: Δ ≥ 4pp at 90% coverage
- H3b: Combined gating outperforms confidence-only
- H3c: Benefits persist in cross-site evaluation

Key References
--------------
1. Geifman & El-Yaniv (2017). "Selective Classification for Deep Neural Networks"
   - Introduced AURC metric for neural selective prediction

2. El-Yaniv & Wiener (2010). "On the Foundations of Noise-free Selective Classification"
   - Theoretical foundations of selective classification

3. Hendrickx et al. (2021). "Machine Learning with a Reject Option: A Survey"
   - Comprehensive survey of rejection methods

Production Features
-------------------
1. **Numerical Stability**: Handles edge cases (all rejected, all accepted)
2. **Vectorized Operations**: NumPy-based for efficiency
3. **Bootstrap CIs**: Statistical confidence intervals for all metrics
4. **Visualization**: Publication-ready risk-coverage curves
5. **Multi-Strategy Support**: Compare confidence-only, stability-only, combined
6. **Threshold Sweep**: Automatic coverage-risk curves across thresholds
7. **Export Support**: JSON/CSV/YAML output for reporting

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Project: Tri-Objective Robust XAI for Medical Imaging
Phase: 8.5 - Selective Prediction Evaluation Metrics
Date: November 28, 2025
Version: 1.0.0 (Production)
"""

from __future__ import annotations

import json
import logging
import warnings
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from scipy import integrate
from sklearn.metrics import auc

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONSTANTS AND CONFIGURATION
# ============================================================================


# Default number of bootstrap samples for confidence intervals
DEFAULT_N_BOOTSTRAP = 1000

# Default confidence level for CIs
DEFAULT_CONFIDENCE_LEVEL = 0.95

# Minimum samples required for reliable metrics
MIN_SAMPLES_FOR_METRICS = 10

# Number of threshold points for risk-coverage curves
DEFAULT_N_THRESHOLDS = 100

# Numerical stability epsilon
EPSILON = 1e-10

# Publication-ready plot style
PLOT_STYLE = {
    "figure.figsize": (10, 8),
    "font.size": 12,
    "axes.labelsize": 14,
    "axes.titlesize": 16,
    "legend.fontsize": 11,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "lines.linewidth": 2,
    "lines.markersize": 8,
}


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class SelectiveMetrics:
    """
    Comprehensive selective prediction metrics container.

    This dataclass holds all computed metrics for a single evaluation,
    including point estimates, confidence intervals, and metadata.

    Attributes
    ----------
    coverage : float
        Fraction of samples accepted [0, 1].

    selective_accuracy : float
        Accuracy on accepted samples [0, 1].

    selective_risk : float
        Error rate on accepted samples = 1 - selective_accuracy [0, 1].

    overall_accuracy : float
        Accuracy on all samples (before selection) [0, 1].

    improvement : float
        selective_accuracy - overall_accuracy (in percentage points).

    risk_on_rejected : float
        Error rate on rejected samples [0, 1].
        Should be HIGH if selection is working (rejecting hard cases).

    rejection_quality : float
        Ratio of mistake rate in rejected vs accepted.
        Higher = better quality rejection (rejecting mistakes).

    aurc : float
        Area Under Risk-Coverage curve [0, 1]. Lower is better.

    e_aurc : float
        Excess AURC = AURC - AURC_optimal. Lower is better.

    ece_post_selection : float
        Expected Calibration Error on accepted samples.

    n_total : int
        Total number of samples evaluated.

    n_accepted : int
        Number of accepted samples.

    n_rejected : int
        Number of rejected samples.

    n_correct_accepted : int
        Number of correctly classified accepted samples.

    n_incorrect_accepted : int
        Number of incorrectly classified accepted samples.

    n_correct_rejected : int
        Number of correctly classified but rejected samples (Type I error).

    n_incorrect_rejected : int
        Number of incorrectly classified and rejected samples (correct rejection).

    rejection_precision : float
        P(mistake | rejected) = n_incorrect_rejected / n_rejected.

    rejection_recall : float
        P(rejected | mistake) = n_incorrect_rejected / n_incorrect_total.

    confidence_intervals : Dict[str, Tuple[float, float]]
        95% bootstrap CIs for key metrics.

    metadata : Dict[str, Any]
        Additional metadata (thresholds, strategy, etc.).
    """

    # Core metrics
    coverage: float
    selective_accuracy: float
    selective_risk: float
    overall_accuracy: float
    improvement: float

    # Rejection analysis
    risk_on_rejected: float
    rejection_quality: float

    # Risk-coverage metrics
    aurc: float
    e_aurc: float

    # Calibration
    ece_post_selection: float

    # Sample counts
    n_total: int
    n_accepted: int
    n_rejected: int
    n_correct_accepted: int
    n_incorrect_accepted: int
    n_correct_rejected: int
    n_incorrect_rejected: int

    # Rejection classification metrics
    rejection_precision: float
    rejection_recall: float

    # Statistical validation
    confidence_intervals: Dict[str, Tuple[float, float]] = field(default_factory=dict)

    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate metrics after initialization."""
        # Validate ranges
        for metric_name in [
            "coverage",
            "selective_accuracy",
            "selective_risk",
            "overall_accuracy",
            "risk_on_rejected",
            "aurc",
            "e_aurc",
        ]:
            value = getattr(self, metric_name)
            if not (0.0 <= value <= 1.0 or np.isnan(value)):
                logger.warning(
                    f"{metric_name}={value} outside [0,1]. May indicate edge case."
                )

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert to dictionary for serialization.

        Returns
        -------
        dict
            JSON-serializable dictionary of all metrics.
        """
        return {
            "coverage": float(self.coverage),
            "selective_accuracy": float(self.selective_accuracy),
            "selective_risk": float(self.selective_risk),
            "overall_accuracy": float(self.overall_accuracy),
            "improvement": float(self.improvement),
            "improvement_percentage_points": float(self.improvement * 100),
            "risk_on_rejected": float(self.risk_on_rejected),
            "rejection_quality": float(self.rejection_quality),
            "aurc": float(self.aurc),
            "e_aurc": float(self.e_aurc),
            "ece_post_selection": float(self.ece_post_selection),
            "n_total": int(self.n_total),
            "n_accepted": int(self.n_accepted),
            "n_rejected": int(self.n_rejected),
            "n_correct_accepted": int(self.n_correct_accepted),
            "n_incorrect_accepted": int(self.n_incorrect_accepted),
            "n_correct_rejected": int(self.n_correct_rejected),
            "n_incorrect_rejected": int(self.n_incorrect_rejected),
            "rejection_precision": float(self.rejection_precision),
            "rejection_recall": float(self.rejection_recall),
            "confidence_intervals": {
                k: [float(v[0]), float(v[1])]
                for k, v in self.confidence_intervals.items()
            },
            "metadata": self.metadata,
        }

    def to_json(self, filepath: Optional[Union[str, Path]] = None) -> str:
        """
        Export to JSON format.

        Parameters
        ----------
        filepath : str or Path, optional
            If provided, save to file. Otherwise return string.

        Returns
        -------
        str
            JSON string representation.
        """
        json_str = json.dumps(self.to_dict(), indent=2)
        if filepath:
            Path(filepath).write_text(json_str)
            logger.info(f"Saved metrics to {filepath}")
        return json_str

    def summary(self) -> str:
        """
        Generate human-readable summary.

        Returns
        -------
        str
            Formatted summary string for logging/display.
        """
        lines = [
            "=" * 60,
            "SELECTIVE PREDICTION METRICS SUMMARY",
            "=" * 60,
            "",
            "📊 CORE METRICS",
            f"  Coverage:            {self.coverage:.2%} ({self.n_accepted}/{self.n_total} accepted)",
            f"  Selective Accuracy:  {self.selective_accuracy:.2%}",
            f"  Overall Accuracy:    {self.overall_accuracy:.2%}",
            f"  Improvement:         {self.improvement*100:+.2f}pp",
            "",
            "📈 RISK METRICS",
            f"  Selective Risk:      {self.selective_risk:.4f}",
            f"  Risk on Rejected:    {self.risk_on_rejected:.4f}",
            f"  Rejection Quality:   {self.rejection_quality:.2f}x",
            f"  AURC:               {self.aurc:.4f}",
            f"  E-AURC:             {self.e_aurc:.4f}",
            "",
            "🎯 REJECTION ANALYSIS",
            f"  Rejected Samples:    {self.n_rejected}",
            f"  - Correct (Type I):  {self.n_correct_rejected}",
            f"  - Incorrect (Good):  {self.n_incorrect_rejected}",
            f"  Rejection Precision: {self.rejection_precision:.2%}",
            f"  Rejection Recall:    {self.rejection_recall:.2%}",
            "",
            "📐 CALIBRATION",
            f"  ECE Post-Selection:  {self.ece_post_selection:.4f}",
            "",
        ]

        if self.confidence_intervals:
            lines.append("📊 CONFIDENCE INTERVALS (95%)")
            for metric, (lower, upper) in self.confidence_intervals.items():
                lines.append(f"  {metric}: [{lower:.4f}, {upper:.4f}]")
            lines.append("")

        lines.append("=" * 60)
        return "\n".join(lines)

    def passes_hypothesis(self, target_improvement: float = 0.04) -> bool:
        """
        Check if H3a hypothesis is satisfied.

        Parameters
        ----------
        target_improvement : float, optional (default=0.04)
            Minimum improvement required (4pp = 0.04).

        Returns
        -------
        bool
            True if improvement >= target at current coverage.
        """
        return self.improvement >= target_improvement


@dataclass
class RiskCoverageCurve:
    """
    Container for risk-coverage curve data.

    Attributes
    ----------
    coverages : np.ndarray
        Coverage values at each threshold, shape (n_thresholds,).

    risks : np.ndarray
        Risk values at each coverage, shape (n_thresholds,).

    accuracies : np.ndarray
        Accuracy values at each coverage, shape (n_thresholds,).

    thresholds : np.ndarray
        Threshold values used, shape (n_thresholds,).

    aurc : float
        Area under the risk-coverage curve.

    e_aurc : float
        Excess AURC (compared to optimal).

    optimal_risks : np.ndarray
        Optimal (oracle) risk at each coverage.
    """

    coverages: np.ndarray
    risks: np.ndarray
    accuracies: np.ndarray
    thresholds: np.ndarray
    aurc: float
    e_aurc: float
    optimal_risks: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "coverages": self.coverages.tolist(),
            "risks": self.risks.tolist(),
            "accuracies": self.accuracies.tolist(),
            "thresholds": self.thresholds.tolist(),
            "aurc": float(self.aurc),
            "e_aurc": float(self.e_aurc),
            "optimal_risks": self.optimal_risks.tolist(),
        }


# ============================================================================
# CORE METRIC FUNCTIONS
# ============================================================================


def compute_coverage(is_accepted: np.ndarray) -> float:
    """
    Compute coverage (fraction of samples accepted).

    Parameters
    ----------
    is_accepted : np.ndarray
        Boolean array indicating acceptance, shape (n_samples,).

    Returns
    -------
    float
        Coverage in [0, 1].

    Examples
    --------
    >>> is_accepted = np.array([True, True, False, True, False])
    >>> compute_coverage(is_accepted)
    0.6
    """
    if len(is_accepted) == 0:
        return 0.0
    return float(np.mean(is_accepted))


def compute_selective_accuracy(
    predictions: np.ndarray, labels: np.ndarray, is_accepted: np.ndarray
) -> float:
    """
    Compute accuracy on accepted samples only.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted class labels, shape (n_samples,).

    labels : np.ndarray
        Ground truth labels, shape (n_samples,).

    is_accepted : np.ndarray
        Boolean array indicating acceptance, shape (n_samples,).

    Returns
    -------
    float
        Selective accuracy in [0, 1]. Returns NaN if no samples accepted.

    Examples
    --------
    >>> predictions = np.array([0, 1, 2, 1, 0])
    >>> labels = np.array([0, 1, 1, 1, 1])
    >>> is_accepted = np.array([True, True, False, True, False])
    >>> compute_selective_accuracy(predictions, labels, is_accepted)
    1.0  # All 3 accepted samples are correct
    """
    if len(predictions) == 0 or not np.any(is_accepted):
        return np.nan

    # Boolean: is each prediction correct?
    correct = predictions == labels

    # Accuracy on accepted samples
    accepted_correct = correct[is_accepted]

    if len(accepted_correct) == 0:
        return np.nan

    return float(np.mean(accepted_correct))


def compute_selective_risk(
    predictions: np.ndarray, labels: np.ndarray, is_accepted: np.ndarray
) -> float:
    """
    Compute risk (error rate) on accepted samples.

    Risk = 1 - Selective Accuracy

    Parameters
    ----------
    predictions : np.ndarray
        Predicted class labels, shape (n_samples,).

    labels : np.ndarray
        Ground truth labels, shape (n_samples,).

    is_accepted : np.ndarray
        Boolean array indicating acceptance, shape (n_samples,).

    Returns
    -------
    float
        Selective risk in [0, 1]. Returns NaN if no samples accepted.
    """
    selective_acc = compute_selective_accuracy(predictions, labels, is_accepted)
    if np.isnan(selective_acc):
        return np.nan
    return 1.0 - selective_acc


def compute_risk_on_rejected(
    predictions: np.ndarray, labels: np.ndarray, is_accepted: np.ndarray
) -> float:
    """
    Compute error rate on rejected samples.

    This should be HIGH if the selector is working properly,
    indicating it's rejecting the hard/incorrect cases.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted class labels, shape (n_samples,).

    labels : np.ndarray
        Ground truth labels, shape (n_samples,).

    is_accepted : np.ndarray
        Boolean array indicating acceptance, shape (n_samples,).

    Returns
    -------
    float
        Risk on rejected samples in [0, 1]. Returns NaN if no samples rejected.

    Notes
    -----
    High risk on rejected indicates good rejection quality:
    - Risk_rejected >> Risk_accepted means selector successfully identifies errors.
    - Risk_rejected ≈ Risk_accepted means selector is random.
    """
    is_rejected = ~is_accepted

    if not np.any(is_rejected):
        return np.nan

    correct = predictions == labels
    rejected_errors = ~correct[is_rejected]

    return float(np.mean(rejected_errors))


def compute_improvement(selective_accuracy: float, overall_accuracy: float) -> float:
    """
    Compute improvement from selective prediction.

    Improvement = Selective Accuracy - Overall Accuracy

    Parameters
    ----------
    selective_accuracy : float
        Accuracy on accepted samples.

    overall_accuracy : float
        Accuracy on all samples (before selection).

    Returns
    -------
    float
        Improvement (can be negative if selection hurts performance).

    Notes
    -----
    RQ3 Target: ≥0.04 (4 percentage points) at 90% coverage.
    """
    if np.isnan(selective_accuracy) or np.isnan(overall_accuracy):
        return np.nan
    return selective_accuracy - overall_accuracy


def compute_rejection_quality(risk_on_rejected: float, selective_risk: float) -> float:
    """
    Compute rejection quality ratio.

    Quality = Risk_rejected / Risk_accepted

    Parameters
    ----------
    risk_on_rejected : float
        Error rate on rejected samples.

    selective_risk : float
        Error rate on accepted samples.

    Returns
    -------
    float
        Quality ratio (>1 means rejecting harder cases).

    Notes
    -----
    - Quality > 1: Good rejection (harder cases rejected)
    - Quality ≈ 1: Random rejection
    - Quality < 1: Poor rejection (easier cases rejected)
    """
    if np.isnan(risk_on_rejected) or selective_risk < EPSILON:
        return np.nan
    return risk_on_rejected / (selective_risk + EPSILON)


def compute_rejection_precision_recall(
    predictions: np.ndarray, labels: np.ndarray, is_accepted: np.ndarray
) -> Tuple[float, float]:
    """
    Compute precision and recall for rejection as a classification task.

    Treats rejection as a binary classifier for detecting mistakes:
    - Positive = Mistake (incorrect prediction)
    - Negative = Correct prediction
    - Predicted Positive = Rejected
    - Predicted Negative = Accepted

    Parameters
    ----------
    predictions : np.ndarray
        Predicted class labels, shape (n_samples,).

    labels : np.ndarray
        Ground truth labels, shape (n_samples,).

    is_accepted : np.ndarray
        Boolean array indicating acceptance, shape (n_samples,).

    Returns
    -------
    precision : float
        P(mistake | rejected) = TP / (TP + FP)

    recall : float
        P(rejected | mistake) = TP / (TP + FN)

    Notes
    -----
    Confusion matrix for rejection:
                        Actually Mistake  Actually Correct
    Rejected (Pred+)         TP               FP
    Accepted (Pred-)         FN               TN

    - TP = n_incorrect_rejected (correctly rejected mistakes)
    - FP = n_correct_rejected (incorrectly rejected correct predictions)
    - FN = n_incorrect_accepted (missed mistakes that were accepted)
    - TN = n_correct_accepted (correctly accepted correct predictions)
    """
    is_rejected = ~is_accepted
    is_mistake = predictions != labels

    # True Positives: rejected AND mistake
    tp = np.sum(is_rejected & is_mistake)

    # False Positives: rejected AND correct
    fp = np.sum(is_rejected & ~is_mistake)

    # False Negatives: accepted AND mistake
    fn = np.sum(is_accepted & is_mistake)

    # Precision: P(mistake | rejected)
    precision = tp / (tp + fp + EPSILON) if (tp + fp) > 0 else np.nan

    # Recall: P(rejected | mistake)
    recall = tp / (tp + fn + EPSILON) if (tp + fn) > 0 else np.nan

    return float(precision), float(recall)


# ============================================================================
# RISK-COVERAGE CURVE AND AURC
# ============================================================================


def compute_risk_coverage_curve(
    predictions: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    n_thresholds: int = DEFAULT_N_THRESHOLDS,
) -> RiskCoverageCurve:
    """
    Compute risk-coverage curve by varying selection threshold.

    The curve shows how risk (error rate) varies as we accept more samples
    (increase coverage) by lowering the selection threshold.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted class labels, shape (n_samples,).

    labels : np.ndarray
        Ground truth labels, shape (n_samples,).

    scores : np.ndarray
        Selection scores (higher = more confident/stable), shape (n_samples,).

    n_thresholds : int, optional (default=100)
        Number of threshold points for the curve.

    Returns
    -------
    RiskCoverageCurve
        Container with coverage, risk, accuracy arrays and AURC.

    Notes
    -----
    Algorithm:
    1. Sort samples by score (descending)
    2. For each coverage level, compute risk on top-scoring samples
    3. Integrate to get AURC
    """
    n_samples = len(predictions)

    if n_samples == 0:
        return RiskCoverageCurve(
            coverages=np.array([]),
            risks=np.array([]),
            accuracies=np.array([]),
            thresholds=np.array([]),
            aurc=np.nan,
            e_aurc=np.nan,
            optimal_risks=np.array([]),
        )

    # Sort by score (highest first = most confident)
    sorted_indices = np.argsort(-scores)  # Descending
    sorted_predictions = predictions[sorted_indices]
    sorted_labels = labels[sorted_indices]
    sorted_scores = scores[sorted_indices]

    # Correctness for each sample
    correct = sorted_predictions == sorted_labels

    # Compute risk at each coverage level
    coverages = []
    risks = []
    accuracies = []
    thresholds = []

    # Use unique thresholds to avoid redundant computation
    unique_thresholds = np.unique(sorted_scores)[::-1]  # Descending

    # Add 100% coverage point
    if len(unique_thresholds) == 0 or unique_thresholds[-1] > 0:
        unique_thresholds = np.append(unique_thresholds, 0.0)

    for threshold in unique_thresholds:
        # Accept samples with score >= threshold
        is_accepted = sorted_scores >= threshold
        n_accepted = np.sum(is_accepted)

        if n_accepted == 0:
            continue

        coverage = n_accepted / n_samples
        accuracy = np.mean(correct[:n_accepted])  # Cumulative accuracy
        risk = 1.0 - accuracy

        coverages.append(coverage)
        risks.append(risk)
        accuracies.append(accuracy)
        thresholds.append(threshold)

    coverages = np.array(coverages)
    risks = np.array(risks)
    accuracies = np.array(accuracies)
    thresholds = np.array(thresholds)

    # Ensure we have coverage = 1.0 point
    if len(coverages) == 0 or coverages[-1] < 1.0:
        coverages = np.append(coverages, 1.0)
        overall_risk = 1.0 - np.mean(correct)
        risks = np.append(risks, overall_risk)
        accuracies = np.append(accuracies, np.mean(correct))
        thresholds = np.append(thresholds, 0.0)

    # Compute AURC using trapezoidal integration
    # Sort by coverage for proper integration
    sort_idx = np.argsort(coverages)
    coverages_sorted = coverages[sort_idx]
    risks_sorted = risks[sort_idx]

    aurc = float(np.trapz(risks_sorted, coverages_sorted))

    # Compute optimal (oracle) AURC
    # Oracle rejects mistakes first (sorted by correctness)
    n_errors = np.sum(~correct)
    optimal_risks = []
    for cov in coverages_sorted:
        n_accept = int(np.round(cov * n_samples))
        # Optimal: accept all correct first, then errors
        n_correct = np.sum(correct)
        n_errors_in_accept = max(0, n_accept - n_correct)
        opt_risk = n_errors_in_accept / max(n_accept, 1)
        optimal_risks.append(opt_risk)
    optimal_risks = np.array(optimal_risks)

    optimal_aurc = float(np.trapz(optimal_risks, coverages_sorted))
    e_aurc = aurc - optimal_aurc

    return RiskCoverageCurve(
        coverages=coverages,
        risks=risks,
        accuracies=accuracies,
        thresholds=thresholds,
        aurc=aurc,
        e_aurc=e_aurc,
        optimal_risks=optimal_risks,
    )


def compute_aurc(
    predictions: np.ndarray, labels: np.ndarray, scores: np.ndarray
) -> Tuple[float, float]:
    """
    Compute AURC and E-AURC (Area Under Risk-Coverage Curve).

    Lower AURC indicates better selective prediction capability.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted class labels, shape (n_samples,).

    labels : np.ndarray
        Ground truth labels, shape (n_samples,).

    scores : np.ndarray
        Selection scores (higher = more confident), shape (n_samples,).

    Returns
    -------
    aurc : float
        Area under risk-coverage curve [0, 1]. Lower is better.

    e_aurc : float
        Excess AURC above optimal [0, 1]. Lower is better.

    Notes
    -----
    Reference: Geifman & El-Yaniv (2017). "Selective Classification for
    Deep Neural Networks"
    """
    curve = compute_risk_coverage_curve(predictions, labels, scores)
    return curve.aurc, curve.e_aurc


# ============================================================================
# CALIBRATION POST-SELECTION
# ============================================================================


def compute_ece_post_selection(
    predictions: np.ndarray,
    labels: np.ndarray,
    confidences: np.ndarray,
    is_accepted: np.ndarray,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error on accepted samples only.

    This measures whether confidence is well-calibrated after rejection,
    which is important for trustworthy clinical predictions.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted class labels, shape (n_samples,).

    labels : np.ndarray
        Ground truth labels, shape (n_samples,).

    confidences : np.ndarray
        Confidence scores for predictions, shape (n_samples,).

    is_accepted : np.ndarray
        Boolean array indicating acceptance, shape (n_samples,).

    n_bins : int, optional (default=15)
        Number of bins for ECE computation.

    Returns
    -------
    float
        ECE on accepted samples [0, 1]. Lower is better.

    Notes
    -----
    ECE = Σ_b |B_b|/n × |acc(B_b) - conf(B_b)|

    Where B_b is the set of samples in bin b.
    """
    if not np.any(is_accepted):
        return np.nan

    # Filter to accepted samples
    accepted_predictions = predictions[is_accepted]
    accepted_labels = labels[is_accepted]
    accepted_confidences = confidences[is_accepted]

    n_accepted = len(accepted_predictions)
    correct = accepted_predictions == accepted_labels

    # Create bins
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]

    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (accepted_confidences > bin_lower) & (
            accepted_confidences <= bin_upper
        )
        prop_in_bin = np.sum(in_bin) / n_accepted

        if prop_in_bin > 0:
            # Accuracy in this bin
            acc_in_bin = np.mean(correct[in_bin])
            # Average confidence in this bin
            avg_conf_in_bin = np.mean(accepted_confidences[in_bin])
            # Contribution to ECE
            ece += prop_in_bin * np.abs(acc_in_bin - avg_conf_in_bin)

    return float(ece)


# ============================================================================
# COMPREHENSIVE METRIC COMPUTATION
# ============================================================================


def compute_selective_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    is_accepted: np.ndarray,
    confidences: Optional[np.ndarray] = None,
    scores: Optional[np.ndarray] = None,
    compute_ci: bool = True,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    metadata: Optional[Dict[str, Any]] = None,
) -> SelectiveMetrics:
    """
    Compute comprehensive selective prediction metrics.

    This is the main entry point for evaluating selective prediction.
    Computes all metrics defined in Phase 8.5 checklist.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted class labels, shape (n_samples,).

    labels : np.ndarray
        Ground truth labels, shape (n_samples,).

    is_accepted : np.ndarray
        Boolean array indicating acceptance, shape (n_samples,).

    confidences : np.ndarray, optional
        Confidence scores for predictions, shape (n_samples,).
        Used for ECE computation. If None, uses uniform 0.5.

    scores : np.ndarray, optional
        Selection scores (higher = more confident/stable), shape (n_samples,).
        Used for AURC computation. If None, uses confidences.

    compute_ci : bool, optional (default=True)
        Whether to compute bootstrap confidence intervals.

    n_bootstrap : int, optional (default=1000)
        Number of bootstrap samples for CIs.

    confidence_level : float, optional (default=0.95)
        Confidence level for CIs (e.g., 0.95 for 95% CI).

    metadata : dict, optional
        Additional metadata to include in results.

    Returns
    -------
    SelectiveMetrics
        Comprehensive metrics container.

    Examples
    --------
    >>> predictions = np.array([0, 1, 2, 1, 0, 2, 1])
    >>> labels = np.array([0, 1, 1, 1, 0, 2, 0])
    >>> is_accepted = np.array([True, True, False, True, True, True, False])
    >>> confidences = np.array([0.9, 0.85, 0.6, 0.8, 0.95, 0.88, 0.55])
    >>>
    >>> metrics = compute_selective_metrics(
    ...     predictions, labels, is_accepted, confidences
    ... )
    >>> print(metrics.summary())
    """
    # Input validation
    predictions = np.asarray(predictions)
    labels = np.asarray(labels)
    is_accepted = np.asarray(is_accepted, dtype=bool)

    n_total = len(predictions)

    if n_total == 0:
        logger.warning("Empty input arrays. Returning NaN metrics.")
        return SelectiveMetrics(
            coverage=np.nan,
            selective_accuracy=np.nan,
            selective_risk=np.nan,
            overall_accuracy=np.nan,
            improvement=np.nan,
            risk_on_rejected=np.nan,
            rejection_quality=np.nan,
            aurc=np.nan,
            e_aurc=np.nan,
            ece_post_selection=np.nan,
            n_total=0,
            n_accepted=0,
            n_rejected=0,
            n_correct_accepted=0,
            n_incorrect_accepted=0,
            n_correct_rejected=0,
            n_incorrect_rejected=0,
            rejection_precision=np.nan,
            rejection_recall=np.nan,
            metadata=metadata or {},
        )

    # Set defaults for optional arrays
    if confidences is None:
        confidences = np.full(n_total, 0.5)
    else:
        confidences = np.asarray(confidences)

    if scores is None:
        scores = confidences
    else:
        scores = np.asarray(scores)

    # Sample counts
    is_correct = predictions == labels
    n_accepted = int(np.sum(is_accepted))
    n_rejected = int(np.sum(~is_accepted))

    n_correct_accepted = int(np.sum(is_correct & is_accepted))
    n_incorrect_accepted = int(np.sum(~is_correct & is_accepted))
    n_correct_rejected = int(np.sum(is_correct & ~is_accepted))
    n_incorrect_rejected = int(np.sum(~is_correct & ~is_accepted))

    # Core metrics
    coverage = compute_coverage(is_accepted)
    selective_accuracy = compute_selective_accuracy(predictions, labels, is_accepted)
    selective_risk = compute_selective_risk(predictions, labels, is_accepted)
    overall_accuracy = float(np.mean(is_correct))
    improvement = compute_improvement(selective_accuracy, overall_accuracy)

    # Rejection analysis
    risk_on_rejected = compute_risk_on_rejected(predictions, labels, is_accepted)
    rejection_quality = compute_rejection_quality(risk_on_rejected, selective_risk)
    rejection_precision, rejection_recall = compute_rejection_precision_recall(
        predictions, labels, is_accepted
    )

    # Risk-coverage metrics
    rc_curve = compute_risk_coverage_curve(predictions, labels, scores)
    aurc = rc_curve.aurc
    e_aurc = rc_curve.e_aurc

    # Calibration post-selection
    ece_post_selection = compute_ece_post_selection(
        predictions, labels, confidences, is_accepted
    )

    # Bootstrap confidence intervals
    confidence_intervals = {}
    if compute_ci and n_total >= MIN_SAMPLES_FOR_METRICS:
        confidence_intervals = _compute_bootstrap_cis(
            predictions,
            labels,
            is_accepted,
            confidences,
            scores,
            n_bootstrap,
            confidence_level,
        )

    # Build metadata
    full_metadata = {
        "n_samples": n_total,
        "n_bootstrap": n_bootstrap if compute_ci else 0,
        "confidence_level": confidence_level,
        **(metadata or {}),
    }

    return SelectiveMetrics(
        coverage=coverage,
        selective_accuracy=selective_accuracy,
        selective_risk=selective_risk,
        overall_accuracy=overall_accuracy,
        improvement=improvement,
        risk_on_rejected=risk_on_rejected,
        rejection_quality=rejection_quality,
        aurc=aurc,
        e_aurc=e_aurc,
        ece_post_selection=ece_post_selection,
        n_total=n_total,
        n_accepted=n_accepted,
        n_rejected=n_rejected,
        n_correct_accepted=n_correct_accepted,
        n_incorrect_accepted=n_incorrect_accepted,
        n_correct_rejected=n_correct_rejected,
        n_incorrect_rejected=n_incorrect_rejected,
        rejection_precision=rejection_precision,
        rejection_recall=rejection_recall,
        confidence_intervals=confidence_intervals,
        metadata=full_metadata,
    )


def _compute_bootstrap_cis(
    predictions: np.ndarray,
    labels: np.ndarray,
    is_accepted: np.ndarray,
    confidences: np.ndarray,
    scores: np.ndarray,
    n_bootstrap: int,
    confidence_level: float,
) -> Dict[str, Tuple[float, float]]:
    """
    Compute bootstrap confidence intervals for key metrics.

    Parameters
    ----------
    predictions, labels, is_accepted, confidences, scores : np.ndarray
        Input arrays for metric computation.

    n_bootstrap : int
        Number of bootstrap samples.

    confidence_level : float
        Confidence level (e.g., 0.95).

    Returns
    -------
    dict
        Dictionary mapping metric names to (lower, upper) tuples.
    """
    n_samples = len(predictions)
    rng = np.random.default_rng(seed=42)

    # Storage for bootstrap samples
    boot_coverage = []
    boot_selective_acc = []
    boot_improvement = []
    boot_aurc = []

    for _ in range(n_bootstrap):
        # Resample with replacement
        indices = rng.choice(n_samples, size=n_samples, replace=True)

        boot_preds = predictions[indices]
        boot_labels = labels[indices]
        boot_accepted = is_accepted[indices]
        boot_scores = scores[indices]

        # Compute metrics on bootstrap sample
        boot_coverage.append(compute_coverage(boot_accepted))
        boot_selective_acc.append(
            compute_selective_accuracy(boot_preds, boot_labels, boot_accepted)
        )

        overall_acc = float(np.mean(boot_preds == boot_labels))
        sel_acc = boot_selective_acc[-1]
        if not np.isnan(sel_acc):
            boot_improvement.append(sel_acc - overall_acc)

        curve = compute_risk_coverage_curve(boot_preds, boot_labels, boot_scores)
        if not np.isnan(curve.aurc):
            boot_aurc.append(curve.aurc)

    # Compute percentiles
    alpha = 1 - confidence_level
    lower_p = alpha / 2 * 100
    upper_p = (1 - alpha / 2) * 100

    cis = {}

    for name, samples in [
        ("coverage", boot_coverage),
        ("selective_accuracy", boot_selective_acc),
        ("improvement", boot_improvement),
        ("aurc", boot_aurc),
    ]:
        samples = np.array([s for s in samples if not np.isnan(s)])
        if len(samples) >= 10:
            cis[name] = (
                float(np.percentile(samples, lower_p)),
                float(np.percentile(samples, upper_p)),
            )

    return cis


# ============================================================================
# MULTI-STRATEGY COMPARISON
# ============================================================================


def compare_strategies(
    predictions: np.ndarray,
    labels: np.ndarray,
    confidence_scores: np.ndarray,
    stability_scores: np.ndarray,
    confidence_threshold: float = 0.85,
    stability_threshold: float = 0.75,
    target_coverage: float = 0.90,
) -> Dict[str, SelectiveMetrics]:
    """
    Compare different gating strategies (confidence-only, stability-only, combined).

    This function evaluates three strategies to validate hypothesis H3b:
    "Combined gating outperforms single-signal approaches."

    Parameters
    ----------
    predictions : np.ndarray
        Predicted class labels, shape (n_samples,).

    labels : np.ndarray
        Ground truth labels, shape (n_samples,).

    confidence_scores : np.ndarray
        Confidence scores, shape (n_samples,).

    stability_scores : np.ndarray
        Stability scores, shape (n_samples,).

    confidence_threshold : float, optional (default=0.85)
        Threshold for confidence-only gating.

    stability_threshold : float, optional (default=0.75)
        Threshold for stability-only gating.

    target_coverage : float, optional (default=0.90)
        Target coverage for reporting.

    Returns
    -------
    dict
        Dictionary mapping strategy names to SelectiveMetrics:
        - "confidence_only": Using only confidence for gating
        - "stability_only": Using only stability for gating
        - "combined": Using confidence AND stability
        - "combined_score": Using average of confidence and stability
    """
    results = {}

    # Strategy 1: Confidence-only
    is_accepted_conf = confidence_scores >= confidence_threshold
    results["confidence_only"] = compute_selective_metrics(
        predictions,
        labels,
        is_accepted_conf,
        confidences=confidence_scores,
        scores=confidence_scores,
        metadata={"strategy": "confidence_only", "threshold": confidence_threshold},
    )

    # Strategy 2: Stability-only
    is_accepted_stab = stability_scores >= stability_threshold
    results["stability_only"] = compute_selective_metrics(
        predictions,
        labels,
        is_accepted_stab,
        confidences=confidence_scores,
        scores=stability_scores,
        metadata={"strategy": "stability_only", "threshold": stability_threshold},
    )

    # Strategy 3: Combined (AND)
    is_accepted_combined = (confidence_scores >= confidence_threshold) & (
        stability_scores >= stability_threshold
    )
    combined_scores = (confidence_scores + stability_scores) / 2
    results["combined"] = compute_selective_metrics(
        predictions,
        labels,
        is_accepted_combined,
        confidences=confidence_scores,
        scores=combined_scores,
        metadata={
            "strategy": "combined",
            "confidence_threshold": confidence_threshold,
            "stability_threshold": stability_threshold,
        },
    )

    # Strategy 4: Combined score (weighted average)
    combined_score = 0.5 * confidence_scores + 0.5 * stability_scores
    # Find threshold for target coverage
    threshold_for_coverage = np.percentile(combined_score, (1 - target_coverage) * 100)
    is_accepted_score = combined_score >= threshold_for_coverage
    results["combined_score"] = compute_selective_metrics(
        predictions,
        labels,
        is_accepted_score,
        confidences=confidence_scores,
        scores=combined_score,
        metadata={
            "strategy": "combined_score",
            "score_threshold": float(threshold_for_coverage),
            "target_coverage": target_coverage,
        },
    )

    return results


# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_risk_coverage_curve(
    curves: Dict[str, RiskCoverageCurve],
    title: str = "Risk-Coverage Curves",
    save_path: Optional[Union[str, Path]] = None,
    show_optimal: bool = True,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot risk-coverage curves for multiple strategies.

    Parameters
    ----------
    curves : dict
        Dictionary mapping strategy names to RiskCoverageCurve objects.

    title : str, optional
        Plot title.

    save_path : str or Path, optional
        If provided, save figure to this path.

    show_optimal : bool, optional (default=True)
        Whether to show optimal (oracle) curve.

    figsize : tuple, optional
        Figure size.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.tab10.colors

        for i, (name, curve) in enumerate(curves.items()):
            # Sort by coverage for plotting
            sort_idx = np.argsort(curve.coverages)
            coverages = curve.coverages[sort_idx]
            risks = curve.risks[sort_idx]

            ax.plot(
                coverages,
                risks,
                label=f"{name} (AURC={curve.aurc:.4f})",
                color=colors[i % len(colors)],
                linewidth=2,
            )

        # Show optimal curve
        if show_optimal and len(curves) > 0:
            first_curve = list(curves.values())[0]
            if len(first_curve.optimal_risks) > 0:
                sort_idx = np.argsort(first_curve.coverages)
                ax.plot(
                    first_curve.coverages[sort_idx],
                    first_curve.optimal_risks,
                    "--",
                    color="gray",
                    linewidth=1.5,
                    label="Optimal (Oracle)",
                )

        ax.set_xlabel("Coverage", fontsize=14)
        ax.set_ylabel("Risk (Error Rate)", fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.legend(loc="upper left", fontsize=11)
        ax.set_xlim([0, 1])
        ax.set_ylim([0, max(0.3, ax.get_ylim()[1])])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved risk-coverage plot to {save_path}")

        return fig


def plot_accuracy_coverage_curve(
    curves: Dict[str, RiskCoverageCurve],
    title: str = "Accuracy-Coverage Curves",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> plt.Figure:
    """
    Plot accuracy vs coverage curves for multiple strategies.

    Parameters
    ----------
    curves : dict
        Dictionary mapping strategy names to RiskCoverageCurve objects.

    title : str, optional
        Plot title.

    save_path : str or Path, optional
        If provided, save figure to this path.

    figsize : tuple, optional
        Figure size.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, ax = plt.subplots(figsize=figsize)

        colors = plt.cm.tab10.colors

        for i, (name, curve) in enumerate(curves.items()):
            # Sort by coverage for plotting
            sort_idx = np.argsort(curve.coverages)
            coverages = curve.coverages[sort_idx]
            accuracies = curve.accuracies[sort_idx]

            ax.plot(
                coverages,
                accuracies,
                label=name,
                color=colors[i % len(colors)],
                linewidth=2,
                marker="o",
                markersize=3,
                markevery=max(1, len(coverages) // 10),
            )

        # Add reference line for overall accuracy
        if len(curves) > 0:
            first_curve = list(curves.values())[0]
            overall_acc = (
                first_curve.accuracies[-1] if len(first_curve.accuracies) > 0 else 0.5
            )
            ax.axhline(
                y=overall_acc,
                color="gray",
                linestyle="--",
                linewidth=1.5,
                label=f"Overall Accuracy ({overall_acc:.1%})",
            )

        ax.set_xlabel("Coverage", fontsize=14)
        ax.set_ylabel("Selective Accuracy", fontsize=14)
        ax.set_title(title, fontsize=16)
        ax.legend(loc="lower left", fontsize=11)
        ax.set_xlim([0, 1])
        ax.set_ylim([0.5, 1.0])
        ax.grid(True, alpha=0.3)

        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved accuracy-coverage plot to {save_path}")

        return fig


def plot_strategy_comparison(
    results: Dict[str, SelectiveMetrics],
    title: str = "Strategy Comparison",
    save_path: Optional[Union[str, Path]] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> plt.Figure:
    """
    Plot bar chart comparing metrics across strategies.

    Parameters
    ----------
    results : dict
        Dictionary mapping strategy names to SelectiveMetrics.

    title : str, optional
        Plot title.

    save_path : str or Path, optional
        If provided, save figure to this path.

    figsize : tuple, optional
        Figure size.

    Returns
    -------
    Figure
        Matplotlib figure object.
    """
    with plt.style.context("seaborn-v0_8-whitegrid"):
        fig, axes = plt.subplots(1, 3, figsize=figsize)

        strategies = list(results.keys())
        colors = plt.cm.Set2.colors

        # Plot 1: Coverage
        coverages = [results[s].coverage for s in strategies]
        axes[0].bar(strategies, coverages, color=colors[: len(strategies)])
        axes[0].set_ylabel("Coverage", fontsize=12)
        axes[0].set_title("Coverage by Strategy", fontsize=14)
        axes[0].set_ylim([0, 1])
        for i, v in enumerate(coverages):
            axes[0].text(i, v + 0.02, f"{v:.1%}", ha="center", fontsize=10)

        # Plot 2: Selective Accuracy
        sel_accs = [results[s].selective_accuracy for s in strategies]
        axes[1].bar(strategies, sel_accs, color=colors[: len(strategies)])
        axes[1].set_ylabel("Selective Accuracy", fontsize=12)
        axes[1].set_title("Selective Accuracy by Strategy", fontsize=14)
        axes[1].set_ylim([0.5, 1])
        for i, v in enumerate(sel_accs):
            if not np.isnan(v):
                axes[1].text(i, v + 0.01, f"{v:.1%}", ha="center", fontsize=10)

        # Plot 3: Improvement
        improvements = [results[s].improvement * 100 for s in strategies]
        bar_colors = [
            "green" if imp >= 4 else "orange" if imp >= 0 else "red"
            for imp in improvements
        ]
        axes[2].bar(strategies, improvements, color=bar_colors)
        axes[2].set_ylabel("Improvement (pp)", fontsize=12)
        axes[2].set_title("Accuracy Improvement", fontsize=14)
        axes[2].axhline(y=4, color="red", linestyle="--", label="H3a Target (4pp)")
        axes[2].legend(fontsize=10)
        for i, v in enumerate(improvements):
            if not np.isnan(v):
                axes[2].text(i, v + 0.3, f"{v:+.1f}pp", ha="center", fontsize=10)

        plt.suptitle(title, fontsize=16, y=1.02)
        plt.tight_layout()

        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches="tight")
            logger.info(f"Saved strategy comparison plot to {save_path}")

        return fig


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def find_threshold_for_coverage(scores: np.ndarray, target_coverage: float) -> float:
    """
    Find the score threshold that achieves a target coverage.

    Parameters
    ----------
    scores : np.ndarray
        Selection scores, shape (n_samples,).

    target_coverage : float
        Desired coverage in [0, 1].

    Returns
    -------
    float
        Threshold such that coverage(scores >= threshold) ≈ target_coverage.
    """
    if target_coverage >= 1.0:
        return 0.0
    if target_coverage <= 0.0:
        return float(np.max(scores)) + EPSILON

    # Find threshold as percentile
    percentile = (1 - target_coverage) * 100
    return float(np.percentile(scores, percentile))


def compute_metrics_at_coverage(
    predictions: np.ndarray,
    labels: np.ndarray,
    scores: np.ndarray,
    target_coverage: float,
    confidences: Optional[np.ndarray] = None,
) -> SelectiveMetrics:
    """
    Compute metrics at a specific target coverage.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted class labels, shape (n_samples,).

    labels : np.ndarray
        Ground truth labels, shape (n_samples,).

    scores : np.ndarray
        Selection scores, shape (n_samples,).

    target_coverage : float
        Desired coverage in [0, 1].

    confidences : np.ndarray, optional
        Confidence scores for ECE computation.

    Returns
    -------
    SelectiveMetrics
        Metrics computed at target coverage.
    """
    threshold = find_threshold_for_coverage(scores, target_coverage)
    is_accepted = scores >= threshold

    return compute_selective_metrics(
        predictions,
        labels,
        is_accepted,
        confidences=confidences,
        scores=scores,
        metadata={"target_coverage": target_coverage, "threshold": threshold},
    )


def validate_hypothesis_h3a(
    metrics: SelectiveMetrics,
    target_improvement: float = 0.04,
    target_coverage: float = 0.90,
) -> Dict[str, Any]:
    """
    Validate hypothesis H3a: ≥4pp improvement at 90% coverage.

    Parameters
    ----------
    metrics : SelectiveMetrics
        Computed selective metrics.

    target_improvement : float, optional (default=0.04)
        Minimum improvement required (4pp = 0.04).

    target_coverage : float, optional (default=0.90)
        Required coverage level.

    Returns
    -------
    dict
        Validation results including:
        - passed: bool
        - improvement: float
        - coverage: float
        - margin: float (how much above/below target)
    """
    passed = (
        metrics.improvement >= target_improvement
        and metrics.coverage >= target_coverage - 0.05  # Allow 5% tolerance
    )

    return {
        "hypothesis": "H3a",
        "description": f"≥{target_improvement*100:.0f}pp improvement at {target_coverage:.0%} coverage",
        "passed": passed,
        "improvement": metrics.improvement,
        "improvement_pp": metrics.improvement * 100,
        "coverage": metrics.coverage,
        "target_improvement": target_improvement,
        "target_coverage": target_coverage,
        "margin": metrics.improvement - target_improvement,
        "confidence_interval": metrics.confidence_intervals.get("improvement"),
    }


# ============================================================================
# MODULE EXPORTS
# ============================================================================


__all__ = [
    # Data classes
    "SelectiveMetrics",
    "RiskCoverageCurve",
    # Core metric functions
    "compute_coverage",
    "compute_selective_accuracy",
    "compute_selective_risk",
    "compute_risk_on_rejected",
    "compute_improvement",
    "compute_rejection_quality",
    "compute_rejection_precision_recall",
    # Risk-coverage
    "compute_risk_coverage_curve",
    "compute_aurc",
    # Calibration
    "compute_ece_post_selection",
    # Main entry point
    "compute_selective_metrics",
    # Comparison
    "compare_strategies",
    # Visualization
    "plot_risk_coverage_curve",
    "plot_accuracy_coverage_curve",
    "plot_strategy_comparison",
    # Utilities
    "find_threshold_for_coverage",
    "compute_metrics_at_coverage",
    "validate_hypothesis_h3a",
]
