"""
Statistical Tests Module for Comprehensive Model Evaluation.

This module provides production-grade statistical testing capabilities:
- Paired t-test for model comparison
- Cohen's d effect size computation
- Bootstrap confidence intervals
- McNemar's test for classifier comparison
- Wilcoxon signed-rank test
- Result formatting and reporting

Phase 9.1: Comprehensive Evaluation Infrastructure
Author: Viraj Jain
MSc Dissertation - University of Glasgow
Date: November 2024
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)

# ============================================================================
# CONSTANTS
# ============================================================================

DEFAULT_CONFIDENCE_LEVEL = 0.95
DEFAULT_N_BOOTSTRAP = 1000
DEFAULT_RANDOM_SEED = 42

# Effect size thresholds (Cohen, 1988)
EFFECT_SIZE_THRESHOLDS = {
    "negligible": 0.2,
    "small": 0.5,
    "medium": 0.8,
    "large": float("inf"),
}


# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class StatisticalTestResult:
    """
    Container for statistical test results.

    Attributes
    ----------
    test_name : str
        Name of the statistical test performed.
    statistic : float
        Test statistic value.
    p_value : float
        P-value for the test.
    significant : bool
        Whether result is statistically significant at alpha level.
    alpha : float
        Significance level used.
    effect_size : Optional[float]
        Effect size measure (e.g., Cohen's d).
    effect_size_interpretation : Optional[str]
        Interpretation of effect size (small/medium/large).
    confidence_interval : Optional[Tuple[float, float]]
        Confidence interval for the difference.
    additional_info : Dict[str, Any]
        Additional test-specific information.
    """

    test_name: str
    statistic: float
    p_value: float
    significant: bool
    alpha: float = 0.05
    effect_size: Optional[float] = None
    effect_size_interpretation: Optional[str] = None
    confidence_interval: Optional[Tuple[float, float]] = None
    additional_info: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "test_name": self.test_name,
            "statistic": (
                float(self.statistic) if not np.isnan(self.statistic) else None
            ),
            "p_value": float(self.p_value) if not np.isnan(self.p_value) else None,
            "significant": self.significant,
            "alpha": self.alpha,
            "effect_size": (
                float(self.effect_size) if self.effect_size is not None else None
            ),
            "effect_size_interpretation": self.effect_size_interpretation,
            "confidence_interval": (
                list(self.confidence_interval) if self.confidence_interval else None
            ),
            "additional_info": self.additional_info,
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            f"STATISTICAL TEST: {self.test_name}",
            "=" * 60,
            f"  Statistic: {self.statistic:.4f}",
            f"  P-value: {self.p_value:.4e}",
            f"  Significance level (α): {self.alpha}",
            f"  Significant: {'YES ✓' if self.significant else 'NO ✗'}",
        ]

        if self.effect_size is not None:
            lines.append(
                f"  Effect size: {self.effect_size:.4f} ({self.effect_size_interpretation})"
            )

        if self.confidence_interval is not None:
            lines.append(
                f"  95% CI: [{self.confidence_interval[0]:.4f}, {self.confidence_interval[1]:.4f}]"
            )

        lines.append("=" * 60)
        return "\n".join(lines)


@dataclass
class BootstrapResult:
    """
    Container for bootstrap analysis results.

    Attributes
    ----------
    point_estimate : float
        Original sample statistic.
    mean : float
        Mean of bootstrap distribution.
    std : float
        Standard deviation of bootstrap distribution.
    ci_lower : float
        Lower bound of confidence interval.
    ci_upper : float
        Upper bound of confidence interval.
    confidence_level : float
        Confidence level used.
    n_bootstrap : int
        Number of bootstrap samples.
    bootstrap_distribution : np.ndarray
        Full bootstrap distribution.
    """

    point_estimate: float
    mean: float
    std: float
    ci_lower: float
    ci_upper: float
    confidence_level: float = 0.95
    n_bootstrap: int = 1000
    bootstrap_distribution: np.ndarray = field(default_factory=lambda: np.array([]))

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary (excluding distribution for size)."""
        return {
            "point_estimate": float(self.point_estimate),
            "mean": float(self.mean),
            "std": float(self.std),
            "ci_lower": float(self.ci_lower),
            "ci_upper": float(self.ci_upper),
            "confidence_level": self.confidence_level,
            "n_bootstrap": self.n_bootstrap,
        }

    def summary(self) -> str:
        """Generate human-readable summary."""
        return (
            f"Point estimate: {self.point_estimate:.4f}\n"
            f"Bootstrap mean: {self.mean:.4f} ± {self.std:.4f}\n"
            f"{self.confidence_level*100:.0f}% CI: [{self.ci_lower:.4f}, {self.ci_upper:.4f}]"
        )


# ============================================================================
# COHEN'S D EFFECT SIZE
# ============================================================================


def compute_cohens_d(
    group1: np.ndarray, group2: np.ndarray, pooled: bool = True
) -> float:
    """
    Compute Cohen's d effect size between two groups.

    Cohen's d measures the standardized difference between two means.

    Parameters
    ----------
    group1 : np.ndarray
        First group of values.
    group2 : np.ndarray
        Second group of values.
    pooled : bool, optional (default=True)
        If True, use pooled standard deviation.
        If False, use group1's standard deviation.

    Returns
    -------
    float
        Cohen's d effect size.

    Notes
    -----
    Interpretation (Cohen, 1988):
    - |d| < 0.2: Negligible
    - 0.2 <= |d| < 0.5: Small
    - 0.5 <= |d| < 0.8: Medium
    - |d| >= 0.8: Large
    """
    mean1 = np.mean(group1)
    mean2 = np.mean(group2)

    if pooled:
        n1, n2 = len(group1), len(group2)
        var1 = np.var(group1, ddof=1)
        var2 = np.var(group2, ddof=1)

        # Pooled standard deviation
        pooled_std = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))

        if pooled_std == 0:
            return 0.0

        return float((mean1 - mean2) / pooled_std)
    else:
        std1 = np.std(group1, ddof=1)
        if std1 == 0:
            return 0.0
        return float((mean1 - mean2) / std1)


def interpret_effect_size(d: float) -> str:
    """
    Interpret Cohen's d effect size.

    Parameters
    ----------
    d : float
        Cohen's d value.

    Returns
    -------
    str
        Interpretation: 'negligible', 'small', 'medium', or 'large'.
    """
    abs_d = abs(d)

    if abs_d < EFFECT_SIZE_THRESHOLDS["negligible"]:
        return "negligible"
    elif abs_d < EFFECT_SIZE_THRESHOLDS["small"]:
        return "small"
    elif abs_d < EFFECT_SIZE_THRESHOLDS["medium"]:
        return "medium"
    else:
        return "large"


def compute_glass_delta(treatment: np.ndarray, control: np.ndarray) -> float:
    """
    Compute Glass's delta effect size.

    Uses control group's standard deviation as denominator.
    Preferred when group variances are unequal.

    Parameters
    ----------
    treatment : np.ndarray
        Treatment group values.
    control : np.ndarray
        Control group values.

    Returns
    -------
    float
        Glass's delta effect size.
    """
    std_control = np.std(control, ddof=1)

    if std_control == 0:
        return 0.0

    return float((np.mean(treatment) - np.mean(control)) / std_control)


def compute_hedges_g(group1: np.ndarray, group2: np.ndarray) -> float:
    """
    Compute Hedges' g effect size (bias-corrected Cohen's d).

    Hedges' g applies a correction factor for small sample sizes.

    Parameters
    ----------
    group1 : np.ndarray
        First group of values.
    group2 : np.ndarray
        Second group of values.

    Returns
    -------
    float
        Hedges' g effect size.
    """
    d = compute_cohens_d(group1, group2, pooled=True)
    n1, n2 = len(group1), len(group2)

    # Correction factor for small samples
    df = n1 + n2 - 2
    correction = 1 - (3 / (4 * df - 1))

    return float(d * correction)


# ============================================================================
# PAIRED T-TEST
# ============================================================================


def paired_t_test(
    values1: np.ndarray,
    values2: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> StatisticalTestResult:
    """
    Perform paired t-test between two related samples.

    Used to compare two models on the same test set (paired observations).

    Parameters
    ----------
    values1 : np.ndarray
        First set of values (e.g., model 1 accuracies per sample).
    values2 : np.ndarray
        Second set of values (e.g., model 2 accuracies per sample).
    alpha : float, optional (default=0.05)
        Significance level.
    alternative : str, optional (default='two-sided')
        'two-sided', 'greater', or 'less'.

    Returns
    -------
    StatisticalTestResult
        Complete test results including effect size and CI.

    Raises
    ------
    ValueError
        If arrays have different lengths or insufficient samples.
    """
    values1 = np.asarray(values1).flatten()
    values2 = np.asarray(values2).flatten()

    if len(values1) != len(values2):
        raise ValueError(
            f"Arrays must have same length: {len(values1)} vs {len(values2)}"
        )

    if len(values1) < 2:
        raise ValueError("Need at least 2 paired samples")

    # Compute differences
    differences = values1 - values2

    # Perform paired t-test
    statistic, p_value = stats.ttest_rel(values1, values2, alternative=alternative)

    # Compute effect size (Cohen's d for paired samples)
    mean_diff = np.mean(differences)
    std_diff = np.std(differences, ddof=1)

    if std_diff > 0:
        d = mean_diff / std_diff
    else:
        d = 0.0

    # Confidence interval for mean difference
    n = len(differences)
    se = std_diff / np.sqrt(n)
    t_crit = stats.t.ppf(1 - alpha / 2, df=n - 1)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    return StatisticalTestResult(
        test_name="Paired t-test",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=float(d),
        effect_size_interpretation=interpret_effect_size(d),
        confidence_interval=(float(ci_lower), float(ci_upper)),
        additional_info={
            "mean_difference": float(mean_diff),
            "std_difference": float(std_diff),
            "n_samples": n,
            "alternative": alternative,
            "degrees_of_freedom": n - 1,
        },
    )


def independent_t_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
    equal_var: bool = True,
    alternative: str = "two-sided",
) -> StatisticalTestResult:
    """
    Perform independent samples t-test.

    Used to compare two independent groups.

    Parameters
    ----------
    group1 : np.ndarray
        First group of values.
    group2 : np.ndarray
        Second group of values.
    alpha : float, optional (default=0.05)
        Significance level.
    equal_var : bool, optional (default=True)
        If True, assumes equal variances (Student's t).
        If False, uses Welch's t-test.
    alternative : str, optional (default='two-sided')
        'two-sided', 'greater', or 'less'.

    Returns
    -------
    StatisticalTestResult
        Complete test results.
    """
    group1 = np.asarray(group1).flatten()
    group2 = np.asarray(group2).flatten()

    # Perform t-test
    statistic, p_value = stats.ttest_ind(
        group1, group2, equal_var=equal_var, alternative=alternative
    )

    # Effect size
    d = compute_cohens_d(group1, group2, pooled=True)

    # Mean difference CI
    mean_diff = np.mean(group1) - np.mean(group2)
    n1, n2 = len(group1), len(group2)

    if equal_var:
        # Pooled variance
        sp2 = (
            (n1 - 1) * np.var(group1, ddof=1) + (n2 - 1) * np.var(group2, ddof=1)
        ) / (n1 + n2 - 2)
        se = np.sqrt(sp2 * (1 / n1 + 1 / n2))
        df = n1 + n2 - 2
    else:
        # Welch's approximation
        var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
        se = np.sqrt(var1 / n1 + var2 / n2)
        # Welch-Satterthwaite degrees of freedom
        num = (var1 / n1 + var2 / n2) ** 2
        denom = (var1 / n1) ** 2 / (n1 - 1) + (var2 / n2) ** 2 / (n2 - 1)
        df = num / denom if denom > 0 else n1 + n2 - 2

    t_crit = stats.t.ppf(1 - alpha / 2, df=df)
    ci_lower = mean_diff - t_crit * se
    ci_upper = mean_diff + t_crit * se

    return StatisticalTestResult(
        test_name="Independent t-test" + (" (Welch)" if not equal_var else ""),
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=float(d),
        effect_size_interpretation=interpret_effect_size(d),
        confidence_interval=(float(ci_lower), float(ci_upper)),
        additional_info={
            "mean_group1": float(np.mean(group1)),
            "mean_group2": float(np.mean(group2)),
            "mean_difference": float(mean_diff),
            "n_group1": n1,
            "n_group2": n2,
            "degrees_of_freedom": float(df),
            "equal_variance_assumed": equal_var,
        },
    )


# ============================================================================
# MCNEMAR'S TEST
# ============================================================================


def mcnemars_test(
    predictions1: np.ndarray,
    predictions2: np.ndarray,
    labels: np.ndarray,
    alpha: float = 0.05,
    correction: bool = True,
) -> StatisticalTestResult:
    """
    Perform McNemar's test for comparing two classifiers.

    McNemar's test determines if there's a significant difference between
    the predictions of two classifiers on the same dataset.

    Parameters
    ----------
    predictions1 : np.ndarray
        Predictions from classifier 1.
    predictions2 : np.ndarray
        Predictions from classifier 2.
    labels : np.ndarray
        Ground truth labels.
    alpha : float, optional (default=0.05)
        Significance level.
    correction : bool, optional (default=True)
        Apply Edwards' continuity correction.

    Returns
    -------
    StatisticalTestResult
        Complete test results.

    Notes
    -----
    The test uses a 2x2 contingency table:
    - n00: Both wrong
    - n01: Classifier 1 wrong, 2 correct
    - n10: Classifier 1 correct, 2 wrong
    - n11: Both correct

    The test focuses on discordant pairs (n01 and n10).
    """
    predictions1 = np.asarray(predictions1).flatten()
    predictions2 = np.asarray(predictions2).flatten()
    labels = np.asarray(labels).flatten()

    # Correctness arrays
    correct1 = predictions1 == labels
    correct2 = predictions2 == labels

    # Contingency table
    n00 = np.sum(~correct1 & ~correct2)  # Both wrong
    n01 = np.sum(~correct1 & correct2)  # 1 wrong, 2 correct
    n10 = np.sum(correct1 & ~correct2)  # 1 correct, 2 wrong
    n11 = np.sum(correct1 & correct2)  # Both correct

    # McNemar's statistic
    if correction:
        # Edwards' continuity correction
        if n01 + n10 == 0:
            chi2 = 0.0
        else:
            chi2 = (abs(n01 - n10) - 1) ** 2 / (n01 + n10)
    else:
        if n01 + n10 == 0:
            chi2 = 0.0
        else:
            chi2 = (n01 - n10) ** 2 / (n01 + n10)

    # P-value (chi-squared distribution with 1 df)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)

    # Odds ratio for effect size
    if n10 > 0:
        odds_ratio = n01 / n10
    else:
        odds_ratio = float("inf") if n01 > 0 else 1.0

    return StatisticalTestResult(
        test_name="McNemar's test" + (" (corrected)" if correction else ""),
        statistic=float(chi2),
        p_value=float(p_value),
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=float(odds_ratio) if np.isfinite(odds_ratio) else None,
        effect_size_interpretation=None,  # Odds ratio doesn't use Cohen's scale
        additional_info={
            "contingency_table": {
                "both_wrong": int(n00),
                "model1_wrong_model2_correct": int(n01),
                "model1_correct_model2_wrong": int(n10),
                "both_correct": int(n11),
            },
            "discordant_pairs": int(n01 + n10),
            "odds_ratio": float(odds_ratio) if np.isfinite(odds_ratio) else "inf",
            "accuracy_model1": float(correct1.mean()),
            "accuracy_model2": float(correct2.mean()),
            "continuity_correction": correction,
        },
    )


# ============================================================================
# WILCOXON SIGNED-RANK TEST
# ============================================================================


def wilcoxon_signed_rank_test(
    values1: np.ndarray,
    values2: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> StatisticalTestResult:
    """
    Perform Wilcoxon signed-rank test for paired samples.

    Non-parametric alternative to paired t-test when normality assumption
    is violated.

    Parameters
    ----------
    values1 : np.ndarray
        First set of paired values.
    values2 : np.ndarray
        Second set of paired values.
    alpha : float, optional (default=0.05)
        Significance level.
    alternative : str, optional (default='two-sided')
        'two-sided', 'greater', or 'less'.

    Returns
    -------
    StatisticalTestResult
        Complete test results.
    """
    values1 = np.asarray(values1).flatten()
    values2 = np.asarray(values2).flatten()

    # Remove ties (pairs where difference is 0)
    differences = values1 - values2
    non_zero = differences != 0

    if np.sum(non_zero) < 2:
        return StatisticalTestResult(
            test_name="Wilcoxon signed-rank test",
            statistic=float("nan"),
            p_value=1.0,
            significant=False,
            alpha=alpha,
            additional_info={"error": "Insufficient non-zero differences"},
        )

    # Perform test
    statistic, p_value = stats.wilcoxon(
        values1, values2, alternative=alternative, zero_method="wilcox"
    )

    # Effect size: r = Z / sqrt(N)
    n = len(values1)
    z = stats.norm.ppf(1 - p_value / 2) if p_value < 1 else 0
    r = abs(z) / np.sqrt(n) if n > 0 else 0

    # Interpret r as effect size
    if r < 0.1:
        r_interp = "negligible"
    elif r < 0.3:
        r_interp = "small"
    elif r < 0.5:
        r_interp = "medium"
    else:
        r_interp = "large"

    return StatisticalTestResult(
        test_name="Wilcoxon signed-rank test",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=float(r),
        effect_size_interpretation=r_interp,
        additional_info={
            "n_pairs": n,
            "n_non_zero_differences": int(np.sum(non_zero)),
            "median_difference": float(np.median(differences)),
            "alternative": alternative,
        },
    )


# ============================================================================
# MANN-WHITNEY U TEST
# ============================================================================


def mann_whitney_u_test(
    group1: np.ndarray,
    group2: np.ndarray,
    alpha: float = 0.05,
    alternative: str = "two-sided",
) -> StatisticalTestResult:
    """
    Perform Mann-Whitney U test for independent samples.

    Non-parametric alternative to independent t-test.

    Parameters
    ----------
    group1 : np.ndarray
        First group of values.
    group2 : np.ndarray
        Second group of values.
    alpha : float, optional (default=0.05)
        Significance level.
    alternative : str, optional (default='two-sided')
        'two-sided', 'greater', or 'less'.

    Returns
    -------
    StatisticalTestResult
        Complete test results.
    """
    group1 = np.asarray(group1).flatten()
    group2 = np.asarray(group2).flatten()

    statistic, p_value = stats.mannwhitneyu(group1, group2, alternative=alternative)

    # Effect size: r = Z / sqrt(N)
    n1, n2 = len(group1), len(group2)
    n_total = n1 + n2
    z = stats.norm.ppf(1 - p_value / 2) if p_value < 1 else 0
    r = abs(z) / np.sqrt(n_total) if n_total > 0 else 0

    # Interpret r
    if r < 0.1:
        r_interp = "negligible"
    elif r < 0.3:
        r_interp = "small"
    elif r < 0.5:
        r_interp = "medium"
    else:
        r_interp = "large"

    return StatisticalTestResult(
        test_name="Mann-Whitney U test",
        statistic=float(statistic),
        p_value=float(p_value),
        significant=p_value < alpha,
        alpha=alpha,
        effect_size=float(r),
        effect_size_interpretation=r_interp,
        additional_info={
            "n_group1": n1,
            "n_group2": n2,
            "median_group1": float(np.median(group1)),
            "median_group2": float(np.median(group2)),
            "alternative": alternative,
        },
    )


# ============================================================================
# BOOTSTRAP CONFIDENCE INTERVALS
# ============================================================================


def bootstrap_confidence_interval(
    data: np.ndarray,
    statistic_fn: callable = np.mean,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
    method: str = "percentile",
) -> BootstrapResult:
    """
    Compute bootstrap confidence interval for a statistic.

    Parameters
    ----------
    data : np.ndarray
        Data to bootstrap.
    statistic_fn : callable, optional (default=np.mean)
        Function to compute the statistic.
    n_bootstrap : int, optional (default=1000)
        Number of bootstrap samples.
    confidence_level : float, optional (default=0.95)
        Confidence level for CI.
    random_seed : int, optional
        Random seed for reproducibility.
    method : str, optional (default='percentile')
        Method for CI: 'percentile', 'bca', or 'basic'.

    Returns
    -------
    BootstrapResult
        Bootstrap analysis results.
    """
    data = np.asarray(data).flatten()
    n = len(data)

    rng = np.random.RandomState(random_seed)

    # Original statistic
    point_estimate = float(statistic_fn(data))

    # Bootstrap resampling
    bootstrap_stats = []
    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        sample = data[indices]
        try:
            stat = statistic_fn(sample)
            if np.isfinite(stat):
                bootstrap_stats.append(stat)
        except Exception:
            continue

    bootstrap_stats = np.array(bootstrap_stats)

    if len(bootstrap_stats) == 0:
        return BootstrapResult(
            point_estimate=point_estimate,
            mean=float("nan"),
            std=float("nan"),
            ci_lower=float("nan"),
            ci_upper=float("nan"),
            confidence_level=confidence_level,
            n_bootstrap=0,
            bootstrap_distribution=np.array([]),
        )

    alpha = 1 - confidence_level

    if method == "percentile":
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))
    elif method == "basic":
        # Basic bootstrap CI
        ci_lower = 2 * point_estimate - np.percentile(
            bootstrap_stats, 100 * (1 - alpha / 2)
        )
        ci_upper = 2 * point_estimate - np.percentile(bootstrap_stats, 100 * alpha / 2)
    else:
        # Default to percentile
        ci_lower = np.percentile(bootstrap_stats, 100 * alpha / 2)
        ci_upper = np.percentile(bootstrap_stats, 100 * (1 - alpha / 2))

    return BootstrapResult(
        point_estimate=point_estimate,
        mean=float(np.mean(bootstrap_stats)),
        std=float(np.std(bootstrap_stats)),
        ci_lower=float(ci_lower),
        ci_upper=float(ci_upper),
        confidence_level=confidence_level,
        n_bootstrap=len(bootstrap_stats),
        bootstrap_distribution=bootstrap_stats,
    )


def bootstrap_paired_difference(
    values1: np.ndarray,
    values2: np.ndarray,
    statistic_fn: callable = np.mean,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
) -> BootstrapResult:
    """
    Compute bootstrap CI for the difference between paired samples.

    Parameters
    ----------
    values1 : np.ndarray
        First set of paired values.
    values2 : np.ndarray
        Second set of paired values.
    statistic_fn : callable, optional (default=np.mean)
        Function to compute statistic on differences.
    n_bootstrap : int, optional (default=1000)
        Number of bootstrap samples.
    confidence_level : float, optional (default=0.95)
        Confidence level.
    random_seed : int, optional
        Random seed.

    Returns
    -------
    BootstrapResult
        Bootstrap results for the difference.
    """
    values1 = np.asarray(values1).flatten()
    values2 = np.asarray(values2).flatten()

    differences = values1 - values2

    return bootstrap_confidence_interval(
        differences,
        statistic_fn=statistic_fn,
        n_bootstrap=n_bootstrap,
        confidence_level=confidence_level,
        random_seed=random_seed,
    )


def bootstrap_metric_comparison(
    predictions1: np.ndarray,
    predictions2: np.ndarray,
    labels: np.ndarray,
    metric_fn: callable,
    n_bootstrap: int = DEFAULT_N_BOOTSTRAP,
    confidence_level: float = DEFAULT_CONFIDENCE_LEVEL,
    random_seed: Optional[int] = DEFAULT_RANDOM_SEED,
) -> Dict[str, Any]:
    """
    Bootstrap comparison of two models on a metric.

    Parameters
    ----------
    predictions1 : np.ndarray
        Predictions from model 1.
    predictions2 : np.ndarray
        Predictions from model 2.
    labels : np.ndarray
        Ground truth labels.
    metric_fn : callable
        Function that takes (predictions, labels) and returns a metric.
    n_bootstrap : int, optional
        Number of bootstrap samples.
    confidence_level : float, optional
        Confidence level.
    random_seed : int, optional
        Random seed.

    Returns
    -------
    dict
        Comparison results with CIs for both models and difference.
    """
    predictions1 = np.asarray(predictions1)
    predictions2 = np.asarray(predictions2)
    labels = np.asarray(labels)

    n = len(labels)
    rng = np.random.RandomState(random_seed)

    # Original metrics
    metric1 = metric_fn(predictions1, labels)
    metric2 = metric_fn(predictions2, labels)

    # Bootstrap
    boot_metrics1 = []
    boot_metrics2 = []
    boot_diffs = []

    for _ in range(n_bootstrap):
        indices = rng.choice(n, size=n, replace=True)
        try:
            m1 = metric_fn(predictions1[indices], labels[indices])
            m2 = metric_fn(predictions2[indices], labels[indices])
            if np.isfinite(m1) and np.isfinite(m2):
                boot_metrics1.append(m1)
                boot_metrics2.append(m2)
                boot_diffs.append(m1 - m2)
        except Exception:
            continue

    alpha = 1 - confidence_level

    return {
        "model1": {
            "value": float(metric1),
            "ci_lower": float(np.percentile(boot_metrics1, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(boot_metrics1, 100 * (1 - alpha / 2))),
        },
        "model2": {
            "value": float(metric2),
            "ci_lower": float(np.percentile(boot_metrics2, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(boot_metrics2, 100 * (1 - alpha / 2))),
        },
        "difference": {
            "value": float(metric1 - metric2),
            "ci_lower": float(np.percentile(boot_diffs, 100 * alpha / 2)),
            "ci_upper": float(np.percentile(boot_diffs, 100 * (1 - alpha / 2))),
            "significant": (
                np.percentile(boot_diffs, 100 * alpha / 2) > 0
                or np.percentile(boot_diffs, 100 * (1 - alpha / 2)) < 0
            ),
        },
        "n_bootstrap": len(boot_diffs),
        "confidence_level": confidence_level,
    }


# ============================================================================
# MULTIPLE COMPARISON CORRECTION
# ============================================================================


def bonferroni_correction(p_values: List[float], alpha: float = 0.05) -> Dict[str, Any]:
    """
    Apply Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    p_values : List[float]
        List of p-values from multiple tests.
    alpha : float, optional (default=0.05)
        Family-wise error rate.

    Returns
    -------
    dict
        Corrected results.
    """
    n = len(p_values)
    corrected_alpha = alpha / n

    significant = [p < corrected_alpha for p in p_values]

    return {
        "original_alpha": alpha,
        "corrected_alpha": corrected_alpha,
        "n_tests": n,
        "p_values": p_values,
        "significant": significant,
        "n_significant": sum(significant),
    }


def benjamini_hochberg_correction(
    p_values: List[float], alpha: float = 0.05
) -> Dict[str, Any]:
    """
    Apply Benjamini-Hochberg procedure for FDR control.

    Parameters
    ----------
    p_values : List[float]
        List of p-values.
    alpha : float, optional (default=0.05)
        False discovery rate threshold.

    Returns
    -------
    dict
        Corrected results with adjusted p-values.
    """
    n = len(p_values)
    p_values = np.asarray(p_values)

    # Sort p-values
    sorted_indices = np.argsort(p_values)
    sorted_p = p_values[sorted_indices]

    # Compute BH critical values
    bh_critical = (np.arange(1, n + 1) / n) * alpha

    # Find largest k where p_k <= k/n * alpha
    significant_mask = sorted_p <= bh_critical

    if np.any(significant_mask):
        max_k = np.max(np.where(significant_mask)[0])
        # All tests with rank <= max_k are significant
        significant_sorted = np.zeros(n, dtype=bool)
        significant_sorted[: max_k + 1] = True
    else:
        significant_sorted = np.zeros(n, dtype=bool)

    # Reorder to original order
    significant = np.zeros(n, dtype=bool)
    significant[sorted_indices] = significant_sorted

    # Compute adjusted p-values (BH-adjusted)
    adjusted_p = np.zeros(n)
    adjusted_p[sorted_indices[-1]] = sorted_p[-1]

    for i in range(n - 2, -1, -1):
        idx = sorted_indices[i]
        adjusted_p[idx] = min(
            adjusted_p[sorted_indices[i + 1]], sorted_p[i] * n / (i + 1)
        )

    adjusted_p = np.minimum(adjusted_p, 1.0)

    return {
        "original_alpha": alpha,
        "n_tests": n,
        "p_values": p_values.tolist(),
        "adjusted_p_values": adjusted_p.tolist(),
        "significant": significant.tolist(),
        "n_significant": int(np.sum(significant)),
    }


# ============================================================================
# RESULT FORMATTING AND REPORTING
# ============================================================================


def format_p_value(p: float, threshold: float = 0.001) -> str:
    """Format p-value for display."""
    if p < threshold:
        return f"< {threshold}"
    else:
        return f"{p:.4f}"


def format_ci(ci: Tuple[float, float], decimals: int = 4) -> str:
    """Format confidence interval for display."""
    return f"[{ci[0]:.{decimals}f}, {ci[1]:.{decimals}f}]"


def generate_comparison_report(
    results: List[StatisticalTestResult],
    model_names: Optional[List[str]] = None,
    title: str = "Statistical Comparison Report",
) -> str:
    """
    Generate a formatted comparison report.

    Parameters
    ----------
    results : List[StatisticalTestResult]
        List of test results.
    model_names : List[str], optional
        Names of models being compared.
    title : str, optional
        Report title.

    Returns
    -------
    str
        Formatted report.
    """
    lines = [
        "=" * 80,
        title.center(80),
        "=" * 80,
        "",
    ]

    for i, result in enumerate(results):
        if model_names and len(model_names) > i:
            lines.append(f"Comparison {i + 1}: {model_names[i]}")
        else:
            lines.append(f"Test {i + 1}: {result.test_name}")

        lines.append("-" * 40)
        lines.append(f"  Statistic: {result.statistic:.4f}")
        lines.append(f"  P-value: {format_p_value(result.p_value)}")
        lines.append(
            f"  Significant (α={result.alpha}): {'Yes ✓' if result.significant else 'No ✗'}"
        )

        if result.effect_size is not None:
            lines.append(
                f"  Effect size: {result.effect_size:.4f} ({result.effect_size_interpretation})"
            )

        if result.confidence_interval is not None:
            lines.append(f"  95% CI: {format_ci(result.confidence_interval)}")

        lines.append("")

    # Summary
    n_significant = sum(1 for r in results if r.significant)
    lines.extend(
        [
            "-" * 80,
            "SUMMARY",
            "-" * 80,
            f"  Total comparisons: {len(results)}",
            f"  Significant results: {n_significant} ({n_significant/len(results)*100:.1f}%)",
            "=" * 80,
        ]
    )

    return "\n".join(lines)


def save_results(
    results: Union[StatisticalTestResult, List[StatisticalTestResult]],
    filepath: Union[str, Path],
    format: str = "json",
) -> None:
    """
    Save statistical test results to file.

    Parameters
    ----------
    results : StatisticalTestResult or List
        Results to save.
    filepath : str or Path
        Output file path.
    format : str, optional (default='json')
        Output format: 'json' or 'txt'.
    """
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    if isinstance(results, StatisticalTestResult):
        results = [results]

    if format == "json":
        data = [r.to_dict() for r in results]
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
    else:
        report = generate_comparison_report(results)
        with open(filepath, "w") as f:
            f.write(report)

    logger.info(f"Results saved to {filepath}")


# ============================================================================
# COMPREHENSIVE MODEL COMPARISON
# ============================================================================


def comprehensive_model_comparison(
    predictions1: np.ndarray,
    predictions2: np.ndarray,
    labels: np.ndarray,
    model1_name: str = "Model 1",
    model2_name: str = "Model 2",
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """
    Perform comprehensive statistical comparison between two models.

    Runs multiple tests to robustly compare model performance.

    Parameters
    ----------
    predictions1 : np.ndarray
        Predictions from model 1.
    predictions2 : np.ndarray
        Predictions from model 2.
    labels : np.ndarray
        Ground truth labels.
    model1_name : str, optional
        Name of model 1.
    model2_name : str, optional
        Name of model 2.
    alpha : float, optional (default=0.05)
        Significance level.

    Returns
    -------
    dict
        Comprehensive comparison results.
    """
    predictions1 = np.asarray(predictions1).flatten()
    predictions2 = np.asarray(predictions2).flatten()
    labels = np.asarray(labels).flatten()

    # Per-sample correctness
    correct1 = (predictions1 == labels).astype(float)
    correct2 = (predictions2 == labels).astype(float)

    results = {
        "model1_name": model1_name,
        "model2_name": model2_name,
        "n_samples": len(labels),
        "accuracy_model1": float(correct1.mean()),
        "accuracy_model2": float(correct2.mean()),
        "accuracy_difference": float(correct1.mean() - correct2.mean()),
    }

    # McNemar's test
    mcnemar_result = mcnemars_test(predictions1, predictions2, labels, alpha=alpha)
    results["mcnemars_test"] = mcnemar_result.to_dict()

    # Paired t-test on correctness
    if len(correct1) > 1:
        ttest_result = paired_t_test(correct1, correct2, alpha=alpha)
        results["paired_t_test"] = ttest_result.to_dict()

    # Wilcoxon test
    if len(correct1) > 1:
        wilcoxon_result = wilcoxon_signed_rank_test(correct1, correct2, alpha=alpha)
        results["wilcoxon_test"] = wilcoxon_result.to_dict()

    # Bootstrap CI for accuracy difference
    boot_diff = bootstrap_paired_difference(correct1, correct2)
    results["bootstrap_accuracy_difference"] = boot_diff.to_dict()

    # Effect size
    if np.std(correct1 - correct2) > 0:
        d = compute_cohens_d(correct1, correct2)
        results["cohens_d"] = float(d)
        results["effect_size_interpretation"] = interpret_effect_size(d)

    # Overall conclusion
    tests_significant = []
    if "mcnemars_test" in results:
        tests_significant.append(results["mcnemars_test"]["significant"])
    if "paired_t_test" in results:
        tests_significant.append(results["paired_t_test"]["significant"])
    if "wilcoxon_test" in results:
        tests_significant.append(results["wilcoxon_test"]["significant"])

    results["any_test_significant"] = any(tests_significant)
    results["all_tests_significant"] = (
        all(tests_significant) if tests_significant else False
    )

    return results


# ============================================================================
# MODULE EXPORTS
# ============================================================================


__all__ = [
    # Data classes
    "StatisticalTestResult",
    "BootstrapResult",
    # Effect size
    "compute_cohens_d",
    "interpret_effect_size",
    "compute_glass_delta",
    "compute_hedges_g",
    # Parametric tests
    "paired_t_test",
    "independent_t_test",
    # Non-parametric tests
    "mcnemars_test",
    "wilcoxon_signed_rank_test",
    "mann_whitney_u_test",
    # Bootstrap
    "bootstrap_confidence_interval",
    "bootstrap_paired_difference",
    "bootstrap_metric_comparison",
    # Multiple comparisons
    "bonferroni_correction",
    "benjamini_hochberg_correction",
    # Reporting
    "format_p_value",
    "format_ci",
    "generate_comparison_report",
    "save_results",
    # Comprehensive
    "comprehensive_model_comparison",
]
