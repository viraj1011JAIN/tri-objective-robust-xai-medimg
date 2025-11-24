"""
RQ1 Cross-Site Generalization Hypothesis Test
==============================================

Research Question 1 (RQ1): Can adversarial robustness and cross-site
generalization be jointly optimized?

Hypothesis H1c: PGD-AT alone does NOT improve cross-site generalization.
    - Baseline cross-site drop: ~15pp
    - PGD-AT cross-site drop: ~15pp (no significant difference)
    - Tri-objective cross-site drop: <8pp (significant improvement)

This test is CRITICAL for your dissertation because it validates the
need for your tri-objective approach.

Statistical Approach:
1. Compute AUROC drops for each method
2. Paired t-test (same test sets across methods)
3. Effect size (Cohen's d)
4. Confidence intervals (bootstrap for n=3)
5. Bonferroni correction (testing 4 datasets)
6. Power analysis

Author: Fixed by Ruthless Mentor
Date: November 24, 2025
Version: 5.2.2
"""

from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests


def compute_auroc_drop(source_test_auroc: float, target_test_auroc: float) -> float:
    """
    Compute AUROC drop from source to target domain.

    Args:
        source_test_auroc: AUROC on source domain test set
        target_test_auroc: AUROC on target domain test set

    Returns:
        AUROC drop in percentage points (positive = degradation)

    Example:
        >>> compute_auroc_drop(85.0, 70.0)
        15.0  # 15pp drop
    """
    return source_test_auroc - target_test_auroc


def compute_cross_site_drops(
    source_aurocs: List[float], target_aurocs_dict: Dict[str, List[float]]
) -> Dict[str, List[float]]:
    """
    Compute AUROC drops for multiple target datasets.

    Args:
        source_aurocs: Source test AUROCs (one per seed)
        target_aurocs_dict: Dict of {target_name: [aurocs per seed]}

    Returns:
        Dict of {target_name: [drops per seed]}

    Example:
        >>> source = [85.0, 86.0, 85.5]
        >>> targets = {
        ...     'isic2019': [70.0, 71.0, 70.5],
        ...     'isic2020': [68.0, 69.0, 68.5]
        ... }
        >>> compute_cross_site_drops(source, targets)
        {
            'isic2019': [15.0, 15.0, 15.0],
            'isic2020': [17.0, 17.0, 17.0]
        }
    """
    drops = {}

    for target_name, target_aurocs in target_aurocs_dict.items():
        if len(target_aurocs) != len(source_aurocs):
            raise ValueError(
                f"Mismatched seeds: source has {len(source_aurocs)}, "
                f"{target_name} has {len(target_aurocs)}"
            )

        target_drops = [
            compute_auroc_drop(src, tgt)
            for src, tgt in zip(source_aurocs, target_aurocs)
        ]
        drops[target_name] = target_drops

    return drops


def test_rq1_hypothesis(
    baseline_drops: Dict[str, List[float]],
    pgd_at_drops: Dict[str, List[float]],
    alpha: float = 0.01,
    bonferroni_correction: bool = True,
) -> Dict:
    """
    Test H1c: PGD-AT does NOT improve cross-site generalization.

    This is the CORE test for RQ1. We want to show that:
    - PGD-AT improves robustness (tested elsewhere)
    - PGD-AT does NOT improve cross-site (tested here)
    - Therefore, need tri-objective approach

    Args:
        baseline_drops: {target_name: [drops per seed]} for baseline
        pgd_at_drops: {target_name: [drops per seed]} for PGD-AT
        alpha: Significance level (0.01 for your work)
        bonferroni_correction: Apply Bonferroni correction for multiple tests

    Returns:
        Comprehensive statistical report

    Expected Result for H1c:
        p_value > 0.05 (no significant difference)
        → Confirms PGD-AT doesn't help cross-site
        → Justifies need for tri-objective
    """
    # Validate inputs
    if set(baseline_drops.keys()) != set(pgd_at_drops.keys()):
        raise ValueError("Baseline and PGD-AT must test same datasets")

    target_datasets = list(baseline_drops.keys())
    n_tests = len(target_datasets)

    # Apply Bonferroni correction if requested
    corrected_alpha = alpha / n_tests if bonferroni_correction else alpha

    # Test each target dataset
    per_dataset_results = {}
    all_p_values = []

    for target_name in target_datasets:
        baseline_drop = np.array(baseline_drops[target_name])
        pgd_at_drop = np.array(pgd_at_drops[target_name])

        # Paired t-test (same test sets)
        t_stat, p_value = stats.ttest_rel(baseline_drop, pgd_at_drop)
        all_p_values.append(p_value)

        # Effect size (Cohen's d)
        mean_diff = np.mean(pgd_at_drop) - np.mean(baseline_drop)
        pooled_std = np.sqrt(
            (np.var(pgd_at_drop, ddof=1) + np.var(baseline_drop, ddof=1)) / 2
        )
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

        # Bootstrap confidence interval (critical with n=3)
        ci = bootstrap_confidence_interval(
            pgd_at_drop, baseline_drop, n_bootstrap=10000
        )

        # Interpretation
        significant = p_value < corrected_alpha

        per_dataset_results[target_name] = {
            "baseline_drop_mean": float(np.mean(baseline_drop)),
            "baseline_drop_std": float(np.std(baseline_drop, ddof=1)),
            "pgd_at_drop_mean": float(np.mean(pgd_at_drop)),
            "pgd_at_drop_std": float(np.std(pgd_at_drop, ddof=1)),
            "difference_mean": float(mean_diff),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "p_value_corrected": float(corrected_alpha),
            "significant": significant,
            "cohens_d": float(cohens_d),
            "effect_interpretation": interpret_effect_size(cohens_d),
            "ci_95_lower": ci["lower"],
            "ci_95_upper": ci["upper"],
            "ci_contains_zero": ci["contains_zero"],
        }

    # Apply Bonferroni correction to all p-values
    if bonferroni_correction:
        reject, pvals_corrected, _, _ = multipletests(
            all_p_values, alpha=alpha, method="bonferroni"
        )

        for i, target_name in enumerate(target_datasets):
            per_dataset_results[target_name]["p_value_bonferroni"] = float(
                pvals_corrected[i]
            )
            per_dataset_results[target_name]["significant_bonferroni"] = bool(reject[i])

    # Overall H1c test
    # H1c: PGD-AT does NOT improve cross-site
    # Expected: p > 0.05 for all datasets (no significant difference)

    all_p_values_arr = np.array(all_p_values)
    any_significant = np.any(all_p_values_arr < corrected_alpha)

    if bonferroni_correction:
        any_significant_bonf = np.any(reject)
        h1c_confirmed = not any_significant_bonf
    else:
        h1c_confirmed = not any_significant

    # Power analysis (did we have enough seeds?)
    power_analyses = {}
    for target_name in target_datasets:
        result = per_dataset_results[target_name]
        power = compute_statistical_power(
            effect_size=result["cohens_d"],
            n_per_group=len(baseline_drops[target_name]),
            alpha=corrected_alpha,
        )
        power_analyses[target_name] = float(power)

    # Generate interpretation
    interpretation = generate_h1c_interpretation(
        h1c_confirmed=h1c_confirmed,
        per_dataset_results=per_dataset_results,
        alpha=alpha,
        bonferroni_correction=bonferroni_correction,
    )

    return {
        "hypothesis": "H1c: PGD-AT does NOT improve cross-site generalization",
        "h1c_confirmed": h1c_confirmed,
        "alpha": alpha,
        "bonferroni_correction": bonferroni_correction,
        "n_datasets": n_tests,
        "per_dataset_results": per_dataset_results,
        "power_analyses": power_analyses,
        "interpretation": interpretation,
        "recommendations": generate_recommendations(
            h1c_confirmed=h1c_confirmed, power_analyses=power_analyses
        ),
    }


def bootstrap_confidence_interval(
    group1: np.ndarray,
    group2: np.ndarray,
    n_bootstrap: int = 10000,
    confidence: float = 0.95,
) -> Dict[str, float]:
    """
    Compute bootstrap confidence interval for difference in means.

    Critical with n=3 seeds (parametric CI unreliable).

    Args:
        group1: First group (e.g., PGD-AT drops)
        group2: Second group (e.g., baseline drops)
        n_bootstrap: Number of bootstrap iterations
        confidence: Confidence level (0.95 for 95% CI)

    Returns:
        Dict with lower, upper, mean, contains_zero
    """
    differences = []

    np.random.seed(42)  # Reproducibility
    for _ in range(n_bootstrap):
        # Resample with replacement
        sample1 = np.random.choice(group1, size=len(group1), replace=True)
        sample2 = np.random.choice(group2, size=len(group2), replace=True)
        differences.append(np.mean(sample1) - np.mean(sample2))

    differences = np.array(differences)
    alpha = 1 - confidence
    ci_lower = float(np.percentile(differences, 100 * alpha / 2))
    ci_upper = float(np.percentile(differences, 100 * (1 - alpha / 2)))

    return {
        "lower": ci_lower,
        "upper": ci_upper,
        "mean": float(np.mean(differences)),
        "contains_zero": ci_lower <= 0 <= ci_upper,
    }


def compute_statistical_power(
    effect_size: float, n_per_group: int, alpha: float = 0.01
) -> float:
    """
    Compute statistical power for paired t-test.

    Power = P(reject H0 | H0 is false)
    Want power ≥ 0.8 (80% chance of detecting true effect)

    Args:
        effect_size: Cohen's d
        n_per_group: Sample size per group (number of seeds)
        alpha: Significance level

    Returns:
        Statistical power (0 to 1)
    """
    from scipy.stats import nct
    from scipy.stats import t as t_dist

    # Non-centrality parameter for paired t-test
    ncp = effect_size * np.sqrt(n_per_group)

    # Degrees of freedom
    df = n_per_group - 1

    # Critical value (two-tailed)
    t_crit = t_dist.ppf(1 - alpha / 2, df)

    # Power calculation using non-central t-distribution
    power = 1 - nct.cdf(t_crit, df, ncp) + nct.cdf(-t_crit, df, ncp)

    return float(power)


def interpret_effect_size(d: float) -> str:
    """
    Interpret Cohen's d effect size.

    Args:
        d: Cohen's d effect size

    Returns:
        Interpretation string
    """
    abs_d = abs(d)
    if abs_d < 0.2:
        return "negligible"
    elif abs_d < 0.5:
        return "small"
    elif abs_d < 0.8:
        return "medium"
    else:
        return "large"


def generate_h1c_interpretation(
    h1c_confirmed: bool,
    per_dataset_results: Dict,
    alpha: float,
    bonferroni_correction: bool,
) -> str:
    """Generate plain English interpretation of H1c test."""
    lines = []

    if h1c_confirmed:
        lines.append(
            "✅ H1c CONFIRMED: PGD-AT does NOT improve " "cross-site generalization"
        )
        lines.append("")
        lines.append(
            "This is the EXPECTED result and supports your " "tri-objective approach:"
        )
        lines.append("- PGD-AT improves adversarial robustness (tested separately)")
        lines.append(
            "- PGD-AT does NOT improve cross-site generalization " "(tested here)"
        )
        lines.append("- Therefore, you need tri-objective training to achieve both")
    else:
        lines.append("❌ H1c REJECTED: PGD-AT DOES affect cross-site generalization")
        lines.append("")
        lines.append("This is UNEXPECTED. Possible explanations:")
        lines.append("- PGD-AT significantly improved cross-site (good for PGD-AT)")
        lines.append("- PGD-AT significantly degraded cross-site (bad for PGD-AT)")
        lines.append(
            "- Need to investigate which datasets show significant differences"
        )

    lines.append("")
    lines.append("Per-Dataset Results:")

    for dataset_name, result in per_dataset_results.items():
        lines.append(f"\n{dataset_name}:")
        lines.append(
            f"  Baseline drop: {result['baseline_drop_mean']:.2f} ± "
            f"{result['baseline_drop_std']:.2f} pp"
        )
        lines.append(
            f"  PGD-AT drop:   {result['pgd_at_drop_mean']:.2f} ± "
            f"{result['pgd_at_drop_std']:.2f} pp"
        )
        lines.append(f"  Difference:    {result['difference_mean']:.2f} pp")
        lines.append(f"  p-value:       {result['p_value']:.4f} (α={alpha})")

        if bonferroni_correction:
            lines.append(f"  p-value (Bonf): {result['p_value_bonferroni']:.4f}")
            lines.append(f"  Significant:    {result['significant_bonferroni']}")
        else:
            lines.append(f"  Significant:    {result['significant']}")

        lines.append(
            f"  Cohen's d:     {result['cohens_d']:.3f} "
            f"({result['effect_interpretation']})"
        )
        lines.append(
            f"  95% CI:        [{result['ci_95_lower']:.2f}, "
            f"{result['ci_95_upper']:.2f}]"
        )

    return "\n".join(lines)


def generate_recommendations(
    h1c_confirmed: bool, power_analyses: Dict[str, float]
) -> List[str]:
    """Generate recommendations based on test results."""
    recommendations = []

    # Check statistical power
    low_power_datasets = [name for name, power in power_analyses.items() if power < 0.8]

    if low_power_datasets:
        recommendations.append(
            f"⚠️  Low statistical power (<0.8) for: "
            f"{', '.join(low_power_datasets)}. "
            f"Consider increasing from 3 to 5 seeds for better reliability."
        )

    # Recommendations based on H1c result
    if h1c_confirmed:
        recommendations.append(
            "✅ H1c confirmed as expected. Use this result to justify "
            "tri-objective approach in dissertation."
        )
        recommendations.append(
            "📊 Create visualization: Baseline vs PGD-AT cross-site drops "
            "(bar chart with error bars)"
        )
        recommendations.append(
            "📝 Emphasize in writing: PGD-AT alone is insufficient, "
            "motivating need for explainability regularization"
        )
    else:
        recommendations.append(
            "⚠️  H1c rejected - investigate which datasets show "
            "significant differences"
        )
        recommendations.append(
            "🔍 Check if PGD-AT improved or degraded cross-site performance"
        )
        recommendations.append(
            "💡 May need to revise hypothesis or investigate " "confounding factors"
        )

    return recommendations


# ============================================================================
# Example Usage
# ============================================================================

if __name__ == "__main__":
    print("=" * 70)
    print("RQ1 Cross-Site Hypothesis Test Example")
    print("=" * 70)
    print()

    # Example data (3 seeds each)
    # Scenario: PGD-AT has similar cross-site drops to baseline

    baseline_drops = {
        "isic2019": [15.0, 14.5, 15.5],  # Mean ~15pp drop
        "isic2020": [16.0, 15.5, 16.5],  # Mean ~16pp drop
        "derm7pt": [14.0, 13.5, 14.5],  # Mean ~14pp drop
    }

    pgd_at_drops = {
        "isic2019": [14.8, 14.3, 15.3],  # Mean ~14.8pp drop (similar)
        "isic2020": [16.2, 15.7, 16.7],  # Mean ~16.2pp drop (similar)
        "derm7pt": [14.1, 13.6, 14.6],  # Mean ~14.1pp drop (similar)
    }

    # Run hypothesis test
    results = test_rq1_hypothesis(
        baseline_drops=baseline_drops,
        pgd_at_drops=pgd_at_drops,
        alpha=0.01,
        bonferroni_correction=True,
    )

    # Print results
    print(f"Hypothesis: {results['hypothesis']}")
    print(f"H1c Confirmed: {results['h1c_confirmed']}")
    print(f"Bonferroni Correction: {results['bonferroni_correction']}")
    print(f"Number of Datasets: {results['n_datasets']}")
    print()

    print("Per-Dataset Results:")
    print("-" * 70)
    for dataset_name, dataset_results in results["per_dataset_results"].items():  # noqa
        print(f"\n{dataset_name}:")
        print(
            f"  Baseline: {dataset_results['baseline_drop_mean']:.2f} ± "
            f"{dataset_results['baseline_drop_std']:.2f} pp"
        )
        print(
            f"  PGD-AT:   {dataset_results['pgd_at_drop_mean']:.2f} ± "
            f"{dataset_results['pgd_at_drop_std']:.2f} pp"
        )
        print(f"  p-value:  {dataset_results['p_value']:.4f}")
        print(f"  p (Bonf): {dataset_results['p_value_bonferroni']:.4f}")
        print(f"  Significant: {dataset_results['significant_bonferroni']}")
        print(
            f"  Cohen's d: {dataset_results['cohens_d']:.3f} "
            f"({dataset_results['effect_interpretation']})"
        )

    print("\n" + "=" * 70)
    print("INTERPRETATION")
    print("=" * 70)
    print(results["interpretation"])

    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    for rec in results["recommendations"]:
        print(f"  {rec}")
