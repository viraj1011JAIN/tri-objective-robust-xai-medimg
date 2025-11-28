"""
Phase 8.5 Selective Metrics Demonstration Script.

This script demonstrates all the functionality of Phase 8.5
Selective Prediction Evaluation Metrics implementation.
"""

import numpy as np

from src.selection import (
    compare_strategies,
    compute_risk_coverage_curve,
    compute_selective_metrics,
    validate_hypothesis_h3a,
)


def main():
    """Run Phase 8.5 demonstration."""
    # Generate sample data
    np.random.seed(42)
    n = 200

    labels = np.random.randint(0, 3, size=n)
    predictions = labels.copy()

    # 15% error rate
    errors = np.random.choice(n, size=30, replace=False)
    predictions[errors] = (labels[errors] + 1) % 3

    # Scores: errors have lower confidence/stability
    confidences = np.random.uniform(0.7, 0.99, size=n)
    confidences[errors] = np.random.uniform(0.4, 0.7, size=len(errors))

    stability = np.random.uniform(0.7, 0.95, size=n)
    stability[errors] = np.random.uniform(0.35, 0.65, size=len(errors))

    # Combined score at 90% coverage
    scores = 0.5 * confidences + 0.5 * stability
    threshold = np.percentile(scores, 10)
    is_accepted = scores >= threshold

    # Compute metrics
    print("=" * 60)
    print("PHASE 8.5: SELECTIVE METRICS DEMONSTRATION")
    print("=" * 60)

    metrics = compute_selective_metrics(
        predictions,
        labels,
        is_accepted,
        confidences=confidences,
        scores=scores,
        compute_ci=True,
        n_bootstrap=100,
    )

    print(metrics.summary())

    # Validate H3a
    h3a = validate_hypothesis_h3a(metrics)
    status = "YES" if h3a["passed"] else "NO"
    print(f"\nH3a VALIDATION: {status}")
    print(f"   Improvement: {h3a['improvement_pp']:.2f}pp (target: 4pp)")
    print(f"   Coverage: {h3a['coverage']:.1%}")

    # Compare strategies
    print("\nSTRATEGY COMPARISON")
    print("-" * 60)
    results = compare_strategies(predictions, labels, confidences, stability)
    for name, m in results.items():
        print(
            f"   {name:20s}: Coverage={m.coverage:.1%}, "
            f"Acc={m.selective_accuracy:.1%}, "
            f"Improvement={m.improvement*100:+.1f}pp, "
            f"AURC={m.aurc:.4f}"
        )

    # Risk-coverage curves
    print("\nRISK-COVERAGE ANALYSIS")
    print("-" * 60)
    for name, s in [
        ("confidence", confidences),
        ("stability", stability),
        ("combined", scores),
    ]:
        curve = compute_risk_coverage_curve(predictions, labels, s)
        print(f"   {name:12s}: AURC={curve.aurc:.4f}, E-AURC={curve.e_aurc:.4f}")

    print("\n" + "=" * 60)
    print("✅ Phase 8.5 Implementation Complete!")
    print("=" * 60)

    return metrics, results


if __name__ == "__main__":
    main()
