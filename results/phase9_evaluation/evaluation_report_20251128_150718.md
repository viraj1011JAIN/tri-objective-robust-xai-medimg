# Tri-Objective Model Evaluation Report
**Generated:** 2025-11-28 15:07:18
---
## Executive Summary
This report presents a comprehensive evaluation of multi-objective optimization models using rigorous statistical testing and Pareto analysis.

## 1. Statistical Summary
| Model | Objective | Mean ± Std | 95% CI |
|-------|-----------|------------|--------|
| TriObjective | accuracy | 0.8445 ± 0.0193 | [0.8159, 0.8805] |
| TriObjective | robustness | 0.6700 ± 0.0347 | [0.6187, 0.7349] |
| TriObjective | interpretability | 0.7280 ± 0.0268 | [0.6879, 0.7607] |
| Baseline | accuracy | 0.8834 ± 0.0206 | [0.8575, 0.9170] |
| Baseline | robustness | 0.5724 ± 0.0343 | [0.5292, 0.6283] |
| Baseline | interpretability | 0.5814 ± 0.0652 | [0.4909, 0.6972] |
| Adversarial | accuracy | 0.8042 ± 0.0143 | [0.7826, 0.8260] |
| Adversarial | robustness | 0.7184 ± 0.0287 | [0.6752, 0.7619] |
| Adversarial | interpretability | 0.4874 ± 0.0427 | [0.4150, 0.5493] |

## 2. Pairwise Statistical Comparisons

### TriObjective vs Baseline
| Objective | Δ Mean | Cohen's d | Interpretation | p-value | Significant |
|-----------|--------|-----------|----------------|---------|-------------|
| accuracy | -0.0390 | -1.854 | large | 0.0013 | ✓ Yes |
| robustness | +0.0976 | 2.683 | large | 0.0004 | ✓ Yes |
| interpretability | +0.1466 | 2.791 | large | 0.0006 | ✓ Yes |

### TriObjective vs Adversarial
| Objective | Δ Mean | Cohen's d | Interpretation | p-value | Significant |
|-----------|--------|-----------|----------------|---------|-------------|
| accuracy | +0.0403 | 2.247 | large | 0.0006 | ✓ Yes |
| robustness | -0.0484 | -1.440 | large | 0.0091 | ✓ Yes |
| interpretability | +0.2406 | 6.403 | large | 0.0002 | ✓ Yes |

### Baseline vs Adversarial
| Objective | Δ Mean | Cohen's d | Interpretation | p-value | Significant |
|-----------|--------|-----------|----------------|---------|-------------|
| accuracy | +0.0793 | 4.240 | large | 0.0002 | ✓ Yes |
| robustness | -0.1460 | -4.382 | large | 0.0002 | ✓ Yes |
| interpretability | +0.0940 | 1.619 | large | 0.0046 | ✓ Yes |

## 3. Pareto Analysis
- **Pareto-optimal models:** TriObjective, Baseline, Adversarial
- **Number of optimal solutions:** 3
- **Dominance ratio:** 100.00%

## 4. Recommendations
1. Large significant difference in accuracy for TriObjective_vs_Baseline: d = -1.854, p = 0.0013
2. Large significant difference in robustness for TriObjective_vs_Baseline: d = 2.683, p = 0.0004
3. Large significant difference in interpretability for TriObjective_vs_Baseline: d = 2.791, p = 0.0006
4. Large significant difference in accuracy for TriObjective_vs_Adversarial: d = 2.247, p = 0.0006
5. Large significant difference in robustness for TriObjective_vs_Adversarial: d = -1.440, p = 0.0091
6. Large significant difference in interpretability for TriObjective_vs_Adversarial: d = 6.403, p = 0.0002
7. Large significant difference in accuracy for Baseline_vs_Adversarial: d = 4.240, p = 0.0002
8. Large significant difference in robustness for Baseline_vs_Adversarial: d = -4.382, p = 0.0002
9. Large significant difference in interpretability for Baseline_vs_Adversarial: d = 1.619, p = 0.0046

## 5. Methodology

This evaluation employs state-of-the-art statistical methods:

- **Effect Sizes:** Cohen's d and Hedges' g for standardized comparison
- **Hypothesis Testing:** Mann-Whitney U test (non-parametric)
- **Confidence Intervals:** BCa bootstrap with 1000 resamples
- **Multiple Comparison Correction:** Benjamini-Hochberg FDR control
- **Pareto Analysis:** Exhaustive dominance checking with knee-point detection
