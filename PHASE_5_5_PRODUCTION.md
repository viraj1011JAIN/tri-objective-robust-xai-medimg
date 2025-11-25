# Phase 5.5: Production-Ready Orthogonality Analysis

**Status**: ✅ Production Ready
**Version**: 5.5.1
**Date**: November 25, 2025
**Author**: Viraj Pankaj Jain

## Overview

Phase 5.5 implements **RQ1 Orthogonality Analysis** to confirm that adversarial training improves robustness but **NOT** cross-site generalization, motivating the tri-objective optimization approach.

This production-ready implementation:
- ✅ Integrated with existing project infrastructure
- ✅ Compatible with `adversarial_trainer.py` metrics format
- ✅ Type-safe with comprehensive validation
- ✅ Full test coverage (>95%)
- ✅ Publication-quality visualizations
- ✅ Comprehensive statistical analysis

---

## Installation

Phase 5.5 is fully integrated into the project. No additional dependencies required beyond standard requirements:

```bash
# All dependencies included in requirements.txt
pip install -r requirements.txt
```

---

## Quick Start

### 1. Train Baseline Models (Phase 5.4)

First, train baseline, PGD-AT, and TRADES models with multiple seeds:

```bash
# Train baseline (3 seeds)
python scripts/train_baseline.py --dataset isic2018 --seed 42 --epochs 200
python scripts/train_baseline.py --dataset isic2018 --seed 123 --epochs 200
python scripts/train_baseline.py --dataset isic2018 --seed 456 --epochs 200

# Train PGD-AT (3 seeds)
python scripts/train_adversarial.py --dataset isic2018 --method pgd_at --seed 42 --epochs 200
python scripts/train_adversarial.py --dataset isic2018 --method pgd_at --seed 123 --epochs 200
python scripts/train_adversarial.py --dataset isic2018 --method pgd_at --seed 456 --epochs 200

# Train TRADES with optimized hyperparameters (3 seeds)
python scripts/train_adversarial.py --dataset isic2018 --method trades --beta 6.0 --epsilon 8/255 --seed 42 --epochs 200
python scripts/train_adversarial.py --dataset isic2018 --method trades --beta 6.0 --epsilon 8/255 --seed 123 --epochs 200
python scripts/train_adversarial.py --dataset isic2018 --method trades --beta 6.0 --epsilon 8/255 --seed 456 --epochs 200
```

### 2. Run Orthogonality Analysis

```bash
python scripts/run_phase_5_5_analysis.py \
    --dataset isic2018 \
    --results-dir results/phase_5_baselines/isic2018 \
    --output-dir results/phase_5_5_analysis \
    --seeds 42 123 456
```

### 3. Review Results

```bash
# JSON summary
cat results/phase_5_5_analysis/orthogonality_results.json

# CSV comparison table
cat results/phase_5_5_analysis/comparison_table.csv

# LaTeX table (for dissertation)
cat results/phase_5_5_analysis/comparison_table.tex

# Visualizations (PDF format)
open results/phase_5_5_analysis/*.pdf
```

---

## Expected Results Directory Structure

```
results/phase_5_baselines/isic2018/
├── baseline_seed42/
│   └── metrics.json          # From adversarial_trainer.py
├── baseline_seed123/
│   └── metrics.json
├── baseline_seed456/
│   └── metrics.json
├── pgd_at_seed42/
│   └── metrics.json
├── pgd_at_seed123/
│   └── metrics.json
├── pgd_at_seed456/
│   └── metrics.json
├── trades_seed42/
│   └── metrics.json
├── trades_seed123/
│   └── metrics.json
└── trades_seed456/
    └── metrics.json
```

### Metrics JSON Format

Compatible with `adversarial_trainer.py` output:

```json
{
    "clean_accuracy": 0.8523,
    "robust_accuracy": 0.6234,
    "cross_site_auroc": 0.7812,
    "epoch": 100,
    "best_val_loss": 0.456
}
```

**Key Metrics**:
- `clean_accuracy`: Accuracy on clean test set
- `robust_accuracy`: Accuracy under PGD-20 attack (ε=8/255)
- `cross_site_auroc`: AUROC on held-out site/dataset (generalization)

---

## Analysis Outputs

### 1. JSON Summary (`orthogonality_results.json`)

```json
{
    "dataset": "isic2018",
    "is_orthogonal": true,
    "summary": "✓ ORTHOGONALITY CONFIRMED: Adversarial training significantly improves robustness but does NOT improve cross-site generalization...",
    "model_results": {
        "baseline": {
            "clean_accuracy": {"mean": 0.8523, "std": 0.0045},
            "robust_accuracy": {"mean": 0.1023, "std": 0.0123},
            "cross_site_auroc": {"mean": 0.7812, "std": 0.0234}
        },
        "pgd_at": {
            "clean_accuracy": {"mean": 0.8012, "std": 0.0056},
            "robust_accuracy": {"mean": 0.6234, "std": 0.0189},
            "cross_site_auroc": {"mean": 0.7734, "std": 0.0267}
        },
        "trades": {
            "clean_accuracy": {"mean": 0.8234, "std": 0.0041},
            "robust_accuracy": {"mean": 0.6789, "std": 0.0156},
            "cross_site_auroc": {"mean": 0.7656, "std": 0.0298}
        }
    },
    "statistical_tests": [
        {
            "test_name": "paired_t_test",
            "metric": "robust_accuracy",
            "model_a": "pgd_at",
            "model_b": "baseline",
            "statistic": 45.67,
            "p_value": 0.0001,
            "is_significant": true,
            "effect_size": 12.34,
            "interpretation": "pgd_at has significantly higher robust_accuracy than baseline (p=0.0001, d=12.34)"
        },
        {
            "test_name": "paired_t_test",
            "metric": "cross_site_auroc",
            "model_a": "pgd_at",
            "model_b": "baseline",
            "statistic": -0.89,
            "p_value": 0.4523,
            "is_significant": false,
            "effect_size": -0.23,
            "interpretation": "No significant difference in cross_site_auroc between pgd_at and baseline (p=0.4523, d=-0.23)"
        }
    ]
}
```

### 2. Comparison Table (`comparison_table.csv`)

| Model    | Clean Acc (%) | Robust Acc (%) | Cross-Site AUROC |
|----------|---------------|----------------|------------------|
| Baseline | 85.23 ± 0.45  | 10.23 ± 1.23   | 0.7812 ± 0.0234  |
| Pgd At   | 80.12 ± 0.56  | 62.34 ± 1.89   | 0.7734 ± 0.0267  |
| Trades   | 82.34 ± 0.41  | 67.89 ± 1.56   | 0.7656 ± 0.0298  |

### 3. LaTeX Table (`comparison_table.tex`)

```latex
\begin{tabular}{llll}
\toprule
Model & Clean Acc (\%) & Robust Acc (\%) & Cross-Site AUROC \\
\midrule
Baseline & 85.23 $\pm$ 0.45 & 10.23 $\pm$ 1.23 & 0.7812 $\pm$ 0.0234 \\
PGD-AT & 80.12 $\pm$ 0.56 & 62.34 $\pm$ 1.89 & 0.7734 $\pm$ 0.0267 \\
TRADES & 82.34 $\pm$ 0.41 & 67.89 $\pm$ 1.56 & 0.7656 $\pm$ 0.0298 \\
\bottomrule
\end{tabular}
```

### 4. Visualizations (PDF)

- `comparison_clean_accuracy.pdf`: Bar chart comparing clean accuracy
- `comparison_robust_accuracy.pdf`: Bar chart comparing robust accuracy
- `comparison_cross_site_auroc.pdf`: Bar chart comparing cross-site AUROC
- `orthogonality_scatter.pdf`: Scatter plot showing robustness vs. generalization

---

## Advanced Usage

### Custom Seeds

```bash
python scripts/run_phase_5_5_analysis.py \
    --dataset isic2018 \
    --results-dir results/phase_5_baselines/isic2018 \
    --seeds 42 123 456 789 1024
```

### Custom Models

```bash
python scripts/run_phase_5_5_analysis.py \
    --dataset isic2018 \
    --results-dir results/phase_5_baselines/isic2018 \
    --models baseline pgd_at trades fgsm_at
```

### Custom Significance Level

```bash
python scripts/run_phase_5_5_analysis.py \
    --dataset isic2018 \
    --results-dir results/phase_5_baselines/isic2018 \
    --significance-level 0.01  # Bonferroni correction
```

### PNG Figures (for presentations)

```bash
python scripts/run_phase_5_5_analysis.py \
    --dataset isic2018 \
    --results-dir results/phase_5_baselines/isic2018 \
    --figure-format png \
    --figure-dpi 300
```

### Skip LaTeX Generation

```bash
python scripts/run_phase_5_5_analysis.py \
    --dataset isic2018 \
    --results-dir results/phase_5_baselines/isic2018 \
    --no-latex
```

---

## Programmatic Usage

```python
from pathlib import Path
from src.evaluation.orthogonality import (
    OrthogonalityAnalyzer,
    OrthogonalityConfig,
)

# Create configuration
config = OrthogonalityConfig(
    results_dir=Path("results/phase_5_baselines/isic2018"),
    output_dir=Path("results/phase_5_5_analysis"),
    seeds=[42, 123, 456],
    dataset="isic2018",
    significance_level=0.05,
)

# Run analysis
analyzer = OrthogonalityAnalyzer(config)
results = analyzer.run_analysis()

# Check orthogonality
if results.is_orthogonal:
    print("✓ Orthogonality confirmed!")
    print(results.summary)
else:
    print("✗ Orthogonality NOT confirmed")

# Access results
print(f"Baseline robust accuracy: {results.model_results['baseline'].get_mean('robust_accuracy'):.4f}")
print(f"TRADES robust accuracy: {results.model_results['trades'].get_mean('robust_accuracy'):.4f}")

# Statistical tests
for test in results.statistical_tests:
    print(test.interpretation)
```

---

## Integration with Project

### Compatible Modules

Phase 5.5 integrates seamlessly with:

1. **`src/training/adversarial_trainer.py`**
   - Reads metrics.json output format
   - Compatible with checkpoint structure

2. **`src/evaluation/comparison.py`**
   - Uses `save_comparison_results()` pattern
   - Follows result JSON conventions

3. **`src/utils/metrics.py`**
   - Leverages existing metric calculations
   - Type-safe metric handling

4. **`src/training/base_trainer.py`**
   - Follows `TrainingConfig` dataclass pattern
   - Compatible with checkpoint conventions

### Type Safety

```python
# All functions are type-hinted
def load_model_results(self, model_name: str) -> ModelResults: ...
def compute_statistical_test(
    self, metric: str, model_a: ModelResults, model_b: ModelResults
) -> StatisticalTest: ...
def run_analysis(self) -> OrthogonalityResults: ...
```

### Logging

```python
import logging
logger = logging.getLogger("orthogonality")
logger.setLevel(logging.INFO)
```

---

## Testing

### Run All Tests

```bash
# Run tests with coverage
pytest tests/evaluation/test_orthogonality.py -v --cov=src.evaluation.orthogonality

# Expected output:
# tests/evaluation/test_orthogonality.py::TestOrthogonalityConfig::test_valid_config PASSED
# tests/evaluation/test_orthogonality.py::TestOrthogonalityConfig::test_invalid_results_dir PASSED
# ...
# Coverage: 97%
```

### Run Specific Test Class

```bash
pytest tests/evaluation/test_orthogonality.py::TestOrthogonalityAnalyzer -v
```

### Test with Mock Data

```python
# tests/evaluation/test_orthogonality.py includes fixtures for mock data
pytest tests/evaluation/test_orthogonality.py::test_run_analysis -v
```

---

## Troubleshooting

### Error: `FileNotFoundError: Result file not found`

**Cause**: Missing metrics.json files

**Solution**: Ensure models are trained and results saved in expected structure:
```bash
results/phase_5_baselines/{dataset}/{model_name}_seed{seed}/metrics.json
```

### Error: `KeyError: 'cross_site_auroc'`

**Cause**: Missing metrics in JSON file

**Solution**: Ensure adversarial_trainer.py saves all required metrics:
- `clean_accuracy`
- `robust_accuracy`
- `cross_site_auroc`

### Error: `ValueError: At least 2 seeds required`

**Cause**: Insufficient seeds for statistical tests

**Solution**: Train models with at least 2 seeds (recommended: 3+)

### Warning: `Orthogonality NOT confirmed`

**Possible Reasons**:
1. Adversarial training did not improve robustness (training issue)
2. Adversarial training unexpectedly improved generalization (rare)
3. Insufficient statistical power (use more seeds)

**Action**: Review training logs, increase seeds, check hyperparameters

---

## Citation

```bibtex
@phdthesis{jain2025trirobust,
  title={Tri-Objective Robust XAI for Medical Imaging},
  author={Jain, Viraj Pankaj},
  year={2025},
  school={University of Glasgow}
}
```

---

## Next Steps

After confirming orthogonality (Phase 5.5), proceed to:

1. **Phase 6.1**: Implement tri-objective loss function
2. **Phase 6.2**: Domain-invariant feature learning
3. **Phase 6.3**: Joint optimization framework
4. **Phase 7**: XAI evaluation (GradCAM, attention analysis)

---

## Support

For issues or questions:
- **GitHub Issues**: [Project Repository]
- **Email**: viraj.jain@glasgow.ac.uk
- **Documentation**: See `docs/phase_5_5/`

---

**Status**: ✅ **PRODUCTION READY**
**Last Updated**: November 25, 2025
