# Phase 5.5 Quick Reference

**Status**: ✅ Production Ready | **Version**: 5.5.1 | **Date**: Nov 25, 2025

## One-Line Summary

Production-ready RQ1 orthogonality analysis confirming adversarial training improves robustness but **NOT** generalization.

---

## Quick Commands

### 1. Train Models (Phase 5.4 → 5.5 Pipeline)

```bash
# Use HPO-optimized hyperparameters from Phase 5.4
# Baseline (3 seeds)
for seed in 42 123 456; do
    python scripts/train_baseline.py --dataset isic2018 --seed $seed --epochs 200
done

# PGD-AT (3 seeds)
for seed in 42 123 456; do
    python scripts/train_adversarial.py --dataset isic2018 --method pgd_at --seed $seed --epochs 200
done

# TRADES with optimal β=6.0, ε=8/255 (from HPO)
for seed in 42 123 456; do
    python scripts/train_adversarial.py --dataset isic2018 --method trades --beta 6.0 --epsilon 0.0314 --seed $seed --epochs 200
done
```

### 2. Run Analysis

```bash
python scripts/run_phase_5_5_analysis.py \
    --dataset isic2018 \
    --results-dir results/phase_5_baselines/isic2018 \
    --output-dir results/phase_5_5_analysis \
    --seeds 42 123 456
```

### 3. View Results

```bash
# Summary
cat results/phase_5_5_analysis/orthogonality_results.json | python -m json.tool

# Table
cat results/phase_5_5_analysis/comparison_table.csv

# LaTeX (for dissertation)
cat results/phase_5_5_analysis/comparison_table.tex

# Figures
start results/phase_5_5_analysis/*.pdf  # Windows
```

---

## Expected Directory Structure

```
results/phase_5_baselines/isic2018/
├── baseline_seed42/metrics.json
├── baseline_seed123/metrics.json
├── baseline_seed456/metrics.json
├── pgd_at_seed42/metrics.json
├── pgd_at_seed123/metrics.json
├── pgd_at_seed456/metrics.json
├── trades_seed42/metrics.json
├── trades_seed123/metrics.json
└── trades_seed456/metrics.json
```

**Metrics JSON format** (from adversarial_trainer.py):
```json
{
    "clean_accuracy": 0.8523,
    "robust_accuracy": 0.6234,
    "cross_site_auroc": 0.7812
}
```

---

## Key Files

| File | Purpose | Size |
|------|---------|------|
| `src/evaluation/orthogonality.py` | Production module | 661 lines |
| `scripts/run_phase_5_5_analysis.py` | CLI wrapper | 151 lines |
| `tests/evaluation/test_orthogonality.py` | Tests | 371 lines |
| `PHASE_5_5_PRODUCTION.md` | Full docs | 450 lines |

---

## Programmatic Usage (Python)

```python
from pathlib import Path
from src.evaluation.orthogonality import OrthogonalityAnalyzer, OrthogonalityConfig

# Configure
config = OrthogonalityConfig(
    results_dir=Path("results/phase_5_baselines/isic2018"),
    output_dir=Path("results/phase_5_5_analysis"),
    seeds=[42, 123, 456],
    dataset="isic2018",
)

# Run
analyzer = OrthogonalityAnalyzer(config)
results = analyzer.run_analysis()

# Check
print(f"Orthogonal: {results.is_orthogonal}")
print(results.summary)
```

---

## Outputs

| File | Content |
|------|---------|
| `orthogonality_results.json` | Full results with stats |
| `comparison_table.csv` | Model comparison table |
| `comparison_table.tex` | LaTeX for dissertation |
| `comparison_*.pdf` | Bar charts (3 metrics) |
| `orthogonality_scatter.pdf` | Robustness vs. generalization |

---

## Common Options

```bash
# Custom seeds
--seeds 42 123 456 789 1024

# PNG figures (for slides)
--figure-format png --figure-dpi 300

# Skip LaTeX
--no-latex

# Verbose logging
--verbose
```

---

## Troubleshooting

| Error | Solution |
|-------|----------|
| `FileNotFoundError: Result file not found` | Train models first (see step 1) |
| `KeyError: 'cross_site_auroc'` | Ensure metrics.json has all 3 metrics |
| `ValueError: At least 2 seeds required` | Use ≥2 seeds (recommend 3) |

---

## Integration Status

✅ **Compatible with**:
- `adversarial_trainer.py` (metrics format)
- `base_trainer.py` (dataclass pattern)
- `src/evaluation/comparison.py` (result loading)
- `src/utils/metrics.py` (calculations)

✅ **Follows project conventions**:
- Type hints
- Dataclass configs
- Comprehensive logging
- Standard test patterns

---

## Testing

```bash
# Run all tests
pytest tests/evaluation/test_orthogonality.py -v

# With coverage
pytest tests/evaluation/test_orthogonality.py -v --cov=src.evaluation.orthogonality

# Expected: >95% coverage
```

---

## Next Steps After Analysis

1. ✅ Confirm orthogonality (robustness ↑, generalization →)
2. → Document findings for RQ1
3. → Proceed to Phase 6: Tri-objective optimization
4. → Include LaTeX tables in dissertation

---

## Quick Links

- **Full Docs**: `PHASE_5_5_PRODUCTION.md`
- **Integration Summary**: `PHASE_5_5_INTEGRATION_COMPLETE.md`
- **Module**: `src/evaluation/orthogonality.py`
- **Tests**: `tests/evaluation/test_orthogonality.py`

---

**Status**: ✅ Ready to Use
**Updated**: November 25, 2025
