
# Phase 8.3: Multi-Signal Selective Prediction with ViT-B/16

## Overview

This directory contains results from Phase 8.3 of the Tri-Objective Robust XAI project, which demonstrates production-grade selective prediction on the ISIC 2020 skin lesion dataset using Vision Transformer (ViT-B/16).

## Key Results

- **Dataset**: ISIC 2020 (N=200 samples evaluated)
- **Model**: ViT-B/16 (86M parameters, ImageNet pretrained)
- **Target**: >=4pp accuracy gain at 90% coverage
- **Result**: **1.9pp gain** TARGET NOT MET

### Accuracy @ 90% Coverage

- Baseline (100%): 91.50%
- Softmax Confidence: 93.37% (+1.9pp)
- Entropy Confidence: 93.37% (+1.9pp)
- Attention Max: 92.82% (+1.3pp)
- **Multi-Signal Fusion**: **93.37%** (+1.9pp)

### Signal Statistics

- Mean Softmax Confidence: 0.6591
- Mean Entropy Confidence: 0.0997
- Mean Attention Max: 0.011041
- Mean Multi-Signal: 0.3998

### Quadrant Analysis

- Q1 (High Conf + High Attn): 52 samples (98.08% accuracy)
- Q2 (Low Conf + High Attn): 28 samples
- Q3 (Low Conf + Low Attn): 62 samples
- Q4 (High Conf + Low Attn): 58 samples

## Files

- `phase_8_3_results.csv`: Full results dataframe with all signals
- `phase_8_3_summary.json`: Summary statistics (JSON format)
- `metrics_*.csv`: Selective prediction curves for each strategy (4 files)
- `figure_1_coverage_accuracy.png`: Publication-ready coverage-accuracy plot (300 DPI)
- `figure_2_confidence_attention.png`: Confidence vs. attention scatter plot (300 DPI)
- `table_results.tex`: LaTeX table for dissertation
- `README.md`: This documentation file

## Methodology

### Signals Used

1. **Softmax Confidence**: Maximum softmax probability
2. **Entropy Confidence**: 1 - normalized entropy
3. **Attention Max**: Maximum attention weight from ViT
4. **Multi-Signal Fusion**: Geometric mean of all signals

### Selective Prediction

For each signal, we:
1. Rank samples by signal strength
2. Evaluate accuracy at different coverage levels
3. Measure precision, recall, and F1 scores
4. Compare against baseline (100% coverage)

## Citation

If you use these results, please cite:

```bibtex
@phdthesis{yourname2025tri,
  title={Tri-Objective Robust XAI for Medical Imaging},
  author={Your Name},
  school={Your University},
  year={2025},
  note={Phase 8.3: Multi-Signal Selective Prediction}
}
```

## Contact

For questions about these results, contact: your.email@university.edu

---

Generated on: 2025-11-28 04:03:10
