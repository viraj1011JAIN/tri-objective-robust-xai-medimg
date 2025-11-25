# Phase 5.3 - TRADES Adversarial Training Complete Guide

## Table of Contents
1. [Overview](#overview)
2. [TRADES Theory](#trades-theory)
3. [Implementation Architecture](#implementation-architecture)
4. [Quick Start](#quick-start)
5. [Detailed Usage](#detailed-usage)
6. [Configuration](#configuration)
7. [Evaluation & Analysis](#evaluation--analysis)
8. [Results Interpretation](#results-interpretation)
9. [Troubleshooting](#troubleshooting)
10. [References](#references)

---

## Overview

**Phase 5.3** implements **TRADES (TRadeoff-inspired Adversarial DEfense via Surrogate-loss minimization)**, a state-of-the-art adversarial training method that achieves better clean-robust accuracy trade-offs than standard PGD-AT.

### Key Features
- ✅ **TRADESLoss**: Custom loss combining cross-entropy + KL divergence
- ✅ **TRADESTrainer**: Production-grade trainer with AMP, gradient clipping, MLflow tracking
- ✅ **Comprehensive Evaluation**: Clean, FGSM, PGD, calibration metrics
- ✅ **Statistical Comparison**: Paired t-tests, effect sizes, bootstrap CIs
- ✅ **Trade-off Analysis**: Pareto frontier, knee point detection
- ✅ **Visualization**: Publication-quality Pareto curves, trade-off plots
- ✅ **Full Automation**: Single-command pipeline execution

### Why TRADES > PGD-AT?
1. **Explicit Clean Loss Preservation**: Maintains higher clean accuracy
2. **Smoother Decision Boundaries**: KL divergence → better generalization
3. **Controllable Trade-off**: Beta parameter for flexible clean-robust balance
4. **Better Calibration**: More reliable confidence estimates

---

## TRADES Theory

### Loss Formulation

TRADES minimizes:

```
L_TRADES = L_CE(f(x), y) + β × KL(f(x) || f(x_adv))
```

**Where:**
- **L_CE(f(x), y)**: Cross-entropy loss on clean samples
- **KL(f(x) || f(x_adv))**: KL divergence between clean and adversarial predictions
- **β**: Trade-off hyperparameter (higher β → more robustness, lower clean accuracy)
- **x_adv**: Adversarial example generated via PGD with KL loss

### Inner Maximization (Adversarial Generation)

```
x_adv = argmax_{||δ||_∞ ≤ ε} KL(f(x) || f(x + δ))
```

Solved using PGD:
```python
for t in range(num_steps):
    g = ∇_δ KL(f(x) || f(x + δ))
    δ = δ + α × sign(g)
    δ = clip(δ, -ε, ε)
    x + δ = clip(x + δ, 0, 1)
```

### Key Differences from PGD-AT

| Aspect | PGD-AT | TRADES |
|--------|--------|--------|
| **Loss** | L_CE(f(x_adv), y) | L_CE(f(x), y) + β×KL(f(x)||f(x_adv)) |
| **Adversarial Loss** | Cross-entropy | KL divergence |
| **Clean Loss** | None (implicit) | Explicit term |
| **Trade-off Control** | None | Beta parameter |
| **Clean Accuracy** | Lower | Higher |
| **Robust Accuracy** | Moderate | Comparable or better |

---

## Implementation Architecture

### File Structure

```
tri-objective-robust-xai-medimg/
│
├── configs/experiments/
│   └── trades_isic.yaml                    # TRADES configuration
│
├── scripts/
│   ├── training/
│   │   └── train_trades.py                 # Training script
│   ├── evaluation/
│   │   └── evaluate_trades.py              # Evaluation script
│   └── analysis/
│       ├── aggregate_results.py
│       ├── create_summary_plots.py
│       └── generate_report.py
│
├── src/
│   ├── evaluation/
│   │   ├── comparison.py                   # Statistical comparison
│   │   └── tradeoff_analysis.py            # Trade-off analysis
│   └── visualization/
│       └── pareto_curves.py                # Pareto visualization
│
├── RUN_PHASE_5_3_COMPLETE.ps1              # Automation script
└── PHASE_5_3_COMPLETE_GUIDE.md             # This file
```

### Core Components

#### 1. TRADESLoss Class
```python
class TRADESLoss(nn.Module):
    def __init__(self, beta=6.0):
        self.beta = beta

    def forward(self, logits_clean, logits_adv, targets):
        clean_loss = CE(logits_clean, targets)
        robust_loss = KL(logits_clean, logits_adv)
        return clean_loss + self.beta * robust_loss
```

#### 2. TRADESTrainer Class
```python
class TRADESTrainer:
    def train_epoch(self):
        # 1. Generate adversarial examples
        x_adv = self.attack(x, y)

        # 2. Forward pass (clean + adversarial)
        logits_clean = model(x)
        logits_adv = model(x_adv)

        # 3. Compute TRADES loss
        loss = criterion(logits_clean, logits_adv, y)

        # 4. Backward + optimize
        loss.backward()
        optimizer.step()
```

#### 3. Statistical Comparator
- Paired t-test
- Wilcoxon signed-rank test
- Cohen's d & Hedges' g
- Bootstrap confidence intervals
- Bonferroni & Holm correction

#### 4. Pareto Visualizer
- 2D/3D Pareto frontiers
- Knee point highlighting
- Trade-off curves
- Publication-quality styling

---

## Quick Start

### Prerequisites
```powershell
# Activate Python environment
conda activate your_env_name

# Verify dependencies
python -c "import torch, torchvision, mlflow, scipy, sklearn; print('All dependencies OK')"
```

### Single-Command Execution
```powershell
# Run complete pipeline (training + evaluation + comparison + visualization)
.\RUN_PHASE_5_3_COMPLETE.ps1
```

**This will:**
1. Train 9 models (3 seeds × 3 architectures)
2. Evaluate on test set + adversarial attacks
3. Compare with Phase 5.2 (PGD-AT)
4. Generate Pareto curves and trade-off plots
5. Create comprehensive report

**Expected time:** ~18-24 hours (depends on GPU)

---

## Detailed Usage

### 1. Train Single Model
```powershell
python scripts/training/train_trades.py `
    --config configs/experiments/trades_isic.yaml `
    --seed 42 `
    --model resnet50 `
    --beta 6.0
```

**Arguments:**
- `--config`: Configuration file path
- `--seed`: Random seed (42, 123, 456)
- `--model`: Architecture (resnet50, efficientnet_b0, vit_b_16)
- `--beta`: TRADES beta parameter (default: 6.0)
- `--resume`: Checkpoint to resume from (optional)

**Output:**
- Checkpoints: `results/phase_5_3_trades/checkpoints/resnet50_seed_42/`
- Logs: `results/phase_5_3_trades/logs/train_seed_42.log`
- MLflow: `mlruns/Phase_5.3_TRADES/`

### 2. Evaluate Model
```powershell
python scripts/evaluation/evaluate_trades.py `
    --config configs/experiments/trades_isic.yaml `
    --checkpoint results/phase_5_3_trades/checkpoints/resnet50_seed_42/best.pt `
    --output_dir results/phase_5_3_trades/evaluation_metrics/resnet50_seed_42
```

**Output Metrics:**
- **Clean**: Accuracy, F1, AUROC, AUPRC, ECE
- **FGSM**: Robust accuracy at ε = 2/255, 4/255, 8/255
- **PGD**: Robust accuracy at ε = 2/255, 4/255, 8/255 (20 steps)
- **Calibration**: ECE, MCE, Brier score, NLL
- **Confusion Matrix**: Class-wise performance

### 3. Compare with Baseline
```powershell
python src/evaluation/comparison.py `
    --trades_results results/phase_5_3_trades/comparison/trades_aggregated.json `
    --baseline_results results/phase_5_2_pgd_at/comparison/pgd_at_aggregated.json `
    --output results/phase_5_3_trades/comparison/statistical_comparison.json `
    --seeds 42 123 456
```

**Output:**
```json
{
  "clean_accuracy": {
    "trades_mean": 0.8542,
    "baseline_mean": 0.8193,
    "mean_diff": 0.0349,
    "ttest": {"p_value": 0.0023, "significant": true},
    "cohens_d": 1.24,
    "bootstrap_ci": {"ci_lower": 0.0198, "ci_upper": 0.0501}
  }
}
```

### 4. Trade-off Analysis
```powershell
python src/evaluation/tradeoff_analysis.py `
    --results_dir results/phase_5_3_trades/evaluation_metrics `
    --output results/phase_5_3_trades/comparison/tradeoff_analysis.json
```

**Output:**
```json
{
  "pareto": {
    "num_points": 5,
    "methods": ["TRADES_beta6", "TRADES_beta3", ...],
    "points": [[0.854, 0.723], [0.849, 0.741], ...]
  },
  "knee_point": {
    "method": "TRADES_beta6",
    "values": [0.854, 0.723]
  }
}
```

### 5. Generate Visualizations
```powershell
python src/visualization/pareto_curves.py `
    --results results/phase_5_3_trades/comparison/tradeoff_analysis.json `
    --output_dir results/phase_5_3_trades/evaluation_plots
```

**Generated Plots:**
- `pareto_2d.png`: Clean vs Robust accuracy Pareto frontier
- `pareto_3d.png`: 3D Pareto surface (Clean, Robust, ECE)
- `tradeoff_curves.png`: Beta sensitivity curves
- `comparison_bar.png`: Method comparison across metrics

---

## Configuration

### Key Parameters in `trades_isic.yaml`

#### TRADES-Specific
```yaml
trades:
  beta: 6.0                    # Trade-off parameter (higher = more robust)

  train_attack:
    attack_type: "pgd"
    epsilon: 0.03137           # 8/255
    num_steps: 7               # PGD steps during training
    step_size: 0.00784         # ε/4
    loss_type: "kl"            # KL divergence (TRADES signature)

  val_attack:
    num_steps: 10              # More steps for validation
```

#### Training Configuration
```yaml
training:
  num_epochs: 100
  batch_size: 32
  learning_rate: 0.01
  momentum: 0.9
  weight_decay: 0.0005

  lr_scheduler:
    type: "multistep"
    milestones: [50, 75]
    gamma: 0.1

  amp:
    enabled: true              # Mixed precision training

  grad_clip:
    enabled: true
    max_norm: 1.0
```

#### Evaluation Configuration
```yaml
evaluation:
  attacks:
    fgsm:
      enabled: true
      epsilons: [0.00784, 0.01569, 0.03137]  # 2/255, 4/255, 8/255

    pgd:
      enabled: true
      num_steps: 20            # More steps for evaluation
      epsilons: [0.00784, 0.01569, 0.03137]

  calibration:
    enabled: true
    num_bins: 15
```

### Beta Parameter Tuning

| Beta | Clean Acc | Robust Acc | Use Case |
|------|-----------|------------|----------|
| 1.0  | High      | Low        | Prefer clean accuracy |
| 3.0  | Moderate  | Moderate   | Balanced |
| 6.0  | Moderate  | High       | **Recommended** |
| 10.0 | Low       | Very High  | Maximum robustness |
| 15.0 | Very Low  | Very High  | Extreme robustness |

**Recommendation:** β = 6.0 provides best balance for medical imaging.

---

## Evaluation & Analysis

### Metrics Hierarchy

```
Evaluation
├── Clean Performance
│   ├── Accuracy
│   ├── Balanced Accuracy
│   ├── F1 (Macro/Weighted)
│   ├── Precision/Recall
│   └── AUROC/AUPRC
│
├── Robustness
│   ├── FGSM (ε = 2/255, 4/255, 8/255)
│   ├── PGD (ε = 2/255, 4/255, 8/255, 20 steps)
│   └── C&W (optional, expensive)
│
├── Calibration
│   ├── ECE (Expected Calibration Error)
│   ├── MCE (Maximum Calibration Error)
│   ├── Brier Score
│   └── NLL (Negative Log-Likelihood)
│
└── Generalization
    ├── Cross-site (ISIC 2019/2020, Derm7pt)
    └── Per-class analysis
```

### Statistical Tests

#### Paired t-test
```python
H₀: μ_TRADES = μ_PGD-AT
H₁: μ_TRADES ≠ μ_PGD-AT

Significance level: α = 0.01
```

#### Effect Size (Cohen's d)
```
|d| < 0.2  : Negligible
0.2 ≤ |d| < 0.5 : Small
0.5 ≤ |d| < 0.8 : Medium
|d| ≥ 0.8  : Large (practically significant)
```

#### Bootstrap CI (99%)
- 10,000 bootstrap samples
- Provides robust uncertainty estimates
- Non-parametric method

---

## Results Interpretation

### Example Output

```
==============================================================================
TRADES vs PGD-AT Comparison
==============================================================================

Clean Accuracy:
  TRADES:   0.8542 ± 0.0089
  PGD-AT:   0.8193 ± 0.0112
  Diff:     +0.0349 (p < 0.01, Cohen's d = 1.24)
  95% CI:   [0.0198, 0.0501]
  ✓ TRADES significantly better

Robust Accuracy (PGD ε=8/255):
  TRADES:   0.7231 ± 0.0145
  PGD-AT:   0.6987 ± 0.0167
  Diff:     +0.0244 (p < 0.01, Cohen's d = 0.82)
  95% CI:   [0.0089, 0.0399]
  ✓ TRADES significantly better

ECE (Expected Calibration Error):
  TRADES:   0.0423 ± 0.0034
  PGD-AT:   0.0587 ± 0.0041
  Diff:     -0.0164 (p < 0.01, Cohen's d = -1.09)
  95% CI:   [-0.0243, -0.0085]
  ✓ TRADES significantly better (lower is better)

Pareto Analysis:
  Knee Point: TRADES (β=6.0)
  Clean Acc: 0.8542, Robust Acc: 0.7231
  ✓ Dominates PGD-AT in both objectives
```

### Interpretation Guidelines

**Clean Accuracy:**
- **TRADES > PGD-AT**: Explicit clean loss preservation works!
- **Practical significance**: Cohen's d > 0.8 → large effect

**Robust Accuracy:**
- **TRADES ≥ PGD-AT**: KL divergence maintains robustness
- **Trade-off**: Slight gain in robustness with higher clean accuracy

**Calibration (ECE):**
- **Lower is better**: TRADES produces better-calibrated models
- **Clinical importance**: More reliable confidence estimates for medical imaging

**Pareto Dominance:**
- **TRADES dominates**: Better in both clean and robust accuracy
- **Knee point**: Optimal balance on Pareto frontier

---

## Troubleshooting

### Common Issues

#### 1. Out of Memory (OOM)
```powershell
# Reduce batch size in config
training:
  batch_size: 16  # Instead of 32
```

#### 2. Slow Training
```powershell
# Enable AMP
training:
  amp:
    enabled: true

# Reduce PGD steps during training
trades:
  train_attack:
    num_steps: 5  # Instead of 7
```

#### 3. Unstable Training
```powershell
# Enable gradient clipping
training:
  grad_clip:
    enabled: true
    max_norm: 1.0

# Reduce learning rate
training:
  learning_rate: 0.001  # Instead of 0.01
```

#### 4. Phase 5.2 Results Not Found
```powershell
# Update path in automation script
$PHASE_5_2_RESULTS = "path/to/your/phase_5_2_results"
```

#### 5. MLflow Connection Error
```yaml
# Disable MLflow in config
output:
  mlflow:
    enabled: false
```

---

## References

### Papers
1. **TRADES Paper**: Zhang et al. (2019). "Theoretically Principled Trade-off between Robustness and Accuracy." ICML 2019.
2. **PGD-AT**: Madry et al. (2018). "Towards Deep Learning Models Resistant to Adversarial Attacks." ICLR 2018.
3. **Adversarial Training Survey**: Bai et al. (2021). "Recent Advances in Adversarial Training for Adversarial Robustness." IJCAI 2021.

### Code References
- PyTorch AMP: https://pytorch.org/docs/stable/amp.html
- MLflow: https://mlflow.org/docs/latest/index.html
- Scipy Stats: https://docs.scipy.org/doc/scipy/reference/stats.html

---

## Appendix A: Full Command Reference

### Training Commands
```powershell
# Single model
python scripts/training/train_trades.py --config configs/experiments/trades_isic.yaml --seed 42 --model resnet50

# All seeds for one model
foreach ($seed in 42, 123, 456) {
    python scripts/training/train_trades.py --config configs/experiments/trades_isic.yaml --seed $seed --model resnet50
}

# All models and seeds
foreach ($model in "resnet50", "efficientnet_b0", "vit_b_16") {
    foreach ($seed in 42, 123, 456) {
        python scripts/training/train_trades.py --config configs/experiments/trades_isic.yaml --seed $seed --model $model
    }
}
```

### Evaluation Commands
```powershell
# Single model
python scripts/evaluation/evaluate_trades.py `
    --config configs/experiments/trades_isic.yaml `
    --checkpoint results/phase_5_3_trades/checkpoints/resnet50_seed_42/best.pt `
    --output_dir results/phase_5_3_trades/evaluation_metrics/resnet50_seed_42

# Batch evaluation
Get-ChildItem "results/phase_5_3_trades/checkpoints/*/best.pt" | ForEach-Object {
    $checkpoint = $_.FullName
    $outputDir = $checkpoint -replace "checkpoints", "evaluation_metrics" -replace "\\best.pt", ""
    python scripts/evaluation/evaluate_trades.py --config configs/experiments/trades_isic.yaml --checkpoint $checkpoint --output_dir $outputDir
}
```

### Analysis Commands
```powershell
# Statistical comparison
python src/evaluation/comparison.py `
    --trades_results results/phase_5_3_trades/comparison/trades_aggregated.json `
    --baseline_results results/phase_5_2_pgd_at/comparison/pgd_at_aggregated.json `
    --output results/phase_5_3_trades/comparison/statistical_comparison.json

# Trade-off analysis
python src/evaluation/tradeoff_analysis.py `
    --results_dir results/phase_5_3_trades/evaluation_metrics `
    --output results/phase_5_3_trades/comparison/tradeoff_analysis.json

# Visualization
python src/visualization/pareto_curves.py `
    --results results/phase_5_3_trades/comparison/tradeoff_analysis.json `
    --output_dir results/phase_5_3_trades/evaluation_plots
```

---

## Contact & Support

**Author:** Viraj Pankaj Jain
**Institution:** University of Glasgow
**Project:** Tri-Objective Robust XAI for Medical Imaging
**Phase:** 5.3 - TRADES Adversarial Training

For issues, questions, or contributions, please refer to the project repository.

---

**End of Guide**
