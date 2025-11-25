# Phase 5.4: TRADES Hyperparameter Optimization - Complete Guide

**Author:** Viraj Pankaj Jain
**Institution:** University of Glasgow, School of Computing Science
**Date:** November 24, 2025
**Version:** 5.4.0

---

## Table of Contents

1. [Overview](#overview)
2. [Prerequisites](#prerequisites)
3. [Installation](#installation)
4. [Quick Start](#quick-start)
5. [Detailed Usage](#detailed-usage)
6. [Architecture](#architecture)
7. [Configuration](#configuration)
8. [Troubleshooting](#troubleshooting)
9. [Results Interpretation](#results-interpretation)
10. [Advanced Usage](#advanced-usage)

---

## Overview

Phase 5.4 implements **hyperparameter optimization (HPO)** for TRADES adversarial training using **Optuna**. The system optimizes three key hyperparameters:

- **β** (TRADES trade-off parameter): [3.0, 10.0]
- **ε** (perturbation budget): {4/255, 6/255, 8/255}
- **Learning rate**: [1e-4, 1e-3]

**Objective Function:**
```
Maximize: 0.4 × robust_acc + 0.3 × clean_acc + 0.3 × cross_site_AUROC
```

**Key Features:**
- 50 trials with TPE (Tree-structured Parzen Estimator) sampler
- Median pruning (n_startup_trials=10) for early stopping
- Comprehensive analysis and visualization
- Automatic retraining with optimal hyperparameters
- MLflow integration for experiment tracking

---

## Prerequisites

### System Requirements
- **OS:** Windows 10/11, Linux, macOS
- **Python:** 3.8+
- **GPU:** CUDA-capable GPU recommended (optional)
- **RAM:** 8GB minimum, 16GB recommended
- **Storage:** 10GB for datasets + checkpoints

### Python Dependencies
```bash
# Core dependencies
torch>=2.0.0
torchvision>=0.15.0
optuna>=3.0.0
numpy>=1.24.0
pandas>=2.0.0

# Visualization
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0

# Optional
mlflow>=2.9.0  # For experiment tracking
openpyxl>=3.1.0  # For Excel export
```

---

## Installation

### 1. Install Dependencies
```powershell
# Install from requirements.txt
pip install -r requirements.txt

# Or install individually
pip install torch torchvision optuna numpy pandas matplotlib seaborn plotly mlflow
```

### 2. Verify Installation
```powershell
python -c "import torch; import optuna; print('Environment ready!')"
```

### 3. Create Required Directories
```powershell
# Automatically created by scripts, but you can manually create:
mkdir logs, checkpoints, results/phase_5_4
```

---

## Quick Start

### Option 1: Full Pipeline (Recommended)
```powershell
# Complete HPO workflow: Study → Analysis → Retrain
.\RUN_PHASE_5_4_COMPLETE.ps1
```

### Option 2: Quick Test
```powershell
# Fast test (10 trials, 2 epochs, subset data)
.\RUN_PHASE_5_4_COMPLETE.ps1 -QuickTest
```

### Option 3: Individual Steps
```powershell
# Step 1: HPO Study
python scripts/run_hpo_study.py --n-trials 50 --n-epochs 10

# Step 2: Retrain with Optimal Hyperparameters
python scripts/retrain_optimal.py --study-name trades_hpo_phase54 --n-epochs 200
```

---

## Detailed Usage

### HPO Study Execution

#### Basic Usage
```powershell
python scripts/run_hpo_study.py `
    --study-name trades_hpo_phase54 `
    --n-trials 50 `
    --n-epochs 10 `
    --dataset cifar10 `
    --model resnet18
```

#### All Parameters
```powershell
python scripts/run_hpo_study.py `
    # Study configuration
    --study-name trades_hpo_phase54 `
    --n-trials 50 `
    --n-epochs 10 `
    --storage sqlite:///hpo_study.db `

    # Data configuration
    --dataset cifar10 `
    --data-dir data `
    --batch-size 128 `
    --num-workers 4 `

    # Model configuration
    --model resnet18 `

    # Device configuration
    --device auto `
    --gpu-id 0 `

    # Output configuration
    --output-dir results/phase_5_4 `
    --checkpoint-dir checkpoints/hpo `

    # Objective weights
    --robust-weight 0.4 `
    --clean-weight 0.3 `
    --auroc-weight 0.3 `

    # Flags
    --quick-test `
    --skip-analysis
```

### Model Retraining

#### Basic Usage
```powershell
python scripts/retrain_optimal.py `
    --study-name trades_hpo_phase54 `
    --n-epochs 200
```

#### With Advanced Options
```powershell
python scripts/retrain_optimal.py `
    --study-name trades_hpo_phase54 `
    --storage sqlite:///hpo_study.db `
    --n-epochs 200 `
    --use-scheduler `
    --use-mlflow `
    --save-frequency 10 `
    --checkpoint-dir checkpoints/final_model
```

### PowerShell Automation Script

#### Full Pipeline
```powershell
.\RUN_PHASE_5_4_COMPLETE.ps1 `
    -NTrials 50 `
    -RetrainEpochs 200 `
    -UseScheduler `
    -UseMLflow
```

#### Execution Modes
```powershell
# Quick test
.\RUN_PHASE_5_4_COMPLETE.ps1 -QuickTest

# HPO only (skip retraining)
.\RUN_PHASE_5_4_COMPLETE.ps1 -SkipRetrain

# Retrain only (skip HPO)
.\RUN_PHASE_5_4_COMPLETE.ps1 -SkipHPO

# Skip analysis
.\RUN_PHASE_5_4_COMPLETE.ps1 -SkipAnalysis
```

#### Custom Configuration
```powershell
.\RUN_PHASE_5_4_COMPLETE.ps1 `
    -StudyName my_custom_study `
    -Dataset cifar100 `
    -Model resnet34 `
    -NTrials 100 `
    -RetrainEpochs 300 `
    -BatchSize 256 `
    -Device cuda `
    -GpuId 1
```

---

## Architecture

### Module Structure
```
tri-objective-robust-xai-medimg/
├── src/training/
│   ├── hpo_config.py          # Configuration classes
│   ├── hpo_objective.py       # Objective functions
│   ├── hpo_trainer.py         # HPO trainer with Optuna
│   └── hpo_analysis.py        # Analysis and visualization
├── scripts/
│   ├── run_hpo_study.py       # Main HPO execution
│   └── retrain_optimal.py     # Retrain with optimal params
├── RUN_PHASE_5_4_COMPLETE.ps1 # Complete automation
└── PHASE_5_4_COMPLETE_GUIDE.md
```

### Component Descriptions

#### 1. `hpo_config.py`
- **Classes:** `HPOConfig`, `SearchSpace`, `PrunerConfig`, `SamplerConfig`
- **Purpose:** Configuration management for HPO studies
- **Key Functions:**
  - `create_default_hpo_config()`: Factory for default configuration
  - `TRADESSearchSpace()`: Dissertation-specified search space
  - YAML serialization/deserialization

#### 2. `hpo_objective.py`
- **Classes:** `TrialMetrics`, `WeightedTriObjective`, `MultiObjectiveEvaluator`
- **Purpose:** Objective function computation
- **Key Functions:**
  - `WeightedTriObjective.__call__()`: Compute weighted objective
  - `get_intermediate_value()`: For pruning decisions
  - Pareto frontier analysis

#### 3. `hpo_trainer.py`
- **Classes:** `HPOTrainer`, `TRADESHPOTrainer`
- **Purpose:** Training loop with Optuna integration
- **Key Functions:**
  - `create_study()`: Initialize Optuna study
  - `train_and_evaluate()`: Train and evaluate trial
  - `report_intermediate_value()`: Enable pruning
  - TRADES loss computation with PGD attack

#### 4. `hpo_analysis.py`
- **Classes:** `HPOAnalyzer`
- **Purpose:** Results analysis and visualization
- **Key Functions:**
  - `plot_optimization_history()`: Trial progression
  - `plot_parameter_importance()`: fANOVA importance
  - `plot_trade_offs()`: Objective trade-off analysis
  - `create_interactive_plots()`: Plotly visualizations
  - `export_results()`: CSV/JSON/Excel export

#### 5. `run_hpo_study.py`
- **Purpose:** Complete HPO workflow orchestration
- **Features:**
  - Dataset loading (CIFAR-10/100, SVHN)
  - Model factory (ResNet18/34/50)
  - Study execution
  - Automatic analysis

#### 6. `retrain_optimal.py`
- **Purpose:** Retrain with optimal hyperparameters
- **Features:**
  - Load best trial from Optuna database
  - Full training (200 epochs)
  - Learning rate scheduler
  - MLflow integration
  - Checkpoint management

---

## Configuration

### HPO Configuration (hpo_config.yaml)

```yaml
# Study configuration
study_name: trades_hpo_phase54
n_trials: 50
direction: maximize
storage_url: sqlite:///hpo_study.db

# Search space (TRADES)
search_space:
  beta:
    type: float
    low: 3.0
    high: 10.0
    log: false
  epsilon:
    type: categorical
    choices: [0.01568627, 0.02352941, 0.03137255]  # 4/255, 6/255, 8/255
  learning_rate:
    type: float
    low: 0.0001
    high: 0.001
    log: true

# Pruner configuration
pruner:
  pruner_type: median
  n_startup_trials: 10
  n_warmup_steps: 5

# Sampler configuration
sampler:
  sampler_type: tpe
  n_startup_trials: 10
  seed: 42

# Objective weights
objective_weights:
  robust_accuracy: 0.4
  clean_accuracy: 0.3
  cross_site_auroc: 0.3
```

### Loading Configuration
```python
from src.training.hpo_config import HPOConfig

# Load from YAML
config = HPOConfig.from_yaml("configs/hpo_config.yaml")

# Create default
config = create_default_hpo_config()

# Modify
config.n_trials = 100
config.save_yaml("configs/my_config.yaml")
```

---

## Troubleshooting

### Common Issues

#### 1. ModuleNotFoundError
**Error:** `ModuleNotFoundError: No module named 'src'`

**Solution:**
```powershell
# Add project root to PYTHONPATH
$env:PYTHONPATH = "C:\Users\Dissertation\tri-objective-robust-xai-medimg;$env:PYTHONPATH"

# Or use absolute imports in scripts
sys.path.insert(0, str(Path(__file__).parent.parent))
```

#### 2. Optuna Not Found
**Error:** `ModuleNotFoundError: No module named 'optuna'`

**Solution:**
```powershell
pip install optuna
```

#### 3. CUDA Out of Memory
**Error:** `RuntimeError: CUDA out of memory`

**Solution:**
```powershell
# Reduce batch size
python scripts/run_hpo_study.py --batch-size 64

# Use CPU
python scripts/run_hpo_study.py --device cpu
```

#### 4. Study Already Exists
**Error:** `DuplicatedStudyError: Another study with name '...' already exists`

**Solution:**
```powershell
# Use different study name
python scripts/run_hpo_study.py --study-name my_new_study

# Or delete existing database
Remove-Item hpo_study.db
```

#### 5. Missing Logs Directory
**Error:** `FileNotFoundError: [Errno 2] No such file or directory: 'logs/hpo_study.log'`

**Solution:**
```powershell
mkdir logs
```

#### 6. KeyError: 'learning_rate'
**Error:** `KeyError: 'learning_rate'`

**Solution:** Ensure `learning_rate` is in search space:
```python
# In hpo_config.py, TRADESSearchSpace should have:
"learning_rate": SearchSpace(
    space_type=SearchSpaceType.FLOAT,
    low=1e-4,
    high=1e-3,
    log=True
)
```

---

## Results Interpretation

### HPO Summary Report

Located at: `results/phase_5_4/analysis/hpo_summary.json`

```json
{
  "study_name": "trades_hpo_phase54",
  "n_trials": 50,
  "n_complete": 45,
  "n_pruned": 5,
  "best_value": 0.7234,
  "best_params": {
    "beta": 6.42,
    "epsilon": 0.02352941,
    "learning_rate": 0.000432
  },
  "best_trial_number": 37
}
```

**Interpretation:**
- **best_value (0.7234):** Weighted objective = 0.4×robust + 0.3×clean + 0.3×AUROC
- **best_params:** Optimal hyperparameters found
- **n_pruned (5):** Trials stopped early by median pruner
- **Trial 37:** Best performing trial

### Visualizations

#### 1. Optimization History
**File:** `results/phase_5_4/analysis/optimization_history.png`

Shows trial-by-trial objective values and running best. Look for:
- **Convergence:** Best value plateaus
- **Exploration:** Wide scatter indicates good exploration
- **Efficiency:** Early trials should improve quickly

#### 2. Parameter Importance
**File:** `results/phase_5_4/analysis/parameter_importance.png`

fANOVA importance scores. Interpretation:
- **High importance (>0.5):** Parameter strongly affects objective
- **Low importance (<0.1):** Parameter has minimal impact
- **Typical:** β is most important, ε second, LR third

#### 3. Parameter Relationships
**File:** `results/phase_5_4/analysis/parameter_relationships.png`

Scatter plots of parameters vs objective. Look for:
- **Linear trends:** Direct relationship
- **Sweet spots:** Clusters of high-performing values
- **Interactions:** Non-linear patterns suggest parameter interactions

#### 4. Objective Trade-offs
**File:** `results/phase_5_4/analysis/objective_tradeoffs.png`

Pairwise plots of robust_acc, clean_acc, cross_site_AUROC. Look for:
- **Trade-offs:** Negative correlation (e.g., robust vs clean)
- **Pareto front:** Non-dominated solutions
- **Best trial:** Should be near Pareto front

#### 5. Convergence Analysis
**File:** `results/phase_5_4/analysis/convergence_analysis.png`

Shows cumulative best value and improvement per trial. Look for:
- **Plateau:** Indicates convergence
- **Late improvements:** May need more trials
- **Steady decline in improvements:** Good convergence

### Final Model Performance

Located at: `results/phase_5_4/final_model/training_summary.json`

```json
{
  "best_epoch": 142,
  "best_robust_accuracy": 0.6845,
  "final_clean_accuracy": 0.8923,
  "final_robust_accuracy": 0.6812
}
```

**Expected Ranges (CIFAR-10):**
- **Clean accuracy:** 85-92%
- **Robust accuracy (ε=6/255):** 60-70%
- **Cross-site AUROC:** 75-85%

---

## Advanced Usage

### Custom Objective Function

```python
from src.training.hpo_objective import ObjectiveFunction, TrialMetrics

class CustomObjective(ObjectiveFunction):
    def __call__(self, metrics: TrialMetrics) -> float:
        # Custom weighting
        return (
            0.5 * metrics.robust_accuracy +
            0.3 * metrics.clean_accuracy +
            0.2 * metrics.explanation_stability
        )

    def get_intermediate_value(self, metrics, epoch):
        return self(metrics)
```

### Multi-Objective Optimization

```python
from optuna.samplers import NSGAIISampler

# Create multi-objective study
study = optuna.create_study(
    directions=["maximize", "maximize", "maximize"],  # 3 objectives
    sampler=NSGAIISampler()
)

# Define multi-objective function
def objective(trial):
    # ... training code ...
    return robust_acc, clean_acc, auroc  # Return tuple
```

### Custom Search Space

```python
from src.training.hpo_config import SearchSpace, SearchSpaceType

custom_search = {
    "beta": SearchSpace(
        space_type=SearchSpaceType.FLOAT,
        low=1.0,
        high=15.0
    ),
    "epsilon": SearchSpace(
        space_type=SearchSpaceType.FLOAT,
        low=0.01,
        high=0.05
    ),
    "learning_rate": SearchSpace(
        space_type=SearchSpaceType.FLOAT,
        low=1e-5,
        high=1e-2,
        log=True
    ),
    "weight_decay": SearchSpace(
        space_type=SearchSpaceType.FLOAT,
        low=1e-5,
        high=1e-3,
        log=True
    )
}
```

### MLflow Integration

```python
import mlflow

# Enable MLflow tracking
mlflow.set_tracking_uri("mlruns")
mlflow.set_experiment("phase_5_4_hpo")

# In training loop
with mlflow.start_run():
    mlflow.log_params(hyperparams)
    mlflow.log_metrics(metrics)
    mlflow.log_artifact("model.pt")
```

### Distributed HPO

```python
# On multiple machines, run:
python scripts/run_hpo_study.py \
    --storage postgresql://user:pass@host/db \
    --study-name shared_study \
    --n-trials 20  # Each machine runs 20 trials

# Total: N_machines × 20 trials
```

---

## Performance Tips

### 1. Speed Up HPO
- Use `--quick-test` for initial testing
- Reduce `--n-epochs` (e.g., 5 instead of 10)
- Use smaller `--batch-size` if GPU memory limited
- Enable pruning with aggressive settings

### 2. Improve Results
- Increase `--n-trials` (e.g., 100-200)
- Use learning rate scheduler in retraining
- Experiment with different objective weights
- Try different samplers (CMA-ES for continuous, NSGA-II for multi-objective)

### 3. Memory Management
- Use `pin_memory=False` in DataLoader
- Reduce `--num-workers`
- Clear CUDA cache between trials:
  ```python
  torch.cuda.empty_cache()
  ```

---

## Citation

If you use this code in your research, please cite:

```bibtex
@mastersthesis{jain2025trades_hpo,
  author = {Viraj Pankaj Jain},
  title = {Tri-Objective Robust Explainable AI for Medical Imaging},
  school = {University of Glasgow},
  year = {2025},
  type = {MSc Dissertation},
  note = {Phase 5.4: TRADES Hyperparameter Optimization}
}
```

---

## Support

For issues or questions:
1. Check [Troubleshooting](#troubleshooting) section
2. Review logs in `logs/` directory
3. Verify environment with `pip list`
4. Contact: [your-email@glasgow.ac.uk]

---

## License

This project is part of an MSc dissertation at the University of Glasgow.
All rights reserved.

---

**Last Updated:** November 24, 2025
**Version:** 5.4.0
**Status:** Production Ready ✅
