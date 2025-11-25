# Google Colab Training Guide - Complete Workflow
## Baseline to Phase 5.2 PGD-AT Training with 3 Seeds

**Last Updated:** November 24, 2025
**Purpose:** Complete command reference for training all models in Google Colab from scratch

---

## 📋 Table of Contents

1. [Setup & Environment](#1-setup--environment)
2. [Dataset Preparation](#2-dataset-preparation)
3. [Baseline Training (3 Seeds)](#3-baseline-training-3-seeds)
4. [Phase 5.2 - PGD Adversarial Training (3 Seeds)](#4-phase-52---pgd-adversarial-training-3-seeds)
5. [Evaluation & Testing](#5-evaluation--testing)
6. [Results Collection](#6-results-collection)

---

## 1. Setup & Environment

### 1.1 Mount Google Drive
```python
from google.colab import drive
drive.mount('/content/drive')
```

### 1.2 Clone Repository
```bash
# Navigate to your drive
cd /content/drive/MyDrive

# Clone the repository (if not already cloned)
!git clone https://github.com/viraj1011JAIN/tri-objective-robust-xai-medimg.git

# Navigate to project directory
cd tri-objective-robust-xai-medimg
```

### 1.3 Install Dependencies
```bash
# Install PyTorch with CUDA support
!pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# Install project dependencies
!pip install -r requirements.txt

# Install additional dependencies
!pip install timm tensorboard scikit-learn pandas numpy matplotlib seaborn pillow pyyaml tqdm mlflow dvc
```

### 1.4 Verify CUDA and Environment
```python
import torch
print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"CUDA version: {torch.version.cuda}")
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None'}")

# Set environment variable for dataset path
import os
os.environ['DATASET_ROOT'] = '/content/drive/MyDrive/data'
```

---

## 2. Dataset Preparation

### 2.1 Verify Dataset Structure
```bash
# Check if datasets are accessible
!ls -lh /content/drive/MyDrive/data/

# Expected structure:
# /content/drive/MyDrive/data/
# ├── isic_2018/
# │   ├── metadata.csv
# │   └── ISIC2018_Task3_Training_Input/
# ├── isic_2019/
# │   ├── metadata.csv
# │   └── train-image/
# ├── isic_2020/
# │   ├── metadata.csv
# │   └── train-image/
# ├── derm7pt/
# ├── nih_cxr/
# └── padchest/
```

### 2.2 Data Preprocessing (If Not Already Done)
```bash
# Preprocess ISIC 2018
!python scripts/data/preprocess_data.py \
  --dataset isic2018 \
  --root /content/drive/MyDrive/data/isic_2018 \
  --csv-path /content/drive/MyDrive/data/isic_2018/metadata.csv \
  --output-dir data/processed/isic2018 \
  --to-hdf5

# Preprocess ISIC 2019
!python scripts/data/preprocess_data.py \
  --dataset isic2019 \
  --root /content/drive/MyDrive/data/isic_2019 \
  --csv-path /content/drive/MyDrive/data/isic_2019/metadata.csv \
  --output-dir data/processed/isic2019 \
  --to-hdf5

# Preprocess ISIC 2020
!python scripts/data/preprocess_data.py \
  --dataset isic2020 \
  --root /content/drive/MyDrive/data/isic_2020 \
  --csv-path /content/drive/MyDrive/data/isic_2020/metadata.csv \
  --output-dir data/processed/isic2020 \
  --to-hdf5
```

### 2.3 Verify Processed Data
```bash
!ls -lh data/processed/isic2018/
!ls -lh data/processed/isic2019/
!ls -lh data/processed/isic2020/
```

---

## 3. Baseline Training (3 Seeds)

### 3.1 Baseline Training - ResNet50 (ISIC 2018)

#### Seed 42
```bash
!python scripts/training/train_resnet50_phase3.py \
  --config configs/experiments/rq1_robustness/baseline_isic2018_resnet50.yaml \
  --seed 42 \
  --experiment-name baseline_resnet50_isic2018_seed42 \
  --output-dir checkpoints/baseline/resnet50_isic2018_seed42 \
  --device cuda \
  --num-workers 4 \
  --mixed-precision
```

#### Seed 123
```bash
!python scripts/training/train_resnet50_phase3.py \
  --config configs/experiments/rq1_robustness/baseline_isic2018_resnet50.yaml \
  --seed 123 \
  --experiment-name baseline_resnet50_isic2018_seed123 \
  --output-dir checkpoints/baseline/resnet50_isic2018_seed123 \
  --device cuda \
  --num-workers 4 \
  --mixed-precision
```

#### Seed 999
```bash
!python scripts/training/train_resnet50_phase3.py \
  --config configs/experiments/rq1_robustness/baseline_isic2018_resnet50.yaml \
  --seed 999 \
  --experiment-name baseline_resnet50_isic2018_seed999 \
  --output-dir checkpoints/baseline/resnet50_isic2018_seed999 \
  --device cuda \
  --num-workers 4 \
  --mixed-precision
```

### 3.2 Baseline Training - EfficientNet-B0 (ISIC 2018)

#### Seed 42
```bash
!python scripts/training/train_efficientnet_phase3.py \
  --config configs/experiments/baseline_isic2018.yaml \
  --seed 42 \
  --experiment-name baseline_efficientnet_isic2018_seed42 \
  --output-dir checkpoints/baseline/efficientnet_isic2018_seed42 \
  --device cuda \
  --num-workers 4 \
  --mixed-precision
```

#### Seed 123
```bash
!python scripts/training/train_efficientnet_phase3.py \
  --config configs/experiments/baseline_isic2018.yaml \
  --seed 123 \
  --experiment-name baseline_efficientnet_isic2018_seed123 \
  --output-dir checkpoints/baseline/efficientnet_isic2018_seed123 \
  --device cuda \
  --num-workers 4 \
  --mixed-precision
```

#### Seed 999
```bash
!python scripts/training/train_efficientnet_phase3.py \
  --config configs/experiments/baseline_isic2018.yaml \
  --seed 999 \
  --experiment-name baseline_efficientnet_isic2018_seed999 \
  --output-dir checkpoints/baseline/efficientnet_isic2018_seed999 \
  --device cuda \
  --num-workers 4 \
  --mixed-precision
```

### 3.3 Baseline Training - ViT (ISIC 2018)

#### Seed 42
```bash
!python scripts/training/train_vit_phase3.py \
  --config configs/experiments/baseline_isic2018.yaml \
  --seed 42 \
  --experiment-name baseline_vit_isic2018_seed42 \
  --output-dir checkpoints/baseline/vit_isic2018_seed42 \
  --device cuda \
  --num-workers 4 \
  --mixed-precision
```

#### Seed 123
```bash
!python scripts/training/train_vit_phase3.py \
  --config configs/experiments/baseline_isic2018.yaml \
  --seed 123 \
  --experiment-name baseline_vit_isic2018_seed123 \
  --output-dir checkpoints/baseline/vit_isic2018_seed123 \
  --device cuda \
  --num-workers 4 \
  --mixed-precision
```

#### Seed 999
```bash
!python scripts/training/train_vit_phase3.py \
  --config configs/experiments/baseline_isic2018.yaml \
  --seed 999 \
  --experiment-name baseline_vit_isic2018_seed999 \
  --output-dir checkpoints/baseline/vit_isic2018_seed999 \
  --device cuda \
  --num-workers 4 \
  --mixed-precision
```

---

## 4. Phase 5.2 - PGD Adversarial Training (3 Seeds)

### 4.1 PGD-AT Training - ResNet50 (ISIC 2018)

#### Seed 42
```bash
!python scripts/training/train_pgd_at.py \
  --config configs/experiments/pgd_at_isic.yaml \
  --seed 42 \
  --experiment-name pgd_at_resnet50_isic2018_seed42 \
  --output-dir checkpoints/pgd_at/resnet50_isic2018_seed42 \
  --device cuda \
  --num-workers 4 \
  --mixed-precision \
  --attack-eps 0.03137 \
  --attack-alpha 0.00784 \
  --attack-steps 10 \
  --attack-random-start
```

#### Seed 123
```bash
!python scripts/training/train_pgd_at.py \
  --config configs/experiments/pgd_at_isic.yaml \
  --seed 123 \
  --experiment-name pgd_at_resnet50_isic2018_seed123 \
  --output-dir checkpoints/pgd_at/resnet50_isic2018_seed123 \
  --device cuda \
  --num-workers 4 \
  --mixed-precision \
  --attack-eps 0.03137 \
  --attack-alpha 0.00784 \
  --attack-steps 10 \
  --attack-random-start
```

#### Seed 999
```bash
!python scripts/training/train_pgd_at.py \
  --config configs/experiments/pgd_at_isic.yaml \
  --seed 999 \
  --experiment-name pgd_at_resnet50_isic2018_seed999 \
  --output-dir checkpoints/pgd_at/resnet50_isic2018_seed999 \
  --device cuda \
  --num-workers 4 \
  --mixed-precision \
  --attack-eps 0.03137 \
  --attack-alpha 0.00784 \
  --attack-steps 10 \
  --attack-random-start
```

### 4.2 PGD-AT Training - EfficientNet-B0 (ISIC 2018)

#### Seed 42
```bash
!python scripts/training/train_pgd_at.py \
  --config configs/experiments/pgd_at_isic.yaml \
  --model efficientnet_b0 \
  --seed 42 \
  --experiment-name pgd_at_efficientnet_isic2018_seed42 \
  --output-dir checkpoints/pgd_at/efficientnet_isic2018_seed42 \
  --device cuda \
  --num-workers 4 \
  --mixed-precision \
  --attack-eps 0.03137 \
  --attack-alpha 0.00784 \
  --attack-steps 10 \
  --attack-random-start
```

#### Seed 123
```bash
!python scripts/training/train_pgd_at.py \
  --config configs/experiments/pgd_at_isic.yaml \
  --model efficientnet_b0 \
  --seed 123 \
  --experiment-name pgd_at_efficientnet_isic2018_seed123 \
  --output-dir checkpoints/pgd_at/efficientnet_isic2018_seed123 \
  --device cuda \
  --num-workers 4 \
  --mixed-precision \
  --attack-eps 0.03137 \
  --attack-alpha 0.00784 \
  --attack-steps 10 \
  --attack-random-start
```

#### Seed 999
```bash
!python scripts/training/train_pgd_at.py \
  --config configs/experiments/pgd_at_isic.yaml \
  --model efficientnet_b0 \
  --seed 999 \
  --experiment-name pgd_at_efficientnet_isic2018_seed999 \
  --output-dir checkpoints/pgd_at/efficientnet_isic2018_seed999 \
  --device cuda \
  --num-workers 4 \
  --mixed-precision \
  --attack-eps 0.03137 \
  --attack-alpha 0.00784 \
  --attack-steps 10 \
  --attack-random-start
```

### 4.3 PGD-AT Training - ViT (ISIC 2018)

#### Seed 42
```bash
!python scripts/training/train_pgd_at.py \
  --config configs/experiments/pgd_at_isic.yaml \
  --model vit_b_16 \
  --seed 42 \
  --experiment-name pgd_at_vit_isic2018_seed42 \
  --output-dir checkpoints/pgd_at/vit_isic2018_seed42 \
  --device cuda \
  --num-workers 4 \
  --mixed-precision \
  --attack-eps 0.03137 \
  --attack-alpha 0.00784 \
  --attack-steps 10 \
  --attack-random-start
```

#### Seed 123
```bash
!python scripts/training/train_pgd_at.py \
  --config configs/experiments/pgd_at_isic.yaml \
  --model vit_b_16 \
  --seed 123 \
  --experiment-name pgd_at_vit_isic2018_seed123 \
  --output-dir checkpoints/pgd_at/vit_isic2018_seed123 \
  --device cuda \
  --num-workers 4 \
  --mixed-precision \
  --attack-eps 0.03137 \
  --attack-alpha 0.00784 \
  --attack-steps 10 \
  --attack-random-start
```

#### Seed 999
```bash
!python scripts/training/train_pgd_at.py \
  --config configs/experiments/pgd_at_isic.yaml \
  --model vit_b_16 \
  --seed 999 \
  --experiment-name pgd_at_vit_isic2018_seed999 \
  --output-dir checkpoints/pgd_at/vit_isic2018_seed999 \
  --device cuda \
  --num-workers 4 \
  --mixed-precision \
  --attack-eps 0.03137 \
  --attack-alpha 0.00784 \
  --attack-steps 10 \
  --attack-random-start
```

---

## 5. Evaluation & Testing

### 5.1 Evaluate Baseline Models

#### ResNet50 - All Seeds
```bash
# Seed 42
!python scripts/evaluation/evaluate_baseline.py \
  --checkpoint checkpoints/baseline/resnet50_isic2018_seed42/best.pt \
  --config configs/experiments/rq1_robustness/baseline_isic2018_resnet50.yaml \
  --output-dir results/baseline/resnet50_seed42 \
  --device cuda

# Seed 123
!python scripts/evaluation/evaluate_baseline.py \
  --checkpoint checkpoints/baseline/resnet50_isic2018_seed123/best.pt \
  --config configs/experiments/rq1_robustness/baseline_isic2018_resnet50.yaml \
  --output-dir results/baseline/resnet50_seed123 \
  --device cuda

# Seed 999
!python scripts/evaluation/evaluate_baseline.py \
  --checkpoint checkpoints/baseline/resnet50_isic2018_seed999/best.pt \
  --config configs/experiments/rq1_robustness/baseline_isic2018_resnet50.yaml \
  --output-dir results/baseline/resnet50_seed999 \
  --device cuda
```

#### EfficientNet-B0 - All Seeds
```bash
# Seed 42
!python scripts/evaluation/evaluate_baseline.py \
  --checkpoint checkpoints/baseline/efficientnet_isic2018_seed42/best.pt \
  --config configs/experiments/baseline_isic2018.yaml \
  --output-dir results/baseline/efficientnet_seed42 \
  --device cuda

# Seed 123
!python scripts/evaluation/evaluate_baseline.py \
  --checkpoint checkpoints/baseline/efficientnet_isic2018_seed123/best.pt \
  --config configs/experiments/baseline_isic2018.yaml \
  --output-dir results/baseline/efficientnet_seed123 \
  --device cuda

# Seed 999
!python scripts/evaluation/evaluate_baseline.py \
  --checkpoint checkpoints/baseline/efficientnet_isic2018_seed999/best.pt \
  --config configs/experiments/baseline_isic2018.yaml \
  --output-dir results/baseline/efficientnet_seed999 \
  --device cuda
```

#### ViT - All Seeds
```bash
# Seed 42
!python scripts/evaluation/evaluate_baseline.py \
  --checkpoint checkpoints/baseline/vit_isic2018_seed42/best.pt \
  --config configs/experiments/baseline_isic2018.yaml \
  --output-dir results/baseline/vit_seed42 \
  --device cuda

# Seed 123
!python scripts/evaluation/evaluate_baseline.py \
  --checkpoint checkpoints/baseline/vit_isic2018_seed123/best.pt \
  --config configs/experiments/baseline_isic2018.yaml \
  --output-dir results/baseline/vit_seed123 \
  --device cuda

# Seed 999
!python scripts/evaluation/evaluate_baseline.py \
  --checkpoint checkpoints/baseline/vit_isic2018_seed999/best.pt \
  --config configs/experiments/baseline_isic2018.yaml \
  --output-dir results/baseline/vit_seed999 \
  --device cuda
```

### 5.2 Evaluate PGD-AT Models

#### ResNet50 - All Seeds
```bash
# Seed 42
!python scripts/evaluation/evaluate_pgd_at.py \
  --checkpoint checkpoints/pgd_at/resnet50_isic2018_seed42/best.pt \
  --config configs/experiments/pgd_at_isic.yaml \
  --output-dir results/pgd_at/resnet50_seed42 \
  --device cuda \
  --attack-eps 0.03137 \
  --attack-alpha 0.00784 \
  --attack-steps 20

# Seed 123
!python scripts/evaluation/evaluate_pgd_at.py \
  --checkpoint checkpoints/pgd_at/resnet50_isic2018_seed123/best.pt \
  --config configs/experiments/pgd_at_isic.yaml \
  --output-dir results/pgd_at/resnet50_seed123 \
  --device cuda \
  --attack-eps 0.03137 \
  --attack-alpha 0.00784 \
  --attack-steps 20

# Seed 999
!python scripts/evaluation/evaluate_pgd_at.py \
  --checkpoint checkpoints/pgd_at/resnet50_isic2018_seed999/best.pt \
  --config configs/experiments/pgd_at_isic.yaml \
  --output-dir results/pgd_at/resnet50_seed999 \
  --device cuda \
  --attack-eps 0.03137 \
  --attack-alpha 0.00784 \
  --attack-steps 20
```

#### EfficientNet-B0 - All Seeds
```bash
# Seed 42
!python scripts/evaluation/evaluate_pgd_at.py \
  --checkpoint checkpoints/pgd_at/efficientnet_isic2018_seed42/best.pt \
  --config configs/experiments/pgd_at_isic.yaml \
  --output-dir results/pgd_at/efficientnet_seed42 \
  --device cuda \
  --attack-eps 0.03137 \
  --attack-alpha 0.00784 \
  --attack-steps 20

# Seed 123
!python scripts/evaluation/evaluate_pgd_at.py \
  --checkpoint checkpoints/pgd_at/efficientnet_isic2018_seed123/best.pt \
  --config configs/experiments/pgd_at_isic.yaml \
  --output-dir results/pgd_at/efficientnet_seed123 \
  --device cuda \
  --attack-eps 0.03137 \
  --attack-alpha 0.00784 \
  --attack-steps 20

# Seed 999
!python scripts/evaluation/evaluate_pgd_at.py \
  --checkpoint checkpoints/pgd_at/efficientnet_isic2018_seed999/best.pt \
  --config configs/experiments/pgd_at_isic.yaml \
  --output-dir results/pgd_at/efficientnet_seed999 \
  --device cuda \
  --attack-eps 0.03137 \
  --attack-alpha 0.00784 \
  --attack-steps 20
```

#### ViT - All Seeds
```bash
# Seed 42
!python scripts/evaluation/evaluate_pgd_at.py \
  --checkpoint checkpoints/pgd_at/vit_isic2018_seed42/best.pt \
  --config configs/experiments/pgd_at_isic.yaml \
  --output-dir results/pgd_at/vit_seed42 \
  --device cuda \
  --attack-eps 0.03137 \
  --attack-alpha 0.00784 \
  --attack-steps 20

# Seed 123
!python scripts/evaluation/evaluate_pgd_at.py \
  --checkpoint checkpoints/pgd_at/vit_isic2018_seed123/best.pt \
  --config configs/experiments/pgd_at_isic.yaml \
  --output-dir results/pgd_at/vit_seed123 \
  --device cuda \
  --attack-eps 0.03137 \
  --attack-alpha 0.00784 \
  --attack-steps 20

# Seed 999
!python scripts/evaluation/evaluate_pgd_at.py \
  --checkpoint checkpoints/pgd_at/vit_isic2018_seed999/best.pt \
  --config configs/experiments/pgd_at_isic.yaml \
  --output-dir results/pgd_at/vit_seed999 \
  --device cuda \
  --attack-eps 0.03137 \
  --attack-alpha 0.00784 \
  --attack-steps 20
```

### 5.3 Robustness Testing - Multiple Attack Strengths

```bash
# Test with multiple epsilon values
for eps in 0.01 0.02 0.03137 0.05 0.1; do
  for seed in 42 123 999; do
    echo "Testing ResNet50 seed ${seed} with eps ${eps}"
    !python scripts/evaluation/evaluate_pgd_at.py \
      --checkpoint checkpoints/pgd_at/resnet50_isic2018_seed${seed}/best.pt \
      --config configs/experiments/pgd_at_isic.yaml \
      --output-dir results/robustness/resnet50_seed${seed}_eps${eps} \
      --device cuda \
      --attack-eps ${eps} \
      --attack-alpha $(echo "scale=5; ${eps}/4" | bc) \
      --attack-steps 20
  done
done
```

---

## 6. Results Collection

### 6.1 Copy Results to Google Drive
```bash
# Create results backup
!mkdir -p /content/drive/MyDrive/dissertation_results/

# Copy all results
!cp -r checkpoints/ /content/drive/MyDrive/dissertation_results/
!cp -r results/ /content/drive/MyDrive/dissertation_results/
!cp -r logs/ /content/drive/MyDrive/dissertation_results/

# Create timestamp
!date > /content/drive/MyDrive/dissertation_results/last_backup.txt
```

### 6.2 Generate Summary Report
```python
import pandas as pd
import json
from pathlib import Path

# Collect all results
results_summary = []

# Baseline results
for model in ['resnet50', 'efficientnet', 'vit']:
    for seed in [42, 123, 999]:
        result_path = Path(f'results/baseline/{model}_seed{seed}/metrics.json')
        if result_path.exists():
            with open(result_path) as f:
                metrics = json.load(f)
                results_summary.append({
                    'Model': model,
                    'Method': 'Baseline',
                    'Seed': seed,
                    'Clean Accuracy': metrics.get('test_accuracy', 0),
                    'Robust Accuracy': metrics.get('robust_accuracy', 0)
                })

# PGD-AT results
for model in ['resnet50', 'efficientnet', 'vit']:
    for seed in [42, 123, 999]:
        result_path = Path(f'results/pgd_at/{model}_seed{seed}/metrics.json')
        if result_path.exists():
            with open(result_path) as f:
                metrics = json.load(f)
                results_summary.append({
                    'Model': model,
                    'Method': 'PGD-AT',
                    'Seed': seed,
                    'Clean Accuracy': metrics.get('test_accuracy', 0),
                    'Robust Accuracy': metrics.get('robust_accuracy', 0)
                })

# Create DataFrame and save
df = pd.DataFrame(results_summary)
df.to_csv('/content/drive/MyDrive/dissertation_results/training_summary.csv', index=False)
print(df)
print(f"\n✅ Results saved to: /content/drive/MyDrive/dissertation_results/training_summary.csv")
```

### 6.3 Aggregate Multi-Seed Results
```python
# Calculate mean and std across seeds
grouped = df.groupby(['Model', 'Method']).agg({
    'Clean Accuracy': ['mean', 'std'],
    'Robust Accuracy': ['mean', 'std']
}).round(4)

print("\n📊 Aggregated Results (Mean ± Std across 3 seeds):")
print(grouped)

# Save aggregated results
grouped.to_csv('/content/drive/MyDrive/dissertation_results/aggregated_results.csv')
```

---

## 📝 Notes & Best Practices

### Training Tips:
1. **Monitor GPU Memory**: Use `nvidia-smi` to check GPU usage
2. **Use Mixed Precision**: Reduces memory usage and speeds up training
3. **Checkpoint Frequently**: Colab sessions can disconnect unexpectedly
4. **Batch Size**: Adjust based on available GPU memory (default: 32-64)
5. **Early Stopping**: Models typically converge in 50-100 epochs

### Attack Parameters:
- **ε (epsilon)**: 0.03137 (8/255) for medical images
- **α (alpha)**: 0.00784 (2/255) - step size
- **Steps**: 10 for training, 20 for evaluation
- **Random Start**: Enabled for stronger adversarial examples

### Expected Training Time (per model):
- **Baseline Training**: ~2-3 hours (50 epochs)
- **PGD-AT Training**: ~4-6 hours (50 epochs)
- **Evaluation**: ~15-30 minutes per checkpoint

### Troubleshooting:
1. **Out of Memory**: Reduce batch size or use gradient accumulation
2. **CUDA Errors**: Restart runtime and clear GPU memory
3. **Dataset Not Found**: Verify path `/content/drive/MyDrive/data/`
4. **Slow Training**: Ensure GPU runtime is enabled in Colab settings

---

## 🎯 Quick Command Summary

### Full Pipeline (All Models, All Seeds)
```bash
# 1. Setup
cd /content/drive/MyDrive/tri-objective-robust-xai-medimg
!pip install -r requirements.txt

# 2. Train Baseline (9 runs total: 3 models × 3 seeds)
for seed in 42 123 999; do
  # ResNet50
  !python scripts/training/train_resnet50_phase3.py --config configs/experiments/rq1_robustness/baseline_isic2018_resnet50.yaml --seed ${seed} --output-dir checkpoints/baseline/resnet50_seed${seed}

  # EfficientNet-B0
  !python scripts/training/train_efficientnet_phase3.py --config configs/experiments/baseline_isic2018.yaml --seed ${seed} --output-dir checkpoints/baseline/efficientnet_seed${seed}

  # ViT
  !python scripts/training/train_vit_phase3.py --config configs/experiments/baseline_isic2018.yaml --seed ${seed} --output-dir checkpoints/baseline/vit_seed${seed}
done

# 3. Train PGD-AT (9 runs total: 3 models × 3 seeds)
for seed in 42 123 999; do
  # ResNet50
  !python scripts/training/train_pgd_at.py --config configs/experiments/pgd_at_isic.yaml --seed ${seed} --output-dir checkpoints/pgd_at/resnet50_seed${seed}

  # EfficientNet-B0
  !python scripts/training/train_pgd_at.py --config configs/experiments/pgd_at_isic.yaml --model efficientnet_b0 --seed ${seed} --output-dir checkpoints/pgd_at/efficientnet_seed${seed}

  # ViT
  !python scripts/training/train_pgd_at.py --config configs/experiments/pgd_at_isic.yaml --model vit_b_16 --seed ${seed} --output-dir checkpoints/pgd_at/vit_seed${seed}
done

# 4. Evaluate All
# (Run evaluation commands from Section 5)

# 5. Backup Results
!cp -r checkpoints/ results/ logs/ /content/drive/MyDrive/dissertation_results/
```

---

## 📊 Expected Results Structure

```
/content/drive/MyDrive/dissertation_results/
├── checkpoints/
│   ├── baseline/
│   │   ├── resnet50_seed42/
│   │   │   ├── best.pt
│   │   │   ├── last.pt
│   │   │   └── training_history.json
│   │   ├── resnet50_seed123/
│   │   ├── resnet50_seed999/
│   │   ├── efficientnet_seed42/
│   │   ├── efficientnet_seed123/
│   │   ├── efficientnet_seed999/
│   │   ├── vit_seed42/
│   │   ├── vit_seed123/
│   │   └── vit_seed999/
│   └── pgd_at/
│       ├── resnet50_seed42/
│       ├── resnet50_seed123/
│       ├── resnet50_seed999/
│       ├── efficientnet_seed42/
│       ├── efficientnet_seed123/
│       ├── efficientnet_seed999/
│       ├── vit_seed42/
│       ├── vit_seed123/
│       └── vit_seed999/
├── results/
│   ├── baseline/
│   │   └── [evaluation results for each model/seed]
│   └── pgd_at/
│       └── [evaluation results for each model/seed]
├── logs/
│   └── [tensorboard logs]
├── training_summary.csv
├── aggregated_results.csv
└── last_backup.txt
```

---

**Document Version:** 1.0
**Compatible with:** Phase 5.2 PGD-AT Implementation
**Dataset Path:** `/content/drive/MyDrive/data/`
**Repository:** https://github.com/viraj1011JAIN/tri-objective-robust-xai-medimg
