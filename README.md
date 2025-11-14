=======
# tri-objective-robust-xai-medimg

## Docker environment


Adversarially robust and explainable deep learning for medical imaging, with a fully reproducible MLOps backbone.

This repository is structured as a **research-grade** codebase:
- PyTorch training pipelines
- MLflow for experiment tracking
- DVC for data versioning
- Docker for reproducible environments
- pre-commit (black, isort, flake8, mypy) for code quality

At the moment, a **CIFAR-10 debug pipeline** is used as a fast, safe smoke test for the infrastructure. Medical imaging datasets plug into the same framework later.

---

## 1. Environment Setup (Windows / PowerShell)

### 1.1 Clone and create virtual environment
```powershell
# From a suitable workspace directory
cd C:\Users\Dissertation

# Clone the repository
git clone https://github.com/viraj1011JAIN/tri-objective-robust-xai-medimg.git
cd .\tri-objective-robust-xai-medimg

# Create and activate virtualenv
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# Upgrade pip and install dependencies
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## 1.2 Install pre-commit hooks
```powershell
pre-commit install
```

This ensures code quality checks run automatically before each commit.

---

## 2. Project Structure
```
tri-objective-robust-xai-medimg/
├── .dvc/                     # DVC configuration
├── .github/                  # CI/CD workflows
├── configs/                  # YAML configuration files
│   ├── datasets/            # Dataset configs
│   ├── models/              # Model configs
│   └── experiments/         # Experiment configs
├── data/                     # Data directory (DVC-tracked)
│   ├── raw/                 # Raw datasets
│   ├── processed/           # Preprocessed data
│   └── concepts/            # Concept banks for TCAV
├── src/                      # Source code
│   ├── datasets/            # Data loaders
│   ├── models/              # Model architectures
│   ├── losses/              # Loss functions
│   ├── attacks/             # Adversarial attacks
│   ├── xai/                 # Explainability methods
│   ├── training/            # Training loops
│   ├── evaluation/          # Evaluation metrics
│   └── utils/               # Utilities
├── scripts/                  # Executable scripts
│   ├── training/            # Training scripts
│   ├── evaluation/          # Evaluation scripts
│   └── data/                # Data processing scripts
├── tests/                    # Unit tests
├── notebooks/               # Jupyter notebooks
├── results/                 # Experiment results
│   ├── checkpoints/        # Model checkpoints
│   ├── logs/               # Training logs
│   ├── metrics/            # Evaluation metrics
│   └── plots/              # Visualizations
├── docs/                    # Documentation
├── .pre-commit-config.yaml  # Pre-commit configuration
├── requirements.txt         # Python dependencies
├── setup.py                # Package setup
├── pyproject.toml          # Project metadata
└── README.md               # This file
```

---

## 3. Quick Start: CIFAR-10 Smoke Test

### 3.1 Train baseline model
```powershell
python scripts/training/train_baseline.py --config configs/experiments/cifar10_baseline.yaml
```

### 3.2 Train adversarially robust model (TRADES)
```powershell
python scripts/training/train_adversarial.py --config configs/experiments/cifar10_trades.yaml
```

### 3.3 Evaluate models
```powershell
python scripts/evaluation/evaluate_robustness.py --checkpoint results/checkpoints/baseline/model_best.pth
```

---

## 4. MLflow Tracking

Start the MLflow UI to view experiment results:
```powershell
mlflow ui
```

Then open http://localhost:5000 in your browser.

---

## 5. DVC Data Versioning

### 5.1 Initialize DVC (first time only)
```powershell
dvc init
```

### 5.2 Track data files
```powershell
dvc add data/raw/ISIC2018
git add data/raw/ISIC2018.dvc .gitignore
git commit -m "Track ISIC2018 dataset with DVC"
```

### 5.3 Pull data from remote
```powershell
dvc pull
```

---

## 6. Running Tests
```powershell
# Run all tests
pytest tests/ -v

# Run with coverage
pytest tests/ --cov=src --cov-report=html

# Run specific test file
pytest tests/test_models.py -v
```

---

## 7. Code Quality

### 7.1 Format code
```powershell
black src/ tests/ scripts/
```

### 7.2 Sort imports
```powershell
isort src/ tests/ scripts/
```

### 7.3 Lint code
```powershell
flake8 src/ tests/ scripts/
```

### 7.4 Type checking
```powershell
mypy src/
```

### 7.5 Run all checks (via pre-commit)
```powershell
pre-commit run --all-files
```

---

## 8. Docker Support

### 8.1 Build Docker image
```powershell
docker build -t tri-objective-xai:latest .
```

### 8.2 Run container
```powershell
docker run --gpus all -it -v C:\Users\Dissertation\tri-objective-robust-xai-medimg/data:/workspace/data -v C:\Users\Dissertation\tri-objective-robust-xai-medimg/results:/workspace/results tri-objective-xai:latest
```

---

## 9. Research Questions (Dissertation Focus)

### RQ1: Joint Optimization of Robustness & Generalization
Can adversarial robustness and cross-site generalization be jointly optimized through unified training objectives?

**Hypotheses:**
- H1a: Tri-objective training improves robust accuracy by ≥35pp over baseline
- H1b: Tri-objective training reduces cross-site AUROC drop by ≥50%
- H1c: Baseline adversarial training does NOT improve cross-site generalization

### RQ2: Concept-Grounded Explanation Stability
Does TCAV-based concept regularization produce explanations that are both adversarially stable and medically grounded?

**Hypotheses:**
- H2a: Explanation SSIM increases from 0.60 to ≥0.75 under adversarial perturbation
- H2b: Artifact TCAV scores decrease from 0.45 to ≤0.20
- H2c: Medical concept TCAV scores increase from 0.58 to ≥0.68

### RQ3: Safe Selective Prediction
Can combining confidence and explanation stability enable safe, reliable clinical deployment?

**Hypotheses:**
- H3a: At 90% coverage, selective accuracy improves by ≥4pp over overall accuracy
- H3b: Error rate on rejected cases is ≥3× higher than on accepted cases
- H3c: Selective prediction provides greater benefit on cross-site test sets
- H3d: ECE decreases after selective rejection

---

## 10. Datasets

### Dermoscopy
- **ISIC 2018**: Melanoma classification (10,015 images)
- **ISIC 2019**: 8 skin lesion types (25,331 images)
- **ISIC 2020**: Melanoma classification (33,126 images)
- **Derm7pt**: 7-point checklist (2,000 images)

### Chest X-Ray
- **NIH ChestX-ray14**: 14 thoracic diseases (112,120 images)
- **PadChest**: Multiple pathologies (160,000 images)

**Note**: CIFAR-10 is used for initial smoke testing before medical datasets are integrated.

---

## 11. Key Components

### Tri-Objective Loss
```
L_total = L_task + λ_rob × L_rob + λ_expl × L_expl
```

Where:
- **L_task**: Cross-entropy with temperature scaling
- **L_rob**: TRADES robustness loss (KL divergence)
- **L_expl**: SSIM stability + TCAV concept regularization

### Selective Prediction
```
Accept if: (confidence > τ_conf) AND (stability > τ_stab)
```

---

## 12. Results (Expected)

### Baseline Performance
- Clean Accuracy: ~87% (ISIC 2018)
- Robust Accuracy (PGD ε=8/255): ~10-15%
- Cross-site AUROC drop: ~15pp
- Explanation SSIM: ~0.60

### Tri-Objective Performance (Target)
- Clean Accuracy: ~85% (±2pp trade-off)
- Robust Accuracy: ~47% (+35pp improvement)
- Cross-site AUROC drop: <8pp (-50% reduction)
- Explanation SSIM: ~0.76 (+0.16 improvement)
- Artifact TCAV: ~0.18 (-60% reduction)
- Selective Accuracy @ 90% coverage: +4pp improvement

---

## 13. Citation

If you use this code, please cite:
```bibtex
@misc{jain2025triobjective,
  title={Tri-Objective Robust XAI for Medical Imaging},
  author={Jain, Viraj Pankaj},
  year={2025},
  institution={University of Glasgow, School of Computing Science},
  howpublished={\url{https://github.com/viraj1011JAIN/tri-objective-robust-xai-medimg}}
}
```

---

## 14. License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 15. Contact

**Author**: Viraj Pankaj Jain
**Institution**: University of Glasgow, School of Computing Science
**Email**: v.jain.1@research.gla.ac.uk (or your actual email)
**GitHub**: https://github.com/viraj1011JAIN

---

## 16. Acknowledgments

- University of Glasgow School of Computing Science
- Medical imaging datasets: ISIC Archive, NIH Clinical Center
- PyTorch, MLflow, DVC communities

---

## 17. Troubleshooting

### Issue: Pre-commit hooks failing
```powershell
# Update pre-commit hooks
pre-commit autoupdate
pre-commit run --all-files
```

### Issue: DVC remote not configured
```powershell
dvc remote add -d myremote /path/to/remote/storage
# or for cloud storage:
dvc remote add -d myremote s3://my-bucket/dvc-storage
```

### Issue: CUDA out of memory
- Reduce batch size in config files
- Use gradient accumulation
- Enable mixed precision training (set use_amp: true in config)

### Issue: MLflow tracking URI not found
```powershell
# Set tracking URI explicitly
$env:MLFLOW_TRACKING_URI = "file:///C:/Users/Dissertation/tri-objective-robust-xai-medimg/mlruns"
mlflow ui
```

---

## 18. Development Roadmap

- [x] Project structure and MLOps infrastructure
- [x] CIFAR-10 smoke test pipeline
- [ ] ISIC 2018 data loader and baseline
- [ ] Adversarial training (PGD-AT, TRADES)
- [ ] Grad-CAM implementation
- [ ] TCAV and concept bank creation
- [ ] Tri-objective loss implementation
- [ ] Selective prediction mechanism
- [ ] Comprehensive evaluation pipeline
- [ ] Cross-site generalization analysis
- [ ] Ablation studies
- [ ] Publication-ready figures and tables

---

**Status**: Infrastructure complete, ready for medical imaging integration
**Last Updated**: November 15, 2025

**Good luck with your dissertation! 🚀**
