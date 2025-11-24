"""
Environment Setup Verification for Tri-Objective Robust XAI.

Checks:
1. CUDA availability and GPU info
2. PyTorch version and configuration
3. Required packages
4. MLflow setup
5. Folder structure
6. Dummy data generation
7. Model instantiation
8. Training loop (1 epoch dry run)

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Project: Tri-Objective Robust XAI for Medical Imaging
Target: A1+ Grade | Publication-Ready (NeurIPS/MICCAI/TMI)
Deadline: November 28, 2025

Usage
-----
Run this script to verify the environment is ready:
    $ python scripts/verify_environment.py
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import torch
import torch.nn as nn

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)8s | %(message)s",
)
logger = logging.getLogger(__name__)


def print_section(title: str):
    """Print a section header."""
    print("\n" + "=" * 80)
    print(f"  {title}")
    print("=" * 80)


def check_cuda():
    """Check CUDA availability and GPU info."""
    print_section("1. CUDA and GPU Check")

    print(f"PyTorch version: {torch.__version__}")
    print(f"CUDA available: {torch.cuda.is_available()}")

    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"cuDNN version: {torch.backends.cudnn.version()}")
        print(f"Number of GPUs: {torch.cuda.device_count()}")

        for i in range(torch.cuda.device_count()):
            print(f"\nGPU {i}: {torch.cuda.get_device_name(i)}")
            props = torch.cuda.get_device_properties(i)
            print(f"  Total memory: {props.total_memory / 1e9:.2f} GB")
            print(f"  Compute capability: {props.major}.{props.minor}")

        # Test a simple operation
        try:
            x = torch.randn(100, 100).cuda()
            y = torch.matmul(x, x)
            print("\n✓ CUDA test passed (matrix multiplication)")
        except Exception as e:
            print(f"\n✗ CUDA test failed: {e}")
            return False
    else:
        print("\n⚠ CUDA not available. Training will use CPU (slow).")
        return False

    return True


def check_packages():
    """Check required packages are installed."""
    print_section("2. Package Check")

    required = [
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("sklearn", "scikit-learn"),
        ("mlflow", "MLflow (optional)"),
        ("fastapi", "FastAPI (optional)"),
        ("uvicorn", "Uvicorn (optional)"),
    ]

    all_ok = True
    for module_name, display_name in required:
        try:
            module = __import__(module_name)
            version = getattr(module, "__version__", "unknown")
            print(f"✓ {display_name:20s}: {version}")
        except ImportError:
            print(f"✗ {display_name:20s}: NOT INSTALLED")
            if "optional" not in display_name.lower():
                all_ok = False

    return all_ok


def check_folder_structure():
    """Check folder structure is correct."""
    print_section("3. Folder Structure Check")

    project_root = Path(__file__).parent.parent
    required_folders = [
        "src/losses",
        "src/models",
        "src/training",
        "src/attacks",
        "src/xai",
        "src/api",
        "src/utils",
        "data/raw",
        "data/processed",
        "checkpoints",
        "logs",
        "mlruns",
    ]

    all_ok = True
    for folder in required_folders:
        folder_path = project_root / folder
        if folder_path.exists():
            print(f"✓ {folder:30s}: EXISTS")
        else:
            print(f"✗ {folder:30s}: MISSING (will create)")
            folder_path.mkdir(parents=True, exist_ok=True)
            all_ok = False

    return all_ok


def test_real_datasets():
    """Test real dataset loading."""
    print_section("4. Real Dataset Loading Test")

    try:
        from pathlib import Path

        from src.datasets.isic import ISICDataset
        from src.datasets.transforms import build_transforms

        # Check if ISIC 2018 is available
        isic_root = Path("/content/drive/MyDrive/data/isic2018")
        if not isic_root.exists():
            print("⚠ ISIC 2018 not found at/content/drive/MyDrive/data/isic2018, skipping test")
            return True

        # Create transforms and dataset
        transforms = build_transforms("isic", "train", 224)
        dataset = ISICDataset(root=isic_root, split="train", transforms=transforms)

        # Create dataloader
        from torch.utils.data import DataLoader

        train_loader = DataLoader(dataset, batch_size=16, shuffle=True)

        # Get one batch
        sample = next(iter(train_loader))
        images = sample.image
        labels = sample.label

        print(f"✓ Batch shape: images={images.shape}, labels={labels.shape}")
        print(f"  Image range: [{images.min():.3f}, {images.max():.3f}]")
        print(f"  Label range: [{labels.min():.0f}, {labels.max():.0f}]")
        print(f"  Dataset size: {len(dataset)} samples")

        return True
    except Exception as e:
        print(f"✗ Dummy data test failed: {e}")
        return False


def test_tri_objective_loss():
    """Test tri-objective loss instantiation."""
    print_section("5. Tri-Objective Loss Test")

    try:
        from src.losses.tri_objective import TriObjectiveLoss

        # Create loss
        criterion = TriObjectiveLoss(
            num_classes=7,
            task_type="multi_class",
            lambda_rob=0.3,
            lambda_expl=0.2,
        )

        # Test forward pass
        batch_size = 16
        num_classes = 7

        logits_clean = torch.randn(batch_size, num_classes)
        logits_adv = torch.randn(batch_size, num_classes)
        labels = torch.randint(0, num_classes, (batch_size,))
        embeddings = torch.randn(batch_size, 2048)

        outputs = criterion(
            logits_clean=logits_clean,
            logits_adv=logits_adv,
            labels=labels,
            embeddings=embeddings,
        )

        print(f"✓ Loss forward pass successful")
        print(f"  Total loss: {outputs['loss'].item():.4f}")
        print(f"  Task loss: {outputs['task'].item():.4f}")
        print(f"  Robustness loss: {outputs['robustness'].item():.4f}")
        print(f"  Explanation loss: {outputs['explanation'].item():.4f}")
        print(f"  Temperature: {outputs['temperature'].item():.4f}")

        return True
    except Exception as e:
        print(f"✗ Tri-objective loss test failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_model_instantiation():
    """Test model instantiation."""
    print_section("6. Model Instantiation Test")

    try:
        from torchvision.models import resnet50

        # Create model
        model = resnet50(pretrained=False)
        num_classes = 7
        model.fc = nn.Linear(model.fc.in_features, num_classes)

        # Test forward pass
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model = model.to(device)

        dummy_input = torch.randn(2, 3, 224, 224).to(device)
        with torch.no_grad():
            output = model(dummy_input)

        print(f"✓ Model instantiation successful")
        print(f"  Architecture: ResNet-50")
        print(f"  Output shape: {output.shape}")
        print(f"  Device: {device}")

        # Count parameters
        num_params = sum(p.numel() for p in model.parameters())
        print(f"  Parameters: {num_params:,}")

        return True
    except Exception as e:
        print(f"✗ Model instantiation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_training_loop():
    """Test training loop with real data."""
    print_section("7. Training Loop Dry Run (1 Epoch)")

    try:
        from pathlib import Path

        from torch.utils.data import DataLoader
        from torchvision.models import resnet50

        from src.attacks.pgd import PGD, PGDConfig
        from src.datasets.isic import ISICDataset
        from src.datasets.transforms import build_transforms
        from src.losses.tri_objective import TriObjectiveLoss

        # Check if ISIC 2018 is available
        isic_root = Path("/content/drive/MyDrive/data/isic2018")
        if not isic_root.exists():
            print("⚠ ISIC 2018 not found at/content/drive/MyDrive/data/isic2018, skipping test")
            return True

        # Setup
        device = "cuda" if torch.cuda.is_available() else "cpu"
        num_classes = 7

        # Create model
        model = resnet50(pretrained=False)
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        model = model.to(device)

        # Create real data loader
        transforms = build_transforms("isic", "train", 224)
        dataset = ISICDataset(root=isic_root, split="train", transforms=transforms)
        train_loader = DataLoader(
            dataset, batch_size=8, shuffle=True, num_workers=0  # Simple for quick test
        )

        # Create loss
        criterion = TriObjectiveLoss(
            num_classes=num_classes,
            task_type="multi_class",
        ).to(device)

        # Create optimizer
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-4)

        # Create PGD attack
        pgd_config = PGDConfig(
            epsilon=8.0 / 255.0,
            num_steps=10,
            step_size=2.0 / 255.0,
        )
        pgd_attack = PGD(pgd_config)

        # Training loop (1 epoch)
        model.train()
        total_loss = 0.0
        num_batches = 0

        print("\nRunning training loop...")
        for batch_idx, sample in enumerate(train_loader):
            images = sample.image.to(device)
            labels = sample.label.to(device)

            # Generate adversarial examples
            model.eval()
            with torch.enable_grad():
                images_adv = pgd_attack(model, images, labels)
            model.train()

            # Forward pass
            logits_clean = model(images)
            logits_adv = model(images_adv)

            # Compute loss
            loss_outputs = criterion(
                logits_clean=logits_clean,
                logits_adv=logits_adv,
                labels=labels,
            )

            loss = loss_outputs["loss"]

            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            num_batches += 1

            print(
                f"  Batch {batch_idx + 1}/{len(train_loader)}: Loss = {loss.item():.4f}"
            )

        avg_loss = total_loss / num_batches
        print(f"\n✓ Training loop successful")
        print(f"  Average loss: {avg_loss:.4f}")
        print(f"  Batches processed: {num_batches}")

        return True
    except Exception as e:
        print(f"\n✗ Training loop failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def test_mlflow_setup():
    """Test MLflow setup."""
    print_section("8. MLflow Setup (Optional)")

    try:
        import mlflow

        # Check MLflow tracking URI
        tracking_uri = mlflow.get_tracking_uri()
        print(f"✓ MLflow installed")
        print(f"  Tracking URI: {tracking_uri}")

        # Try to create an experiment
        experiment_name = "test_tri_objective"
        experiment = mlflow.get_experiment_by_name(experiment_name)
        if experiment is None:
            experiment_id = mlflow.create_experiment(experiment_name)
            print(f"  Created test experiment: {experiment_name}")
        else:
            print(f"  Experiment exists: {experiment_name}")

        return True
    except ImportError:
        print("⚠ MLflow not installed (optional)")
        return True
    except Exception as e:
        print(f"⚠ MLflow setup failed: {e}")
        return True  # Not critical


def main():
    """Run all checks."""
    print("\n" + "=" * 80)
    print("  TRI-OBJECTIVE ROBUST XAI - ENVIRONMENT VERIFICATION")
    print("  War Room Sprint: Day 1 - Shadow Execution")
    print("=" * 80)

    results = {
        "CUDA": check_cuda(),
        "Packages": check_packages(),
        "Folders": check_folder_structure(),
        "Real Datasets": test_real_datasets(),
        "Tri-Objective Loss": test_tri_objective_loss(),
        "Model": test_model_instantiation(),
        "Training Loop": test_training_loop(),
        "MLflow": test_mlflow_setup(),
    }

    # Print summary
    print_section("SUMMARY")

    all_passed = True
    for check, passed in results.items():
        status = "✓ PASS" if passed else "✗ FAIL"
        print(f"  {check:25s}: {status}")
        if not passed:
            all_passed = False

    print("\n" + "=" * 80)
    if all_passed:
        print("  🎉 ALL CHECKS PASSED - READY FOR DATA ARRIVAL")
        print("  Next: Wait for datasets, then launch Day 2 training")
    else:
        print("  ⚠ SOME CHECKS FAILED - FIX ISSUES BEFORE TRAINING")
        print("  Review logs above and resolve errors")
    print("=" * 80 + "\n")

    return all_passed


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
