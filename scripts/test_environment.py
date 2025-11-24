"""
Quick Python Environment Test
==============================
Run this script to verify your Python environment is correctly set up
for the tri-objective-robust-xai-medimg project.

Usage:
    python scripts/test_environment.py
"""

import sys
from pathlib import Path

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


def test_python_version():
    """Check Python version."""
    print("\n[1/7] Python Version")
    print("-" * 50)
    version = sys.version_info
    print(f"Python {version.major}.{version.minor}.{version.micro}")

    if version.major == 3 and version.minor >= 10:
        print("✓ Python version OK (3.10+)")
        return True
    else:
        print(f"✗ Python 3.10+ required, found {version.major}.{version.minor}")
        return False


def test_core_packages():
    """Test core scientific packages."""
    print("\n[2/7] Core Packages")
    print("-" * 50)

    packages = {
        "torch": "PyTorch",
        "torchvision": "TorchVision",
        "numpy": "NumPy",
        "pandas": "Pandas",
        "scikit-learn": "Scikit-learn",
        "PIL": "Pillow",
    }

    all_ok = True
    for pkg, name in packages.items():
        try:
            if pkg == "scikit-learn":
                __import__("sklearn")
            else:
                __import__(pkg)
            print(f"✓ {name}")
        except ImportError:
            print(f"✗ {name} - NOT INSTALLED")
            all_ok = False

    return all_ok


def test_torch_cuda():
    """Test PyTorch CUDA availability."""
    print("\n[3/7] PyTorch & CUDA")
    print("-" * 50)

    try:
        import torch

        print(f"PyTorch version: {torch.__version__}")

        if torch.cuda.is_available():
            print(f"✓ CUDA available: {torch.cuda.get_device_name(0)}")
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
        else:
            print("⚠ CUDA not available (CPU mode)")

        return True
    except ImportError:
        print("✗ PyTorch not installed")
        return False


def test_project_imports():
    """Test project module imports."""
    print("\n[4/7] Project Imports")
    print("-" * 50)

    imports = [
        ("src.datasets.isic", "ISICDataset", "Dataset classes"),
        ("src.models.resnet", "ResNet50Classifier", "Model classes"),
        ("src.training.baseline_trainer", "BaselineTrainer", "Training classes"),
        ("src.utils.config", "load_experiment_config", "Config utilities"),
        ("src.utils.reproducibility", "set_global_seed", "Reproducibility"),
    ]

    all_ok = True
    for module, cls, description in imports:
        try:
            mod = __import__(module, fromlist=[cls])
            getattr(mod, cls)
            print(f"✓ {description}")
        except (ImportError, AttributeError) as e:
            print(f"✗ {description} - {e}")
            all_ok = False

    return all_ok


def test_data_path():
    """Test data directory access."""
    print("\n[5/7] Data Directory")
    print("-" * 50)

    import os

    data_root = Path(os.environ.get("DATA_ROOT", "/content/drive/MyDrive/data"))

    if data_root.exists():
        print(f"✓ Data root exists: {data_root}")

        # Count datasets
        datasets = [
            d for d in data_root.iterdir() if d.is_dir() and not d.name.startswith(".")
        ]
        print(f"  Found {len(datasets)} dataset directories:")
        for ds in sorted(datasets)[:10]:  # Show first 10
            print(f"    - {ds.name}")

        return True
    else:
        print(f"✗ Data root not found: {data_root}")
        return False


def test_config_files():
    """Test configuration file access."""
    print("\n[6/7] Configuration Files")
    print("-" * 50)

    configs_dir = PROJECT_ROOT / "configs"

    if not configs_dir.exists():
        print(f"✗ Configs directory not found: {configs_dir}")
        return False

    # Check key config files
    base_config = configs_dir / "base.yaml"
    if base_config.exists():
        print(f"✓ Base config: {base_config.name}")
    else:
        print("✗ Base config not found")
        return False

    # Count configs
    dataset_configs = (
        list((configs_dir / "datasets").glob("*.yaml"))
        if (configs_dir / "datasets").exists()
        else []
    )
    model_configs = (
        list((configs_dir / "models").glob("*.yaml"))
        if (configs_dir / "models").exists()
        else []
    )

    print(f"✓ Dataset configs: {len(dataset_configs)}")
    print(f"✓ Model configs: {len(model_configs)}")

    return True


def test_write_permissions():
    """Test write permissions in key directories."""
    print("\n[7/7] Write Permissions")
    print("-" * 50)

    test_dirs = [
        PROJECT_ROOT / "logs",
        PROJECT_ROOT / "results",
        PROJECT_ROOT / "checkpoints",
        PROJECT_ROOT / "mlruns",
    ]

    all_ok = True
    for test_dir in test_dirs:
        test_dir.mkdir(parents=True, exist_ok=True)
        test_file = test_dir / ".write_test"

        try:
            test_file.write_text("test")
            test_file.unlink()
            print(f"✓ {test_dir.name}/")
        except Exception as e:
            print(f"✗ {test_dir.name}/ - {e}")
            all_ok = False

    return all_ok


def main():
    """Run all tests."""
    print("=" * 50)
    print("Python Environment Test")
    print("=" * 50)

    results = {
        "Python Version": test_python_version(),
        "Core Packages": test_core_packages(),
        "PyTorch & CUDA": test_torch_cuda(),
        "Project Imports": test_project_imports(),
        "Data Directory": test_data_path(),
        "Config Files": test_config_files(),
        "Write Permissions": test_write_permissions(),
    }

    # Summary
    print("\n" + "=" * 50)
    print("Summary")
    print("=" * 50)

    passed = sum(results.values())
    total = len(results)

    for test, result in results.items():
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{status:8} {test}")

    print("-" * 50)
    print(f"Result: {passed}/{total} tests passed")

    if passed == total:
        print("\n✓ Environment is ready for development!")
        return 0
    else:
        print("\n✗ Some tests failed. Please check the output above.")
        print("\nTroubleshooting:")
        if not results["Core Packages"]:
            print("  - Run: pip install -r requirements.txt")
        if not results["Project Imports"]:
            print("  - Ensure you're in the project root directory")
        if not results["Data Directory"]:
            print("  - Set DATA_ROOT: $env:DATA_ROOT='/content/drive/MyDrive/data'")
        return 1


if __name__ == "__main__":
    sys.exit(main())
