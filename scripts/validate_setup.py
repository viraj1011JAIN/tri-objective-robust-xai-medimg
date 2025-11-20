"""
Validate entire project setup and configuration.

Run this before starting experiments to catch issues early.
"""

import logging
import sys
from pathlib import Path
from typing import List, Tuple

import yaml

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def check_directories() -> List[Tuple[str, bool]]:
    """Check required directories exist."""
    required_dirs = [
        "data/raw",
        "data/processed",
        "configs/experiments",
        "results/metrics",
        "results/figures",
        "checkpoints",
        "mlruns",
        "logs",
    ]

    results = []
    for dir_path in required_dirs:
        full_path = PROJECT_ROOT / dir_path
        exists = full_path.exists()
        results.append((dir_path, exists))

        if not exists:
            logger.warning(f"Missing directory: {dir_path}")
            full_path.mkdir(parents=True, exist_ok=True)
            logger.info(f"✓ Created: {dir_path}")

    return results


def check_config_files() -> List[Tuple[str, bool]]:
    """Check required config files exist."""
    required_configs = [
        "configs/base.yaml",
        "configs/datasets/isic2018.yaml",
        "configs/models/resnet50.yaml",
        "configs/experiments/rq1_robustness/baseline_isic2018_resnet50.yaml",
    ]

    results = []
    for config_path in required_configs:
        full_path = PROJECT_ROOT / config_path
        exists = full_path.exists()
        results.append((config_path, exists))

        if not exists:
            logger.error(f"Missing config: {config_path}")
        else:
            # Validate YAML syntax
            try:
                with open(full_path) as f:
                    yaml.safe_load(f)
                logger.info(f"✓ Valid: {config_path}")
            except Exception as e:
                logger.error(f"Invalid YAML in {config_path}: {e}")
                results[-1] = (config_path, False)

    return results


def check_data_availability() -> List[Tuple[str, bool]]:
    """Check if datasets are downloaded."""
    data_paths = [
        "data/raw/isic2018/ISIC2018_Task3_Training_Input",
        "data/raw/isic2018/ISIC2018_Task3_Training_GroundTruth.csv",
    ]

    results = []
    for data_path in data_paths:
        full_path = PROJECT_ROOT / data_path
        exists = full_path.exists()
        results.append((data_path, exists))

        if not exists:
            logger.warning(f"Dataset not found: {data_path}")
            logger.info("Run: python scripts/data/download_isic2018.py")

    return results


def check_python_dependencies() -> bool:
    """Check critical Python packages are installed."""
    required_packages = [
        "torch",
        "torchvision",
        "numpy",
        "pandas",
        "scikit-learn",
        "matplotlib",
        "seaborn",
        "pyyaml",
        "mlflow",
        "tqdm",
    ]

    missing = []
    for package in required_packages:
        try:
            __import__(package)
        except ImportError:
            missing.append(package)

    if missing:
        logger.error(f"Missing packages: {', '.join(missing)}")
        logger.info("Install with: pip install -r requirements.txt")
        return False

    logger.info("✓ All required packages installed")
    return True


def check_cuda_availability() -> bool:
    """Check CUDA/GPU setup."""
    try:
        import torch

        cuda_available = torch.cuda.is_available()
        if cuda_available:
            device_name = torch.cuda.get_device_name(0)
            logger.info(f"✓ CUDA available: {device_name}")
            return True
        else:
            logger.warning("⚠ CUDA not available, training will use CPU (slower)")
            return False
    except Exception as e:
        logger.error(f"Error checking CUDA: {e}")
        return False


def main():
    """Run all validation checks."""
    logger.info("=" * 80)
    logger.info("PROJECT SETUP VALIDATION")
    logger.info("=" * 80)

    all_checks_passed = True

    # 1. Check directories
    logger.info("\n1. Checking directories...")
    check_directories()

    # 2. Check config files
    logger.info("\n2. Checking configuration files...")
    config_results = check_config_files()
    if not all(exists for _, exists in config_results):
        all_checks_passed = False

    # 3. Check data
    logger.info("\n3. Checking datasets...")
    check_data_availability()

    # 4. Check dependencies
    logger.info("\n4. Checking Python dependencies...")
    deps_ok = check_python_dependencies()
    if not deps_ok:
        all_checks_passed = False

    # 5. Check CUDA
    logger.info("\n5. Checking CUDA/GPU...")
    check_cuda_availability()

    # Summary
    logger.info("\n" + "=" * 80)
    if all_checks_passed:
        logger.info("✓ ALL CRITICAL CHECKS PASSED")
        logger.info("✓ Project is ready for experiments")
    else:
        logger.error("✗ SOME CHECKS FAILED")
        logger.error("✗ Fix issues above before running experiments")
        sys.exit(1)
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
