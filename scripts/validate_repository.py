"""
Repository Validation Script
=============================
Validates the entire repository setup including:
- Import structure
- Dataset paths
- Configuration files
- Required dependencies
"""

import logging
import sys
from pathlib import Path
from typing import List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def check_imports() -> Tuple[bool, List[str]]:
    """Check if all critical imports work."""
    issues = []
    try:
        from src.datasets.base_dataset import BaseMedicalDataset, Split  # noqa: F401
        from src.datasets.chest_xray import ChestXRayDataset  # noqa: F401
        from src.datasets.derm7pt import Derm7ptDataset  # noqa: F401
        from src.datasets.isic import ISICDataset  # noqa: F401

        logger.info("✓ Dataset imports successful")
    except ImportError as e:
        issues.append(f"Dataset import error: {e}")

    try:
        from src.models.build import build_model  # noqa: F401
        from src.models.efficientnet import EfficientNetB0Classifier  # noqa: F401
        from src.models.resnet import ResNet50Classifier  # noqa: F401
        from src.models.vit import ViTB16Classifier  # noqa: F401

        logger.info("✓ Model imports successful")
    except ImportError as e:
        issues.append(f"Model import error: {e}")

    try:
        from src.training.base_trainer import BaseTrainer, TrainingConfig  # noqa: F401
        from src.training.baseline_trainer import BaselineTrainer  # noqa: F401

        logger.info("✓ Training imports successful")
    except ImportError as e:
        issues.append(f"Training import error: {e}")

    try:
        from src.utils.config import load_experiment_config  # noqa: F401
        from src.utils.mlflow_utils import init_mlflow  # noqa: F401
        from src.utils.reproducibility import set_global_seed  # noqa: F401

        logger.info("✓ Utils imports successful")
    except ImportError as e:
        issues.append(f"Utils import error: {e}")

    return len(issues) == 0, issues


def check_dataset_paths() -> Tuple[bool, List[str]]:
    """Check if dataset directories exist and are accessible."""
    issues = []
    data_root = Path("/content/drive/MyDrive/data")

    if not data_root.exists():
        issues.append(
            f"Data root directory does not exist: {data_root}\n"
            "  Please ensure/content/drive/MyDrive/data exists and contains your datasets"
        )
        return False, issues

    logger.info(f"✓ Data root exists: {data_root}")

    # Check expected dataset directories (based on actual/content/drive/MyDrive/data contents)
    expected_datasets = {
        "isic_2018": "ISIC 2018 dermoscopy dataset",
        "isic_2019": "ISIC 2019 dermoscopy dataset",
        "isic_2020": "ISIC 2020 dermoscopy dataset",
        "derm7pt": "Derm7pt dermoscopy dataset",
        "nih_cxr": "NIH ChestX-ray14 dataset",
        "padchest": "PadChest dataset",
    }

    found = []
    missing = []

    for dataset_dir, description in expected_datasets.items():
        full_path = data_root / dataset_dir
        if full_path.exists():
            found.append(f"  ✓ {dataset_dir} ({description})")
        else:
            missing.append(f"  ✗ {dataset_dir} ({description})")

    if found:
        logger.info("Found datasets:")
        for item in found:
            logger.info(item)

    if missing:
        logger.warning("Missing expected datasets:")
        for item in missing:
            logger.warning(item)

    # Check for additional datasets not in expected list
    try:
        actual_dirs = [d.name for d in data_root.iterdir() if d.is_dir()]
        extra_datasets = [
            d
            for d in actual_dirs
            if d not in expected_datasets
            and not d.startswith(".")
            and d not in [".dvc_cache", "processed", "raw"]
        ]
        if extra_datasets:
            logger.info("Additional datasets found:")
            for dataset in extra_datasets:
                logger.info(f"  • {dataset}")
    except Exception as e:
        logger.debug(f"Could not scan for additional datasets: {e}")

    return True, issues


def check_configs() -> Tuple[bool, List[str]]:
    """Check if configuration files are valid."""
    issues = []
    configs_dir = PROJECT_ROOT / "configs"

    if not configs_dir.exists():
        issues.append(f"Configs directory not found: {configs_dir}")
        return False, issues

    # Check base config
    base_config = configs_dir / "base.yaml"
    if not base_config.exists():
        issues.append(f"Base config not found: {base_config}")
    else:
        logger.info(f"✓ Base config exists: {base_config}")

    # Check dataset configs
    dataset_configs = configs_dir / "datasets"
    if dataset_configs.exists():
        yaml_files = list(dataset_configs.glob("*.yaml"))
        logger.info(
            f"✓ Found {len(yaml_files)} dataset configs: "
            f"{[f.stem for f in yaml_files]}"
        )
    else:
        issues.append(f"Dataset configs directory not found: {dataset_configs}")

    # Check model configs
    model_configs = configs_dir / "models"
    if model_configs.exists():
        yaml_files = list(model_configs.glob("*.yaml"))
        logger.info(
            f"✓ Found {len(yaml_files)} model configs: "
            f"{[f.stem for f in yaml_files]}"
        )
    else:
        issues.append(f"Model configs directory not found: {model_configs}")

    return len(issues) == 0, issues


def check_environment() -> Tuple[bool, List[str]]:
    """Check environment setup."""
    issues = []
    import os

    # Check if DATA_ROOT is set (optional but recommended)
    data_root_env = os.environ.get("DATA_ROOT")
    if data_root_env:
        logger.info(f"✓ DATA_ROOT environment variable set: {data_root_env}")
    else:
        logger.warning(
            "⚠ DATA_ROOT environment variable not set (using/content/drive/MyDrive/data as default)"
        )

    # Check Python version
    py_version = sys.version_info
    if py_version >= (3, 10):
        logger.info(f"✓ Python version: {py_version.major}.{py_version.minor}")
    else:
        issues.append(
            f"Python 3.10+ required, found: " f"{py_version.major}.{py_version.minor}"
        )

    return len(issues) == 0, issues


def main():
    """Run all validation checks."""
    logger.info("=" * 70)
    logger.info("Repository Validation")
    logger.info("=" * 70)

    all_passed = True

    # Check imports
    logger.info("\n[1/4] Checking imports...")
    passed, issues = check_imports()
    if not passed:
        all_passed = False
        for issue in issues:
            logger.error(f"  ✗ {issue}")

    # Check dataset paths
    logger.info("\n[2/4] Checking dataset paths...")
    passed, issues = check_dataset_paths()
    if not passed:
        all_passed = False
        for issue in issues:
            logger.error(f"  ✗ {issue}")

    # Check configs
    logger.info("\n[3/4] Checking configuration files...")
    passed, issues = check_configs()
    if not passed:
        all_passed = False
        for issue in issues:
            logger.error(f"  ✗ {issue}")

    # Check environment
    logger.info("\n[4/4] Checking environment setup...")
    passed, issues = check_environment()
    if not passed:
        all_passed = False
        for issue in issues:
            logger.error(f"  ✗ {issue}")

    # Summary
    logger.info("\n" + "=" * 70)
    if all_passed:
        logger.info("✓ All validation checks passed!")
        logger.info("=" * 70)
        return 0
    else:
        logger.error("✗ Some validation checks failed. See details above.")
        logger.info("=" * 70)
        return 1


if __name__ == "__main__":
    sys.exit(main())
