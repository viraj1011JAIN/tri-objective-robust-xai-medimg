#!/usr/bin/env python3
"""
Production Readiness Validation Script
======================================

Comprehensive sanity check for sections 1.1, 1.2, and 1.3.

This script validates:
- Environment setup (Python, packages, structure)
- MLOps infrastructure (DVC, MLflow)
- Code quality & CI/CD (pre-commit, tests, workflows)

Run this before proceeding to Section 1.4.

Usage:
    python scripts/validate_production_readiness.py

Author: Viraj Pankaj Jain
Institution: University of Glasgow
"""

import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


class Colors:
    """ANSI color codes for terminal output."""

    GREEN = "\033[92m"
    YELLOW = "\033[93m"
    RED = "\033[91m"
    CYAN = "\033[96m"
    RESET = "\033[0m"
    BOLD = "\033[1m"


def print_header(text: str) -> None:
    """Print section header."""
    print(f"\n{Colors.CYAN}{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{text}{Colors.RESET}")
    print(f"{Colors.CYAN}{Colors.BOLD}{'=' * 70}{Colors.RESET}\n")


def print_success(text: str) -> None:
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.RESET}")


def print_warning(text: str) -> None:
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.RESET}")


def print_error(text: str) -> None:
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.RESET}")


def check_python_environment() -> Tuple[bool, List[str]]:
    """Validate Python environment setup."""
    issues = []

    # Check Python version
    if sys.version_info < (3, 10):
        issues.append(f"Python version {sys.version_info} < 3.10 (required)")
    else:
        print_success(
            f"Python {sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        )

    # Check virtual environment
    if hasattr(sys, "real_prefix") or (
        hasattr(sys, "base_prefix") and sys.base_prefix != sys.prefix
    ):
        print_success("Virtual environment active")
    else:
        issues.append("No virtual environment detected")

    return len(issues) == 0, issues


def check_critical_packages() -> Tuple[bool, List[str]]:
    """Validate critical package installations."""
    issues = []
    packages = {
        "torch": "PyTorch",
        "mlflow": "MLflow",
        "dvc": "DVC",
        "pytest": "pytest",
        "albumentations": "Albumentations",
    }

    for module, name in packages.items():
        try:
            pkg = __import__(module)
            version = getattr(pkg, "__version__", "unknown")
            print_success(f"{name} {version}")
        except ImportError:
            issues.append(f"{name} not installed")

    # Check PyTorch CUDA
    try:
        import torch

        if torch.cuda.is_available():
            print_success(f"CUDA available: {torch.cuda.get_device_name(0)}")
        else:
            print_warning("CUDA not available (CPU-only, acceptable for dev)")
    except Exception:
        pass

    return len(issues) == 0, issues


def check_directory_structure() -> Tuple[bool, List[str]]:
    """Validate project directory structure."""
    issues = []
    required_dirs = [
        "src",
        "tests",
        "configs",
        "scripts",
        "data",
        "docs",
        "notebooks",
        "results",
        "mlruns",
        ".github",
    ]

    for dir_name in required_dirs:
        dir_path = PROJECT_ROOT / dir_name
        if dir_path.exists():
            print_success(f"Directory: {dir_name}/")
        else:
            issues.append(f"Missing directory: {dir_name}/")

    return len(issues) == 0, issues


def check_configuration_files() -> Tuple[bool, List[str]]:
    """Validate configuration files."""
    issues = []
    required_files = [
        "pyproject.toml",
        "requirements.txt",
        "environment.yml",
        "pytest.ini",
        ".pre-commit-config.yaml",
        ".dvcignore",
        ".gitignore",
        "Dockerfile",
        "dvc.yaml",
    ]

    for file_name in required_files:
        file_path = PROJECT_ROOT / file_name
        if file_path.exists():
            print_success(f"Config file: {file_name}")
        else:
            issues.append(f"Missing config: {file_name}")

    return len(issues) == 0, issues


def check_dvc_setup() -> Tuple[bool, List[str]]:
    """Validate DVC configuration."""
    issues = []

    # Check DVC directory
    dvc_dir = PROJECT_ROOT / ".dvc"
    if dvc_dir.exists():
        print_success("DVC initialized (.dvc/ directory exists)")
    else:
        issues.append("DVC not initialized")
        return False, issues

    # Check DVC config
    dvc_config = dvc_dir / "config"
    if dvc_config.exists():
        print_success("DVC config present")
        # Try to read default remote
        try:
            with open(dvc_config, "r") as f:
                content = f.read()
                if "remote" in content:
                    print_success("DVC remote configured")
                else:
                    print_warning("No DVC remote configured")
        except Exception:
            pass
    else:
        print_warning("DVC config missing")

    return len(issues) == 0, issues


def check_mlflow_setup() -> Tuple[bool, List[str]]:
    """Validate MLflow configuration."""
    issues = []

    # Check MLflow directory
    mlruns_dir = PROJECT_ROOT / "mlruns"
    if mlruns_dir.exists():
        print_success("MLflow runs directory exists")
    else:
        issues.append("MLflow not initialized (mlruns/ missing)")
        return False, issues

    # Check for experiments
    try:
        import mlflow

        mlflow.set_tracking_uri("file:./mlruns")
        client = mlflow.tracking.MlflowClient()
        experiments = client.search_experiments()
        exp_count = len([e for e in experiments if e.name != "Default"])
        if exp_count > 0:
            print_success(f"MLflow: {exp_count} experiments tracked")
        else:
            print_warning("No MLflow experiments created yet")
    except Exception as e:
        print_warning(f"Could not query MLflow: {e}")

    return len(issues) == 0, issues


def check_precommit_hooks() -> Tuple[bool, List[str]]:
    """Validate pre-commit hooks installation."""
    issues = []

    # Check pre-commit config
    config_file = PROJECT_ROOT / ".pre-commit-config.yaml"
    if not config_file.exists():
        issues.append("No .pre-commit-config.yaml")
        return False, issues

    print_success(".pre-commit-config.yaml present")

    # Check if hooks are installed
    hook_file = PROJECT_ROOT / ".git" / "hooks" / "pre-commit"
    if hook_file.exists():
        print_success("Pre-commit hooks installed")
    else:
        print_warning("Pre-commit hooks not installed (run: pre-commit install)")

    return len(issues) == 0, issues


def check_github_workflows() -> Tuple[bool, List[str]]:
    """Validate GitHub Actions workflows."""
    issues = []
    workflows_dir = PROJECT_ROOT / ".github" / "workflows"

    if not workflows_dir.exists():
        issues.append("No .github/workflows/ directory")
        return False, issues

    required_workflows = ["tests.yml", "lint.yml", "docs.yml"]
    for workflow in required_workflows:
        workflow_path = workflows_dir / workflow
        if workflow_path.exists():
            print_success(f"Workflow: {workflow}")
        else:
            issues.append(f"Missing workflow: {workflow}")

    return len(issues) == 0, issues


def check_imports() -> Tuple[bool, List[str]]:
    """Validate critical imports."""
    issues = []
    critical_imports = [
        ("src.datasets.isic", "ISICDataset"),
        ("src.models.resnet", "ResNet50Classifier"),
        ("src.losses.task_loss", "TaskLoss"),
        ("src.training.baseline_trainer", "BaselineTrainer"),
        ("src.utils.config", "load_experiment_config"),
        ("src.utils.mlflow_utils", "init_mlflow"),
    ]

    for module_name, class_name in critical_imports:
        try:
            module = __import__(module_name, fromlist=[class_name])
            getattr(module, class_name)
            print_success(f"Import: {module_name}.{class_name}")
        except ImportError as e:
            issues.append(f"Cannot import {module_name}.{class_name}: {e}")
        except AttributeError:
            issues.append(f"Module {module_name} has no attribute {class_name}")

    return len(issues) == 0, issues


def check_datasets() -> Tuple[bool, List[str]]:
    """Validate dataset accessibility."""
    issues = []
    datasets = {
        "ISIC 2018": Path("F:/data/isic_2018"),
        "ISIC 2019": Path("F:/data/isic_2019"),
        "ISIC 2020": Path("F:/data/isic_2020"),
        "Derm7pt": Path("F:/data/derm7pt"),
        "NIH CXR": Path("F:/data/nih_cxr"),
        "PadChest": Path("F:/data/padchest"),
    }

    for name, path in datasets.items():
        if path.exists():
            file_count = sum(1 for _ in path.rglob("*") if _.is_file())
            print_success(f"{name}: {file_count:,} files at {path}")
        else:
            print_warning(f"{name}: NOT FOUND at {path}")

    return len(issues) == 0, issues


def main() -> int:
    """Run all validation checks."""
    print(f"\n{Colors.BOLD}{'=' * 70}{Colors.RESET}")
    print(f"{Colors.BOLD}Production Readiness Validation{Colors.RESET}")
    print(f"{Colors.BOLD}Tri-Objective Robust XAI for Medical Imaging{Colors.RESET}")
    print(f"{Colors.BOLD}{'=' * 70}{Colors.RESET}")

    checks: Dict[str, Tuple[bool, List[str]]] = {}

    # Section 1.1: Environment Setup
    print_header("Section 1.1: Environment Setup")
    checks["Python Environment"] = check_python_environment()
    checks["Critical Packages"] = check_critical_packages()
    checks["Directory Structure"] = check_directory_structure()
    checks["Configuration Files"] = check_configuration_files()

    # Section 1.2: MLOps Infrastructure
    print_header("Section 1.2: MLOps Infrastructure")
    checks["DVC Setup"] = check_dvc_setup()
    checks["MLflow Setup"] = check_mlflow_setup()
    checks["Dataset Access"] = check_datasets()

    # Section 1.3: Code Quality & CI/CD
    print_header("Section 1.3: Code Quality & CI/CD")
    checks["Pre-commit Hooks"] = check_precommit_hooks()
    checks["GitHub Workflows"] = check_github_workflows()
    checks["Import Structure"] = check_imports()

    # Summary
    print_header("Validation Summary")

    passed = sum(1 for success, _ in checks.values() if success)
    total = len(checks)

    print(f"\n{Colors.BOLD}Results: {passed}/{total} checks passed{Colors.RESET}\n")

    all_issues = []
    for check_name, (success, issues) in checks.items():
        if success:
            print_success(f"{check_name}: PASSED")
        else:
            print_error(f"{check_name}: FAILED")
            all_issues.extend(issues)

    if all_issues:
        print_header("Issues Found")
        for issue in all_issues:
            print_error(issue)

    # Final verdict
    print_header("Final Verdict")

    if passed == total:
        print_success("✓ ALL CHECKS PASSED - PRODUCTION READY")
        print_success("✓ Cleared for Section 1.4: Dataset Preparation & Validation")
        return 0
    else:
        print_error(f"✗ {total - passed} CHECK(S) FAILED")
        print_error("✗ Fix issues before proceeding to Section 1.4")
        return 1


if __name__ == "__main__":
    sys.exit(main())
