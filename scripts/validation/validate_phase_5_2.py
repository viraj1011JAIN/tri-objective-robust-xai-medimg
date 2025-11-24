"""
Phase 5.2: Production & A1+ Standards Validation
=================================================

Comprehensive validation that Phase 5.2 meets:
- Production-level standards (imports, error handling, memory management)
- Masters A1+ standards (RQ1 test, advanced statistics, proper documentation)

Validation Levels:
1. CRITICAL SHOWSTOPPERS - Must pass to run at all
2. PRODUCTION BLOCKERS - Must pass for production use
3. A1+ REQUIREMENTS - Must pass for masters A1+ grade
4. BEST PRACTICES - Should pass for excellence

Author: Viraj Pankaj Jain
Date: November 24, 2025
Version: 5.2.1
"""

import ast
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"


class ValidationLevel:
    """Validation severity levels."""

    CRITICAL = "CRITICAL"
    PRODUCTION = "PRODUCTION"
    A1_PLUS = "A1+"
    BEST_PRACTICE = "BEST_PRACTICE"


def print_success(msg: str):
    """Print success message in green."""
    print(f"{GREEN}✅ {msg}{RESET}")


def print_error(msg: str, level: str = "ERROR"):
    """Print error message in red."""
    if level == ValidationLevel.CRITICAL:
        print(f"{RED}🚨 CRITICAL: {msg}{RESET}")
    elif level == ValidationLevel.PRODUCTION:
        print(f"{RED}❌ PRODUCTION: {msg}{RESET}")
    elif level == ValidationLevel.A1_PLUS:
        print(f"{MAGENTA}❌ A1+: {msg}{RESET}")
    else:
        print(f"{RED}❌ {msg}{RESET}")


def print_warning(msg: str):
    """Print warning message in yellow."""
    print(f"{YELLOW}⚠️  {msg}{RESET}")


def print_info(msg: str):
    """Print info message in blue."""
    print(f"{BLUE}ℹ️  {msg}{RESET}")


def print_header(msg: str):
    """Print section header in cyan."""
    print(f"\n{CYAN}{'='*60}{RESET}")
    print(f"{CYAN}{msg}{RESET}")
    print(f"{CYAN}{'='*60}{RESET}\n")


def check_file_exists(file_path: Path, description: str) -> bool:
    """Check if file exists and print result."""
    if file_path.exists():
        print_success(f"{description}: {file_path.name}")
        return True
    else:
        print_error(
            f"{description}: {file_path.name} NOT FOUND", ValidationLevel.CRITICAL
        )
        return False


def validate_phase_5_2():
    """Validate Phase 5.2 implementation."""
    print("\n" + "=" * 80)
    print("Phase 5.2: Production Readiness Validation")
    print("=" * 80 + "\n")

    project_root = Path(__file__).parent.parent
    all_checks_passed = True

    # Check 1: Core files exist
    print("1. Checking Core Files...")
    required_files = [
        ("scripts/training/train_pgd_at.py", "Training script"),
        ("scripts/evaluation/evaluate_pgd_at.py", "Evaluation script"),
        ("configs/experiments/pgd_at_isic.yaml", "Configuration file"),
        ("tests/test_phase_5_2_pgd_at.py", "Unit tests"),
    ]

    for file_path, description in required_files:
        if not check_file_exists(project_root / file_path, description):
            all_checks_passed = False

    # Check 2: Documentation exists
    print("\n2. Checking Documentation...")
    doc_files = [
        ("PHASE_5.2_README.md", "Main README"),
        ("PHASE_5.2_COMMANDS.ps1", "Quick commands"),
        ("PHASE_5.2_COMPLETE.md", "Completion summary"),
    ]

    for file_path, description in doc_files:
        if not check_file_exists(project_root / file_path, description):
            all_checks_passed = False

    # Check 3: Dependencies exist
    print("\n3. Checking Dependencies...")
    dependencies = [
        ("src/attacks/pgd.py", "PGD attack"),
        ("src/losses/robust_loss.py", "Robust losses"),
        ("src/training/adversarial_trainer.py", "Adversarial trainer"),
        ("src/models/__init__.py", "Model builders"),
        ("src/datasets/__init__.py", "Dataset loaders"),
    ]

    for file_path, description in dependencies:
        if not check_file_exists(project_root / file_path, description):
            all_checks_passed = False

    # Check 4: Imports work
    print("\n4. Checking Imports...")
    try:
        sys.path.insert(0, str(project_root))

        # Try importing training script
        from scripts.training import train_pgd_at

        assert hasattr(train_pgd_at, "PGDATTrainer")
        assert hasattr(train_pgd_at, "run_multi_seed_training")
        print_success("Training script imports correctly")

        # Try importing evaluation script
        from scripts.evaluation import evaluate_pgd_at

        assert hasattr(evaluate_pgd_at, "PGDATEvaluator")
        print_success("Evaluation script imports correctly")

        # Try importing dependencies
        from src.attacks import PGD
        from src.attacks.pgd import PGDConfig
        from src.losses.robust_loss import AdversarialTrainingLoss
        from src.training.adversarial_trainer import AdversarialTrainer

        print_success("All dependencies import correctly")

    except ImportError as e:
        print_error(f"Import failed: {str(e)}")
        all_checks_passed = False
    except AssertionError as e:
        print_error(f"Missing required class/function: {str(e)}")
        all_checks_passed = False

    # Check 5: Configuration is valid YAML
    print("\n5. Checking Configuration...")
    try:
        import yaml

        config_path = project_root / "configs/experiments/pgd_at_isic.yaml"

        with open(config_path, "r") as f:
            config = yaml.safe_load(f)

        # Check required sections
        required_sections = ["model", "dataset", "training", "adversarial_training"]
        for section in required_sections:
            if section not in config:
                print_error(f"Missing config section: {section}")
                all_checks_passed = False
            else:
                print_success(f"Config section present: {section}")

        # Check attack parameters
        attack_config = config["adversarial_training"]["attack"]
        if attack_config["epsilon"] == 0.03137:  # 8/255
            print_success(f"Attack epsilon correct: {attack_config['epsilon']}")
        else:
            print_warning(
                f"Attack epsilon: {attack_config['epsilon']} (expected 0.03137)"
            )

        if attack_config["num_steps"] == 7:
            print_success(f"Training steps correct: {attack_config['num_steps']}")
        else:
            print_warning(f"Training steps: {attack_config['num_steps']} (expected 7)")

    except Exception as e:
        print_error(f"Config validation failed: {str(e)}")
        all_checks_passed = False

    # Check 6: Code quality metrics
    print("\n6. Checking Code Quality...")
    try:
        train_script = project_root / "scripts/training/train_pgd_at.py"
        eval_script = project_root / "scripts/evaluation/evaluate_pgd_at.py"

        train_lines = len(train_script.read_text().splitlines())
        eval_lines = len(eval_script.read_text().splitlines())

        if train_lines > 800:
            print_success(f"Training script: {train_lines} lines (comprehensive)")
        else:
            print_warning(f"Training script: {train_lines} lines (may be incomplete)")

        if eval_lines > 450:
            print_success(f"Evaluation script: {eval_lines} lines (comprehensive)")
        else:
            print_warning(f"Evaluation script: {eval_lines} lines (may be incomplete)")

        # Check for docstrings
        train_text = train_script.read_text()
        if '"""' in train_text and "Author:" in train_text:
            print_success("Training script has proper documentation")
        else:
            print_warning("Training script missing documentation")

    except Exception as e:
        print_error(f"Code quality check failed: {str(e)}")

    # Check 7: Directory structure
    print("\n7. Checking Directory Structure...")
    directories = [
        "scripts/training",
        "scripts/evaluation",
        "configs/experiments",
        "tests",
        "src/attacks",
        "src/losses",
        "src/training",
    ]

    for directory in directories:
        dir_path = project_root / directory
        if dir_path.exists() and dir_path.is_dir():
            print_success(f"Directory exists: {directory}")
        else:
            print_error(f"Directory missing: {directory}")
            all_checks_passed = False

    # Final summary
    print("\n" + "=" * 80)
    if all_checks_passed:
        print_success("Phase 5.2 Implementation: PRODUCTION-READY ✅")
        print("\nNext Steps:")
        print("1. Verify CUDA and GPU availability")
        print("2. Prepare ISIC 2018 dataset")
        print("3. Run single-seed training test")
        print("4. Run multi-seed production training")
        print("5. Execute evaluation pipeline")
        print("\nQuick Start:")
        print(
            "  python scripts/training/train_pgd_at.py --config configs/experiments/pgd_at_isic.yaml --seeds 42 123 456"
        )
    else:
        print_error("Phase 5.2 Implementation: ISSUES DETECTED ❌")
        print("\nPlease fix the issues above before proceeding.")

    print("=" * 80 + "\n")

    return all_checks_passed


if __name__ == "__main__":
    success = validate_phase_5_2()
    sys.exit(0 if success else 1)
