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
from typing import Dict, List

# Color codes for output
GREEN = "\033[92m"
RED = "\033[91m"
YELLOW = "\033[93m"
BLUE = "\033[94m"
MAGENTA = "\033[95m"
CYAN = "\033[96m"
RESET = "\033[0m"


class ValidationResults:
    """Track validation results."""

    def __init__(self):
        self.critical_failures = []
        self.production_failures = []
        self.a1_plus_failures = []
        self.best_practice_warnings = []
        self.total_checks = 0
        self.passed_checks = 0

    def add_failure(self, level: str, msg: str):
        """Add a failure at specified level."""
        self.total_checks += 1
        if level == "CRITICAL":
            self.critical_failures.append(msg)
        elif level == "PRODUCTION":
            self.production_failures.append(msg)
        elif level == "A1+":
            self.a1_plus_failures.append(msg)
        elif level == "BEST_PRACTICE":
            self.best_practice_warnings.append(msg)

    def add_pass(self):
        """Add a passed check."""
        self.total_checks += 1
        self.passed_checks += 1

    def get_grade(self) -> str:
        """Calculate grade based on failures."""
        if self.critical_failures:
            return "F (Cannot Run)"
        if self.production_failures:
            return "C (Not Production-Ready)"
        if self.a1_plus_failures:
            return "B+ to A- (Good but Not A1+)"
        if self.best_practice_warnings:
            return "A+ (Excellent with Minor Improvements)"
        return "A1+ (Beyond Masters Standards)"


def print_success(msg: str):
    """Print success message in green."""
    print(f"{GREEN}✅ {msg}{RESET}")


def print_error(msg: str, level: str = "ERROR"):
    """Print error message."""
    if level == "CRITICAL":
        print(f"{RED}🚨 CRITICAL: {msg}{RESET}")
    elif level == "PRODUCTION":
        print(f"{RED}❌ PRODUCTION: {msg}{RESET}")
    elif level == "A1+":
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
    print(f"\n{CYAN}{'='*70}{RESET}")
    print(f"{CYAN}{msg:^70}{RESET}")
    print(f"{CYAN}{'='*70}{RESET}\n")


def check_file_has_pattern(file_path: Path, pattern: str, description: str) -> bool:
    """Check if file contains a pattern."""
    try:
        content = file_path.read_text()
        if re.search(pattern, content, re.MULTILINE | re.DOTALL):
            return True
        return False
    except Exception:
        return False


def check_critical_imports(project_root: Path, results: ValidationResults):
    """Check that all critical imports work."""
    print_header("CRITICAL: Import Validation")

    try:
        sys.path.insert(0, str(project_root))

        # Test training script imports
        try:
            from scripts.training import train_pgd_at

            assert hasattr(train_pgd_at, "PGDATTrainer")
            assert hasattr(train_pgd_at, "run_multi_seed_training")
            print_success("Training script imports successfully")
            results.add_pass()
        except (ImportError, AssertionError) as e:
            print_error(f"Training script import failed: {e}", "CRITICAL")
            results.add_failure("CRITICAL", "Training script imports failed")

        # Test evaluation script imports
        try:
            from scripts.evaluation import evaluate_pgd_at

            assert hasattr(evaluate_pgd_at, "PGDATEvaluator")
            print_success("Evaluation script imports successfully")
            results.add_pass()
        except (ImportError, AssertionError) as e:
            print_error(f"Evaluation script import failed: {e}", "CRITICAL")
            results.add_failure("CRITICAL", "Evaluation script imports failed")

        # Test dependency imports
        try:
            from src.attacks import FGSM, PGD, AutoAttack, CarliniWagner
            from src.attacks.pgd import PGDConfig
            from src.models import build_model
            from src.training.adversarial_trainer import (
                train_adversarial_epoch,
                validate_robust,
            )
            from src.utils.metrics import calculate_metrics

            print_success("All dependency imports work")
            results.add_pass()
        except ImportError as e:
            print_error(f"Dependency import failed: {e}", "CRITICAL")
            results.add_failure("CRITICAL", f"Dependency import failed: {e}")

    except Exception as e:
        print_error(f"Import validation crashed: {e}", "CRITICAL")
        results.add_failure("CRITICAL", "Import validation crashed")


def check_error_handling(project_root: Path, results: ValidationResults):
    """Check for proper error handling in code."""
    print_header("PRODUCTION: Error Handling Validation")

    train_script = project_root / "scripts/training/train_pgd_at.py"
    eval_script = project_root / "scripts/evaluation/evaluate_pgd_at.py"

    train_content = train_script.read_text()
    eval_content = eval_script.read_text()

    # Check for try-except blocks
    train_try_count = len(re.findall(r"\btry:", train_content))
    eval_try_count = len(re.findall(r"\btry:", eval_content))

    if train_try_count >= 3:
        print_success(f"Training script has {train_try_count} error handlers")
        results.add_pass()
    else:
        print_error(
            f"Training script has only {train_try_count} error handlers "
            f"(need ≥3 for checkpoint loading, data loading, training)",
            "PRODUCTION",
        )
        results.add_failure(
            "PRODUCTION", "Insufficient error handling in training script"
        )

    if eval_try_count >= 3:
        print_success(f"Evaluation script has {eval_try_count} error handlers")
        results.add_pass()
    else:
        print_error(
            f"Evaluation script has only {eval_try_count} error handlers "
            f"(need ≥3 for checkpoint loading, data loading, evaluation)",
            "PRODUCTION",
        )
        results.add_failure(
            "PRODUCTION", "Insufficient error handling in evaluation script"
        )

    # Check for Path.exists() validations
    train_exists_count = len(re.findall(r"\.exists\(\)", train_content))
    eval_exists_count = len(re.findall(r"\.exists\(\)", eval_content))

    if train_exists_count >= 2:
        print_success(
            f"Training script validates file existence ({train_exists_count}x)"
        )
        results.add_pass()
    else:
        print_error(
            f"Training script lacks file validation ({train_exists_count}x)",
            "PRODUCTION",
        )
        results.add_failure(
            "PRODUCTION", "Missing file existence validation in training"
        )


def check_rq1_hypothesis_test(project_root: Path, results: ValidationResults):
    """Check if RQ1 cross-site hypothesis test is implemented."""
    print_header("A1+: RQ1 Hypothesis Test Implementation")

    eval_script = project_root / "scripts/evaluation/evaluate_pgd_at.py"
    eval_content = eval_script.read_text()

    # Check for RQ1 test function
    has_rq1_function = bool(
        re.search(
            r"def\s+.*rq1.*\(|def\s+.*cross.*site.*\(|" r"def\s+.*hypothesis.*test.*\(",
            eval_content,
            re.IGNORECASE,
        )
    )

    if has_rq1_function:
        print_success("RQ1 hypothesis test function exists")
        results.add_pass()
    else:
        print_error(
            "RQ1 hypothesis test function NOT FOUND - This is the core "
            "research question!",
            "A1+",
        )
        results.add_failure("A1+", "RQ1 cross-site hypothesis test not implemented")
        print_info(
            "  Expected: Function to compare PGD-AT vs baseline "
            "on cross-site AUROC drops"
        )
        return

    # Check for AUROC drop calculation
    has_auroc_drop = "auroc" in eval_content.lower() and (
        "drop" in eval_content.lower() or "degradation" in eval_content.lower()
    )

    if has_auroc_drop:
        print_success("AUROC drop calculation present")
        results.add_pass()
    else:
        print_error(
            "AUROC drop calculation missing - Cannot test cross-site " "generalization",
            "A1+",
        )
        results.add_failure("A1+", "AUROC drop calculation not implemented")

    # Check for statistical testing
    has_ttest = "ttest" in eval_content.lower() or "t_test" in eval_content

    if has_ttest:
        print_success("Statistical testing (t-test) present")
        results.add_pass()
    else:
        print_error("Statistical testing missing - Cannot validate hypothesis", "A1+")
        results.add_failure("A1+", "Statistical hypothesis testing not implemented")


def check_statistical_rigor(project_root: Path, results: ValidationResults):
    """Check for advanced statistical analysis."""
    print_header("A1+: Statistical Analysis Rigor")

    eval_script = project_root / "scripts/evaluation/evaluate_pgd_at.py"
    eval_content = eval_script.read_text()

    # Check for Bonferroni correction
    has_bonferroni = (
        "bonferroni" in eval_content.lower() or "multipletests" in eval_content
    )

    if has_bonferroni:
        print_success("Bonferroni/multiple comparison correction present")
        results.add_pass()
    else:
        print_error(
            "Bonferroni correction missing - Testing multiple datasets "
            "requires correction",
            "A1+",
        )
        results.add_failure("A1+", "Multiple comparison correction not implemented")

    # Check for confidence intervals
    has_confidence_intervals = (
        "confidence" in eval_content.lower() and "interval" in eval_content.lower()
    )

    if has_confidence_intervals:
        print_success("Confidence interval calculation present")
        results.add_pass()
    else:
        print_warning("Confidence intervals missing - Effect sizes need uncertainty")
        results.add_failure(
            "A1+", "Confidence intervals on effect sizes not implemented"
        )

    # Check for normality testing
    has_normality_test = (
        "shapiro" in eval_content.lower() or "normaltest" in eval_content.lower()
    )

    if has_normality_test:
        print_success("Normality testing present")
        results.add_pass()
    else:
        print_warning(
            "Normality testing missing - Should verify parametric " "test assumptions"
        )
        results.add_failure("BEST_PRACTICE", "Normality testing not implemented")

    # Check for power analysis
    has_power_analysis = "power" in eval_content.lower() and (
        "analysis" in eval_content.lower() or "calculate" in eval_content.lower()
    )

    if has_power_analysis:
        print_success("Statistical power analysis present")
        results.add_pass()
    else:
        print_warning("Power analysis missing - Cannot assess if sample size adequate")
        results.add_failure(
            "BEST_PRACTICE", "Statistical power analysis not implemented"
        )


def check_dataset_handling(project_root: Path, results: ValidationResults):
    """Check if all datasets are properly handled."""
    print_header("PRODUCTION: Dataset Handling Validation")

    train_script = project_root / "scripts/training/train_pgd_at.py"
    eval_script = project_root / "scripts/evaluation/evaluate_pgd_at.py"

    train_content = train_script.read_text()
    eval_content = eval_script.read_text()

    # Check for dataset support
    required_datasets = ["isic2018", "isic2019", "isic2020", "derm7pt"]

    for dataset in required_datasets:
        if dataset in eval_content.lower():
            print_success(f"Dataset {dataset} referenced in evaluation")
            results.add_pass()
        else:
            print_error(
                f"Dataset {dataset} not found in evaluation script", "PRODUCTION"
            )
            results.add_failure("PRODUCTION", f"Dataset {dataset} not properly handled")

    # Check for dataset error handling
    has_dataset_validation = "dataset" in eval_content.lower() and (
        "validate" in eval_content.lower()
        or "check" in eval_content.lower()
        or "exists" in eval_content.lower()
    )

    if has_dataset_validation:
        print_success("Dataset validation logic present")
        results.add_pass()
    else:
        print_error(
            "Dataset validation missing - May fail silently on " "unsupported datasets",
            "PRODUCTION",
        )
        results.add_failure("PRODUCTION", "Dataset validation not implemented")


def check_memory_management(project_root: Path, results: ValidationResults):
    """Check for proper memory management."""
    print_header("PRODUCTION: Memory Management Validation")

    eval_script = project_root / "scripts/evaluation/evaluate_pgd_at.py"
    eval_content = eval_script.read_text()

    # Check for CUDA cache clearing
    has_cuda_clear = "torch.cuda.empty_cache()" in eval_content

    if has_cuda_clear:
        print_success("CUDA cache clearing present")
        results.add_pass()
    else:
        print_error(
            "CUDA cache clearing missing - May cause OOM errors on "
            "multi-dataset evaluation",
            "PRODUCTION",
        )
        results.add_failure("PRODUCTION", "CUDA memory management not implemented")

    # Check for explicit deletions
    has_del_statement = re.search(r"\bdel\s+\w+", eval_content)

    if has_del_statement:
        print_success("Explicit object cleanup present")
        results.add_pass()
    else:
        print_warning("No explicit cleanup (del statements) - Memory may accumulate")
        results.add_failure("BEST_PRACTICE", "Explicit memory cleanup not implemented")

    # Check for garbage collection
    has_gc = "gc.collect()" in eval_content

    if has_gc:
        print_success("Garbage collection explicitly triggered")
        results.add_pass()
    else:
        print_warning(
            "Manual garbage collection not triggered - May help with " "memory pressure"
        )
        results.add_failure("BEST_PRACTICE", "Manual garbage collection not used")


def check_documentation_quality(project_root: Path, results: ValidationResults):
    """Check documentation completeness."""
    print_header("A1+: Documentation Quality")

    readme_path = project_root / "PHASE_5.2_README.md"

    if not readme_path.exists():
        print_error("README file missing", "A1+")
        results.add_failure("A1+", "Documentation file missing")
        return

    readme_content = readme_path.read_text()

    # Check for key sections
    required_sections = [
        "Overview",
        "Research Question",
        "Implementation",
        "Usage",
        "Results",
    ]

    for section in required_sections:
        if section.lower() in readme_content.lower():
            print_success(f"Documentation section present: {section}")
            results.add_pass()
        else:
            print_warning(f"Documentation section missing: {section}")
            results.add_failure(
                "BEST_PRACTICE", f"Documentation missing {section} section"
            )

    # Check documentation length
    readme_lines = len(readme_content.splitlines())

    if readme_lines >= 250:
        print_success(f"Comprehensive documentation: {readme_lines} lines")
        results.add_pass()
    else:
        print_warning(f"Documentation brief: {readme_lines} lines (expected ≥250)")
        results.add_failure(
            "BEST_PRACTICE", "Documentation could be more comprehensive"
        )


def check_code_quality_metrics(project_root: Path, results: ValidationResults):
    """Check code quality metrics."""
    print_header("BEST PRACTICE: Code Quality Metrics")

    train_script = project_root / "scripts/training/train_pgd_at.py"
    eval_script = project_root / "scripts/evaluation/evaluate_pgd_at.py"

    train_content = train_script.read_text()
    eval_content = eval_script.read_text()

    # Check for type hints
    train_type_hints = len(
        re.findall(r":\s*\w+\s*=|def\s+\w+\([^)]*:\s*\w+", train_content)
    )  # noqa
    eval_type_hints = len(
        re.findall(r":\s*\w+\s*=|def\s+\w+\([^)]*:\s*\w+", eval_content)
    )  # noqa

    if train_type_hints >= 10 and eval_type_hints >= 10:
        print_success(
            f"Type hints present (train: {train_type_hints}, eval: {eval_type_hints})"
        )  # noqa
        results.add_pass()
    else:
        print_warning(
            f"Limited type hints (train: {train_type_hints}, eval: {eval_type_hints})"
        )  # noqa
        results.add_failure("BEST_PRACTICE", "Limited type hint usage")

    # Check for docstrings
    train_docstrings = train_content.count('"""')
    eval_docstrings = eval_content.count('"""')

    if train_docstrings >= 10 and eval_docstrings >= 8:
        print_success(
            f"Comprehensive docstrings (train: {train_docstrings//2}, eval: {eval_docstrings//2})"
        )  # noqa
        results.add_pass()
    else:
        print_warning(
            f"Limited docstrings (train: {train_docstrings//2}, eval: {eval_docstrings//2})"
        )  # noqa
        results.add_failure("BEST_PRACTICE", "More docstrings recommended")


def print_summary_report(results: ValidationResults):
    """Print comprehensive summary report."""
    print_header("VALIDATION SUMMARY REPORT")

    print(f"Total Checks: {results.total_checks}")
    print(f"Passed: {GREEN}{results.passed_checks}{RESET}")
    print(f"Failed: {RED}{results.total_checks - results.passed_checks}{RESET}")
    print()

    if results.critical_failures:
        print(
            f"{RED}🚨 CRITICAL FAILURES ({len(results.critical_failures)}):{RESET}"
        )  # noqa
        for failure in results.critical_failures:
            print(f"  - {failure}")
        print()

    if results.production_failures:
        print(
            f"{RED}❌ PRODUCTION BLOCKERS ({len(results.production_failures)}):{RESET}"
        )  # noqa
        for failure in results.production_failures:
            print(f"  - {failure}")
        print()

    if results.a1_plus_failures:
        print(
            f"{MAGENTA}❌ A1+ REQUIREMENTS ({len(results.a1_plus_failures)}):{RESET}"
        )  # noqa
        for failure in results.a1_plus_failures:
            print(f"  - {failure}")
        print()

    if results.best_practice_warnings:
        print(
            f"{YELLOW}⚠️  BEST PRACTICE IMPROVEMENTS ({len(results.best_practice_warnings)}):{RESET}"
        )  # noqa
        for warning in results.best_practice_warnings:
            print(f"  - {warning}")
        print()

    # Calculate and display grade
    grade = results.get_grade()

    print(f"\n{CYAN}{'='*70}{RESET}")
    print(f"{CYAN}FINAL GRADE: {grade:^54}{RESET}")
    print(f"{CYAN}{'='*70}{RESET}\n")

    # Provide recommendations
    if results.critical_failures:
        print(f"{RED}IMMEDIATE ACTION REQUIRED:{RESET}")
        print("  Phase 5.2 cannot run. Fix critical issues first.")
    elif results.production_failures:
        print(f"{RED}NOT PRODUCTION-READY:{RESET}")
        print("  Code runs but lacks production safeguards.")
    elif results.a1_plus_failures:
        print(f"{YELLOW}GOOD BUT NOT A1+:{RESET}")
        print("  Code is production-ready but misses research requirements.")
    elif results.best_practice_warnings:
        print(f"{GREEN}EXCELLENT WITH MINOR IMPROVEMENTS:{RESET}")
        print("  Meets all requirements, some best practices could be added.")
    else:
        print(f"{GREEN}🎉 BEYOND MASTERS STANDARDS!{RESET}")
        print("  Production-ready, research-complete, best practices followed.")


def main():
    """Main validation entry point."""
    print(f"\n{CYAN}{'='*70}{RESET}")
    print(f"{CYAN}Phase 5.2: Production & A1+ Standards Validation{RESET:^70}")
    print(f"{CYAN}{'='*70}{RESET}\n")

    project_root = Path(__file__).parent.parent.parent
    results = ValidationResults()

    # Run all validation checks
    check_critical_imports(project_root, results)
    check_error_handling(project_root, results)
    check_rq1_hypothesis_test(project_root, results)
    check_statistical_rigor(project_root, results)
    check_dataset_handling(project_root, results)
    check_memory_management(project_root, results)
    check_documentation_quality(project_root, results)
    check_code_quality_metrics(project_root, results)

    # Print summary
    print_summary_report(results)

    # Exit with appropriate code
    if results.critical_failures:
        sys.exit(1)
    elif results.production_failures or results.a1_plus_failures:
        sys.exit(2)
    else:
        sys.exit(0)


if __name__ == "__main__":
    main()
