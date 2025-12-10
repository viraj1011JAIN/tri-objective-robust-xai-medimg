"""
Execute complete RQ2 TCAV evaluation pipeline.
Production-ready execution script.
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path


def run_command(cmd, description, timeout=None):
    """Execute command with progress tracking."""
    print(f"\n{'='*60}")
    print(f"EXECUTING: {description}")
    print(f"Command: {cmd}")
    print(f"{'='*60}")

    start_time = time.time()

    try:
        result = subprocess.run(
            cmd,
            shell=True,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=Path.cwd(),
        )

        duration = time.time() - start_time

        if result.returncode == 0:
            print(f"✓ SUCCESS ({duration:.1f}s)")
            if result.stdout.strip():
                print("Output:", result.stdout.strip())
            return True
        else:
            print(f"✗ FAILED ({duration:.1f}s)")
            print("STDOUT:", result.stdout)
            print("STDERR:", result.stderr)
            return False

    except subprocess.TimeoutExpired:
        print(f"✗ TIMEOUT after {timeout}s")
        return False
    except Exception as e:
        print(f"✗ ERROR: {e}")
        return False


def check_prerequisites():
    """Check if models and data are ready."""
    print("\n" + "=" * 60)
    print("CHECKING PREREQUISITES")
    print("=" * 60)

    required_files = [
        "checkpoints/baseline/best.pt",
        "checkpoints/tri_objective/best.pt",
        "data/ISIC2018/test",
    ]

    missing = []
    for file_path in required_files:
        if not Path(file_path).exists():
            missing.append(file_path)

    if missing:
        print("✗ Missing required files:")
        for f in missing:
            print(f"  - {f}")
        return False

    print("✓ All prerequisites met")
    return True


def main():
    parser = argparse.ArgumentParser(description="Execute RQ2 TCAV Evaluation Pipeline")
    parser.add_argument(
        "--skip-concept-bank",
        action="store_true",
        help="Skip concept bank creation if already exists",
    )
    parser.add_argument(
        "--skip-cav-training",
        action="store_true",
        help="Skip CAV training if already exists",
    )
    parser.add_argument(
        "--quick", action="store_true", help="Quick execution with reduced samples"
    )

    args = parser.parse_args()

    print("\n" + "=" * 80)
    print("RQ2 TCAV EVALUATION PIPELINE")
    print("Production Execution Script")
    print("=" * 80)

    # Check prerequisites
    if not check_prerequisites():
        print("\nERROR: Prerequisites not met. Please ensure models are trained.")
        sys.exit(1)

    # Create output directories
    Path("results/rq2_complete").mkdir(parents=True, exist_ok=True)
    Path("results/tables").mkdir(parents=True, exist_ok=True)
    Path("results/statistics").mkdir(parents=True, exist_ok=True)
    Path("data/concept_bank").mkdir(parents=True, exist_ok=True)

    steps = []

    # Step 1: Create concept bank
    if (
        not args.skip_concept_bank
        or not Path("data/concept_bank/concept_bank.pkl").exists()
    ):
        steps.append(
            {
                "cmd": "python scripts/data/create_concept_bank.py --data-dir data/ISIC2018/test --output-dir data/concept_bank --max-samples 1000",
                "desc": "Creating concept bank",
                "timeout": 1800,  # 30 minutes
            }
        )
    else:
        print("✓ Skipping concept bank creation (already exists)")

    # Step 2: Train CAVs
    if not args.skip_cav_training:
        steps.append(
            {
                "cmd": "python scripts/evaluation/train_cavs.py",
                "desc": "Training Concept Activation Vectors",
                "timeout": 3600,  # 1 hour
            }
        )
    else:
        print("✓ Skipping CAV training")

    # Step 3: Run RQ2 evaluation
    eval_cmd = "python scripts/evaluation/evaluate_rq2_complete.py"
    if args.quick:
        eval_cmd += " --quick"

    steps.append(
        {
            "cmd": eval_cmd,
            "desc": "Running RQ2 comprehensive evaluation",
            "timeout": 7200,  # 2 hours
        }
    )

    # Step 4: Generate tables
    steps.append(
        {
            "cmd": "python scripts/results/generate_rq2_tables.py",
            "desc": "Generating result tables",
            "timeout": 300,  # 5 minutes
        }
    )

    # Step 5: Statistical tests
    steps.append(
        {
            "cmd": "python scripts/results/rq2_statistical_tests.py",
            "desc": "Running statistical hypothesis tests",
            "timeout": 300,  # 5 minutes
        }
    )

    # Execute pipeline
    start_time = time.time()
    failed_steps = []

    for i, step in enumerate(steps, 1):
        print(f"\n\nSTEP {i}/{len(steps)}")
        success = run_command(step["cmd"], step["desc"], step["timeout"])

        if not success:
            failed_steps.append(f"Step {i}: {step['desc']}")

    total_duration = time.time() - start_time

    # Final summary
    print("\n" + "=" * 80)
    print("PIPELINE EXECUTION SUMMARY")
    print("=" * 80)

    print(f"Total execution time: {total_duration/3600:.1f} hours")

    if failed_steps:
        print(f"\n✗ {len(failed_steps)} steps failed:")
        for step in failed_steps:
            print(f"  - {step}")
        print("\nPipeline completed with errors.")
        sys.exit(1)
    else:
        print("\n✓ All steps completed successfully!")

        # Show results location
        print("\nResults generated:")
        print("  - Raw results: results/rq2_complete/")
        print("  - Tables: results/tables/")
        print("  - Statistics: results/statistics/")

        print("\nRQ2 evaluation pipeline completed successfully!")


if __name__ == "__main__":
    main()
