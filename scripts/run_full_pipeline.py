"""
Fully automated pipeline for Tri-Objective Robust XAI experiments.

Runs:
1. Environment validation
2. Data preparation
3. Baseline training (multi-seed)
4. Results aggregation
5. Report generation
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def run_command(cmd: list, description: str) -> bool:
    """Run shell command and handle errors."""
    logger.info(f"\n{'='*80}")
    logger.info(f"RUNNING: {description}")
    logger.info(f"{'='*80}")
    logger.info(f"Command: {' '.join(cmd)}")

    try:
        subprocess.run(cmd, check=True, cwd=PROJECT_ROOT)
        logger.info(f"✓ {description} completed successfully")
        return True
    except subprocess.CalledProcessError as e:
        logger.error(f"✗ {description} failed with exit code {e.returncode}")
        return False


def main(args):
    """Run full pipeline."""
    logger.info("=" * 80)
    logger.info("TRI-OBJECTIVE ROBUST XAI - AUTOMATED PIPELINE")
    logger.info("=" * 80)

    # Step 1: Validate setup
    if not args.skip_validation:
        if not run_command(
            [sys.executable, "scripts/validate_setup.py"], "Environment validation"
        ):
            logger.error("Setup validation failed. Fix errors before continuing.")
            return False

    # Step 2: Download data (if needed)
    if not args.skip_data:
        data_path = PROJECT_ROOT / "data/raw/isic2018"
        if not data_path.exists() or not any(data_path.iterdir()):
            if not run_command(
                [sys.executable, "scripts/data/download_isic2018.py"],
                "ISIC2018 dataset download",
            ):
                logger.error("Data download failed")
                return False

    # Step 3: Prepare data
    if not args.skip_data:
        if not run_command(
            [sys.executable, "scripts/data/prepare_isic2018.py"],
            "ISIC2018 data preparation",
        ):
            logger.error("Data preparation failed")
            return False

    # Step 4: Train baseline models (multi-seed)
    seeds = args.seeds or [42, 123, 456]
    config_path = "configs/experiments/rq1_robustness/baseline_isic2018_resnet50.yaml"

    for seed in seeds:
        if not run_command(
            [
                sys.executable,
                "-m",
                "src.training.train_baseline",
                "--config",
                config_path,
                "--seed",
                str(seed),
            ],
            f"Baseline training (seed={seed})",
        ):
            logger.error(f"Training failed for seed {seed}")
            if not args.continue_on_error:
                return False

    # Step 5: Generate results summary
    if not run_command(
        [sys.executable, "scripts/results/generate_baseline_table.py"],
        "Results aggregation",
    ):
        logger.error("Results aggregation failed")
        return False

    # Step 6: Validate results
    if not run_command(
        [sys.executable, "scripts/results/validate_baseline_isic2018.py"],
        "Results validation",
    ):
        logger.warning("Results validation found issues (check above)")

    # Step 7: Generate plots (if requested)
    if args.generate_plots:
        logger.info("\n" + "=" * 80)
        logger.info("Generating visualization plots...")
        logger.info("=" * 80)
        # Add plot generation commands here

    logger.info("\n" + "=" * 80)
    logger.info("✓ PIPELINE COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)
    logger.info("Results saved to: results/metrics/rq1_robustness/")
    logger.info("Checkpoints saved to: checkpoints/baseline/")

    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run full experiment pipeline")
    parser.add_argument(
        "--seeds",
        nargs="+",
        type=int,
        default=[42, 123, 456],
        help="Random seeds for multi-seed experiments",
    )
    parser.add_argument(
        "--skip-validation", action="store_true", help="Skip environment validation"
    )
    parser.add_argument(
        "--skip-data", action="store_true", help="Skip data download/preparation"
    )
    parser.add_argument(
        "--continue-on-error",
        action="store_true",
        help="Continue pipeline even if some seeds fail",
    )
    parser.add_argument(
        "--generate-plots", action="store_true", help="Generate visualization plots"
    )

    args = parser.parse_args()

    success = main(args)
    sys.exit(0 if success else 1)
