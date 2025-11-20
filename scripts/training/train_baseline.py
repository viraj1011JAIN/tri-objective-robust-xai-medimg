from __future__ import annotations

"""
CLI entrypoint for baseline training.

This thin wrapper is responsible only for:
- Parsing command line arguments
- Ensuring the project root is on sys.path
- Calling `src.training.train_baseline.main(args)` with a rich Namespace

All core training logic lives in `src/training/train_baseline.py`.
"""

import argparse
import logging
import sys
from pathlib import Path

# -----------------------------------------------------------------------------
# Make sure project root is importable as a package
# -----------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.training import train_baseline as tb  # noqa: E402

logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
def _build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Baseline training script (ISIC 2018 / dermoscopy).",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Core config
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment YAML configuration.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )

    # Device / debug
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help=(
            "Device override (e.g. 'cuda', 'cpu'). "
            "If omitted, the training module will choose automatically."
        ),
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help=(
            "Enable debug mode (may reduce dataset size / epochs, "
            "depending on training implementation)."
        ),
    )

    # Dataloader overrides (provide both naming variants to be safe)
    parser.add_argument(
        "--override-batch-size",
        dest="override_batch_size",
        type=int,
        default=None,
        help="Optional batch size override.",
    )
    parser.add_argument(
        "--override-num-workers",
        dest="override_num_workers",
        type=int,
        default=None,
        help="Optional DataLoader num_workers override.",
    )

    # Experiment naming / logging
    parser.add_argument(
        "--experiment-name",
        type=str,
        default=None,
        help="Optional experiment name override (used for MLflow / checkpoints).",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="results/logs/baseline_isic2018_resnet50",
        help="Base directory for training logs.",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="results/checkpoints/baseline_isic2018_resnet50",
        help="Base directory for checkpoints (per-seed subfolders will be created).",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/metrics/baseline_isic2018_resnet50",
        help="Base directory for training summaries/metrics.",
    )

    # MLflow convenience flag
    parser.add_argument(
        "--no-mlflow",
        dest="use_mlflow",
        action="store_false",
        help="Disable MLflow logging even if enabled in the config.",
    )
    parser.set_defaults(use_mlflow=True)

    # Optional run suffix for easier multi-run organisation
    parser.add_argument(
        "--run-suffix",
        type=str,
        default=None,
        help="Optional suffix appended to the run name (e.g. 'debug', 'try1').",
    )

    return parser


def _parse_cli_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = _build_arg_parser()
    args = parser.parse_args(argv)

    # ------------------------------------------------------------------
    # Normalise / duplicate fields so `src.training.train_baseline.main`
    # can use whichever naming convention it expects.
    # ------------------------------------------------------------------

    # Mirror batch size override
    if not hasattr(args, "batch_size_override"):
        args.batch_size_override = args.override_batch_size

    # Mirror num_workers override
    if not hasattr(args, "num_workers_override"):
        args.num_workers_override = args.override_num_workers

    # Mirror experiment name override
    if not hasattr(args, "experiment_name_override"):
        args.experiment_name_override = args.experiment_name

    # Ensure directories exist on the Namespace
    if not hasattr(args, "log_dir"):
        args.log_dir = "results/logs/baseline_isic2018_resnet50"
    if not hasattr(args, "checkpoint_dir"):
        args.checkpoint_dir = "results/checkpoints/baseline_isic2018_resnet50"
    if not hasattr(args, "results_dir"):
        args.results_dir = "results/metrics/baseline_isic2018_resnet50"

    # Ensure MLflow + run_suffix fields exist
    if not hasattr(args, "use_mlflow"):
        args.use_mlflow = True
    if not hasattr(args, "run_suffix"):
        args.run_suffix = None

    return args


# -----------------------------------------------------------------------------
# Entrypoint
# -----------------------------------------------------------------------------
def main(argv: list[str] | None = None) -> None:
    """
    Parse CLI args and delegate to `src.training.train_baseline.main`.
    """
    args = _parse_cli_args(argv)
    tb.main(args)


if __name__ == "__main__":
    main()
