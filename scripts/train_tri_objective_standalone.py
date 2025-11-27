"""
Standalone Tri-Objective Training Script for Phase 7.7.

This script trains a model with tri-objective optimization:
- L_task: Classification loss with calibration
- L_rob: TRADES adversarial robustness
- L_expl: SSIM stability + TCAV concept alignment

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Phase: 7.7 - Initial Tri-Objective Validation
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
from src.datasets.isic import ISICDataset
from src.datasets.transforms import get_isic_transforms
from src.models.build import build_model
from src.training.tri_objective_trainer import TriObjectiveConfig, TriObjectiveTrainer
from src.utils.reproducibility import set_global_seed

# Optional MLflow import
try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    mlflow = None
    MLFLOW_AVAILABLE = False

logger = logging.getLogger(__name__)


def setup_logging(log_dir: Path) -> None:
    """Configure logging."""
    log_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = log_dir / f"tri_objective_train_{timestamp}.log"

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(),
        ],
    )


def create_dataloaders(
    data_root: Path,
    batch_size: int = 32,
    num_workers: int = 4,
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """Create train, val, test dataloaders."""
    logger.info("Creating ISIC 2018 datasets...")

    # Get transforms for each split
    train_transforms = get_isic_transforms(split="train", image_size=224)
    val_transforms = get_isic_transforms(split="val", image_size=224)
    test_transforms = get_isic_transforms(split="test", image_size=224)

    train_dataset = ISICDataset(
        root=data_root,
        split="train",
        transforms=train_transforms,
    )

    val_dataset = ISICDataset(
        root=data_root,
        split="val",
        transforms=val_transforms,
    )

    test_dataset = ISICDataset(
        root=data_root,
        split="test",
        transforms=test_transforms,
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(
        f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}"
    )

    return train_loader, val_loader, test_loader


def main(args: argparse.Namespace) -> None:
    """Main training function."""

    # Setup logging
    log_dir = Path(args.log_dir)
    setup_logging(log_dir)

    logger.info("=" * 80)
    logger.info("TRI-OBJECTIVE TRAINING - PHASE 7.7")
    logger.info("=" * 80)
    logger.info(f"Seed: {args.seed}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Data root: {args.data_root}")
    logger.info(f"lambda_rob: {args.lambda_rob}, lambda_expl: {args.lambda_expl}")

    # Set seed
    set_global_seed(args.seed)

    # Create dataloaders
    train_loader, val_loader, test_loader = create_dataloaders(
        data_root=Path(args.data_root),
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Build model
    logger.info(f"Building model: {args.model_name}")
    model = build_model(
        name=args.model_name,
        num_classes=7,  # ISIC 2018 has 7 classes
        pretrained=args.pretrained,
    )
    model = model.to(args.device)

    # Create optimizer
    optimizer = Adam(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )

    # Create scheduler
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.max_epochs,
        eta_min=1e-6,
    )

    # Create training config
    config = TriObjectiveConfig(
        max_epochs=args.max_epochs,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        early_stopping_patience=args.patience,
        lambda_rob=args.lambda_rob,
        lambda_expl=args.lambda_expl,
        lambda_ssim=args.lambda_ssim,
        lambda_tcav=args.lambda_tcav,
        temperature=args.temperature,
        trades_beta=args.trades_beta,
        pgd_epsilon=args.pgd_epsilon,
        pgd_step_size=args.pgd_step_size,
        pgd_num_steps=args.pgd_num_steps,
        pgd_random_start=args.pgd_random_start,
        generate_heatmaps=args.generate_heatmaps,
        extract_embeddings=args.extract_embeddings,
        batch_size=args.batch_size,
        device=args.device,
        use_mlflow=args.use_mlflow,
    )

    # Create trainer
    logger.info("Initializing TriObjectiveTrainer...")
    trainer = TriObjectiveTrainer(
        model=model,
        optimizer=optimizer,
        train_loader=train_loader,
        config=config,
        val_loader=val_loader,
        scheduler=scheduler,
        device=args.device,
    )

    # Setup MLflow if available
    if MLFLOW_AVAILABLE and args.use_mlflow:
        # End any active run before starting new one
        if mlflow.active_run() is not None:
            logger.warning("Active MLflow run detected, ending it...")
            mlflow.end_run()

        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run(run_name=f"tri_obj_seed{args.seed}")
        mlflow.log_params(
            {
                "seed": args.seed,
                "model": args.model_name,
                "batch_size": args.batch_size,
                "learning_rate": args.learning_rate,
                "lambda_rob": args.lambda_rob,
                "lambda_expl": args.lambda_expl,
                "max_epochs": args.max_epochs,
            }
        )

    # Train model
    logger.info("Starting training...")
    history = trainer.fit()

    # Save results
    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    results_file = results_dir / f"tri_objective_seed{args.seed}_results.json"
    with open(results_file, "w") as f:
        json.dump(history, f, indent=2)

    logger.info(f"Results saved to: {results_file}")

    # Evaluate on test set (if evaluate method exists)
    logger.info("Evaluating on test set...")
    try:
        test_metrics = trainer.evaluate(test_loader)

        logger.info("Test Metrics:")
        for metric, value in test_metrics.items():
            logger.info(f"  {metric}: {value:.4f}")

        # Log test metrics
        if MLFLOW_AVAILABLE and args.use_mlflow:
            for metric, value in test_metrics.items():
                mlflow.log_metric(f"test_{metric}", value)
    except AttributeError:
        logger.warning(
            "Trainer does not have evaluate() method. Skipping test evaluation."
        )

    if MLFLOW_AVAILABLE and args.use_mlflow:
        mlflow.end_run()

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETE")
    logger.info("=" * 80)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Tri-Objective Training Script",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Data arguments
    parser.add_argument(
        "--data-root", type=str, required=True, help="Path to ISIC 2018 data"
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/tri_objective",
        help="Results directory",
    )
    parser.add_argument(
        "--log-dir", type=str, default="logs/tri_objective", help="Log directory"
    )

    # Model arguments
    parser.add_argument(
        "--model-name", type=str, default="resnet50", help="Model architecture"
    )
    parser.add_argument(
        "--pretrained", action="store_true", default=True, help="Use pretrained weights"
    )

    # Training arguments
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", type=str, default="cuda", help="Device (cuda/cpu)")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )
    parser.add_argument("--max-epochs", type=int, default=100, help="Maximum epochs")
    parser.add_argument(
        "--learning-rate", type=float, default=1e-4, help="Learning rate"
    )
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay")
    parser.add_argument(
        "--patience", type=int, default=15, help="Early stopping patience"
    )

    # Tri-objective weights
    parser.add_argument(
        "--lambda-rob", type=float, default=0.3, help="Robustness loss weight"
    )
    parser.add_argument(
        "--lambda-expl", type=float, default=0.1, help="Explanation loss weight"
    )
    parser.add_argument(
        "--lambda-ssim", type=float, default=0.7, help="SSIM weight in explanation loss"
    )
    parser.add_argument(
        "--lambda-tcav", type=float, default=0.3, help="TCAV weight in explanation loss"
    )
    parser.add_argument(
        "--temperature", type=float, default=1.5, help="Calibration temperature"
    )
    parser.add_argument(
        "--trades-beta", type=float, default=6.0, help="TRADES beta parameter"
    )

    # PGD attack parameters
    parser.add_argument(
        "--pgd-epsilon", type=float, default=8.0 / 255.0, help="PGD epsilon (L-inf)"
    )
    parser.add_argument(
        "--pgd-step-size", type=float, default=2.0 / 255.0, help="PGD step size"
    )
    parser.add_argument(
        "--pgd-num-steps", type=int, default=7, help="PGD number of steps"
    )
    parser.add_argument(
        "--pgd-random-start", action="store_true", default=True, help="PGD random start"
    )

    # Explanation parameters
    parser.add_argument(
        "--generate-heatmaps",
        action="store_true",
        default=False,
        help="Generate Grad-CAM heatmaps",
    )
    parser.add_argument(
        "--extract-embeddings",
        action="store_true",
        default=True,
        help="Extract embeddings for TCAV",
    )

    # MLflow arguments
    parser.add_argument(
        "--use-mlflow", action="store_true", default=True, help="Use MLflow logging"
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="Tri-Objective-XAI-Dermoscopy",
        help="MLflow experiment name",
    )

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    main(args)
