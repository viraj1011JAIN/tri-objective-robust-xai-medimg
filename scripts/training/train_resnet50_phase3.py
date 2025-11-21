"""
Training script for ResNet-50 with Phase 3.2 production-grade losses.

This script trains ResNet-50 on medical imaging datasets (ISIC 2018, Derm7pt)
with support for:
- TaskLoss (CrossEntropy, BinaryCrossEntropy, FocalLoss)
- CalibrationLoss (temperature scaling + label smoothing)
- GPU acceleration (CUDA)
- Checkpointing & logging (TensorBoard, MLflow)
- Hyperparameter optimization

Usage:
    # Train with TaskLoss (CrossEntropy)
    python scripts/training/train_resnet50_phase3.py --dataset isic2018

    # Train with TaskLoss (FocalLoss for class imbalance)
    python scripts/training/train_resnet50_phase3.py --dataset isic2018 --use-focal-loss --focal-gamma 2.0

    # Train with CalibrationLoss (temperature scaling + label smoothing)
    python scripts/training/train_resnet50_phase3.py --dataset isic2018 --use-calibration --temperature 1.5 --label-smoothing 0.1

Phase 3.3 Baseline Training Integration - Part of Master's Dissertation
Author: Viraj Jain
Date: November 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import Derm7ptDataset, ISIC2018Dataset
from src.models.build import build_classifier
from src.training.baseline_trainer import BaselineTrainer, TrainingConfig
from src.utils.config import load_config
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train ResNet-50 with Phase 3.2 losses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="isic2018",
        choices=["isic2018", "derm7pt"],
        help="Dataset to train on",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="F:/data",
        help="Root directory for datasets",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
    )

    # Model
    parser.add_argument(
        "--pretrained",
        action="store_true",
        help="Use ImageNet pretrained weights",
    )
    parser.add_argument(
        "--dropout",
        type=float,
        default=0.5,
        help="Dropout rate",
    )

    # Loss configuration
    parser.add_argument(
        "--use-focal-loss",
        action="store_true",
        help="Use FocalLoss instead of CrossEntropy",
    )
    parser.add_argument(
        "--focal-gamma",
        type=float,
        default=2.0,
        help="Gamma parameter for FocalLoss",
    )
    parser.add_argument(
        "--use-calibration",
        action="store_true",
        help="Use CalibrationLoss (temperature scaling + label smoothing)",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.5,
        help="Initial temperature for CalibrationLoss",
    )
    parser.add_argument(
        "--label-smoothing",
        type=float,
        default=0.0,
        help="Label smoothing factor (0.0 = no smoothing, 0.1 = typical)",
    )

    # Training
    parser.add_argument(
        "--max-epochs",
        type=int,
        default=100,
        help="Maximum number of training epochs",
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=0.0001,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=0.0001,
        help="Weight decay (L2 regularization)",
    )
    parser.add_argument(
        "--gradient-clip",
        type=float,
        default=1.0,
        help="Gradient clipping value",
    )

    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to train on (cuda/cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )

    # Logging & checkpointing
    parser.add_argument(
        "--log-dir",
        type=str,
        default="results/logs/resnet50_phase3",
        help="Directory for TensorBoard logs",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="results/checkpoints/resnet50_phase3",
        help="Directory for model checkpoints",
    )
    parser.add_argument(
        "--log-every-n-steps",
        type=int,
        default=50,
        help="Log training metrics every N steps",
    )
    parser.add_argument(
        "--eval-every-n-epochs",
        type=int,
        default=1,
        help="Run validation every N epochs",
    )

    # Early stopping
    parser.add_argument(
        "--early-stopping",
        action="store_true",
        help="Enable early stopping",
    )
    parser.add_argument(
        "--early-stopping-patience",
        type=int,
        default=15,
        help="Early stopping patience (epochs)",
    )

    # MLflow
    parser.add_argument(
        "--use-mlflow",
        action="store_true",
        help="Enable MLflow logging",
    )
    parser.add_argument(
        "--experiment-name",
        type=str,
        default="ResNet50_Phase3.3",
        help="MLflow experiment name",
    )

    return parser.parse_args()


def load_dataset(
    dataset_name: str,
    data_root: str,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """
    Load and prepare dataset.

    Args:
        dataset_name: Name of dataset (isic2018, derm7pt)
        data_root: Root directory for data
        batch_size: Batch size for DataLoader
        num_workers: Number of DataLoader workers

    Returns:
        train_loader: Training DataLoader
        val_loader: Validation DataLoader
        test_loader: Test DataLoader
        num_classes: Number of classes in dataset
    """
    logger.info(f"Loading dataset: {dataset_name}")

    if dataset_name == "isic2018":
        # ISIC 2018 (7 classes)
        data_dir = Path(data_root) / "isic_2018"
        train_dataset = ISIC2018Dataset(
            root=str(data_dir),
            split="train",
            download=False,
        )
        val_dataset = ISIC2018Dataset(
            root=str(data_dir),
            split="val",
            download=False,
        )
        test_dataset = ISIC2018Dataset(
            root=str(data_dir),
            split="test",
            download=False,
        )
        num_classes = 7

    elif dataset_name == "derm7pt":
        # Derm7pt (2 classes - benign/malignant)
        data_dir = Path(data_root) / "derm7pt"
        train_dataset = Derm7ptDataset(
            root=str(data_dir),
            split="train",
            download=False,
        )
        val_dataset = Derm7ptDataset(
            root=str(data_dir),
            split="val",
            download=False,
        )
        test_dataset = Derm7ptDataset(
            root=str(data_dir),
            split="test",
            download=False,
        )
        num_classes = 2

    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    # Create DataLoaders
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
        f"Dataset loaded: {len(train_dataset)} train, "
        f"{len(val_dataset)} val, {len(test_dataset)} test"
    )

    return train_loader, val_loader, test_loader, num_classes


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Setup logging
    setup_logging(log_dir=args.log_dir)
    logger.info("=" * 80)
    logger.info("ResNet-50 Phase 3.3 Baseline Training")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max epochs: {args.max_epochs}")
    logger.info(f"Learning rate: {args.lr}")

    # Log loss configuration
    if args.use_calibration:
        logger.info(
            f"Loss: CalibrationLoss (temperature={args.temperature}, "
            f"label_smoothing={args.label_smoothing})"
        )
    elif args.use_focal_loss:
        logger.info(f"Loss: TaskLoss (FocalLoss, gamma={args.focal_gamma})")
    else:
        logger.info("Loss: TaskLoss (CrossEntropy)")

    # Set random seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load dataset
    train_loader, val_loader, test_loader, num_classes = load_dataset(
        dataset_name=args.dataset,
        data_root=args.data_root,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Build model
    logger.info("Building ResNet-50 model...")
    model = build_classifier(
        model_name="resnet50",
        num_classes=num_classes,
        pretrained=args.pretrained,
        dropout=args.dropout,
    )
    logger.info(f"Model created: {model.__class__.__name__}")

    # Create optimizer
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    # Create training config
    config = TrainingConfig(
        max_epochs=args.max_epochs,
        eval_every_n_epochs=args.eval_every_n_epochs,
        log_every_n_steps=args.log_every_n_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        gradient_clip_val=args.gradient_clip,
        early_stopping=args.early_stopping,
        early_stopping_patience=args.early_stopping_patience,
    )

    # Create trainer with Phase 3.2 losses
    logger.info("Creating BaselineTrainer with Phase 3.2 losses...")
    trainer = BaselineTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        config=config,
        num_classes=num_classes,
        device=args.device,
        # Phase 3.2 loss configuration
        task_type="multi_class",
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        use_calibration=args.use_calibration,
        init_temperature=args.temperature,
        label_smoothing=args.label_smoothing,
    )

    logger.info(f"Trainer created with loss: {type(trainer.criterion).__name__}")
    if args.use_calibration:
        temp = trainer.get_temperature()
        logger.info(f"Initial temperature: {temp:.4f}")

    # Train
    logger.info("Starting training...")
    logger.info("-" * 80)

    best_val_loss = float("inf")
    for epoch in range(args.max_epochs):
        # Train epoch
        train_metrics = trainer.train_epoch(epoch)
        logger.info(
            f"Epoch {epoch+1}/{args.max_epochs} - "
            f"Train Loss: {train_metrics['loss']:.4f}, "
            f"Train Acc: {train_metrics['accuracy']:.4f}"
        )

        # Validation
        if (epoch + 1) % args.eval_every_n_epochs == 0:
            val_metrics = trainer.validate(epoch)
            logger.info(
                f"Epoch {epoch+1}/{args.max_epochs} - "
                f"Val Loss: {val_metrics['loss']:.4f}, "
                f"Val Acc: {val_metrics['accuracy']:.4f}"
            )

            # Save best model
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                checkpoint_path = (
                    Path(args.checkpoint_dir) / f"resnet50_best_epoch{epoch+1}.pt"
                )
                checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "val_loss": val_metrics["loss"],
                        "val_accuracy": val_metrics["accuracy"],
                    },
                    checkpoint_path,
                )
                logger.info(f"Saved best model to {checkpoint_path}")

            # Early stopping
            if args.early_stopping:
                # Simple early stopping based on validation loss
                if (
                    epoch > args.early_stopping_patience
                    and val_metrics["loss"] > best_val_loss
                ):
                    logger.info(
                        f"Early stopping triggered at epoch {epoch+1} "
                        f"(patience={args.early_stopping_patience})"
                    )
                    break

        # Log loss statistics (from Phase 3.2 BaseLoss)
        loss_stats = trainer.get_loss_statistics()
        if loss_stats:
            logger.info(f"Loss statistics: {loss_stats}")

    logger.info("-" * 80)
    logger.info("Training complete!")
    logger.info(f"Best validation loss: {best_val_loss:.4f}")

    # Test evaluation
    logger.info("Evaluating on test set...")
    test_metrics = trainer.validate(epoch=-1)  # Use test_loader
    logger.info(
        f"Test Loss: {test_metrics['loss']:.4f}, "
        f"Test Acc: {test_metrics['accuracy']:.4f}"
    )

    logger.info("=" * 80)


if __name__ == "__main__":
    main()
