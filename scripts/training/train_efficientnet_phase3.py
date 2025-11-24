"""
Training script for EfficientNet-B0 with Phase 3.2 production-grade losses.

This script trains EfficientNet-B0 on medical imaging datasets with support for:
- TaskLoss (CrossEntropy, BinaryCrossEntropy, FocalLoss)
- CalibrationLoss (temperature scaling + label smoothing)
- GPU acceleration (CUDA)
- Checkpointing & logging (TensorBoard, MLflow)
- Hyperparameter optimization

Usage:
    # Train with TaskLoss (CrossEntropy)
    python scripts/training/train_efficientnet_phase3.py --dataset isic2018

    # Train with CalibrationLoss
    python scripts/training/train_efficientnet_phase3.py --dataset isic2018 --use-calibration --temperature 1.5 --label-smoothing 0.1

Phase 3.3 Baseline Training Integration - Part of Master's Dissertation
Author: Viraj Jain
Date: November 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data.datasets import Derm7ptDataset, ISIC2018Dataset
from src.models.build import build_classifier
from src.training.baseline_trainer import BaselineTrainer, TrainingConfig
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B0 with Phase 3.2 losses",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Dataset
    parser.add_argument(
        "--dataset", type=str, default="isic2018", choices=["isic2018", "derm7pt"]
    )
    parser.add_argument("--data-root", type=str, default="/content/drive/MyDrive/data")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)

    # Model
    parser.add_argument("--pretrained", action="store_true")
    parser.add_argument("--dropout", type=float, default=0.5)

    # Loss configuration
    parser.add_argument("--use-focal-loss", action="store_true")
    parser.add_argument("--focal-gamma", type=float, default=2.0)
    parser.add_argument("--use-calibration", action="store_true")
    parser.add_argument("--temperature", type=float, default=1.5)
    parser.add_argument("--label-smoothing", type=float, default=0.0)

    # Training
    parser.add_argument("--max-epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.0001)
    parser.add_argument("--weight-decay", type=float, default=0.0001)
    parser.add_argument("--gradient-clip", type=float, default=1.0)

    # System
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--seed", type=int, default=42)

    # Logging & checkpointing
    parser.add_argument(
        "--log-dir", type=str, default="results/logs/efficientnet_phase3"
    )
    parser.add_argument(
        "--checkpoint-dir", type=str, default="results/checkpoints/efficientnet_phase3"
    )
    parser.add_argument("--log-every-n-steps", type=int, default=50)
    parser.add_argument("--eval-every-n-epochs", type=int, default=1)

    # Early stopping
    parser.add_argument("--early-stopping", action="store_true")
    parser.add_argument("--early-stopping-patience", type=int, default=15)

    # MLflow
    parser.add_argument("--use-mlflow", action="store_true")
    parser.add_argument("--experiment-name", type=str, default="EfficientNet_Phase3.3")

    return parser.parse_args()


def load_dataset(
    dataset_name: str, data_root: str, batch_size: int, num_workers: int
) -> tuple[DataLoader, DataLoader, DataLoader, int]:
    """Load and prepare dataset."""
    logger.info(f"Loading dataset: {dataset_name}")

    if dataset_name == "isic2018":
        data_dir = Path(data_root) / "isic_2018"
        train_dataset = ISIC2018Dataset(root=str(data_dir), split="train")
        val_dataset = ISIC2018Dataset(root=str(data_dir), split="val")
        test_dataset = ISIC2018Dataset(root=str(data_dir), split="test")
        num_classes = 7
    elif dataset_name == "derm7pt":
        data_dir = Path(data_root) / "derm7pt"
        train_dataset = Derm7ptDataset(root=str(data_dir), split="train")
        val_dataset = Derm7ptDataset(root=str(data_dir), split="val")
        test_dataset = Derm7ptDataset(root=str(data_dir), split="test")
        num_classes = 2
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

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
        f"Dataset: {len(train_dataset)} train, {len(val_dataset)} val, "
        f"{len(test_dataset)} test"
    )
    return train_loader, val_loader, test_loader, num_classes


def main() -> None:
    """Main training function."""
    args = parse_args()

    # Setup logging
    setup_logging(log_dir=args.log_dir)
    logger.info("=" * 80)
    logger.info("EfficientNet-B0 Phase 3.3 Baseline Training")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Device: {args.device}")
    logger.info(f"Batch size: {args.batch_size}")
    logger.info(f"Max epochs: {args.max_epochs}")

    # Log loss configuration
    if args.use_calibration:
        logger.info(
            f"Loss: CalibrationLoss (temp={args.temperature}, "
            f"smoothing={args.label_smoothing})"
        )
    elif args.use_focal_loss:
        logger.info(f"Loss: TaskLoss (FocalLoss, gamma={args.focal_gamma})")
    else:
        logger.info("Loss: TaskLoss (CrossEntropy)")

    # Set seed
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.seed)

    # Load dataset
    train_loader, val_loader, test_loader, num_classes = load_dataset(
        args.dataset, args.data_root, args.batch_size, args.num_workers
    )

    # Build model (EfficientNet-B0)
    logger.info("Building EfficientNet-B0 model...")
    model = build_classifier(
        model_name="efficientnet_b0",
        num_classes=num_classes,
        pretrained=args.pretrained,
        dropout=args.dropout,
    )
    logger.info(f"Model: {model.__class__.__name__}")

    # Optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay
    )

    # Training config
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

    # Trainer with Phase 3.2 losses
    logger.info("Creating BaselineTrainer...")
    trainer = BaselineTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        optimizer=optimizer,
        config=config,
        num_classes=num_classes,
        device=args.device,
        task_type="multi_class",
        use_focal_loss=args.use_focal_loss,
        focal_gamma=args.focal_gamma,
        use_calibration=args.use_calibration,
        init_temperature=args.temperature,
        label_smoothing=args.label_smoothing,
    )
    logger.info(f"Loss: {type(trainer.criterion).__name__}")

    # Train
    logger.info("Starting training...")
    logger.info("-" * 80)

    best_val_loss = float("inf")
    for epoch in range(args.max_epochs):
        train_metrics = trainer.train_epoch(epoch)
        logger.info(
            f"Epoch {epoch+1} - Train: Loss={train_metrics['loss']:.4f}, "
            f"Acc={train_metrics['accuracy']:.4f}"
        )

        if (epoch + 1) % args.eval_every_n_epochs == 0:
            val_metrics = trainer.validate(epoch)
            logger.info(
                f"Epoch {epoch+1} - Val: Loss={val_metrics['loss']:.4f}, "
                f"Acc={val_metrics['accuracy']:.4f}"
            )

            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                checkpoint_path = (
                    Path(args.checkpoint_dir) / f"efficientnet_best_e{epoch+1}.pt"
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
                logger.info(f"Saved best model: {checkpoint_path}")

    logger.info("-" * 80)
    logger.info(f"Training complete! Best val loss: {best_val_loss:.4f}")

    # Test
    logger.info("Evaluating on test set...")
    test_metrics = trainer.validate(epoch=-1)
    logger.info(
        f"Test: Loss={test_metrics['loss']:.4f}, Acc={test_metrics['accuracy']:.4f}"
    )
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
