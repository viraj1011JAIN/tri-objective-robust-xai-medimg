"""
Calibration Evaluation Script for Phase 3.3 Baseline Training.

This script evaluates model calibration on trained checkpoints:
- Loads trained model checkpoint
- Runs inference on validation/test set
- Calculates ECE and MCE
- Generates reliability diagrams and confidence histograms

Usage:
    # Evaluate ResNet-50 checkpoint
    python scripts/evaluate_calibration.py \\
        --checkpoint results/checkpoints/resnet50_phase3/resnet50_best_e50.pt \\
        --model resnet50 \\
        --dataset isic2018

    # Compare calibrated vs uncalibrated model
    python scripts/evaluate_calibration.py \\
        --checkpoint results/checkpoints/resnet50_phase3/resnet50_calibrated.pt \\
        --model resnet50 \\
        --dataset isic2018 \\
        --output-dir results/calibration/resnet50_calibrated

Phase 3.3 Baseline Training Integration - Master's Dissertation
Author: Viraj Jain
Date: November 2024
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

# Add project root to path
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import torch
from torch.utils.data import DataLoader

from src.data.datasets import Derm7ptDataset, ISIC2018Dataset
from src.evaluation.calibration import evaluate_calibration
from src.models.build import build_classifier
from src.utils.logging_utils import setup_logging

logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate model calibration",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model & checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["resnet50", "efficientnet_b0", "vit_b_16"],
        help="Model architecture",
    )
    parser.add_argument(
        "--num-classes",
        type=int,
        default=7,
        help="Number of output classes",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        default="isic2018",
        choices=["isic2018", "derm7pt"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default="/content/drive/MyDrive/data",
        help="Root directory for datasets",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Dataset split to evaluate",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
    )

    # Calibration settings
    parser.add_argument(
        "--num-bins",
        type=int,
        default=15,
        help="Number of bins for calibration metrics",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/calibration",
        help="Directory to save calibration plots and metrics",
    )

    # System
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run evaluation on",
    )

    return parser.parse_args()


def load_model_checkpoint(
    checkpoint_path: str,
    model_name: str,
    num_classes: int,
    device: str,
) -> torch.nn.Module:
    """
    Load model from checkpoint.

    Args:
        checkpoint_path: Path to checkpoint file
        model_name: Model architecture name
        num_classes: Number of output classes
        device: Device to load model on

    Returns:
        Loaded model in eval mode
    """
    logger.info(f"Loading model from {checkpoint_path}")

    # Build model
    model = build_classifier(
        model_name=model_name,
        num_classes=num_classes,
        pretrained=False,  # Will load from checkpoint
    )

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Extract state dict (handle different checkpoint formats)
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    elif "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
    else:
        state_dict = checkpoint

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    logger.info(f"Model loaded: {model.__class__.__name__}")

    # Log checkpoint info
    if "epoch" in checkpoint:
        logger.info(f"Checkpoint epoch: {checkpoint['epoch']}")
    if "val_loss" in checkpoint:
        logger.info(f"Checkpoint val_loss: {checkpoint['val_loss']:.4f}")
    if "val_accuracy" in checkpoint:
        logger.info(f"Checkpoint val_accuracy: {checkpoint['val_accuracy']:.4f}")

    return model


def load_dataset(
    dataset_name: str,
    data_root: str,
    split: str,
    batch_size: int,
    num_workers: int,
) -> tuple[DataLoader, int]:
    """
    Load dataset for calibration evaluation.

    Args:
        dataset_name: Name of dataset
        data_root: Root directory for data
        split: Dataset split (train/val/test)
        batch_size: Batch size
        num_workers: Number of workers

    Returns:
        data_loader: DataLoader for evaluation
        num_classes: Number of classes
    """
    logger.info(f"Loading {dataset_name} ({split} split)")

    if dataset_name == "isic2018":
        data_dir = Path(data_root) / "isic_2018"
        dataset = ISIC2018Dataset(root=str(data_dir), split=split)
        num_classes = 7
    elif dataset_name == "derm7pt":
        data_dir = Path(data_root) / "derm7pt"
        dataset = Derm7ptDataset(root=str(data_dir), split=split)
        num_classes = 2
    else:
        raise ValueError(f"Unknown dataset: {dataset_name}")

    data_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    logger.info(f"Dataset loaded: {len(dataset)} samples")

    return data_loader, num_classes


@torch.no_grad()
def collect_predictions(
    model: torch.nn.Module,
    data_loader: DataLoader,
    device: str,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Collect predictions and labels from model.

    Args:
        model: Model to evaluate
        data_loader: DataLoader for evaluation
        device: Device to run on

    Returns:
        predictions: Predicted probabilities (N, C)
        labels: True labels (N,)
    """
    logger.info("Collecting predictions...")

    model.eval()
    all_predictions = []
    all_labels = []

    for batch_idx, batch in enumerate(data_loader):
        images = batch["image"].to(device)
        labels = batch["label"].to(device)

        # Forward pass
        outputs = model(images)

        # Apply softmax to get probabilities
        probs = torch.softmax(outputs, dim=1)

        all_predictions.append(probs.cpu())
        all_labels.append(labels.cpu())

        if (batch_idx + 1) % 10 == 0:
            logger.info(f"Processed {batch_idx + 1}/{len(data_loader)} batches")

    # Concatenate
    predictions = torch.cat(all_predictions, dim=0)
    labels = torch.cat(all_labels, dim=0)

    logger.info(f"Collected {len(labels)} predictions")

    return predictions, labels


def main() -> None:
    """Main evaluation function."""
    args = parse_args()

    # Setup logging
    setup_logging()
    logger.info("=" * 80)
    logger.info("Model Calibration Evaluation - Phase 3.3")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset} ({args.split} split)")
    logger.info(f"Device: {args.device}")

    # Load model
    model = load_model_checkpoint(
        checkpoint_path=args.checkpoint,
        model_name=args.model,
        num_classes=args.num_classes,
        device=args.device,
    )

    # Load dataset
    data_loader, num_classes = load_dataset(
        dataset_name=args.dataset,
        data_root=args.data_root,
        split=args.split,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
    )

    # Collect predictions
    predictions, labels = collect_predictions(
        model=model,
        data_loader=data_loader,
        device=args.device,
    )

    # Evaluate calibration
    logger.info("-" * 80)
    logger.info("Evaluating calibration metrics...")

    metrics = evaluate_calibration(
        predictions=predictions,
        labels=labels,
        num_bins=args.num_bins,
        output_dir=args.output_dir,
    )

    # Save metrics to file
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    metrics_file = output_dir / "calibration_metrics.txt"
    with open(metrics_file, "w") as f:
        f.write("=" * 80 + "\n")
        f.write("Calibration Evaluation Results\n")
        f.write("=" * 80 + "\n")
        f.write(f"Checkpoint: {args.checkpoint}\n")
        f.write(f"Model: {args.model}\n")
        f.write(f"Dataset: {args.dataset} ({args.split} split)\n")
        f.write(f"Number of samples: {len(labels)}\n")
        f.write("-" * 80 + "\n")
        f.write(f"Expected Calibration Error (ECE): {metrics['ece']:.6f}\n")
        f.write(f"Maximum Calibration Error (MCE): {metrics['mce']:.6f}\n")
        f.write(f"Accuracy: {metrics['accuracy']:.6f}\n")
        f.write(f"Average Confidence: {metrics['avg_confidence']:.6f}\n")
        f.write("=" * 80 + "\n")

    logger.info(f"Metrics saved to {metrics_file}")

    logger.info("-" * 80)
    logger.info("Calibration evaluation complete!")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
