"""
Baseline Evaluation Script for Phase 3.5: Baseline Evaluation - Dermoscopy.

This script evaluates trained baseline models on:
1. ISIC 2018 test set (same-site)
2. ISIC 2019 (cross-site)
3. ISIC 2020 (cross-site)
4. Derm7pt (cross-site)

Computes:
- Classification metrics: Accuracy, AUROC (per-class + macro), F1, MCC
- Calibration metrics: ECE, MCE, Brier score
- Confusion matrix and per-class precision/recall
- Bootstrap 95% confidence intervals
- Reliability diagrams

Usage:
    # Evaluate on ISIC 2018 test set
    python scripts/evaluation/evaluate_baseline.py \\
        --checkpoint results/checkpoints/rq1_robustness/baseline_isic2018_resnet50/best.pt \\
        --model resnet50 \\
        --dataset isic2018 \\
        --split test \\
        --output-dir results/evaluation/baseline_isic2018

    # Cross-site evaluation on ISIC 2019
    python scripts/evaluation/evaluate_baseline.py \\
        --checkpoint results/checkpoints/rq1_robustness/baseline_isic2018_resnet50/best.pt \\
        --model resnet50 \\
        --dataset isic2019 \\
        --split test \\
        --output-dir results/evaluation/baseline_isic2019_cross_site

Phase 3.5: Baseline Evaluation - Dermoscopy
Author: Viraj Jain
Date: November 2024
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.datasets.derm7pt import Derm7ptDataset
from src.datasets.isic import ISICDataset
from src.evaluation.calibration import evaluate_calibration
from src.evaluation.metrics import (
    compute_bootstrap_ci,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_per_class_metrics,
    plot_confusion_matrix,
    plot_roc_curves,
)
from src.models.build import build_model
from src.utils.reproducibility import set_global_seed

logger = logging.getLogger(__name__)


# Dataset configurations
DATASET_CONFIGS = {
    "isic2018": {
        "root": "/content/drive/MyDrive/data/isic_2018",
        "num_classes": 7,
        "class_names": ["MEL", "NV", "BCC", "AKIEC", "BKL", "DF", "VASC"],
    },
    "isic2019": {
        "root": "/content/drive/MyDrive/data/isic_2019",
        "num_classes": 8,
        "class_names": ["MEL", "NV", "BCC", "AK", "BKL", "DF", "VASC", "SCC"],
    },
    "isic2020": {
        "root": "/content/drive/MyDrive/data/isic_2020",
        "num_classes": 2,
        "class_names": ["benign", "malignant"],
    },
    "derm7pt": {
        "root": "/content/drive/MyDrive/data/derm7pt",
        "num_classes": 2,
        "class_names": ["benign", "malignant"],
    },
}


def setup_logging(output_dir: Path) -> None:
    """Configure logging to file and stdout."""
    log_file = output_dir / "evaluation.log"
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(levelname)s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )


def load_checkpoint(
    checkpoint_path: Path,
    model: nn.Module,
    device: torch.device,
) -> Dict[str, Any]:
    """
    Load model checkpoint.

    Parameters
    ----------
    checkpoint_path : Path
        Path to checkpoint file
    model : nn.Module
        Model instance
    device : torch.device
        Device to load checkpoint to

    Returns
    -------
    checkpoint : dict
        Checkpoint dictionary with metadata
    """
    logger.info(f"Loading checkpoint from {checkpoint_path}")

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model state
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.eval()

    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'N/A')}")
    logger.info(f"Best val loss: {checkpoint.get('best_val_loss', 'N/A'):.4f}")

    return checkpoint


def create_dataloader(
    dataset_name: str,
    split: str,
    batch_size: int = 32,
    num_workers: int = 4,
) -> DataLoader:
    """
    Create data loader for evaluation.

    Parameters
    ----------
    dataset_name : str
        Name of dataset (isic2018, isic2019, isic2020, derm7pt)
    split : str
        Data split (train, val, test)
    batch_size : int
        Batch size
    num_workers : int
        Number of data loading workers

    Returns
    -------
    loader : DataLoader
        Data loader
    """
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Supported: {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[dataset_name]
    root = Path(config["root"])

    logger.info(f"Creating dataloader for {dataset_name} ({split} split)")

    # Create dataset
    if dataset_name.startswith("isic"):
        dataset = ISICDataset(
            root=root,
            split=split,
            transform=None,  # Use default transforms
        )
    elif dataset_name == "derm7pt":
        dataset = Derm7ptDataset(
            root=root,
            split=split,
            transform=None,
        )
    else:
        raise NotImplementedError(f"Dataset {dataset_name} not implemented")

    # Create dataloader
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=False,
    )

    logger.info(f"Created dataloader with {len(dataset)} samples")

    return loader


@torch.no_grad()
def evaluate_model(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Dict[str, np.ndarray]:
    """
    Run inference and collect predictions and labels.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate
    dataloader : DataLoader
        Data loader
    device : torch.device
        Device to run inference on

    Returns
    -------
    results : dict
        Dictionary containing:
        - predictions: Predicted probabilities, shape (N, num_classes)
        - labels: Ground truth labels, shape (N,)
    """
    model.eval()

    all_predictions: List[np.ndarray] = []
    all_labels: List[np.ndarray] = []

    logger.info("Running inference...")

    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(device, non_blocking=True)
        labels = batch["label"]

        # Forward pass
        logits = model(images)

        # Apply softmax to get probabilities
        probs = torch.softmax(logits, dim=1)

        # Collect results
        all_predictions.append(probs.cpu().numpy())
        all_labels.append(labels.numpy())

    # Concatenate all batches
    predictions = np.concatenate(all_predictions, axis=0)
    labels = np.concatenate(all_labels, axis=0)

    logger.info(f"Collected predictions: {predictions.shape}")
    logger.info(f"Collected labels: {labels.shape}")

    return {"predictions": predictions, "labels": labels}


def compute_all_metrics(
    predictions: np.ndarray,
    labels: np.ndarray,
    num_classes: int,
    class_names: List[str],
    n_bootstrap: int = 1000,
) -> Dict[str, Any]:
    """
    Compute all evaluation metrics with bootstrap CI.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted probabilities, shape (N, num_classes)
    labels : np.ndarray
        Ground truth labels, shape (N,)
    num_classes : int
        Number of classes
    class_names : list of str
        Names of classes
    n_bootstrap : int
        Number of bootstrap samples for CI

    Returns
    -------
    metrics : dict
        All computed metrics
    """
    logger.info("Computing classification metrics...")

    # 1. Classification metrics
    classification_metrics = compute_classification_metrics(
        predictions, labels, num_classes, class_names
    )

    # 2. Per-class metrics
    per_class_metrics = compute_per_class_metrics(
        predictions, labels, class_names
    )

    # 3. Confusion matrix
    cm = compute_confusion_matrix(predictions, labels)
    cm_normalized = compute_confusion_matrix(
        predictions, labels, normalize="true"
    )

    # 4. Calibration metrics
    logger.info("Computing calibration metrics...")
    calibration_metrics = evaluate_calibration(
        predictions, labels, num_bins=15, output_dir=None
    )

    # 5. Bootstrap confidence intervals
    logger.info(f"Computing bootstrap CI (n={n_bootstrap})...")
    ci_metrics = compute_bootstrap_ci(
        predictions,
        labels,
        num_classes,
        metric_fn=compute_classification_metrics,
        n_bootstrap=n_bootstrap,
        confidence_level=0.95,
    )

    # Combine all metrics
    all_metrics = {
        "classification": classification_metrics,
        "per_class": per_class_metrics,
        "confusion_matrix": cm.tolist(),
        "confusion_matrix_normalized": cm_normalized.tolist(),
        "calibration": calibration_metrics,
        "bootstrap_ci": ci_metrics,
    }

    return all_metrics


def save_results(
    metrics: Dict[str, Any],
    output_dir: Path,
    dataset_name: str,
    split: str,
    model_name: str,
) -> None:
    """
    Save evaluation results to JSON and CSV.

    Parameters
    ----------
    metrics : dict
        Computed metrics
    output_dir : Path
        Output directory
    dataset_name : str
        Dataset name
    split : str
        Data split
    model_name : str
        Model name
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save full results as JSON
    json_path = output_dir / "evaluation_results.json"
    logger.info(f"Saving results to {json_path}")

    with open(json_path, "w") as f:
        json.dump(metrics, f, indent=2)

    # Create summary CSV
    csv_path = output_dir / "evaluation_summary.csv"
    logger.info(f"Saving summary to {csv_path}")

    summary_rows = []

    # Classification metrics
    for key, value in metrics["classification"].items():
        if not key.startswith("auroc_"):  # Skip per-class AUROC for summary
            summary_rows.append({
                "metric": key,
                "value": value,
                "ci_lower": metrics["bootstrap_ci"].get(f"{key}_ci", (np.nan, np.nan))[0],
                "ci_upper": metrics["bootstrap_ci"].get(f"{key}_ci", (np.nan, np.nan))[1],
            })

    # Calibration metrics
    for key, value in metrics["calibration"].items():
        summary_rows.append({
            "metric": key,
            "value": value,
            "ci_lower": np.nan,
            "ci_upper": np.nan,
        })

    df_summary = pd.DataFrame(summary_rows)
    df_summary.to_csv(csv_path, index=False)

    # Per-class metrics CSV
    per_class_path = output_dir / "per_class_metrics.csv"
    df_per_class = pd.DataFrame(metrics["per_class"]).T
    df_per_class.index.name = "class"
    df_per_class.to_csv(per_class_path)

    logger.info("Results saved successfully")


def generate_plots(
    predictions: np.ndarray,
    labels: np.ndarray,
    class_names: List[str],
    output_dir: Path,
    dataset_name: str,
) -> None:
    """
    Generate and save evaluation plots.

    Parameters
    ----------
    predictions : np.ndarray
        Predicted probabilities
    labels : np.ndarray
        Ground truth labels
    class_names : list of str
        Names of classes
    output_dir : Path
        Output directory
    dataset_name : str
        Dataset name
    """
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("Generating plots...")

    # 1. Confusion matrix
    cm = compute_confusion_matrix(predictions, labels)
    fig = plot_confusion_matrix(
        cm,
        class_names,
        title=f"Confusion Matrix - {dataset_name}",
        normalize=False,
        save_path=str(plots_dir / "confusion_matrix.png"),
    )
    plt.close(fig)

    # Normalized confusion matrix
    cm_norm = compute_confusion_matrix(predictions, labels, normalize="true")
    fig = plot_confusion_matrix(
        cm_norm,
        class_names,
        title=f"Normalized Confusion Matrix - {dataset_name}",
        normalize=True,
        save_path=str(plots_dir / "confusion_matrix_normalized.png"),
    )
    plt.close(fig)

    # 2. ROC curves
    if len(class_names) > 2:  # Multi-class
        fig = plot_roc_curves(
            predictions,
            labels,
            class_names,
            title=f"ROC Curves - {dataset_name}",
            save_path=str(plots_dir / "roc_curves.png"),
        )
        plt.close(fig)

    # 3. Calibration plots (reliability diagram)
    from src.evaluation.calibration import plot_reliability_diagram, plot_confidence_histogram

    fig = plot_reliability_diagram(
        predictions,
        labels,
        num_bins=15,
        title=f"Reliability Diagram - {dataset_name}",
        save_path=str(plots_dir / "reliability_diagram.png"),
    )
    plt.close(fig)

    fig = plot_confidence_histogram(
        predictions,
        labels,
        title=f"Confidence Histogram - {dataset_name}",
        save_path=str(plots_dir / "confidence_histogram.png"),
    )
    plt.close(fig)

    logger.info(f"Plots saved to {plots_dir}")


def main(args: argparse.Namespace) -> None:
    """Main evaluation function."""
    # Setup
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    setup_logging(output_dir)

    logger.info("=" * 80)
    logger.info("Phase 3.5: Baseline Evaluation - Dermoscopy")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Split: {args.split}")
    logger.info(f"Output: {output_dir}")
    logger.info("=" * 80)

    # Set seed
    set_global_seed(args.seed)

    # Device
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # Dataset config
    if args.dataset not in DATASET_CONFIGS:
        raise ValueError(f"Unknown dataset: {args.dataset}")

    config = DATASET_CONFIGS[args.dataset]
    num_classes = config["num_classes"]
    class_names = config["class_names"]

    # Build model
    logger.info(f"Building {args.model} model with {num_classes} classes")
    model = build_model(
        args.model,
        num_classes=num_classes,
        pretrained=False,  # Loading from checkpoint
    )
    model = model.to(device)

    # Load checkpoint
    checkpoint = load_checkpoint(Path(args.checkpoint), model, device)

    # Create dataloader
    try:
        dataloader = create_dataloader(
            args.dataset,
            args.split,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
        )
    except FileNotFoundError as e:
        logger.error(f"Dataset not found: {e}")
        logger.error(
            f"Dataset should be at: {config['root']}"
        )
        logger.error(
            "⚠️  Note: Dataset is on external HDD (/content/drive/MyDrive/data) which is not accessible."
        )
        logger.error(
            "This evaluation will run when the dataset becomes available."
        )
        return

    # Evaluate
    results = evaluate_model(model, dataloader, device)
    predictions = results["predictions"]
    labels = results["labels"]

    # Compute metrics
    metrics = compute_all_metrics(
        predictions,
        labels,
        num_classes,
        class_names,
        n_bootstrap=args.n_bootstrap,
    )

    # Add metadata
    metrics["metadata"] = {
        "checkpoint": str(args.checkpoint),
        "model": args.model,
        "dataset": args.dataset,
        "split": args.split,
        "num_samples": len(labels),
        "num_classes": num_classes,
        "class_names": class_names,
        "device": str(device),
    }

    # Save results
    save_results(
        metrics,
        output_dir,
        args.dataset,
        args.split,
        args.model,
    )

    # Generate plots
    generate_plots(
        predictions,
        labels,
        class_names,
        output_dir,
        args.dataset,
    )

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("EVALUATION SUMMARY")
    logger.info("=" * 80)
    logger.info(f"Dataset: {args.dataset} ({args.split} split)")
    logger.info(f"Samples: {len(labels)}")
    logger.info(f"Classes: {num_classes}")
    logger.info("-" * 80)
    logger.info("Classification Metrics:")
    for key, value in metrics["classification"].items():
        if not key.startswith("auroc_"):
            ci = metrics["bootstrap_ci"].get(f"{key}_ci", (np.nan, np.nan))
            logger.info(f"  {key:20s}: {value:.4f} (95% CI: [{ci[0]:.4f}, {ci[1]:.4f}])")
    logger.info("-" * 80)
    logger.info("Calibration Metrics:")
    for key, value in metrics["calibration"].items():
        logger.info(f"  {key:20s}: {value:.4f}")
    logger.info("=" * 80)

    logger.info(f"\n✅ Evaluation complete! Results saved to {output_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate baseline model on dermoscopy datasets"
    )

    # Model and checkpoint
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="resnet50",
        choices=["resnet50", "efficientnet_b0", "vit_b_16"],
        help="Model architecture",
    )

    # Dataset
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["isic2018", "isic2019", "isic2020", "derm7pt"],
        help="Dataset name",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate on",
    )

    # Evaluation settings
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for CI",
    )

    # Output
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Directory to save evaluation results",
    )

    # Misc
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )

    args = parser.parse_args()
    main(args)
