"""
Baseline evaluation script for NIH ChestX-ray14 multi-label classification.

Evaluates trained baseline models on NIH ChestX-ray14 test set with:
- Multi-label classification metrics (macro/micro AUROC, per-disease AUROC)
- Hamming loss, subset accuracy
- Per-class precision, recall, F1
- Multi-label calibration metrics (ECE, MCE, Brier score)
- Bootstrap 95% confidence intervals
- Reliability diagrams and confusion matrices

Usage:
    python scripts/evaluation/evaluate_baseline_cxr.py \
        --checkpoint results/checkpoints/rq1_robustness/baseline_nih_resnet50/best.pt \
        --model resnet50 \
        --dataset nih_chestxray14 \
        --split test \
        --n-bootstrap 1000 \
        --output-dir results/evaluation/baseline_nih
"""

from __future__ import annotations

import argparse
import json

# Import dataset and model builders
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.append(str(Path(__file__).parent.parent.parent))

from src.datasets.chest_xray import ChestXRayDataset
from src.datasets.transforms import get_transforms
from src.evaluation.multilabel_calibration import (
    compute_multilabel_calibration_metrics,
    plot_multilabel_confidence_histogram,
    plot_multilabel_reliability_diagram,
)
from src.evaluation.multilabel_metrics import (
    compute_bootstrap_ci_multilabel,
    compute_multilabel_auroc,
    compute_multilabel_confusion_matrix,
    compute_multilabel_metrics,
    compute_optimal_thresholds,
    plot_multilabel_auroc_per_class,
    plot_multilabel_roc_curves,
    plot_per_class_confusion_matrices,
)
from src.models.build import build_model

# Dataset configurations for NIH ChestX-ray14 and PadChest
DATASET_CONFIGS = {
    "nih_chestxray14": {
        "data_root": "/content/drive/MyDrive/data/NIH_ChestXray14",
        "csv_path": "/content/drive/MyDrive/data/NIH_ChestXray14/metadata.csv",
        "num_classes": 14,
        "class_names": [
            "Atelectasis",
            "Cardiomegaly",
            "Effusion",
            "Infiltration",
            "Mass",
            "Nodule",
            "Pneumonia",
            "Pneumothorax",
            "Consolidation",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Pleural_Thickening",
            "Hernia",
        ],
        "dataset_class": "ChestXRayDataset",
        "task_type": "multi_label",
    },
    "padchest": {
        "data_root": "/content/drive/MyDrive/data/PadChest",
        "csv_path": "/content/drive/MyDrive/data/PadChest/metadata.csv",
        "num_classes": 14,
        "class_names": [
            "Atelectasis",
            "Cardiomegaly",
            "Effusion",
            "Infiltration",
            "Mass",
            "Nodule",
            "Pneumonia",
            "Pneumothorax",
            "Consolidation",
            "Edema",
            "Emphysema",
            "Fibrosis",
            "Pleural_Thickening",
            "Hernia",
        ],
        "dataset_class": "ChestXRayDataset",
        "task_type": "multi_label",
        # Label harmonization for cross-site evaluation
        "label_harmonization": {
            "atelectasis": "Atelectasis",
            "cardiomegaly": "Cardiomegaly",
            "effusion": "Effusion",
            "infiltration": "Infiltration",
            "mass": "Mass",
            "nodule": "Nodule",
            "pneumonia": "Pneumonia",
            "pneumothorax": "Pneumothorax",
            "consolidation": "Consolidation",
            "edema": "Edema",
            "emphysema": "Emphysema",
            "fibrosis": "Fibrosis",
            "pleural_thickening": "Pleural_Thickening",
            "hernia": "Hernia",
        },
    },
}


def load_checkpoint(
    checkpoint_path: str, model: nn.Module, device: torch.device
) -> Tuple[nn.Module, Dict]:
    """Load model checkpoint and return model + metadata."""
    print(f"Loading checkpoint: {checkpoint_path}")

    checkpoint = torch.load(checkpoint_path, map_location=device)

    # Load model weights
    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model.to(device)
    model.eval()

    print(f"✓ Checkpoint loaded successfully")
    if "epoch" in checkpoint:
        print(f"  Epoch: {checkpoint['epoch']}")
    if "best_metric" in checkpoint:
        print(f"  Best metric: {checkpoint['best_metric']:.4f}")

    return model, checkpoint


def create_dataloader(
    dataset_name: str,
    split: str = "test",
    batch_size: int = 32,
    num_workers: int = 4,
) -> Tuple[DataLoader, int, List[str]]:
    """Create evaluation dataloader for chest X-ray datasets."""
    if dataset_name not in DATASET_CONFIGS:
        raise ValueError(
            f"Unknown dataset: {dataset_name}. "
            f"Available: {list(DATASET_CONFIGS.keys())}"
        )

    config = DATASET_CONFIGS[dataset_name]
    data_root = Path(config["data_root"])
    csv_path = Path(config["csv_path"])

    # Check if data exists
    if not data_root.exists():
        raise FileNotFoundError(
            f"Dataset not found: {data_root}\n"
            f"Please ensure the dataset is available at this location.\n"
            f"This is expected if the external HDD (/content/drive/MyDrive/data) is not connected."
        )

    # Get evaluation transforms (no augmentation)
    transforms = get_transforms(split="val", image_size=224)

    # Create dataset
    dataset_kwargs = {
        "root": data_root,
        "split": split,
        "transforms": transforms,
        "csv_path": csv_path,
        "image_path_column": "image_path",
        "labels_column": "labels",
        "split_column": "split",
        "label_separator": "|",
    }

    # Add label harmonization for PadChest
    if "label_harmonization" in config:
        dataset_kwargs["label_harmonization"] = config["label_harmonization"]

    dataset = ChestXRayDataset(**dataset_kwargs)

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
    )

    num_classes = config["num_classes"]
    class_names = config["class_names"]

    print(f"✓ Created dataloader for {dataset_name} ({split} split)")
    print(f"  Total samples: {len(dataset)}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num classes: {num_classes}")

    return dataloader, num_classes, class_names


@torch.no_grad()
def evaluate_model(
    model: nn.Module, dataloader: DataLoader, device: torch.device
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Run model inference and collect predictions.

    Returns
    -------
    y_true : np.ndarray
        Ground truth binary labels, shape [N, C]
    y_pred : np.ndarray
        Predicted binary labels (threshold=0.5), shape [N, C]
    y_prob : np.ndarray
        Predicted probabilities (after sigmoid), shape [N, C]
    """
    model.eval()

    all_labels = []
    all_probs = []

    print("Running inference...")
    for batch in tqdm(dataloader, desc="Evaluating"):
        images = batch["image"].to(device)
        labels = batch["label"]  # Multi-hot labels [B, C]

        # Forward pass
        logits = model(images)

        # Apply sigmoid for multi-label probabilities
        probs = torch.sigmoid(logits)

        # Collect results
        all_labels.append(labels.cpu().numpy())
        all_probs.append(probs.cpu().numpy())

    # Concatenate all batches
    y_true = np.vstack(all_labels)  # [N, C]
    y_prob = np.vstack(all_probs)  # [N, C]

    # Apply default threshold for binary predictions
    y_pred = (y_prob >= 0.5).astype(int)

    print(f"✓ Inference complete")
    print(f"  Samples: {y_true.shape[0]}")
    print(f"  Classes: {y_true.shape[1]}")
    print(f"  Label distribution: {y_true.sum(axis=0).tolist()}")

    return y_true, y_pred, y_prob


def compute_all_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
    class_names: List[str],
    n_bootstrap: int = 1000,
    threshold: float = 0.5,
) -> Dict[str, Any]:
    """Compute all classification and calibration metrics with bootstrap CI."""
    print("\nComputing metrics...")

    # 1. Multi-label classification metrics
    metrics = compute_multilabel_metrics(y_true, y_pred, y_prob, class_names, threshold)

    # 2. Multi-label calibration metrics
    calibration_metrics = compute_multilabel_calibration_metrics(
        y_true, y_prob, n_bins=15
    )
    metrics.update(calibration_metrics)

    # 3. Optimal per-class thresholds
    optimal_thresholds = compute_optimal_thresholds(y_true, y_prob, metric="f1")
    metrics["optimal_thresholds"] = optimal_thresholds.tolist()

    # 4. Bootstrap confidence intervals
    print("Computing bootstrap confidence intervals...")

    # Helper functions for bootstrap
    def auroc_macro_fn(y_t, y_p):
        return compute_multilabel_auroc(y_t, y_p)["auroc_macro"]

    def auroc_micro_fn(y_t, y_p):
        return compute_multilabel_auroc(y_t, y_p)["auroc_micro"]

    def hamming_fn(y_t, y_p):
        from sklearn.metrics import hamming_loss

        y_pred_boot = (y_p >= threshold).astype(int)
        return hamming_loss(y_t, y_pred_boot)

    # Compute CIs
    auroc_macro_val, auroc_macro_lower, auroc_macro_upper = (
        compute_bootstrap_ci_multilabel(
            y_true, y_prob, auroc_macro_fn, n_bootstrap=n_bootstrap
        )
    )

    auroc_micro_val, auroc_micro_lower, auroc_micro_upper = (
        compute_bootstrap_ci_multilabel(
            y_true, y_prob, auroc_micro_fn, n_bootstrap=n_bootstrap
        )
    )

    hamming_val, hamming_lower, hamming_upper = compute_bootstrap_ci_multilabel(
        y_true, y_prob, hamming_fn, n_bootstrap=n_bootstrap
    )

    # Add CIs to metrics
    metrics["auroc_macro_ci_lower"] = auroc_macro_lower
    metrics["auroc_macro_ci_upper"] = auroc_macro_upper
    metrics["auroc_micro_ci_lower"] = auroc_micro_lower
    metrics["auroc_micro_ci_upper"] = auroc_micro_upper
    metrics["hamming_loss_ci_lower"] = hamming_lower
    metrics["hamming_loss_ci_upper"] = hamming_upper

    print("✓ Metrics computed successfully")

    return metrics


def save_results(metrics: Dict[str, Any], output_dir: Path, dataset_name: str) -> None:
    """Save evaluation results to JSON and CSV files."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Save full results as JSON
    results_file = output_dir / "results.json"
    with open(results_file, "w") as f:
        json.dump(metrics, f, indent=2)
    print(f"✓ Saved full results: {results_file}")

    # 2. Save summary metrics as CSV
    summary_metrics = {
        "dataset": dataset_name,
        "auroc_macro": metrics["auroc_macro"],
        "auroc_macro_ci_lower": metrics.get("auroc_macro_ci_lower", np.nan),
        "auroc_macro_ci_upper": metrics.get("auroc_macro_ci_upper", np.nan),
        "auroc_micro": metrics["auroc_micro"],
        "auroc_micro_ci_lower": metrics.get("auroc_micro_ci_lower", np.nan),
        "auroc_micro_ci_upper": metrics.get("auroc_micro_ci_upper", np.nan),
        "auroc_weighted": metrics["auroc_weighted"],
        "hamming_loss": metrics["hamming_loss"],
        "hamming_loss_ci_lower": metrics.get("hamming_loss_ci_lower", np.nan),
        "hamming_loss_ci_upper": metrics.get("hamming_loss_ci_upper", np.nan),
        "subset_accuracy": metrics["subset_accuracy"],
        "precision_macro": metrics["precision_macro"],
        "recall_macro": metrics["recall_macro"],
        "f1_macro": metrics["f1_macro"],
        "ece_macro": metrics["ece_macro"],
        "mce_macro": metrics["mce_macro"],
        "brier_score_macro": metrics["brier_score_macro"],
    }

    summary_file = output_dir / "summary_metrics.csv"
    pd.DataFrame([summary_metrics]).to_csv(summary_file, index=False)
    print(f"✓ Saved summary metrics: {summary_file}")

    # 3. Save per-class metrics as CSV
    if "per_class_metrics" in metrics:
        per_class_data = []
        for class_name, class_metrics in metrics["per_class_metrics"].items():
            per_class_data.append(
                {
                    "class_name": class_name,
                    "auroc": class_metrics["auroc"],
                    "precision": class_metrics["precision"],
                    "recall": class_metrics["recall"],
                    "f1": class_metrics["f1"],
                    "support": class_metrics["support"],
                }
            )

        per_class_file = output_dir / "per_class_metrics.csv"
        pd.DataFrame(per_class_data).to_csv(per_class_file, index=False)
        print(f"✓ Saved per-class metrics: {per_class_file}")


def generate_plots(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
    num_classes: int,
    class_names: List[str],
    metrics: Dict[str, Any],
    output_dir: Path,
) -> None:
    """Generate all evaluation plots."""
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    print("\nGenerating plots...")

    # 1. Per-class AUROC bar chart
    auroc_scores = np.array(metrics["auroc_per_class"])
    plot_multilabel_auroc_per_class(
        auroc_scores,
        class_names,
        save_path=str(plots_dir / "auroc_per_class.png"),
        title="Per-Class AUROC - Multi-Label Classification",
    )
    print("✓ Saved per-class AUROC plot")

    # 2. ROC curves for all classes
    plot_multilabel_roc_curves(
        y_true,
        y_prob,
        class_names,
        save_path=str(plots_dir / "roc_curves.png"),
        title="Multi-Label ROC Curves",
    )
    print("✓ Saved ROC curves")

    # 3. Per-class confusion matrices
    plot_per_class_confusion_matrices(
        y_true,
        y_pred,
        class_names,
        save_path=str(plots_dir / "confusion_matrices.png"),
        title="Per-Class Confusion Matrices",
    )
    print("✓ Saved confusion matrices")

    # 4. Reliability diagrams
    plot_multilabel_reliability_diagram(
        y_true,
        y_prob,
        class_names,
        n_bins=15,
        save_path=str(plots_dir / "reliability_diagrams.png"),
        title="Multi-Label Reliability Diagrams",
    )
    print("✓ Saved reliability diagrams")

    # 5. Confidence histograms
    plot_multilabel_confidence_histogram(
        y_prob,
        class_names,
        save_path=str(plots_dir / "confidence_histograms.png"),
        title="Multi-Label Confidence Distribution",
    )
    print("✓ Saved confidence histograms")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline model on chest X-ray datasets"
    )
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
        choices=["resnet50", "efficientnet_b4", "vit_base_patch16_224"],
        help="Model architecture",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="nih_chestxray14",
        choices=["nih_chestxray14", "padchest"],
        help="Dataset to evaluate on",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="test",
        choices=["train", "val", "test"],
        help="Data split to evaluate",
    )
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=1000,
        help="Number of bootstrap samples for CI",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
        help="Threshold for binary predictions",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/evaluation/baseline_cxr",
        help="Output directory for results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu)",
    )
    parser.add_argument(
        "--num-workers",
        type=int,
        default=4,
        help="Number of DataLoader workers",
    )

    args = parser.parse_args()

    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")

    # Create dataloader
    dataloader, num_classes, class_names = create_dataloader(
        args.dataset, args.split, args.batch_size, args.num_workers
    )

    # Build model
    print(f"\nBuilding model: {args.model}")
    model = build_model(args.model, num_classes=num_classes, pretrained=False)

    # Load checkpoint
    model, checkpoint = load_checkpoint(args.checkpoint, model, device)

    # Evaluate model
    y_true, y_pred, y_prob = evaluate_model(model, dataloader, device)

    # Compute metrics
    metrics = compute_all_metrics(
        y_true,
        y_pred,
        y_prob,
        num_classes,
        class_names,
        args.n_bootstrap,
        args.threshold,
    )

    # Save results
    output_dir = Path(args.output_dir)
    save_results(metrics, output_dir, args.dataset)

    # Generate plots
    generate_plots(
        y_true, y_pred, y_prob, num_classes, class_names, metrics, output_dir
    )

    print(f"\n{'=' * 60}")
    print("Evaluation Summary")
    print(f"{'=' * 60}")
    print(f"Dataset: {args.dataset}")
    print(f"Split: {args.split}")
    print(f"Samples: {y_true.shape[0]}")
    print(
        f"\nMacro AUROC: {metrics['auroc_macro']:.4f} "
        f"[{metrics.get('auroc_macro_ci_lower', 0):.4f}, "
        f"{metrics.get('auroc_macro_ci_upper', 1):.4f}]"
    )
    print(
        f"Micro AUROC: {metrics['auroc_micro']:.4f} "
        f"[{metrics.get('auroc_micro_ci_lower', 0):.4f}, "
        f"{metrics.get('auroc_micro_ci_upper', 1):.4f}]"
    )
    print(
        f"Hamming Loss: {metrics['hamming_loss']:.4f} "
        f"[{metrics.get('hamming_loss_ci_lower', 0):.4f}, "
        f"{metrics.get('hamming_loss_ci_upper', 1):.4f}]"
    )
    print(f"Subset Accuracy: {metrics['subset_accuracy']:.4f}")
    print(f"ECE (macro): {metrics['ece_macro']:.4f}")
    print(f"MCE (macro): {metrics['mce_macro']:.4f}")
    print(f"Brier Score: {metrics['brier_score_macro']:.4f}")
    print(f"\nResults saved to: {output_dir}")
    print(f"{'=' * 60}\n")


if __name__ == "__main__":
    main()
