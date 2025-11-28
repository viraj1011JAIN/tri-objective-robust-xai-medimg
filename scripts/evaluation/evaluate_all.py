#!/usr/bin/env python
"""
Master Evaluation Script for Tri-Objective Robust XAI Medical Imaging.

This script provides comprehensive evaluation of all trained models:
- Loads all models from checkpoints
- Evaluates on test sets
- Computes classification metrics
- Computes calibration metrics
- Runs statistical comparisons
- Generates Pareto analysis
- Creates evaluation reports

Phase 9.1: Comprehensive Evaluation Infrastructure
Author: Viraj Jain
MSc Dissertation - University of Glasgow
Date: November 2024

Usage
-----
    python scripts/evaluation/evaluate_all.py --config configs/evaluation.yaml
    python scripts/evaluation/evaluate_all.py --checkpoint_dir checkpoints/ --output_dir results/evaluation/
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.evaluation.calibration import (
    calculate_ece,
    calculate_mce,
    evaluate_calibration,
    plot_reliability_diagram,
)
from src.evaluation.metrics import (
    compute_bootstrap_ci,
    compute_classification_metrics,
    compute_confusion_matrix,
    compute_per_class_metrics,
    compute_pr_curve,
    compute_roc_curve,
    plot_confusion_matrix,
    plot_roc_curves,
)
from src.evaluation.pareto_analysis import (
    analyze_tradeoffs,
    compute_hypervolume_2d,
    compute_pareto_frontier,
    find_knee_points,
    plot_pareto_2d,
    save_frontier,
)
from src.evaluation.statistical_tests import comprehensive_model_comparison
from src.evaluation.statistical_tests import save_results as save_statistical_results

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# CONFIGURATION
# ============================================================================


DEFAULT_CONFIG = {
    "checkpoint_dir": "checkpoints",
    "output_dir": "results/evaluation",
    "data_dir": "data",
    "batch_size": 32,
    "num_workers": 4,
    "device": "cuda" if torch.cuda.is_available() else "cpu",
    # Model patterns to evaluate
    "model_patterns": [
        "baseline/*.pt",
        "tri_objective/*.pt",
        "hpo/*.pt",
    ],
    # Datasets to evaluate on
    "test_datasets": ["cifar10"],
    # Metrics configuration
    "metrics": {
        "classification": True,
        "calibration": True,
        "bootstrap_ci": True,
        "bootstrap_n": 1000,
        "confidence_level": 0.95,
    },
    # Statistical tests
    "statistical_tests": {
        "enabled": True,
        "alpha": 0.05,
        "correction_method": "bonferroni",
    },
    # Pareto analysis
    "pareto_analysis": {
        "enabled": True,
        "objectives": ["accuracy", "robustness", "interpretability"],
        "minimize": [False, False, False],  # Maximize all
    },
    # Visualization
    "plots": {
        "confusion_matrix": True,
        "roc_curves": True,
        "reliability_diagrams": True,
        "pareto_frontier": True,
    },
}


def load_config(config_path: Optional[str] = None) -> Dict[str, Any]:
    """Load configuration from file or use defaults."""
    config = DEFAULT_CONFIG.copy()

    if config_path and Path(config_path).exists():
        with open(config_path, "r") as f:
            user_config = yaml.safe_load(f)

        # Deep merge
        for key, value in user_config.items():
            if isinstance(value, dict) and key in config:
                config[key].update(value)
            else:
                config[key] = value

    return config


# ============================================================================
# MODEL LOADING
# ============================================================================


def find_checkpoints(checkpoint_dir: Path, patterns: List[str]) -> List[Path]:
    """Find all checkpoint files matching patterns."""
    checkpoints = []

    for pattern in patterns:
        found = list(checkpoint_dir.glob(pattern))
        checkpoints.extend(found)

    # Remove duplicates and sort
    checkpoints = sorted(set(checkpoints))

    logger.info(f"Found {len(checkpoints)} checkpoints")
    for cp in checkpoints:
        logger.info(f"  - {cp.relative_to(checkpoint_dir)}")

    return checkpoints


def load_checkpoint(checkpoint_path: Path, device: str = "cuda") -> Dict[str, Any]:
    """Load a checkpoint file."""
    try:
        checkpoint = torch.load(
            checkpoint_path, map_location=device, weights_only=False
        )
        return checkpoint
    except Exception as e:
        logger.error(f"Failed to load checkpoint {checkpoint_path}: {e}")
        return None


def create_model_from_checkpoint(checkpoint: Dict[str, Any], device: str = "cuda"):
    """Create model from checkpoint state dict."""
    # This is a placeholder - actual implementation depends on your model architecture
    # Import your model classes here
    try:
        from src.models.tri_objective_network import TriObjectiveNetwork

        config = checkpoint.get("config", checkpoint.get("model_config", {}))
        model = TriObjectiveNetwork(config)

        state_dict = checkpoint.get(
            "model_state_dict", checkpoint.get("state_dict", {})
        )
        if state_dict:
            model.load_state_dict(state_dict)

        model = model.to(device)
        model.eval()

        return model
    except Exception as e:
        logger.error(f"Failed to create model: {e}")
        return None


# ============================================================================
# DATA LOADING
# ============================================================================


def get_test_dataloader(
    dataset_name: str, data_dir: Path, batch_size: int = 32, num_workers: int = 4
) -> DataLoader:
    """Get test dataloader for a dataset."""
    try:
        from src.data.datasets import get_dataset

        dataset = get_dataset(
            name=dataset_name,
            root=data_dir,
            train=False,
            download=True,
        )

        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        return dataloader
    except ImportError:
        # Fallback: use torchvision directly
        import torchvision
        import torchvision.transforms as transforms

        transform = transforms.Compose(
            [
                transforms.ToTensor(),
                transforms.Normalize(
                    (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)
                ),
            ]
        )

        if dataset_name.lower() == "cifar10":
            dataset = torchvision.datasets.CIFAR10(
                root=str(data_dir),
                train=False,
                download=True,
                transform=transform,
            )
        elif dataset_name.lower() == "cifar100":
            dataset = torchvision.datasets.CIFAR100(
                root=str(data_dir),
                train=False,
                download=True,
                transform=transform,
            )
        else:
            logger.error(f"Unknown dataset: {dataset_name}")
            return None

        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )


# ============================================================================
# EVALUATION FUNCTIONS
# ============================================================================


@torch.no_grad()
def evaluate_model(
    model, dataloader: DataLoader, device: str = "cuda", config: Dict[str, Any] = None
) -> Dict[str, Any]:
    """
    Evaluate a single model.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    dataloader : DataLoader
        Test data loader.
    device : str
        Device to use.
    config : dict
        Evaluation configuration.

    Returns
    -------
    dict
        Evaluation results.
    """
    if config is None:
        config = DEFAULT_CONFIG

    model.eval()

    all_probs = []
    all_preds = []
    all_labels = []
    all_logits = []

    logger.info("Running inference...")

    for batch in dataloader:
        if isinstance(batch, (list, tuple)):
            images, labels = batch[0], batch[1]
        else:
            images = batch["image"]
            labels = batch["label"]

        images = images.to(device)
        labels = labels.to(device)

        outputs = model(images)

        # Handle different output formats
        if isinstance(outputs, dict):
            logits = outputs.get("logits", outputs.get("output"))
        elif isinstance(outputs, tuple):
            logits = outputs[0]
        else:
            logits = outputs

        probs = torch.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)

        all_logits.append(logits.cpu().numpy())
        all_probs.append(probs.cpu().numpy())
        all_preds.append(preds.cpu().numpy())
        all_labels.append(labels.cpu().numpy())

    # Concatenate results
    all_logits = np.concatenate(all_logits, axis=0)
    all_probs = np.concatenate(all_probs, axis=0)
    all_preds = np.concatenate(all_preds, axis=0)
    all_labels = np.concatenate(all_labels, axis=0)

    results = {
        "n_samples": len(all_labels),
        "predictions": all_preds,
        "probabilities": all_probs,
        "labels": all_labels,
    }

    # Classification metrics
    if config.get("metrics", {}).get("classification", True):
        logger.info("Computing classification metrics...")

        pred_tensor = torch.tensor(all_preds)
        label_tensor = torch.tensor(all_labels)

        classification_metrics = compute_classification_metrics(
            predictions=pred_tensor,
            labels=label_tensor,
            num_classes=all_probs.shape[1],
        )
        results["classification"] = classification_metrics

        # Per-class metrics
        per_class = compute_per_class_metrics(
            predictions=pred_tensor,
            labels=label_tensor,
            num_classes=all_probs.shape[1],
        )
        results["per_class"] = per_class

        # Confusion matrix
        cm = compute_confusion_matrix(
            predictions=pred_tensor,
            labels=label_tensor,
            num_classes=all_probs.shape[1],
        )
        results["confusion_matrix"] = cm.numpy()

    # Calibration metrics
    if config.get("metrics", {}).get("calibration", True):
        logger.info("Computing calibration metrics...")

        confidences = all_probs.max(axis=1)
        correct = (all_preds == all_labels).astype(float)

        ece = calculate_ece(
            confidences=torch.tensor(confidences),
            accuracies=torch.tensor(correct),
            n_bins=15,
        )
        mce = calculate_mce(
            confidences=torch.tensor(confidences),
            accuracies=torch.tensor(correct),
            n_bins=15,
        )

        results["calibration"] = {
            "ece": float(ece),
            "mce": float(mce),
        }

    # Bootstrap confidence intervals
    if config.get("metrics", {}).get("bootstrap_ci", True):
        logger.info("Computing bootstrap confidence intervals...")

        metrics_fn = lambda p, l: float((p == l).mean())

        ci = compute_bootstrap_ci(
            predictions=torch.tensor(all_preds),
            labels=torch.tensor(all_labels),
            metric_fn=metrics_fn,
            n_bootstrap=config.get("metrics", {}).get("bootstrap_n", 1000),
            confidence_level=config.get("metrics", {}).get("confidence_level", 0.95),
        )

        results["bootstrap_ci"] = {
            "accuracy": {
                "mean": float(ci["mean"]),
                "lower": float(ci["lower"]),
                "upper": float(ci["upper"]),
            }
        }

    return results


def evaluate_robustness(
    model, dataloader: DataLoader, device: str = "cuda", epsilon: float = 0.03
) -> Dict[str, Any]:
    """
    Evaluate model robustness against adversarial attacks.

    Parameters
    ----------
    model : nn.Module
        Model to evaluate.
    dataloader : DataLoader
        Test data loader.
    device : str
        Device to use.
    epsilon : float
        Attack strength.

    Returns
    -------
    dict
        Robustness evaluation results.
    """
    try:
        from src.attacks.pgd import PGDAttack

        attack = PGDAttack(
            model=model,
            epsilon=epsilon,
            alpha=epsilon / 4,
            num_steps=20,
            random_start=True,
        )

        model.eval()

        clean_correct = 0
        robust_correct = 0
        total = 0

        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                images, labels = batch[0], batch[1]
            else:
                images = batch["image"]
                labels = batch["label"]

            images = images.to(device)
            labels = labels.to(device)

            # Clean accuracy
            with torch.no_grad():
                clean_outputs = model(images)
                if isinstance(clean_outputs, dict):
                    clean_logits = clean_outputs.get(
                        "logits", clean_outputs.get("output")
                    )
                else:
                    clean_logits = clean_outputs
                clean_preds = clean_logits.argmax(dim=1)
                clean_correct += (clean_preds == labels).sum().item()

            # Adversarial accuracy
            adv_images = attack(images, labels)
            with torch.no_grad():
                adv_outputs = model(adv_images)
                if isinstance(adv_outputs, dict):
                    adv_logits = adv_outputs.get("logits", adv_outputs.get("output"))
                else:
                    adv_logits = adv_outputs
                adv_preds = adv_logits.argmax(dim=1)
                robust_correct += (adv_preds == labels).sum().item()

            total += len(labels)

        return {
            "clean_accuracy": clean_correct / total,
            "robust_accuracy": robust_correct / total,
            "epsilon": epsilon,
        }
    except ImportError:
        logger.warning("Robustness evaluation skipped: attack module not available")
        return {"error": "Attack module not available"}


# ============================================================================
# REPORTING
# ============================================================================


def generate_evaluation_report(
    all_results: Dict[str, Dict[str, Any]], output_dir: Path
) -> str:
    """Generate comprehensive evaluation report."""

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    lines = [
        "=" * 80,
        "COMPREHENSIVE MODEL EVALUATION REPORT",
        f"Generated: {timestamp}",
        "=" * 80,
        "",
    ]

    # Summary table
    lines.append("## Summary")
    lines.append("")
    lines.append("| Model | Accuracy | ECE | F1 (macro) |")
    lines.append("|-------|----------|-----|------------|")

    for model_name, results in all_results.items():
        acc = results.get("classification", {}).get("accuracy", "N/A")
        ece = results.get("calibration", {}).get("ece", "N/A")
        f1 = results.get("classification", {}).get("f1_macro", "N/A")

        if isinstance(acc, float):
            acc = f"{acc:.4f}"
        if isinstance(ece, float):
            ece = f"{ece:.4f}"
        if isinstance(f1, float):
            f1 = f"{f1:.4f}"

        lines.append(f"| {model_name} | {acc} | {ece} | {f1} |")

    lines.append("")

    # Detailed results for each model
    lines.append("## Detailed Results")
    lines.append("")

    for model_name, results in all_results.items():
        lines.append(f"### {model_name}")
        lines.append("")

        # Classification metrics
        if "classification" in results:
            lines.append("**Classification Metrics:**")
            for metric, value in results["classification"].items():
                if isinstance(value, float):
                    lines.append(f"- {metric}: {value:.4f}")
                else:
                    lines.append(f"- {metric}: {value}")
            lines.append("")

        # Calibration metrics
        if "calibration" in results:
            lines.append("**Calibration Metrics:**")
            for metric, value in results["calibration"].items():
                if isinstance(value, float):
                    lines.append(f"- {metric}: {value:.4f}")
            lines.append("")

        # Bootstrap CI
        if "bootstrap_ci" in results:
            lines.append("**Confidence Intervals (95%):**")
            for metric, ci in results["bootstrap_ci"].items():
                lines.append(
                    f"- {metric}: {ci['mean']:.4f} [{ci['lower']:.4f}, {ci['upper']:.4f}]"
                )
            lines.append("")

    lines.append("=" * 80)

    report = "\n".join(lines)

    # Save report
    report_path = output_dir / "evaluation_report.md"
    with open(report_path, "w") as f:
        f.write(report)

    logger.info(f"Report saved to {report_path}")

    return report


def save_all_results(all_results: Dict[str, Dict[str, Any]], output_dir: Path) -> None:
    """Save all results to JSON files."""

    # Convert numpy arrays to lists for JSON serialization
    def convert_to_serializable(obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, dict):
            return {k: convert_to_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [convert_to_serializable(item) for item in obj]
        return obj

    # Save individual model results
    for model_name, results in all_results.items():
        # Remove large arrays for summary
        summary = {
            k: v
            for k, v in results.items()
            if k not in ["predictions", "probabilities", "labels", "confusion_matrix"]
        }

        model_dir = output_dir / "models" / model_name
        model_dir.mkdir(parents=True, exist_ok=True)

        with open(model_dir / "results.json", "w") as f:
            json.dump(convert_to_serializable(summary), f, indent=2)

    # Save combined summary
    summary = {}
    for model_name, results in all_results.items():
        summary[model_name] = {
            "accuracy": results.get("classification", {}).get("accuracy"),
            "f1_macro": results.get("classification", {}).get("f1_macro"),
            "ece": results.get("calibration", {}).get("ece"),
            "mce": results.get("calibration", {}).get("mce"),
        }

    with open(output_dir / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Results saved to {output_dir}")


# ============================================================================
# MAIN PIPELINE
# ============================================================================


def run_full_evaluation(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run full evaluation pipeline.

    Parameters
    ----------
    config : dict
        Evaluation configuration.

    Returns
    -------
    dict
        All evaluation results.
    """
    # Setup paths
    checkpoint_dir = Path(config["checkpoint_dir"])
    output_dir = Path(config["output_dir"])
    data_dir = Path(config["data_dir"])

    output_dir.mkdir(parents=True, exist_ok=True)

    device = config["device"]
    logger.info(f"Using device: {device}")

    # Find checkpoints
    checkpoints = find_checkpoints(checkpoint_dir, config["model_patterns"])

    if not checkpoints:
        logger.error("No checkpoints found!")
        return {}

    # Get test data
    all_dataloaders = {}
    for dataset_name in config["test_datasets"]:
        dataloader = get_test_dataloader(
            dataset_name=dataset_name,
            data_dir=data_dir,
            batch_size=config["batch_size"],
            num_workers=config["num_workers"],
        )
        if dataloader:
            all_dataloaders[dataset_name] = dataloader

    if not all_dataloaders:
        logger.error("No datasets available!")
        return {}

    # Evaluate each model
    all_results = {}

    for checkpoint_path in checkpoints:
        model_name = checkpoint_path.stem
        logger.info(f"\n{'='*60}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"{'='*60}")

        # Load checkpoint
        checkpoint = load_checkpoint(checkpoint_path, device)
        if checkpoint is None:
            continue

        # Create model
        model = create_model_from_checkpoint(checkpoint, device)
        if model is None:
            # Try to evaluate using stored predictions if available
            if "test_results" in checkpoint:
                all_results[model_name] = checkpoint["test_results"]
                continue
            else:
                logger.warning(f"Skipping {model_name}: Could not create model")
                continue

        # Evaluate on each dataset
        model_results = {}

        for dataset_name, dataloader in all_dataloaders.items():
            logger.info(f"Evaluating on {dataset_name}...")

            results = evaluate_model(
                model=model,
                dataloader=dataloader,
                device=device,
                config=config,
            )

            model_results[dataset_name] = results

        # Use first dataset for primary results
        primary_dataset = list(model_results.keys())[0]
        all_results[model_name] = model_results[primary_dataset]
        all_results[model_name]["all_datasets"] = model_results

        # Clean up
        del model
        torch.cuda.empty_cache() if torch.cuda.is_available() else None

    # Statistical comparisons
    if (
        config.get("statistical_tests", {}).get("enabled", True)
        and len(all_results) >= 2
    ):
        logger.info("\nRunning statistical comparisons...")

        # Prepare data for comparison
        model_predictions = {
            name: results.get("predictions", [])
            for name, results in all_results.items()
            if "predictions" in results
        }
        model_labels = {
            name: results.get("labels", [])
            for name, results in all_results.items()
            if "labels" in results
        }

        if len(model_predictions) >= 2:
            # Get first model's labels as reference
            ref_labels = list(model_labels.values())[0]

            comparison = comprehensive_model_comparison(
                model_predictions=model_predictions,
                ground_truth=ref_labels,
                alpha=config["statistical_tests"]["alpha"],
            )

            # Save statistical results
            stats_dir = output_dir / "statistical_tests"
            save_statistical_results(comparison, stats_dir)

    # Pareto analysis
    if config.get("pareto_analysis", {}).get("enabled", True) and len(all_results) >= 2:
        logger.info("\nRunning Pareto analysis...")

        objectives_config = config["pareto_analysis"]
        objective_names = objectives_config.get(
            "objectives", ["accuracy", "robustness", "interpretability"]
        )

        # Collect objective values
        objectives_list = []
        model_names = []

        for model_name, results in all_results.items():
            obj_values = []

            for obj_name in objective_names:
                if obj_name == "accuracy":
                    value = results.get("classification", {}).get("accuracy", 0.5)
                elif obj_name == "robustness":
                    value = results.get("robustness", {}).get(
                        "robust_accuracy",
                        results.get("classification", {}).get("accuracy", 0.5),
                    )
                elif obj_name == "interpretability":
                    value = results.get("interpretability", {}).get(
                        "score", 1.0 - results.get("calibration", {}).get("ece", 0.1)
                    )
                else:
                    value = results.get(obj_name, 0.5)

                obj_values.append(value)

            objectives_list.append(obj_values)
            model_names.append(model_name)

        if len(objectives_list) >= 2:
            objectives = np.array(objectives_list)

            # Compute Pareto frontier
            frontier = compute_pareto_frontier(
                objectives=objectives,
                minimize=objectives_config.get("minimize", [False, False, False]),
                objective_names=objective_names,
                metadata_list=[{"name": name} for name in model_names],
            )

            # Find knee points
            find_knee_points(frontier, method="distance")

            # Compute hypervolume (for 2D)
            if len(objective_names) == 2:
                compute_hypervolume_2d(frontier)

            # Analyze trade-offs
            tradeoff_analysis = analyze_tradeoffs(frontier)

            # Save Pareto results
            pareto_dir = output_dir / "pareto_analysis"
            pareto_dir.mkdir(parents=True, exist_ok=True)

            save_frontier(frontier, pareto_dir / "frontier.json")

            with open(pareto_dir / "tradeoffs.json", "w") as f:
                json.dump(tradeoff_analysis, f, indent=2)

            # Plot Pareto frontier
            if (
                config.get("plots", {}).get("pareto_frontier", True)
                and len(objective_names) == 2
            ):
                fig = plot_pareto_2d(
                    frontier=frontier,
                    all_solutions=objectives,
                    title="Model Comparison: Pareto Frontier",
                    xlabel=objective_names[0],
                    ylabel=objective_names[1],
                    save_path=str(pareto_dir / "pareto_frontier.png"),
                )
                plt.close(fig)

            logger.info(f"Pareto frontier: {len(frontier)} solutions")
            logger.info(frontier.summary())

    # Generate plots
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    for model_name, results in all_results.items():
        model_plots_dir = plots_dir / model_name
        model_plots_dir.mkdir(parents=True, exist_ok=True)

        # Confusion matrix
        if (
            config.get("plots", {}).get("confusion_matrix", True)
            and "confusion_matrix" in results
        ):
            try:
                fig = plot_confusion_matrix(
                    torch.tensor(results["confusion_matrix"]),
                    title=f"Confusion Matrix: {model_name}",
                )
                fig.savefig(model_plots_dir / "confusion_matrix.png", dpi=300)
                plt.close(fig)
            except Exception as e:
                logger.warning(f"Failed to plot confusion matrix for {model_name}: {e}")

        # Reliability diagram
        if config.get("plots", {}).get("reliability_diagrams", True):
            try:
                probs = results.get("probabilities")
                labels = results.get("labels")
                preds = results.get("predictions")

                if probs is not None and labels is not None and preds is not None:
                    confidences = probs.max(axis=1)
                    correct = (preds == labels).astype(float)

                    fig = plot_reliability_diagram(
                        confidences=torch.tensor(confidences),
                        accuracies=torch.tensor(correct),
                        n_bins=15,
                        title=f"Reliability Diagram: {model_name}",
                    )
                    fig.savefig(model_plots_dir / "reliability_diagram.png", dpi=300)
                    plt.close(fig)
            except Exception as e:
                logger.warning(
                    f"Failed to plot reliability diagram for {model_name}: {e}"
                )

    # Save all results
    save_all_results(all_results, output_dir)

    # Generate report
    generate_evaluation_report(all_results, output_dir)

    logger.info(f"\nEvaluation complete! Results saved to {output_dir}")

    return all_results


# ============================================================================
# CLI
# ============================================================================


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive evaluation of trained models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    python evaluate_all.py --config configs/evaluation.yaml
    python evaluate_all.py --checkpoint_dir checkpoints/ --output_dir results/
    python evaluate_all.py --device cuda --batch_size 64
        """,
    )

    parser.add_argument(
        "--config",
        "-c",
        type=str,
        default=None,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="checkpoints",
        help="Directory containing model checkpoints",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/evaluation",
        help="Directory to save evaluation results",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data",
        help="Directory containing test data",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use for evaluation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for evaluation",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Load configuration
    config = load_config(args.config)

    # Override with CLI arguments
    if args.checkpoint_dir:
        config["checkpoint_dir"] = args.checkpoint_dir
    if args.output_dir:
        config["output_dir"] = args.output_dir
    if args.data_dir:
        config["data_dir"] = args.data_dir
    if args.device:
        config["device"] = args.device
    if args.batch_size:
        config["batch_size"] = args.batch_size

    # Run evaluation
    start_time = time.time()

    try:
        results = run_full_evaluation(config)

        elapsed = time.time() - start_time
        logger.info(f"Evaluation completed in {elapsed:.1f} seconds")

        return 0 if results else 1

    except Exception as e:
        logger.exception(f"Evaluation failed: {e}")
        return 1


if __name__ == "__main__":
    import matplotlib

    matplotlib.use("Agg")  # Non-interactive backend
    import matplotlib.pyplot as plt

    sys.exit(main())
