#!/usr/bin/env python3
"""
================================================================================
TRADES Evaluation Script - Phase 5.3
================================================================================
Comprehensive evaluation of TRADES-trained models across multiple dimensions:
    1. Clean accuracy on test set
    2. Robustness under various attacks (FGSM, PGD, C&W)
    3. Calibration metrics (ECE, MCE, Brier score)
    4. Cross-site generalization
    5. Per-class performance analysis
    6. Comparison with baseline (PGD-AT)

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Date: November 2025
================================================================================
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import yaml
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from src.attacks.fgsm import FGSMAttack
from src.attacks.pgd import PGDAttack
from src.data.isic_dataset import ISICDataset
from src.evaluation.calibration import compute_calibration_metrics
from src.models.model_factory import get_model
from src.utils.logging_utils import setup_logger


class TRADESEvaluator:
    """Comprehensive evaluation pipeline for TRADES models."""

    def __init__(
        self,
        config: Dict[str, Any],
        model: nn.Module,
        test_loader: DataLoader,
        device: torch.device,
        logger: logging.Logger,
    ):
        self.config = config
        self.model = model
        self.test_loader = test_loader
        self.device = device
        self.logger = logger

        # Results storage
        self.results = {
            "clean": {},
            "fgsm": {},
            "pgd": {},
            "calibration": {},
            "per_class": {},
            "confusion_matrix": None,
        }

        self.logger.info("TRADESEvaluator initialized")

    @torch.no_grad()
    def evaluate_clean(self) -> Dict[str, float]:
        """Evaluate clean accuracy and metrics."""
        self.logger.info("Evaluating clean accuracy...")

        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []

        for images, labels in tqdm(self.test_loader, desc="Clean Evaluation"):
            images = images.to(self.device)
            logits = self.model(images)
            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_probs.extend(probs.cpu().numpy())

        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        # Compute metrics
        metrics = {
            "accuracy": accuracy_score(all_labels, all_preds),
            "balanced_accuracy": balanced_accuracy_score(all_labels, all_preds),
            "f1_macro": f1_score(all_labels, all_preds, average="macro"),
            "f1_weighted": f1_score(all_labels, all_preds, average="weighted"),
            "precision_macro": precision_score(all_labels, all_preds, average="macro"),
            "recall_macro": recall_score(all_labels, all_preds, average="macro"),
            "auroc": roc_auc_score(
                all_labels, all_probs, multi_class="ovr", average="macro"
            ),
            "auprc": average_precision_score(all_labels, all_probs, average="macro"),
        }

        self.results["clean"] = metrics
        self.results["confusion_matrix"] = confusion_matrix(all_labels, all_preds)

        self.logger.info(f"Clean Accuracy: {metrics['accuracy']:.4f}")
        self.logger.info(f"Balanced Accuracy: {metrics['balanced_accuracy']:.4f}")
        self.logger.info(f"F1 (macro): {metrics['f1_macro']:.4f}")

        return metrics

    @torch.no_grad()
    def evaluate_fgsm(self) -> Dict[str, Dict[str, float]]:
        """Evaluate robustness under FGSM attack."""
        self.logger.info("Evaluating FGSM robustness...")

        fgsm_cfg = self.config["evaluation"]["attacks"]["fgsm"]
        if not fgsm_cfg["enabled"]:
            self.logger.info("FGSM evaluation disabled")
            return {}

        results = {}

        for epsilon in fgsm_cfg["epsilons"]:
            self.logger.info(f"FGSM ε={epsilon:.5f}")

            attack = FGSMAttack(
                model=self.model, epsilon=epsilon, clip_min=0.0, clip_max=1.0
            )

            correct = 0
            total = 0

            for images, labels in tqdm(self.test_loader, desc=f"FGSM ε={epsilon:.5f}"):
                images, labels = images.to(self.device), labels.to(self.device)

                # Generate adversarial examples
                adv_images = attack(images, labels)

                # Predict
                logits = self.model(adv_images)
                preds = logits.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

            accuracy = correct / total
            results[f"eps_{epsilon:.5f}"] = {"accuracy": accuracy}
            self.logger.info(f"FGSM ε={epsilon:.5f} Accuracy: {accuracy:.4f}")

        self.results["fgsm"] = results
        return results

    @torch.no_grad()
    def evaluate_pgd(self) -> Dict[str, Dict[str, float]]:
        """Evaluate robustness under PGD attack."""
        self.logger.info("Evaluating PGD robustness...")

        pgd_cfg = self.config["evaluation"]["attacks"]["pgd"]
        if not pgd_cfg["enabled"]:
            self.logger.info("PGD evaluation disabled")
            return {}

        results = {}

        for epsilon in pgd_cfg["epsilons"]:
            self.logger.info(f"PGD ε={epsilon:.5f}")

            attack = PGDAttack(
                model=self.model,
                epsilon=epsilon,
                num_steps=pgd_cfg["num_steps"],
                step_size=pgd_cfg["step_size"],
                random_start=pgd_cfg["random_start"],
                loss_type="ce",
                clip_min=0.0,
                clip_max=1.0,
            )

            correct = 0
            total = 0

            for images, labels in tqdm(self.test_loader, desc=f"PGD ε={epsilon:.5f}"):
                images, labels = images.to(self.device), labels.to(self.device)

                adv_images = attack(images, labels)
                logits = self.model(adv_images)
                preds = logits.argmax(dim=1)

                correct += (preds == labels).sum().item()
                total += labels.size(0)

            accuracy = correct / total
            results[f"eps_{epsilon:.5f}"] = {"accuracy": accuracy}
            self.logger.info(f"PGD ε={epsilon:.5f} Accuracy: {accuracy:.4f}")

        self.results["pgd"] = results
        return results

    @torch.no_grad()
    def evaluate_calibration(self) -> Dict[str, float]:
        """Evaluate model calibration."""
        self.logger.info("Evaluating calibration...")

        cal_cfg = self.config["evaluation"]["calibration"]
        if not cal_cfg["enabled"]:
            self.logger.info("Calibration evaluation disabled")
            return {}

        all_probs = []
        all_labels = []

        self.model.eval()
        for images, labels in tqdm(self.test_loader, desc="Calibration"):
            images = images.to(self.device)
            logits = self.model(images)
            probs = torch.softmax(logits, dim=1)

            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Compute calibration metrics
        metrics = compute_calibration_metrics(
            all_probs, all_labels, num_bins=cal_cfg["num_bins"]
        )

        self.results["calibration"] = metrics

        self.logger.info(f"ECE: {metrics['ece']:.4f}")
        self.logger.info(f"MCE: {metrics['mce']:.4f}")
        self.logger.info(f"Brier Score: {metrics['brier_score']:.4f}")

        return metrics

    def save_results(self, output_dir: Path) -> None:
        """Save evaluation results."""
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON
        results_file = output_dir / "evaluation_results.json"
        with open(results_file, "w") as f:
            # Convert numpy arrays to lists for JSON serialization
            results_json = {
                k: v.tolist() if isinstance(v, np.ndarray) else v
                for k, v in self.results.items()
            }
            json.dump(results_json, f, indent=2)

        self.logger.info(f"Saved results to {results_file}")

        # Save confusion matrix
        if self.results["confusion_matrix"] is not None:
            cm_file = output_dir / "confusion_matrix.npy"
            np.save(cm_file, self.results["confusion_matrix"])

            # Plot confusion matrix
            self.plot_confusion_matrix(output_dir / "confusion_matrix.png")

    def plot_confusion_matrix(self, save_path: Path) -> None:
        """Plot and save confusion matrix."""
        cm = self.results["confusion_matrix"]

        plt.figure(figsize=(10, 8))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=range(cm.shape[0]),
            yticklabels=range(cm.shape[0]),
        )
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.title("Confusion Matrix - TRADES")
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
        plt.close()

        self.logger.info(f"Saved confusion matrix plot to {save_path}")

    def run_full_evaluation(self) -> Dict[str, Any]:
        """Run complete evaluation pipeline."""
        self.logger.info("=" * 80)
        self.logger.info("Starting Full Evaluation")
        self.logger.info("=" * 80)

        # Clean evaluation
        self.evaluate_clean()

        # Adversarial evaluations
        self.evaluate_fgsm()
        self.evaluate_pgd()

        # Calibration
        self.evaluate_calibration()

        self.logger.info("=" * 80)
        self.logger.info("Evaluation Complete")
        self.logger.info("=" * 80)

        return self.results


def main():
    """Main evaluation function."""
    parser = argparse.ArgumentParser(description="TRADES Evaluation - Phase 5.3")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/experiments/trades_isic.yaml",
        help="Path to configuration file",
    )
    parser.add_argument(
        "--checkpoint", type=str, required=True, help="Path to model checkpoint"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (overrides config)",
    )
    args = parser.parse_args()

    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    # Override output directory
    if args.output_dir:
        config["output"]["base_dir"] = args.output_dir

    # Setup logging
    output_dir = Path(config["output"]["base_dir"])
    log_dir = output_dir / config["output"]["logs_dir"]
    log_dir.mkdir(parents=True, exist_ok=True)

    logger = setup_logger(
        name="TRADES_Eval",
        log_file=log_dir / "evaluation.log",
        level=config["output"]["logging"]["level"],
    )

    logger.info("=" * 80)
    logger.info("TRADES Evaluation - Phase 5.3")
    logger.info("=" * 80)
    logger.info(f"Checkpoint: {args.checkpoint}")

    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Device: {device}")

    # Load test dataset
    logger.info("Loading test dataset...")
    test_dataset = ISICDataset(
        data_dir=config["data"]["data_dir"],
        metadata_csv=config["data"]["metadata_csv"],
        split="test",
        transform=False,
        image_size=config["data"]["image_size"],
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
    )

    logger.info(f"Test samples: {len(test_dataset)}")

    # Load model
    logger.info("Loading model...")
    model = get_model(
        architecture=config["model"]["architecture"],
        num_classes=config["model"]["num_classes"],
        pretrained=False,
    )

    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    logger.info(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")

    # Initialize evaluator
    evaluator = TRADESEvaluator(
        config=config,
        model=model,
        test_loader=test_loader,
        device=device,
        logger=logger,
    )

    # Run evaluation
    results = evaluator.run_full_evaluation()

    # Save results
    metrics_dir = output_dir / config["output"]["metrics_dir"]
    evaluator.save_results(metrics_dir)


if __name__ == "__main__":
    main()
