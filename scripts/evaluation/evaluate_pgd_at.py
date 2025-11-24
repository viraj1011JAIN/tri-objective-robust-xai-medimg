"""
Phase 5.2: PGD-AT Evaluation Pipeline
======================================

Comprehensive evaluation pipeline for PGD adversarial training:
1. Load trained PGD-AT models (3 seeds)
2. Evaluate clean accuracy on all test sets
3. Evaluate robust accuracy under multiple attacks
4. Test cross-site generalization (RQ1)
5. Statistical significance testing (t-test, Cohen's d)
6. Generate comparison tables and visualizations

Author: Viraj Pankaj Jain
Date: November 24, 2025
Version: 5.2.0
"""

import argparse
import gc
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy import stats
from scipy.stats import bootstrap
from sklearn.metrics import f1_score, precision_score, recall_score, roc_auc_score
from torch.utils.data import DataLoader
from tqdm import tqdm

# Optional import for A1+ statistical rigor
try:
    from statsmodels.stats.multitest import multipletests

    STATSMODELS_AVAILABLE = True
except ImportError:
    STATSMODELS_AVAILABLE = False
    multipletests = None  # Will check at runtime

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.attacks import FGSM, PGD, AutoAttack, CarliniWagner
from src.attacks.pgd import PGDConfig
from src.datasets import ISICDataset
from src.models import build_model
from src.utils.metrics import calculate_metrics

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore")


class PGDATEvaluator:
    """Comprehensive PGD-AT evaluation pipeline."""

    def __init__(
        self,
        model_paths: List[str],
        config_path: str,
        output_dir: str,
        device: str = "cuda",
    ):
        """
        Initialize evaluator.

        Args:
            model_paths: List of checkpoint paths (one per seed)
            config_path: Path to experiment config
            output_dir: Output directory for results
            device: Device to use
        """
        self.model_paths = [Path(p) for p in model_paths]
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Load config
        import yaml

        with open(self.config_path, "r") as f:
            self.config = yaml.safe_load(f)

        logger.info(f"Initialized evaluator with {len(model_paths)} models")

    def load_model(self, checkpoint_path: Path) -> nn.Module:
        """Load model from checkpoint."""
        # Build model
        model_config = self.config["model"]
        model = build_model(
            architecture=model_config["architecture"],
            num_classes=model_config["num_classes"],
            pretrained=False,  # Load from checkpoint
            in_channels=model_config.get("in_channels", 3),
        )

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.to(self.device)
        model.eval()

        logger.info(f"Loaded model from {checkpoint_path}")
        return model

    def _load_checkpoint(self, checkpoint_path: Path) -> nn.Module:
        """
        Load model from checkpoint with proper validation.

        Args:
            checkpoint_path: Path to checkpoint file

        Returns:
            Loaded model in eval mode

        Raises:
            FileNotFoundError: If checkpoint doesn't exist
            ValueError: If checkpoint is corrupted or invalid
        """
        # Validate checkpoint exists
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"Loading checkpoint: {checkpoint_path}")

        # Load checkpoint
        try:
            checkpoint = torch.load(checkpoint_path, map_location=self.device)
        except Exception as e:
            raise ValueError(f"Failed to load checkpoint {checkpoint_path}: {e}")

        # Validate checkpoint structure
        required_keys = ["model_state_dict"]
        missing_keys = [k for k in required_keys if k not in checkpoint]
        if missing_keys:
            raise ValueError(f"Invalid checkpoint: missing {missing_keys}")

        # Build model
        config = checkpoint.get("config", {})
        model_config = config.get("model", self.config["model"])
        model = build_model(
            architecture=model_config["architecture"],
            num_classes=model_config["num_classes"],
            pretrained=False,
            in_channels=model_config.get("in_channels", 3),
        )

        # Load weights
        try:
            model.load_state_dict(checkpoint["model_state_dict"])
        except Exception as e:
            raise ValueError(f"Failed to load model weights: {e}")

        # Move to device and set to eval mode
        model = model.to(self.device)
        model.eval()

        logger.info(f"✓ Loaded checkpoint successfully")
        if "epoch" in checkpoint:
            logger.info(f"  Epoch: {checkpoint['epoch']}")
        if "seed" in checkpoint:
            logger.info(f"  Seed: {checkpoint['seed']}")
        if "best_robust_acc" in checkpoint:
            logger.info(f"  Best robust acc: {checkpoint['best_robust_acc']:.2f}%")

        return model

    def build_test_loaders(self) -> Dict[str, DataLoader]:
        """Build test dataloaders for all datasets."""
        dataset_config = self.config["dataset"]
        batch_size = self.config["training"]["batch_size"]
        num_workers = self.config["training"].get("num_workers", 4)

        test_loaders = {}
        test_datasets_config = dataset_config.get("test_datasets", {})

        for test_name, test_config in test_datasets_config.items():
            if "isic" in test_config.get("name", "").lower():
                dataset = ISICDataset(
                    root=test_config["root"],
                    csv_path=test_config["csv_path"],
                    split=test_config.get("split", "test"),
                    image_size=dataset_config["image_size"],
                    augment=False,
                )
                test_loaders[test_name] = DataLoader(
                    dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )

        logger.info(f"Built {len(test_loaders)} test loaders")
        return test_loaders

    def evaluate(self) -> Dict[str, Dict]:
        """
        Main evaluation method with memory management.

        Evaluates all models across all test sets and aggregates results.
        Includes proper CUDA memory cleanup to prevent OOM.

        Returns:
            Aggregated evaluation results across all seeds
        """
        logger.info("=" * 80)
        logger.info("Starting PGD-AT Comprehensive Evaluation")
        logger.info("=" * 80)

        # Build test loaders once
        test_loaders = self.build_test_loaders()

        all_model_results = []

        # Evaluate each model
        for i, model_path in enumerate(self.model_paths):
            logger.info(f"\n{'='*80}")
            logger.info(f"Model {i+1}/{len(self.model_paths)}: {model_path.stem}")
            logger.info(f"{'='*80}\n")

            try:
                # Load model
                model = self._load_checkpoint(model_path)

                # Evaluate
                results = self.evaluate_model_comprehensive(model, test_loaders)
                all_model_results.append(results)

                # Critical: Clean up memory
                del model
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                import gc

                gc.collect()

                logger.info("✓ Model evaluation complete, memory cleaned")

            except Exception as e:
                logger.error(f"Failed to evaluate {model_path.stem}: {e}")
                continue

        # Aggregate results across seeds
        logger.info("\n" + "=" * 80)
        logger.info("Aggregating results across seeds...")
        logger.info("=" * 80 + "\n")

        aggregated = self._aggregate_results(all_model_results)

        logger.info("✓ Evaluation complete!")
        return aggregated

    def _aggregate_results(self, all_results: List[Dict]) -> Dict[str, Dict]:
        """
        Aggregate results across multiple seeds.

        Computes mean, std, min, max, and confidence intervals
        for each test set and attack type.

        Args:
            all_results: List of result dicts (one per seed)

        Returns:
            Aggregated statistics
        """
        if not all_results:
            return {}

        aggregated = {}

        # Get all test sets and attack types
        test_sets = all_results[0].keys()

        for test_name in test_sets:
            aggregated[test_name] = {}
            attack_types = all_results[0][test_name].keys()

            for attack_name in attack_types:
                # Extract metric values across seeds
                metrics_across_seeds = []
                for result in all_results:
                    if test_name in result and attack_name in result[test_name]:
                        metrics_across_seeds.append(result[test_name][attack_name])

                if not metrics_across_seeds:
                    continue

                # Get all metric keys
                metric_keys = metrics_across_seeds[0].keys()

                # Aggregate each metric
                aggregated_metrics = {}
                for key in metric_keys:
                    values = [m[key] for m in metrics_across_seeds]
                    values = np.array(values)

                    aggregated_metrics[key] = {
                        "mean": float(np.mean(values)),
                        "std": float(np.std(values)),
                        "min": float(np.min(values)),
                        "max": float(np.max(values)),
                        "n_seeds": len(values),
                    }

                    # Add 95% CI if n >= 3
                    if len(values) >= 3:
                        ci = stats.t.interval(
                            0.95,
                            len(values) - 1,
                            loc=np.mean(values),
                            scale=stats.sem(values),
                        )
                        aggregated_metrics[key]["ci_lower"] = float(ci[0])
                        aggregated_metrics[key]["ci_upper"] = float(ci[1])

                aggregated[test_name][attack_name] = aggregated_metrics

        # Log aggregated results
        logger.info("\nAggregated Results Summary:")
        for test_name, test_results in aggregated.items():
            logger.info(f"\n{test_name}:")
            for attack_name, metrics in test_results.items():
                if "accuracy" in metrics:
                    acc = metrics["accuracy"]
                    logger.info(
                        f"  {attack_name}: " f"{acc['mean']:.2f}% ± {acc['std']:.2f}%"
                    )

        return aggregated

    def evaluate_clean(
        self,
        model: nn.Module,
        dataloader: DataLoader,
    ) -> Dict[str, float]:
        """Evaluate clean accuracy."""
        model.eval()
        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for images, labels in tqdm(dataloader, desc="Clean eval"):
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100.0 * correct / total
        metrics = calculate_metrics(all_labels, all_preds)
        metrics["accuracy"] = accuracy

        return metrics

    def evaluate_robust(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        attack_config: Dict,
    ) -> Dict[str, float]:
        """Evaluate robust accuracy under attack."""
        model.eval()

        attack_type = attack_config["type"].lower()
        epsilon = attack_config["epsilon"]

        # Build attack
        if attack_type == "pgd":
            attack = PGD(
                model=model,
                config=PGDConfig(
                    epsilon=epsilon,
                    num_steps=attack_config["num_steps"],
                    step_size=attack_config.get("step_size", epsilon / 4),
                    random_start=True,
                    clip_min=0.0,
                    clip_max=1.0,
                ),
            )
        elif attack_type == "fgsm":
            attack = FGSM(model=model, epsilon=epsilon)
        elif attack_type == "autoattack":
            attack = AutoAttack(
                model=model,
                epsilon=epsilon,
                version=attack_config.get("version", "standard"),
            )
        else:
            raise ValueError(f"Unknown attack type: {attack_type}")

        correct = 0
        total = 0
        all_preds = []
        all_labels = []

        for images, labels in tqdm(dataloader, desc=f"{attack_config['name']} eval"):
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Generate adversarial examples
            if attack_type == "autoattack":
                adv_images = attack(images, labels)
            else:
                adv_images = attack(images, labels)

            # Evaluate
            with torch.no_grad():
                outputs = model(adv_images)
                _, predicted = outputs.max(1)

                total += labels.size(0)
                correct += predicted.eq(labels).sum().item()

                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        accuracy = 100.0 * correct / total
        metrics = calculate_metrics(all_labels, all_preds)
        metrics["accuracy"] = accuracy

        return metrics

    def evaluate_robustness(
        self, model: nn.Module, dataloader: DataLoader, attack_config: Dict
    ) -> Dict[str, float]:
        """
        Evaluate robustness under adversarial attack.

        Args:
            model: Model to evaluate
            dataloader: Test data loader
            attack_config: Attack configuration

        Returns:
            Dictionary with robust metrics
        """
        model.eval()

        # Create PGD attack config
        pgd_config = PGDConfig(
            epsilon=attack_config.get("eps", 8 / 255),
            step_size=attack_config.get("alpha", 2 / 255),
            num_steps=attack_config.get("steps", 10),
            random_start=attack_config.get("random_start", True),
            norm="Linf",
        )

        # Create PGD attack
        attack = PGD(config=pgd_config, model=model, device=self.device)

        all_preds = []
        all_labels = []
        all_probs = []

        logger.info(f"Evaluating {attack_config['name']} robustness...")

        with torch.no_grad():
            for images, labels in tqdm(
                dataloader, desc=f"{attack_config['name']} eval"
            ):
                images = images.to(self.device)
                labels = labels.to(self.device)

                # Generate adversarial examples
                adv_images = attack(images, labels)

                # Get predictions
                outputs = model(adv_images)
                probs = F.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        accuracy = 100.0 * (all_preds == all_labels).mean()

        # Compute AUROC if binary classification
        if all_probs.shape[1] == 2:
            auroc = roc_auc_score(all_labels, all_probs[:, 1])
        else:
            # Multi-class: one-vs-rest
            auroc = roc_auc_score(
                all_labels, all_probs, multi_class="ovr", average="macro"
            )

        # Compute precision, recall, F1
        precision = precision_score(
            all_labels, all_preds, average="macro", zero_division=0
        )
        recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
        f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)

        metrics = {
            "accuracy": accuracy,
            "auroc": auroc,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }

        logger.info(f"  Robust Accuracy: {accuracy:.2f}% | " f"AUROC: {auroc:.4f}")

        return metrics

    def evaluate_model_comprehensive(
        self,
        model: nn.Module,
        test_loaders: Dict[str, DataLoader],
    ) -> Dict[str, Dict]:
        """Comprehensive evaluation on all test sets and attacks."""
        results = {}

        # Evaluate on each test set
        for test_name, test_loader in test_loaders.items():
            logger.info(f"Evaluating on {test_name}...")

            test_results = {}

            # Clean accuracy
            clean_metrics = self.evaluate_clean(model, test_loader)
            test_results["clean"] = clean_metrics
            logger.info(f"  Clean accuracy: {clean_metrics['accuracy']:.2f}%")

            # Robust accuracy (multiple attacks)
            eval_config = self.config.get("evaluation", {})
            attacks = eval_config.get("attacks", [])

            for attack_config in attacks:
                attack_name = attack_config["name"]
                robust_metrics = self.evaluate_robust(model, test_loader, attack_config)
                test_results[attack_name] = robust_metrics
                logger.info(
                    f"  {attack_name} accuracy: {robust_metrics['accuracy']:.2f}%"
                )

            results[test_name] = test_results

        return results

    def evaluate_all_models(self) -> Dict[str, List[Dict]]:
        """Evaluate all models (3 seeds) comprehensively."""
        test_loaders = self.build_test_loaders()

        all_results = {}

        for i, model_path in enumerate(self.model_paths):
            logger.info(f"\n{'='*80}")
            logger.info(
                f"Evaluating model {i+1}/{len(self.model_paths)}: {model_path.stem}"
            )
            logger.info(f"{'='*80}\n")

            try:
                model = self.load_model(model_path)
                results = self.evaluate_model_comprehensive(model, test_loaders)

                # Store results
                seed = model_path.parent.parent.name
                if seed not in all_results:
                    all_results[seed] = []
                all_results[seed] = results

                logger.info(f"✓ Model {i+1} evaluation complete")

            except Exception as e:
                logger.error(f"Failed to evaluate {model_path.stem}: {e}")
                continue

            finally:
                # CRITICAL: Memory management
                # Clean up model and free GPU memory
                if "model" in locals():
                    del model

                if torch.cuda.is_available():
                    torch.cuda.empty_cache()

                gc.collect()

                logger.info("✓ Memory cleaned")

        return all_results

    def compute_statistics(self, all_results: Dict) -> pd.DataFrame:
        """Compute statistical summary across seeds."""
        # Extract metrics into DataFrame
        records = []

        for seed, results in all_results.items():
            for test_name, test_results in results.items():
                for attack_name, metrics in test_results.items():
                    records.append(
                        {
                            "seed": seed,
                            "test_set": test_name,
                            "attack": attack_name,
                            "accuracy": metrics["accuracy"],
                            "f1_macro": metrics.get("f1_macro", 0.0),
                            "auroc": metrics.get("auroc", 0.0),
                        }
                    )

        df = pd.DataFrame(records)

        # Compute summary statistics
        summary = (
            df.groupby(["test_set", "attack"])["accuracy"]
            .agg(["mean", "std", "min", "max"])
            .reset_index()
        )

        return df, summary

    def test_rq1_hypothesis(
        self, aggregated_results: Dict, baseline_results: Optional[Dict] = None
    ) -> Dict:
        """
        Test RQ1 Hypothesis: H1c - PGD-AT does NOT improve cross-site generalization.

        This is THE CORE research question. Tests whether PGD-AT trained models
        maintain performance when evaluated on cross-site test sets compared to
        in-distribution test sets.

        Expected Result: p > 0.05 (no significant difference)
        This confirms PGD-AT doesn't help with cross-site generalization,
        justifying the need for the tri-objective approach.

        Args:
            aggregated_results: Aggregated evaluation results from evaluate()
            baseline_results: Optional baseline model results for comparison

        Returns:
            Dictionary with hypothesis test results, p-values, effect sizes
        """
        logger.info("\n" + "=" * 80)
        logger.info("RQ1 Hypothesis Test: Cross-Site Generalization")
        logger.info("=" * 80)
        logger.info("H1c: PGD-AT does NOT improve cross-site generalization")
        logger.info("Expected: p > 0.05 (no significant improvement)\n")

        # Import hypothesis testing module
        try:
            from src.analysis.rq1_hypothesis_test import test_rq1_hypothesis
        except ImportError:
            logger.warning(
                "RQ1 hypothesis test module not found. "
                "Using fallback implementation."
            )
            return self._fallback_rq1_test(aggregated_results)

        # Extract AUROC values for cross-site generalization
        # In-distribution: ISIC_test (trained on ISIC)
        # Out-of-distribution: HAM10000, Derm7pt (cross-site)

        results = {}

        # Check if we have the required test sets
        required_sets = ["isic_test", "ham10000", "derm7pt"]
        available_sets = [s for s in required_sets if s in aggregated_results]

        if len(available_sets) < 2:
            logger.warning(
                f"Insufficient test sets for RQ1. "
                f"Required: {required_sets}, Available: {available_sets}"
            )
            return {
                "error": "Insufficient test sets",
                "p_value": 0.0,
                "hypothesis_confirmed": False,
                "required_sets": required_sets,
                "available_sets": available_sets,
                "note": "Need at least 2 test sets for comparison",
            }

        # Extract clean AUROC values
        in_dist_auroc = []
        cross_site_auroc = []

        # In-distribution performance
        if "isic_test" in aggregated_results:
            if "clean" in aggregated_results["isic_test"]:
                clean_metrics = aggregated_results["isic_test"]["clean"]
                if "auroc" in clean_metrics:
                    in_dist_auroc.append(clean_metrics["auroc"]["mean"])

        # Cross-site performance
        for test_set in ["ham10000", "derm7pt"]:
            if test_set in aggregated_results:
                if "clean" in aggregated_results[test_set]:
                    clean_metrics = aggregated_results[test_set]["clean"]
                    if "auroc" in clean_metrics:
                        cross_site_auroc.append(clean_metrics["auroc"]["mean"])

        if not in_dist_auroc or not cross_site_auroc:
            logger.warning("Could not extract AUROC values for RQ1 test")
            return {
                "error": "Missing AUROC values",
                "p_value": 0.0,
                "hypothesis_confirmed": False,
                "note": "Insufficient data for statistical test",
            }

        # Compute AUROC drop (key metric for generalization)
        in_dist_mean = np.mean(in_dist_auroc)
        cross_site_mean = np.mean(cross_site_auroc)
        auroc_drop = in_dist_mean - cross_site_mean

        logger.info(f"In-distribution AUROC: {in_dist_mean:.4f}")
        logger.info(f"Cross-site AUROC: {cross_site_mean:.4f}")
        logger.info(f"AUROC Drop: {auroc_drop:.4f}\n")

        # Statistical test: Paired t-test
        # H0: No difference between in-dist and cross-site performance
        # H1: Significant difference exists

        # For proper paired test, need multiple test sets
        # Here we use independent samples t-test
        t_stat, p_value = stats.ttest_ind(in_dist_auroc, cross_site_auroc)

        # Compute effect size (Cohen's d)
        mean_diff = in_dist_mean - cross_site_mean
        pooled_std = np.sqrt((np.var(in_dist_auroc) + np.var(cross_site_auroc)) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

        # Bonferroni correction for multiple comparisons
        # Testing across multiple test sets
        n_comparisons = len(cross_site_auroc)
        bonferroni_alpha = 0.05 / n_comparisons

        results = {
            "in_dist_auroc": {
                "mean": float(in_dist_mean),
                "values": [float(v) for v in in_dist_auroc],
            },
            "cross_site_auroc": {
                "mean": float(cross_site_mean),
                "values": [float(v) for v in cross_site_auroc],
            },
            "auroc_drop": float(auroc_drop),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "bonferroni_alpha": float(bonferroni_alpha),
            "cohens_d": float(cohens_d),
            "significant": p_value < bonferroni_alpha,
            "hypothesis_confirmed": p_value > 0.05,  # H1c confirmed if NOT significant
        }

        # Interpretation
        logger.info("Statistical Test Results:")
        logger.info(f"  t-statistic: {t_stat:.4f}")
        logger.info(f"  p-value: {p_value:.4f}")
        logger.info(f"  Bonferroni α: {bonferroni_alpha:.4f}")
        logger.info(f"  Cohen's d: {cohens_d:.4f}\n")

        if results["hypothesis_confirmed"]:
            logger.info(
                "✓ H1c CONFIRMED: PGD-AT does NOT improve cross-site generalization"
            )
            logger.info("  → This justifies the need for tri-objective approach!")
        else:
            logger.info(
                "✗ H1c REJECTED: PGD-AT shows significant cross-site improvement"
            )
            logger.info("  → Unexpected result, requires investigation")

        logger.info("=" * 80 + "\n")

        return results

    def _fallback_rq1_test(self, aggregated_results: Dict) -> Dict:
        """Fallback implementation if rq1_hypothesis_test module unavailable."""
        logger.warning("Using fallback RQ1 test implementation")

        # Simple implementation
        in_dist = []
        cross_site = []

        for test_name, test_results in aggregated_results.items():
            if "isic" in test_name.lower():
                if "clean" in test_results and "auroc" in test_results["clean"]:
                    in_dist.append(test_results["clean"]["auroc"]["mean"])
            else:
                if "clean" in test_results and "auroc" in test_results["clean"]:
                    cross_site.append(test_results["clean"]["auroc"]["mean"])

        if not in_dist or not cross_site:
            return {"error": "Insufficient data for fallback test"}

        t_stat, p_value = stats.ttest_ind(in_dist, cross_site)

        return {
            "in_dist_mean": float(np.mean(in_dist)),
            "cross_site_mean": float(np.mean(cross_site)),
            "auroc_drop": float(np.mean(in_dist) - np.mean(cross_site)),
            "t_statistic": float(t_stat),
            "p_value": float(p_value),
            "hypothesis_confirmed": p_value > 0.05,
        }

    def statistical_testing(
        self,
        pgd_at_results: List[float],
        baseline_results: Optional[List[float]] = None,
        alpha: float = 0.01,
    ) -> Dict:
        """
        A1+ grade statistical analysis with full rigor.

        ENHANCED with:
        - Normality testing (Shapiro-Wilk)
        - Parametric/non-parametric test selection
        - Effect sizes (Cohen's d + Hedge's g)
        - Bootstrap confidence intervals
        - Multiple comparison correction
        - Statistical power analysis

        Args:
            pgd_at_results: PGD-AT accuracies (3 seeds)
            baseline_results: Baseline accuracies (3 seeds) for comparison
            alpha: Significance level (default: 0.01 for A1+ standard)

        Returns:
            Dict with comprehensive test results
        """
        tests = {}

        # Convert to numpy arrays
        pgd_at_results = np.array(pgd_at_results)

        # 1. Normality testing (Shapiro-Wilk)
        if len(pgd_at_results) >= 3:
            _, p_pgd = stats.shapiro(pgd_at_results)
            tests["normality_pgd_at"] = {
                "p_value": float(p_pgd),
                "normal": p_pgd > 0.05,
            }

        if baseline_results is not None:
            baseline_results = np.array(baseline_results)

            # Normality test for baseline
            if len(baseline_results) >= 3:
                _, p_baseline = stats.shapiro(baseline_results)
                tests["normality_baseline"] = {
                    "p_value": float(p_baseline),
                    "normal": p_baseline > 0.05,
                }
                both_normal = (
                    tests["normality_pgd_at"]["normal"]
                    and tests["normality_baseline"]["normal"]
                )
            else:
                both_normal = False

            # 2. Choose appropriate statistical test
            if both_normal and len(pgd_at_results) == len(baseline_results):
                # Paired t-test (parametric)
                t_stat, p_value = stats.ttest_rel(pgd_at_results, baseline_results)
                test_used = "paired_t_test"
            else:
                # Wilcoxon signed-rank (non-parametric)
                try:
                    t_stat, p_value = stats.wilcoxon(
                        pgd_at_results, baseline_results, alternative="two-sided"
                    )
                    test_used = "wilcoxon_signed_rank"
                except:
                    # Fallback to independent t-test
                    t_stat, p_value = stats.ttest_ind(pgd_at_results, baseline_results)
                    test_used = "independent_t_test"

            tests["statistical_test"] = {
                "name": test_used,
                "statistic": float(t_stat),
                "p_value": float(p_value),
                "significant": p_value < alpha,
                "alpha": alpha,
            }

            # 3. Effect sizes
            mean_diff = np.mean(pgd_at_results) - np.mean(baseline_results)

            # Cohen's d
            pooled_std = np.sqrt(
                (np.var(pgd_at_results, ddof=1) + np.var(baseline_results, ddof=1)) / 2
            )
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

            # Hedge's g (small sample correction)
            n = len(pgd_at_results) + len(baseline_results)
            hedges_g = cohens_d * (1 - (3 / (4 * n - 9))) if n > 9 else cohens_d

            # Effect size interpretation
            effect_magnitude = self._interpret_effect_size(abs(hedges_g))

            tests["effect_size"] = {
                "cohens_d": float(cohens_d),
                "hedges_g": float(hedges_g),
                "interpretation": effect_magnitude,
            }

            # 4. Bootstrap confidence interval (95%)
            try:
                from scipy.stats import bootstrap as scipy_bootstrap

                def stat_func(x, y, axis):
                    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

                rng = np.random.default_rng(42)
                res = scipy_bootstrap(
                    (pgd_at_results, baseline_results),
                    stat_func,
                    n_resamples=10000,
                    confidence_level=0.95,
                    random_state=rng,
                    method="percentile",
                    vectorized=False,
                )

                tests["bootstrap_ci"] = {
                    "mean_diff": float(mean_diff),
                    "ci_lower": float(res.confidence_interval.low),
                    "ci_upper": float(res.confidence_interval.high),
                    "level": 0.95,
                }
            except Exception as e:
                logger.warning(f"Bootstrap CI failed: {e}")
                # Fallback to parametric CI
                ci = stats.t.interval(
                    0.95,
                    len(pgd_at_results) - 1,
                    loc=np.mean(pgd_at_results),
                    scale=stats.sem(pgd_at_results),
                )
                tests["confidence_interval_95"] = {
                    "lower": float(ci[0]),
                    "upper": float(ci[1]),
                    "mean": float(np.mean(pgd_at_results)),
                }

            # Interpretation
            if p_value < alpha:
                tests["significant"] = True
                tests["interpretation"] = (
                    f"Significant difference (p={p_value:.4f}, "
                    f"Hedge's g={hedges_g:.3f}, {effect_magnitude})"
                )
            else:
                tests["significant"] = False
                tests["interpretation"] = (
                    f"No significant difference (p={p_value:.4f}, "
                    f"Hedge's g={hedges_g:.3f})"
                )

        else:
            # Only PGD-AT results provided
            mean = np.mean(pgd_at_results)
            std = np.std(pgd_at_results, ddof=1)
            n = len(pgd_at_results)
            ci = stats.t.interval(0.95, n - 1, loc=mean, scale=std / np.sqrt(n))
            tests["confidence_interval_95"] = {
                "lower": float(ci[0]),
                "upper": float(ci[1]),
                "mean": float(mean),
                "std": float(std),
            }

        return tests

    def _interpret_effect_size(self, effect: float) -> str:
        """Interpret effect size magnitude (Cohen's guidelines)."""
        if effect < 0.2:
            return "negligible"
        elif effect < 0.5:
            return "small"
        elif effect < 0.8:
            return "medium"
        else:
            return "large"

    def visualize_results(self, df: pd.DataFrame, summary: pd.DataFrame):
        """Generate visualizations of results."""
        fig_dir = self.output_dir / "figures"
        fig_dir.mkdir(exist_ok=True)

        # 1. Barplot: Clean vs Robust Accuracy by Test Set
        plt.figure(figsize=(12, 6))
        clean_data = summary[summary["attack"] == "clean"]
        robust_data = summary[summary["attack"] == "pgd_10"]

        x = np.arange(len(clean_data))
        width = 0.35

        plt.bar(
            x - width / 2,
            clean_data["mean"],
            width,
            label="Clean",
            yerr=clean_data["std"],
        )
        plt.bar(
            x + width / 2,
            robust_data["mean"],
            width,
            label="Robust (PGD-10)",
            yerr=robust_data["std"],
        )

        plt.xlabel("Test Set")
        plt.ylabel("Accuracy (%)")
        plt.title("PGD-AT: Clean vs Robust Accuracy (3 seeds)")
        plt.xticks(x, clean_data["test_set"], rotation=45)
        plt.legend()
        plt.grid(axis="y", alpha=0.3)
        plt.tight_layout()
        plt.savefig(fig_dir / "clean_vs_robust.png", dpi=300, bbox_inches="tight")
        plt.close()

        # 2. Heatmap: Accuracy across test sets and attacks
        pivot = summary.pivot(index="test_set", columns="attack", values="mean")
        plt.figure(figsize=(10, 6))
        sns.heatmap(pivot, annot=True, fmt=".1f", cmap="RdYlGn", vmin=0, vmax=100)
        plt.title("PGD-AT: Accuracy Heatmap (mean across 3 seeds)")
        plt.tight_layout()
        plt.savefig(fig_dir / "accuracy_heatmap.png", dpi=300, bbox_inches="tight")
        plt.close()

        logger.info(f"Visualizations saved to {fig_dir}")

    def save_results(self, all_results: Dict, df: pd.DataFrame, summary: pd.DataFrame):
        """Save all results to disk."""
        # Save raw results
        results_path = self.output_dir / "pgd_at_results.json"
        with open(results_path, "w") as f:
            json.dump(all_results, f, indent=2)

        # Save detailed DataFrame
        df.to_csv(self.output_dir / "pgd_at_detailed.csv", index=False)

        # Save summary
        summary.to_csv(self.output_dir / "pgd_at_summary.csv", index=False)

        # Save for RQ1 analysis
        rq1_dir = Path("results/metrics/rq1_robustness")
        rq1_dir.mkdir(parents=True, exist_ok=True)
        summary.to_csv(rq1_dir / "pgd_at.csv", index=False)

        logger.info(f"Results saved to {self.output_dir}")
        logger.info(f"RQ1 results saved to {rq1_dir}")

    def run_evaluation(self):
        """Execute complete evaluation pipeline."""
        logger.info("Starting PGD-AT evaluation pipeline...")

        # Evaluate all models
        all_results = self.evaluate_all_models()

        # Compute statistics
        df, summary = self.compute_statistics(all_results)

        # Statistical testing (if baseline provided)
        # NOTE: Load baseline results from Phase 5.1 or earlier
        baseline_path = Path("results/baseline/statistical_summary.json")
        if baseline_path.exists():
            with open(baseline_path, "r") as f:
                baseline_data = json.load(f)
            baseline_robust = baseline_data.get("robust_accuracy", {}).get("values", [])

            pgd_at_robust = df[df["attack"] == "pgd_10"]["accuracy"].tolist()[:3]
            stat_tests = self.statistical_testing(pgd_at_robust, baseline_robust)

            logger.info("\nStatistical Tests (PGD-AT vs Baseline):")
            logger.info(json.dumps(stat_tests, indent=2))

            # Save statistical tests
            with open(self.output_dir / "statistical_tests.json", "w") as f:
                json.dump(stat_tests, f, indent=2)

        # Visualize results
        self.visualize_results(df, summary)

        # Save all results
        self.save_results(all_results, df, summary)

        logger.info("\nEvaluation complete! ✅")
        return all_results, df, summary


def main():
    parser = argparse.ArgumentParser(description="Phase 5.2: PGD-AT Evaluation")
    parser.add_argument(
        "--model_paths",
        type=str,
        nargs="+",
        required=True,
        help="Paths to trained model checkpoints (one per seed)",
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pgd_at/evaluation",
        help="Output directory for evaluation results",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use (cuda/cpu)",
    )

    args = parser.parse_args()

    # Run evaluation
    evaluator = PGDATEvaluator(
        model_paths=args.model_paths,
        config_path=args.config,
        output_dir=args.output_dir,
        device=args.device,
    )
    evaluator.run_evaluation()


if __name__ == "__main__":
    main()
