"""
Phase 5.2: Complete PGD-AT Evaluation Pipeline
==============================================

This script provides a COMPLETE end-to-end pipeline for Phase 5.2:
1. Evaluates baseline and PGD-AT models on all test sets
2. Tests RQ1 hypothesis: "Does PGD-AT improve cross-site generalization?"
3. Generates publication-quality results
4. Answers RQ1 for dissertation

Author: Viraj Pankaj Jain
Date: November 24, 2025
Version: 1.0 (Production)
"""

import argparse
import json
import logging
import sys
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from torch.utils.data import DataLoader

# Add project root
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scipy import stats

from src.attacks.pgd import PGD, PGDConfig
from src.datasets import ISICDataset
from src.models import build_model
from src.utils.logging_utils import setup_logger
from src.utils.metrics import calculate_metrics

warnings.filterwarnings("ignore")
logger = setup_logger(__name__)


class Phase52Pipeline:
    """
    Complete Phase 5.2 evaluation pipeline.

    Evaluates baseline and PGD-AT models on all test sets,
    tests RQ1 hypothesis, and generates publication results.
    """

    def __init__(self, config_path: str, device: str = "cuda"):
        """Initialize pipeline."""
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        logger.info(f"Using device: {self.device}")

        # Load config
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        # Output directory
        self.output_dir = Path("results/phase_5_2_complete")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Phase 5.2 Pipeline initialized")

    def load_test_datasets(self) -> Dict[str, DataLoader]:
        """
        Load all test datasets for cross-site evaluation.

        Returns:
            Dictionary of test dataloaders
        """
        logger.info("\n" + "=" * 80)
        logger.info("Loading Test Datasets")
        logger.info("=" * 80)

        test_loaders = {}
        data_root = Path("data/processed")

        # Dataset configurations
        datasets_config = {
            "isic2018_test": {
                "root": data_root / "isic2018_test",
                "csv": data_root / "isic2018_test" / "metadata.csv",
                "split": "test",
                "description": "ISIC 2018 (In-Distribution)",
            },
            "isic2019": {
                "root": data_root / "isic2019",
                "csv": data_root / "isic2019" / "metadata.csv",
                "split": "test",
                "description": "ISIC 2019 (Cross-Site)",
            },
            "isic2020": {
                "root": data_root / "isic2020",
                "csv": data_root / "isic2020" / "metadata.csv",
                "split": "test",
                "description": "ISIC 2020 (Cross-Site)",
            },
            "derm7pt": {
                "root": data_root / "derm7pt",
                "csv": data_root / "derm7pt" / "metadata.csv",
                "split": "test",
                "description": "Derm7pt (Cross-Site)",
            },
        }

        for name, cfg in datasets_config.items():
            try:
                if not cfg["root"].exists():
                    logger.warning(f"  ⚠️  {name}: Path not found, skipping")
                    continue

                dataset = ISICDataset(
                    root=str(cfg["root"]),
                    csv_path=str(cfg["csv"]) if cfg["csv"].exists() else None,
                    split=cfg["split"],
                    image_size=224,
                    augment=False,
                )

                test_loaders[name] = DataLoader(
                    dataset,
                    batch_size=32,
                    shuffle=False,
                    num_workers=4,
                    pin_memory=True,
                )

                logger.info(f"  ✓ {cfg['description']}: " f"{len(dataset)} samples")

            except Exception as e:
                logger.warning(f"  ⚠️  Failed to load {name}: {e}")

        if not test_loaders:
            raise RuntimeError("No test datasets loaded!")

        logger.info(f"\nTotal: {len(test_loaders)} test sets loaded")
        return test_loaders

    def load_model(self, checkpoint_path: Path) -> nn.Module:
        """Load model from checkpoint."""
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

        logger.info(f"  Loading: {checkpoint_path}")

        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location=self.device)

        # Build model
        if "config" in checkpoint:
            model_config = checkpoint["config"].get("model", {})
            num_classes = checkpoint["config"].get("data", {}).get("num_classes", 2)
        else:
            model_config = self.config.get("model", {})
            num_classes = 2

        model = build_model(
            architecture=model_config.get("architecture", "resnet50"),
            num_classes=num_classes,
            pretrained=False,
        )

        # Load weights
        if "model_state_dict" in checkpoint:
            model.load_state_dict(checkpoint["model_state_dict"])
        elif "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)

        model = model.to(self.device)
        model.eval()

        return model

    def evaluate_clean(
        self, model: nn.Module, dataloader: DataLoader
    ) -> Dict[str, float]:
        """Evaluate clean accuracy."""
        model.eval()

        all_preds = []
        all_labels = []
        all_probs = []

        with torch.no_grad():
            for images, labels in dataloader:
                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = model(images)
                probs = torch.softmax(outputs, dim=1)
                preds = outputs.argmax(dim=1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())

        # Compute metrics
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)

        metrics = calculate_metrics(
            y_true=all_labels, y_pred=all_preds, y_probs=all_probs
        )

        return metrics

    def evaluate_robust(
        self, model: nn.Module, dataloader: DataLoader, epsilon: float = 8 / 255
    ) -> Dict[str, float]:
        """Evaluate robust accuracy under PGD attack."""
        model.eval()

        # Create PGD attack
        pgd_config = PGDConfig(
            epsilon=epsilon,
            step_size=epsilon / 4,
            num_steps=10,
            random_start=True,
            norm="Linf",
        )
        attack = PGD(config=pgd_config, model=model, device=self.device)

        all_preds = []
        all_labels = []

        for images, labels in dataloader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Generate adversarial examples
            adv_images = attack(images, labels)

            # Evaluate
            with torch.no_grad():
                outputs = model(adv_images)
                preds = outputs.argmax(dim=1)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

        # Compute accuracy
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)

        accuracy = 100.0 * (all_preds == all_labels).mean()

        return {"accuracy": accuracy, "robust": True}

    def evaluate_model(
        self, model_path: Path, test_loaders: Dict[str, DataLoader], model_name: str
    ) -> Dict[str, Dict]:
        """Evaluate a single model on all test sets."""
        logger.info(f"\n{'='*80}")
        logger.info(f"Evaluating: {model_name}")
        logger.info(f"{'='*80}")

        # Load model
        model = self.load_model(model_path)

        results = {}

        for test_name, test_loader in test_loaders.items():
            logger.info(f"\n  Dataset: {test_name}")

            # Clean accuracy
            clean_metrics = self.evaluate_clean(model, test_loader)
            logger.info(
                f"    Clean - Acc: {clean_metrics['accuracy']:.2f}%, "
                f"AUROC: {clean_metrics.get('auroc', 0):.4f}"
            )

            # Robust accuracy
            robust_metrics = self.evaluate_robust(model, test_loader)
            logger.info(
                f"    Robust (ε=8/255) - Acc: {robust_metrics['accuracy']:.2f}%"
            )

            results[test_name] = {"clean": clean_metrics, "robust": robust_metrics}

        # Clean up
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        logger.info(f"\n  ✓ {model_name} evaluation complete")

        return results

    def evaluate_all_models(
        self,
        model_type: str,
        model_paths: List[Path],
        test_loaders: Dict[str, DataLoader],
    ) -> Dict[str, Dict]:
        """Evaluate all seeds of a model type."""
        all_results = {}

        for i, model_path in enumerate(model_paths):
            seed_name = f"{model_type}_seed_{i+1}"
            results = self.evaluate_model(model_path, test_loaders, seed_name)
            all_results[seed_name] = results

        return all_results

    def aggregate_results(self, all_results: Dict[str, Dict]) -> Dict[str, Dict]:
        """Aggregate results across seeds."""
        logger.info("\n" + "=" * 80)
        logger.info("Aggregating Results Across Seeds")
        logger.info("=" * 80)

        # Get all test sets
        first_seed = list(all_results.values())[0]
        test_sets = list(first_seed.keys())

        aggregated = {}

        for test_name in test_sets:
            aggregated[test_name] = {}

            for metric_type in ["clean", "robust"]:
                # Extract values across seeds
                values = {}
                for seed_results in all_results.values():
                    if test_name in seed_results:
                        metrics = seed_results[test_name][metric_type]
                        for key, val in metrics.items():
                            if key not in values:
                                values[key] = []
                            values[key].append(val)

                # Compute statistics
                stats_dict = {}
                for key, vals in values.items():
                    vals_array = np.array(vals)
                    stats_dict[key] = {
                        "mean": float(np.mean(vals_array)),
                        "std": float(np.std(vals_array)),
                        "min": float(np.min(vals_array)),
                        "max": float(np.max(vals_array)),
                        "values": [float(v) for v in vals],
                    }

                aggregated[test_name][metric_type] = stats_dict

        # Log summary
        for test_name, test_results in aggregated.items():
            logger.info(f"\n{test_name}:")
            if "clean" in test_results and "accuracy" in test_results["clean"]:
                acc = test_results["clean"]["accuracy"]
                logger.info(f"  Clean: {acc['mean']:.2f}% ± {acc['std']:.2f}%")
            if "robust" in test_results and "accuracy" in test_results["robust"]:
                acc = test_results["robust"]["accuracy"]
                logger.info(f"  Robust: {acc['mean']:.2f}% ± {acc['std']:.2f}%")

        return aggregated

    def test_rq1_hypothesis(
        self, baseline_agg: Dict[str, Dict], pgd_at_agg: Dict[str, Dict]
    ) -> Dict[str, Any]:
        """
        Test RQ1 Hypothesis: Does PGD-AT improve cross-site generalization?

        H1c: PGD-AT does NOT significantly improve cross-site generalization
        Expected: p > 0.05
        """
        logger.info("\n" + "=" * 80)
        logger.info("RQ1 HYPOTHESIS TEST")
        logger.info("=" * 80)
        logger.info("H1c: PGD-AT does NOT improve cross-site generalization")
        logger.info("Expected: p > 0.05 (no significant difference)\n")

        # Identify test sets
        in_dist_set = "isic2018_test"
        cross_site_sets = ["isic2019", "isic2020", "derm7pt"]

        # Extract AUROC values
        baseline_source = (
            baseline_agg.get(in_dist_set, {})
            .get("clean", {})
            .get("auroc", {})
            .get("mean", 0)
        )

        pgd_at_source = (
            pgd_at_agg.get(in_dist_set, {})
            .get("clean", {})
            .get("auroc", {})
            .get("mean", 0)
        )

        logger.info(f"Source AUROC ({in_dist_set}):")
        logger.info(f"  Baseline: {baseline_source:.4f}")
        logger.info(f"  PGD-AT:   {pgd_at_source:.4f}\n")

        # Compute drops for each cross-site dataset
        baseline_drops = []
        pgd_at_drops = []

        for cross_site in cross_site_sets:
            if cross_site in baseline_agg and cross_site in pgd_at_agg:
                baseline_target = baseline_agg[cross_site]["clean"]["auroc"]["mean"]
                pgd_at_target = pgd_at_agg[cross_site]["clean"]["auroc"]["mean"]

                baseline_drop = baseline_source - baseline_target
                pgd_at_drop = pgd_at_source - pgd_at_target

                baseline_drops.append(baseline_drop)
                pgd_at_drops.append(pgd_at_drop)

                logger.info(f"{cross_site}:")
                logger.info(
                    f"  Baseline: AUROC={baseline_target:.4f}, "
                    f"Drop={baseline_drop:.4f}"
                )
                logger.info(
                    f"  PGD-AT:   AUROC={pgd_at_target:.4f}, "
                    f"Drop={pgd_at_drop:.4f}\n"
                )

        if not baseline_drops or not pgd_at_drops:
            return {
                "error": "Insufficient cross-site data",
                "p_value": None,
                "hypothesis_confirmed": None,
            }

        # Convert to numpy
        baseline_drops = np.array(baseline_drops)
        pgd_at_drops = np.array(pgd_at_drops)

        # Statistical test: Paired t-test
        t_stat, p_value = stats.ttest_rel(baseline_drops, pgd_at_drops)

        # Effect size (Cohen's d)
        mean_diff = pgd_at_drops.mean() - baseline_drops.mean()
        pooled_std = np.sqrt((baseline_drops.var() + pgd_at_drops.var()) / 2)
        cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0.0

        # H1c: Confirmed if NO significant difference (p > 0.05)
        hypothesis_confirmed = p_value > 0.05

        # Log results
        logger.info("=" * 80)
        logger.info("STATISTICAL TEST RESULTS:")
        logger.info(
            f"  Baseline drops (mean): {baseline_drops.mean():.4f} ± {baseline_drops.std():.4f}"
        )
        logger.info(
            f"  PGD-AT drops (mean):   {pgd_at_drops.mean():.4f} ± {pgd_at_drops.std():.4f}"
        )
        logger.info(f"  t-statistic: {t_stat:.4f}")
        logger.info(f"  p-value: {p_value:.4f}")
        logger.info(f"  Cohen's d: {cohens_d:.3f}")
        logger.info("=" * 80)

        if hypothesis_confirmed:
            logger.info("\n✓ H1c CONFIRMED (p > 0.05)")
            logger.info(
                "  PGD-AT does NOT significantly improve cross-site generalization"
            )
            logger.info("  → This justifies the need for tri-objective approach!")
        else:
            logger.info("\n✗ H1c REJECTED (p ≤ 0.05)")
            logger.info("  PGD-AT DOES affect cross-site generalization")
            logger.info(f"  Effect size (Cohen's d): {cohens_d:.3f}")

        logger.info("=" * 80 + "\n")

        return {
            "hypothesis": "H1c: PGD-AT does NOT improve cross-site generalization",
            "p_value": float(p_value),
            "t_statistic": float(t_stat),
            "cohens_d": float(cohens_d),
            "hypothesis_confirmed": hypothesis_confirmed,
            "baseline_drops": {
                "mean": float(baseline_drops.mean()),
                "std": float(baseline_drops.std()),
                "values": baseline_drops.tolist(),
            },
            "pgd_at_drops": {
                "mean": float(pgd_at_drops.mean()),
                "std": float(pgd_at_drops.std()),
                "values": pgd_at_drops.tolist(),
            },
            "interpretation": (
                "H1c CONFIRMED: PGD-AT does NOT improve cross-site generalization. "
                "This validates orthogonality between robustness and generalization."
                if hypothesis_confirmed
                else "H1c REJECTED: PGD-AT does affect cross-site generalization."
            ),
        }

    def generate_results_table(
        self, baseline_agg: Dict[str, Dict], pgd_at_agg: Dict[str, Dict]
    ) -> pd.DataFrame:
        """Generate results table for dissertation."""
        rows = []

        for test_name in baseline_agg.keys():
            # Baseline
            baseline_clean = baseline_agg[test_name]["clean"]["accuracy"]
            baseline_robust = baseline_agg[test_name]["robust"]["accuracy"]
            baseline_auroc = baseline_agg[test_name]["clean"].get("auroc", {})

            # PGD-AT
            pgd_clean = pgd_at_agg[test_name]["clean"]["accuracy"]
            pgd_robust = pgd_at_agg[test_name]["robust"]["accuracy"]
            pgd_auroc = pgd_at_agg[test_name]["clean"].get("auroc", {})

            rows.append(
                {
                    "Test Set": test_name,
                    "Baseline Clean Acc": f"{baseline_clean['mean']:.2f} ± {baseline_clean['std']:.2f}",
                    "Baseline Robust Acc": f"{baseline_robust['mean']:.2f} ± {baseline_robust['std']:.2f}",
                    "Baseline AUROC": f"{baseline_auroc.get('mean', 0):.4f} ± {baseline_auroc.get('std', 0):.4f}",
                    "PGD-AT Clean Acc": f"{pgd_clean['mean']:.2f} ± {pgd_clean['std']:.2f}",
                    "PGD-AT Robust Acc": f"{pgd_robust['mean']:.2f} ± {pgd_robust['std']:.2f}",
                    "PGD-AT AUROC": f"{pgd_auroc.get('mean', 0):.4f} ± {pgd_auroc.get('std', 0):.4f}",
                    "Δ Robust Acc": f"{pgd_robust['mean'] - baseline_robust['mean']:+.2f}",
                }
            )

        df = pd.DataFrame(rows)
        return df

    def save_results(
        self,
        baseline_results: Dict,
        pgd_at_results: Dict,
        baseline_agg: Dict,
        pgd_at_agg: Dict,
        rq1_results: Dict,
        results_table: pd.DataFrame,
    ):
        """Save all results."""
        logger.info("\n" + "=" * 80)
        logger.info("Saving Results")
        logger.info("=" * 80)

        # Save raw results
        with open(self.output_dir / "baseline_results.json", "w") as f:
            json.dump(baseline_results, f, indent=2)
        logger.info(f"  ✓ Baseline raw results saved")

        with open(self.output_dir / "pgd_at_results.json", "w") as f:
            json.dump(pgd_at_results, f, indent=2)
        logger.info(f"  ✓ PGD-AT raw results saved")

        # Save aggregated results
        with open(self.output_dir / "baseline_aggregated.json", "w") as f:
            json.dump(baseline_agg, f, indent=2)
        logger.info(f"  ✓ Baseline aggregated saved")

        with open(self.output_dir / "pgd_at_aggregated.json", "w") as f:
            json.dump(pgd_at_agg, f, indent=2)
        logger.info(f"  ✓ PGD-AT aggregated saved")

        # Save RQ1 results
        with open(self.output_dir / "rq1_hypothesis_test.json", "w") as f:
            json.dump(rq1_results, f, indent=2)
        logger.info(f"  ✓ RQ1 hypothesis test saved")

        # Save results table
        results_table.to_csv(self.output_dir / "results_table.csv", index=False)
        logger.info(f"  ✓ Results table (CSV) saved")

        results_table.to_latex(self.output_dir / "results_table.tex", index=False)
        logger.info(f"  ✓ Results table (LaTeX) saved")

        logger.info(f"\nAll results saved to: {self.output_dir}")

    def run(self, baseline_paths: List[Path], pgd_at_paths: List[Path]):
        """Run complete Phase 5.2 pipeline."""
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 5.2: COMPLETE EVALUATION PIPELINE")
        logger.info("=" * 80)
        logger.info(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")

        # Step 1: Load test datasets
        test_loaders = self.load_test_datasets()

        # Step 2: Evaluate baseline models
        logger.info("\n" + "=" * 80)
        logger.info("STEP 1: Evaluating Baseline Models")
        logger.info("=" * 80)
        baseline_results = self.evaluate_all_models(
            "baseline", baseline_paths, test_loaders
        )

        # Step 3: Evaluate PGD-AT models
        logger.info("\n" + "=" * 80)
        logger.info("STEP 2: Evaluating PGD-AT Models")
        logger.info("=" * 80)
        pgd_at_results = self.evaluate_all_models("pgd_at", pgd_at_paths, test_loaders)

        # Step 4: Aggregate results
        baseline_agg = self.aggregate_results(baseline_results)
        pgd_at_agg = self.aggregate_results(pgd_at_results)

        # Step 5: Test RQ1 hypothesis
        rq1_results = self.test_rq1_hypothesis(baseline_agg, pgd_at_agg)

        # Step 6: Generate results table
        results_table = self.generate_results_table(baseline_agg, pgd_at_agg)

        logger.info("\n" + "=" * 80)
        logger.info("RESULTS TABLE")
        logger.info("=" * 80)
        print(results_table.to_string(index=False))

        # Step 7: Save everything
        self.save_results(
            baseline_results,
            pgd_at_results,
            baseline_agg,
            pgd_at_agg,
            rq1_results,
            results_table,
        )

        # Final summary
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 5.2 COMPLETE")
        logger.info("=" * 80)
        logger.info(f"✓ Evaluated {len(baseline_paths)} baseline models")
        logger.info(f"✓ Evaluated {len(pgd_at_paths)} PGD-AT models")
        logger.info(f"✓ Tested on {len(test_loaders)} test sets")
        logger.info(f"✓ RQ1 hypothesis tested")
        logger.info(f"✓ Results saved to: {self.output_dir}")
        logger.info(f"\nFinished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        logger.info("=" * 80 + "\n")

        return {
            "baseline_agg": baseline_agg,
            "pgd_at_agg": pgd_at_agg,
            "rq1_results": rq1_results,
            "results_table": results_table,
        }


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5.2: Complete PGD-AT Evaluation Pipeline"
    )
    parser.add_argument(
        "--config", type=str, default="configs/base.yaml", help="Path to config file"
    )
    parser.add_argument(
        "--baseline-checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Paths to baseline model checkpoints (3 seeds)",
    )
    parser.add_argument(
        "--pgd-at-checkpoints",
        type=str,
        nargs="+",
        required=True,
        help="Paths to PGD-AT model checkpoints (3 seeds)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda", help="Device to use (cuda/cpu)"
    )

    args = parser.parse_args()

    # Convert to Path objects
    baseline_paths = [Path(p) for p in args.baseline_checkpoints]
    pgd_at_paths = [Path(p) for p in args.pgd_at_checkpoints]

    # Validate paths
    for p in baseline_paths + pgd_at_paths:
        if not p.exists():
            raise FileNotFoundError(f"Checkpoint not found: {p}")

    # Create pipeline
    pipeline = Phase52Pipeline(args.config, args.device)

    # Run pipeline
    results = pipeline.run(baseline_paths, pgd_at_paths)

    logger.info("\n🎉 Phase 5.2 pipeline completed successfully!")
    logger.info("\nRQ1 Answer:")
    logger.info(results["rq1_results"]["interpretation"])


if __name__ == "__main__":
    main()
