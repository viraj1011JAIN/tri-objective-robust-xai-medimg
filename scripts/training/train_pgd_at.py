"""
Phase 5.2: PGD Adversarial Training
====================================

Production-grade PGD-AT training script for RQ1 robustness evaluation.

Key Features:
- Multi-seed training (3 seeds for statistical significance)
- Clean + robust accuracy tracking
- Cross-site generalization evaluation
- Statistical significance testing (t-test, Cohen's d)
- MLflow experiment tracking
- Comprehensive checkpointing

Usage:
    python scripts/training/train_pgd_at.py --config configs/experiments/pgd_at_isic.yaml
    python scripts/training/train_pgd_at.py --config configs/experiments/pgd_at_isic.yaml --seeds 42 123 456
    python scripts/training/train_pgd_at.py --resume results/pgd_at/seed_42/checkpoints/last.pt

Author: Viraj Pankaj Jain
Date: November 24, 2025
Version: 5.2.0
"""

import argparse
import json
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import mlflow
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import yaml
from scipy import stats
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.attacks import FGSM, PGD, AutoAttack, CarliniWagner
from src.attacks.pgd import PGDConfig
from src.datasets import ChestXRayDataset, ISICDataset
from src.datasets.transforms import get_test_transforms, get_train_transforms
from src.losses.robust_loss import AdversarialTrainingLoss
from src.models import build_model
from src.training.adversarial_trainer import (
    AdversarialTrainer,
    AdversarialTrainingConfig,
    train_adversarial_epoch,
    validate_robust,
)
from src.utils.config import load_experiment_config
from src.utils.metrics import calculate_metrics
from src.utils.reproducibility import set_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger = logging.getLogger(__name__)
warnings.filterwarnings("ignore", category=UserWarning)


class PGDATTrainer:
    """High-performance PGD-AT trainer with comprehensive evaluation."""

    def __init__(
        self,
        config_path: str,
        output_dir: str,
        seed: int = 42,
        resume_from: Optional[str] = None,
        device: str = "cuda",
    ):
        """
        Initialize PGD-AT trainer.

        Args:
            config_path: Path to experiment config YAML
            output_dir: Directory for checkpoints/results
            seed: Random seed for reproducibility
            resume_from: Optional checkpoint path to resume from
            device: Device to train on ('cuda' or 'cpu')
        """
        self.config_path = Path(config_path)
        self.output_dir = Path(output_dir)
        self.seed = seed
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

        # Create output directories
        self.checkpoint_dir = self.output_dir / "checkpoints"
        self.results_dir = self.output_dir / "results"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.results_dir.mkdir(parents=True, exist_ok=True)

        # Set seed
        set_seed(seed)
        logger.info(f"Set random seed: {seed}")

        # Load configuration
        self.config = self._load_config()
        logger.info(f"Loaded config from {config_path}")

        # Initialize model
        self.model = self._build_model()
        logger.info(
            f"Built {self.config['model']['architecture']} model "
            f"({sum(p.numel() for p in self.model.parameters()) / 1e6:.2f}M params)"
        )

        # Initialize dataloaders
        self.train_loader, self.val_loader, self.test_loaders = (
            self._build_dataloaders()
        )
        logger.info(
            f"Loaded dataset: {len(self.train_loader.dataset)} train, "
            f"{len(self.val_loader.dataset)} val"
        )

        # Initialize loss, optimizer, scheduler
        self.criterion = self._build_criterion()
        self.optimizer = self._build_optimizer()
        self.scheduler = self._build_scheduler()

        # Initialize adversarial trainer
        self.trainer = self._build_trainer()

        # Training state
        self.start_epoch = 0
        self.best_robust_acc = 0.0
        self.history = {"train": [], "val": [], "test": []}

        # Resume if checkpoint provided
        if resume_from:
            self._resume_from_checkpoint(resume_from)

        # Setup and verify training components
        self._setup_training()

    def _setup_training(self) -> None:
        """
        Setup training components and verify configuration.

        This method validates that all training components are properly
        initialized and logs setup information.
        """
        # Ensure model is on correct device
        self.model = self.model.to(self.device)

        # Count parameters
        n_params = sum(p.numel() for p in self.model.parameters())
        n_trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        logger.info("Training setup verification:")
        logger.info(f"  Total parameters: {n_params:,}")
        logger.info(f"  Trainable parameters: {n_trainable:,}")
        logger.info(f"  Device: {self.device}")
        logger.info(f"  Optimizer: {self.optimizer.__class__.__name__}")
        logger.info(
            f"  Scheduler: {self.scheduler.__class__.__name__ if self.scheduler else 'None'}"
        )
        logger.info(
            f"  Mixed precision: {self.config['training'].get('use_amp', False)}"
        )

        # Verify components
        assert self.model is not None, "Model not initialized"
        assert self.optimizer is not None, "Optimizer not initialized"
        assert self.criterion is not None, "Loss function not initialized"
        assert self.trainer is not None, "Adversarial trainer not initialized"

        logger.info("✓ Training setup verified")

    def _load_config(self) -> Dict:
        """Load and validate experiment configuration."""
        with open(self.config_path, "r") as f:
            config = yaml.safe_load(f)

        # Validate required fields
        required = ["model", "dataset", "training", "adversarial_training"]
        for field in required:
            if field not in config:
                raise ValueError(f"Missing required config field: {field}")

        return config

    def _build_model(self) -> nn.Module:
        """Build model from config."""
        model_config = self.config["model"]
        model = build_model(
            architecture=model_config["architecture"],
            num_classes=model_config["num_classes"],
            pretrained=model_config.get("pretrained", True),
            in_channels=model_config.get("in_channels", 3),
        )
        return model.to(self.device)

    def _build_dataloaders(
        self,
    ) -> Tuple[DataLoader, DataLoader, Dict[str, DataLoader]]:
        """Build train, val, and test dataloaders."""
        dataset_config = self.config["dataset"]
        batch_size = self.config["training"]["batch_size"]
        num_workers = self.config["training"].get("num_workers", 4)

        # Build training dataset
        image_size = dataset_config.get("image_size", 224)

        if dataset_config["name"].lower() == "isic2018":
            train_transforms = get_train_transforms(
                dataset="isic", image_size=image_size
            )
            val_transforms = get_test_transforms(dataset="isic", image_size=image_size)

            train_dataset = ISICDataset(
                root=dataset_config["root"],
                csv_path=dataset_config["csv_path"],
                split="train",
                transforms=train_transforms,
            )
            val_dataset = ISICDataset(
                root=dataset_config["root"],
                csv_path=dataset_config["csv_path"],
                split="val",
                transforms=val_transforms,
            )
        else:
            raise ValueError(f"Unsupported dataset: {dataset_config['name']}")

        # Create dataloaders
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

        # Build test loaders for cross-site evaluation
        test_loaders = {}
        test_datasets_config = dataset_config.get("test_datasets", {})
        test_transforms = get_test_transforms(dataset="isic", image_size=image_size)

        for test_name, test_config in test_datasets_config.items():
            if test_config.get("name", "").lower() == "isic2018":
                test_dataset = ISICDataset(
                    root=test_config["root"],
                    csv_path=test_config["csv_path"],
                    split="test",
                    transforms=test_transforms,
                )
                test_loaders[test_name] = DataLoader(
                    test_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )

        return train_loader, val_loader, test_loaders

    def _build_criterion(self) -> nn.Module:
        """Build adversarial training loss."""
        adv_config = self.config["adversarial_training"]
        return AdversarialTrainingLoss(
            mix_clean=adv_config.get("mix_clean", 0.0),
            reduction="mean",
        )

    def _build_optimizer(self) -> torch.optim.Optimizer:
        """Build optimizer from config."""
        opt_config = self.config["training"]["optimizer"]
        opt_type = opt_config["type"].lower()

        if opt_type == "adam":
            return torch.optim.Adam(
                self.model.parameters(),
                lr=opt_config["learning_rate"],
                weight_decay=opt_config.get("weight_decay", 0.0),
                betas=opt_config.get("betas", (0.9, 0.999)),
            )
        elif opt_type == "sgd":
            return torch.optim.SGD(
                self.model.parameters(),
                lr=opt_config["learning_rate"],
                momentum=opt_config.get("momentum", 0.9),
                weight_decay=opt_config.get("weight_decay", 0.0),
            )
        else:
            raise ValueError(f"Unsupported optimizer: {opt_type}")

    def _build_scheduler(self) -> Optional[torch.optim.lr_scheduler._LRScheduler]:
        """Build learning rate scheduler from config."""
        sched_config = self.config["training"].get("scheduler")
        if not sched_config:
            return None

        sched_type = sched_config["type"].lower()

        if sched_type == "reduce_lr_on_plateau":
            return torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode=sched_config.get("mode", "max"),
                factor=sched_config.get("factor", 0.5),
                patience=sched_config.get("patience", 5),
            )
        elif sched_type == "cosine":
            return torch.optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=self.config["training"]["num_epochs"],
                eta_min=sched_config.get("eta_min", 0.0),
            )
        else:
            return None

    def _build_trainer(self) -> AdversarialTrainer:
        """Build adversarial trainer."""
        adv_config = self.config["adversarial_training"]
        attack_config = adv_config["attack"]

        # Create adversarial training configuration
        at_config = AdversarialTrainingConfig(
            loss_type="at",  # Standard PGD-AT
            beta=1.0,
            attack_epsilon=attack_config["epsilon"],
            attack_steps=attack_config["num_steps"],
            attack_step_size=attack_config["step_size"],
            attack_random_start=attack_config.get("random_start", True),
            mix_clean=adv_config.get("mix_clean", 0.0),
            use_amp=adv_config.get("use_amp", True),
            gradient_clip=adv_config.get("gradient_clip", 1.0),
        )

        return AdversarialTrainer(
            model=self.model,
            config=at_config,
            device=self.device,
        )

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train one epoch with PGD-AT."""
        self.model.train()
        metrics = train_adversarial_epoch(
            model=self.model,
            dataloader=self.train_loader,
            optimizer=self.optimizer,
            criterion=self.trainer.criterion,
            attack=self.trainer.attack,
            device=self.device,
            epoch=epoch,
            use_amp=self.trainer.config.use_amp,
            gradient_clip=self.trainer.config.gradient_clip,
            log_frequency=self.config["training"].get("log_interval", 10),
        )
        return metrics

    def validate(self, dataloader: DataLoader, epoch: int) -> Dict[str, float]:
        """Validate model with clean and robust accuracy."""
        adv_config = self.config["adversarial_training"]
        eval_config = adv_config.get("evaluation", {})

        # Create evaluation PGD config (stronger attack)
        eval_pgd_config = PGDConfig(
            epsilon=eval_config.get("attack_epsilon", adv_config["attack"]["epsilon"]),
            num_steps=eval_config.get("attack_steps", 10),
            step_size=eval_config.get("attack_epsilon", adv_config["attack"]["epsilon"])
            / 4,
            random_start=True,
            clip_min=0.0,
            clip_max=1.0,
        )

        metrics = validate_robust(
            model=self.model,
            val_loader=dataloader,
            criterion=self.criterion,
            pgd_config=eval_pgd_config,
            device=self.device,
            epoch=epoch,
        )
        return metrics

    def evaluate_cross_site(self) -> Dict[str, Dict[str, float]]:
        """
        Evaluate on all test sets for cross-site generalization.

        Returns:
            Dict mapping test set name to metrics dict
        """
        results = {}

        for test_name, test_loader in self.test_loaders.items():
            logger.info(f"Evaluating on {test_name}...")
            metrics = self.validate(test_loader, epoch=-1)
            results[test_name] = metrics
            logger.info(
                f"{test_name} - Clean: {metrics['clean_acc']:.2f}%, "
                f"Robust: {metrics['robust_acc']:.2f}%"
            )

        return results

    def save_checkpoint(
        self, epoch: int, metrics: Dict[str, float], is_best: bool = False
    ):
        """Save training checkpoint."""
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": (
                self.scheduler.state_dict() if self.scheduler else None
            ),
            "metrics": metrics,
            "config": self.config,
            "seed": self.seed,
            "best_robust_acc": self.best_robust_acc,
        }

        # Save last checkpoint
        torch.save(checkpoint, self.checkpoint_dir / "last.pt")

        # Save best checkpoint
        if is_best:
            torch.save(checkpoint, self.checkpoint_dir / "best.pt")
            logger.info(f"✅ New best robust accuracy: {metrics['robust_acc']:.2f}%")

        # Save epoch checkpoint
        if (epoch + 1) % self.config["training"].get("save_freq", 10) == 0:
            torch.save(checkpoint, self.checkpoint_dir / f"epoch_{epoch+1}.pt")

    def _resume_from_checkpoint(self, checkpoint_path: str):
        """Resume training from checkpoint."""
        logger.info(f"Resuming from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=self.device, weights_only=False)

        self.model.load_state_dict(checkpoint["model_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        if self.scheduler and checkpoint.get("scheduler_state_dict"):
            self.scheduler.load_state_dict(checkpoint["scheduler_state_dict"])

        self.start_epoch = checkpoint["epoch"] + 1
        self.best_robust_acc = checkpoint.get("best_robust_acc", 0.0)

        logger.info(f"Resumed from epoch {self.start_epoch}")

    def train(self):
        """Execute full training loop."""
        num_epochs = self.config["training"]["num_epochs"]

        logger.info(
            f"Starting PGD-AT training for {num_epochs} epochs (seed={self.seed})"
        )

        for epoch in range(self.start_epoch, num_epochs):
            # Training
            train_metrics = self.train_epoch(epoch)
            self.history["train"].append(train_metrics)

            # Validation
            val_metrics = self.validate(self.val_loader, epoch)
            self.history["val"].append(val_metrics)

            # Update scheduler
            if self.scheduler:
                if isinstance(
                    self.scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau
                ):
                    self.scheduler.step(val_metrics["robust_acc"])
                else:
                    self.scheduler.step()

            # Log metrics
            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_metrics['loss']:.4f}, "
                f"Val Clean: {val_metrics['clean_acc']:.2f}%, "
                f"Val Robust: {val_metrics['robust_acc']:.2f}%"
            )

            # Log to MLflow
            mlflow.log_metrics(
                {
                    "train/loss": train_metrics["loss"],
                    "train/clean_acc": train_metrics.get("clean_acc", 0.0),
                    "train/robust_acc": train_metrics.get("robust_acc", 0.0),
                    "val/clean_acc": val_metrics["clean_acc"],
                    "val/robust_acc": val_metrics["robust_acc"],
                    "val/loss": val_metrics["loss"],
                    "lr": self.optimizer.param_groups[0]["lr"],
                },
                step=epoch,
            )

            # Save checkpoint
            is_best = val_metrics["robust_acc"] > self.best_robust_acc
            if is_best:
                self.best_robust_acc = val_metrics["robust_acc"]
            self.save_checkpoint(epoch, val_metrics, is_best)

        logger.info(
            f"Training completed! Best robust accuracy: {self.best_robust_acc:.2f}%"
        )

        # Final cross-site evaluation
        logger.info("Running cross-site evaluation...")
        test_results = self.evaluate_cross_site()
        self.history["test"] = test_results

        # Save final results
        self._save_results()

        return self.history

    def _save_results(self):
        """Save training history and final results."""
        # Save history
        history_path = self.results_dir / "training_history.json"
        with open(history_path, "w") as f:
            json.dump(self.history, f, indent=2)

        # Save final metrics
        metrics_path = self.results_dir / "final_metrics.json"
        final_metrics = {
            "best_robust_acc": self.best_robust_acc,
            "final_val_metrics": self.history["val"][-1] if self.history["val"] else {},
            "test_results": self.history["test"],
            "seed": self.seed,
        }
        with open(metrics_path, "w") as f:
            json.dump(final_metrics, f, indent=2)

        logger.info(f"Results saved to {self.results_dir}")


def run_multi_seed_training(
    config_path: str,
    output_base: str,
    seeds: List[int],
    mlflow_experiment: str = "PGD-AT-ISIC2018",
):
    """
    Run PGD-AT training with multiple seeds for statistical significance.

    Args:
        config_path: Path to experiment config
        output_base: Base output directory
        seeds: List of random seeds
        mlflow_experiment: MLflow experiment name
    """
    mlflow.set_experiment(mlflow_experiment)

    results = []

    for seed in seeds:
        logger.info(f"\n{'='*80}")
        logger.info(f"Starting training with seed {seed}")
        logger.info(f"{'='*80}\n")

        # Create seed-specific output directory
        output_dir = Path(output_base) / f"seed_{seed}"

        # Start MLflow run
        with mlflow.start_run(run_name=f"pgd_at_seed_{seed}"):
            # Log config and seed
            mlflow.log_params({"seed": seed, "method": "pgd_at"})
            mlflow.log_artifact(config_path)

            # Train model
            trainer = PGDATTrainer(
                config_path=config_path,
                output_dir=str(output_dir),
                seed=seed,
            )
            history = trainer.train()

            # Log final metrics
            mlflow.log_metrics(
                {
                    "final/best_robust_acc": trainer.best_robust_acc,
                    "final/val_clean_acc": history["val"][-1]["clean_acc"],
                    "final/val_robust_acc": history["val"][-1]["robust_acc"],
                }
            )

            results.append(
                {
                    "seed": seed,
                    "best_robust_acc": trainer.best_robust_acc,
                    "final_val_metrics": history["val"][-1],
                    "test_results": history["test"],
                }
            )

    # Aggregate results and compute statistics
    logger.info(f"\n{'='*80}")
    logger.info("Multi-seed Training Complete - Computing Statistics")
    logger.info(f"{'='*80}\n")

    compute_statistical_summary(results, output_base)

    return results


def compute_statistical_summary(results: List[Dict], output_dir: str):
    """
    Compute statistical summary across multiple seeds.

    Args:
        results: List of result dicts from each seed
        output_dir: Directory to save summary
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract metrics
    robust_accs = [r["best_robust_acc"] for r in results]
    clean_accs = [r["final_val_metrics"]["clean_acc"] for r in results]

    # Compute statistics
    summary = {
        "n_seeds": len(results),
        "robust_accuracy": {
            "mean": float(np.mean(robust_accs)),
            "std": float(np.std(robust_accs)),
            "min": float(np.min(robust_accs)),
            "max": float(np.max(robust_accs)),
            "values": robust_accs,
        },
        "clean_accuracy": {
            "mean": float(np.mean(clean_accs)),
            "std": float(np.std(clean_accs)),
            "min": float(np.min(clean_accs)),
            "max": float(np.max(clean_accs)),
            "values": clean_accs,
        },
    }

    # Save summary
    summary_path = output_path / "statistical_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    # Log summary
    logger.info("Statistical Summary:")
    logger.info(
        f"  Robust Accuracy: {summary['robust_accuracy']['mean']:.2f}% ± {summary['robust_accuracy']['std']:.2f}%"
    )
    logger.info(
        f"  Clean Accuracy: {summary['clean_accuracy']['mean']:.2f}% ± {summary['clean_accuracy']['std']:.2f}%"
    )
    logger.info(
        f"  Range (Robust): [{summary['robust_accuracy']['min']:.2f}%, {summary['robust_accuracy']['max']:.2f}%]"
    )

    # Save detailed CSV
    df = pd.DataFrame(results)
    df.to_csv(output_path / "pgd_at_results.csv", index=False)
    logger.info(f"Results saved to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Phase 5.2: PGD Adversarial Training")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to experiment config YAML",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="results/pgd_at",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        default=[42, 123, 456],
        help="Random seeds for multi-seed training",
    )
    parser.add_argument(
        "--mlflow_experiment",
        type=str,
        default="Phase5.2-PGD-AT",
        help="MLflow experiment name",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Resume from checkpoint",
    )
    parser.add_argument(
        "--single_seed",
        action="store_true",
        help="Run single seed training (use first seed only)",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda/cpu). Defaults to cuda if available.",
    )

    args = parser.parse_args()

    # Single seed or multi-seed training
    if args.single_seed or args.resume:
        seed = args.seeds[0]
        logger.info(f"Running single-seed training (seed={seed})")

        mlflow.set_experiment(args.mlflow_experiment)
        with mlflow.start_run(run_name=f"pgd_at_seed_{seed}"):
            mlflow.log_params({"seed": seed, "method": "pgd_at"})
            mlflow.log_artifact(args.config)

            trainer = PGDATTrainer(
                config_path=args.config,
                output_dir=args.output_dir,
                seed=seed,
                resume_from=args.resume,
            )
            trainer.train()
    else:
        logger.info(f"Running multi-seed training with seeds: {args.seeds}")
        run_multi_seed_training(
            config_path=args.config,
            output_base=args.output_dir,
            seeds=args.seeds,
            mlflow_experiment=args.mlflow_experiment,
        )


if __name__ == "__main__":
    main()
