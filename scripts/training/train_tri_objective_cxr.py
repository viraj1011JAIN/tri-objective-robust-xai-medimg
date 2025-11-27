#!/usr/bin/env python3
"""
Tri-Objective Training Script for Multi-Label Chest X-Ray Classification - Phase 7.6
====================================================================================

Production-grade training script adapting the tri-objective framework for multi-label
classification where each chest X-ray can have multiple disease labels simultaneously.

Key Adaptations:
1. Multi-label task loss (BCE/Focal instead of CE)
2. Multi-label metrics (macro/micro AUROC, Hamming loss)
3. Per-class threshold optimization
4. Same robustness and explanation losses (architecture-agnostic)

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Date: November 27, 2025
Target: A1+ Grade, Publication-Ready

Usage:
    python scripts/training/train_tri_objective_cxr.py \\
        --config configs/experiments/tri_objective_cxr.yaml \\
        --seed 42 \\
        --gpu 0
"""

import argparse
import sys
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import yaml
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from src.attacks import FGSMAttack, PGDAttack
from src.datasets import ChestXRayDataset
from src.evaluation.multilabel_metrics import compute_multilabel_auroc
from src.losses import ExplanationLoss, TRADESLoss
from src.losses.multi_label_task_loss import MultiLabelTaskLoss, OptimalThresholdFinder
from src.models import ResNet50Classifier
from src.utils.logging_utils import setup_logger
from src.utils.reproducibility import set_seed
from src.xai import ConceptBank, GradCAM

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)


class TriObjectiveMultiLabelLoss(nn.Module):
    """Combined tri-objective loss for multi-label classification."""

    def __init__(
        self,
        task_loss: MultiLabelTaskLoss,
        robust_loss: TRADESLoss,
        explanation_loss: ExplanationLoss,
        lambda_rob: float,
        lambda_expl: float,
    ):
        super().__init__()
        self.task_loss = task_loss
        self.robust_loss = robust_loss
        self.explanation_loss = explanation_loss
        self.lambda_rob = lambda_rob
        self.lambda_expl = lambda_expl

    def forward(
        self,
        model: nn.Module,
        images: torch.Tensor,
        labels: torch.Tensor,
        pgd_attack: PGDAttack,
        fgsm_attack: FGSMAttack,
        gradcam: GradCAM,
        concept_bank: ConceptBank,
    ) -> Dict[str, torch.Tensor]:
        """Compute tri-objective loss for multi-label."""
        # Task loss
        outputs = model(images)
        loss_task = self.task_loss(outputs, labels)

        # Robustness loss (TRADES with KL on sigmoid)
        images_adv = pgd_attack.perturb(model, images, labels)
        outputs_adv = model(images_adv)

        probs_clean = torch.sigmoid(outputs)
        probs_adv = torch.sigmoid(outputs_adv)

        eps = 1e-8
        kl_div = probs_clean * torch.log((probs_clean + eps) / (probs_adv + eps)) + (
            1 - probs_clean
        ) * torch.log((1 - probs_clean + eps) / (1 - probs_adv + eps))
        loss_robustness = kl_div.mean()

        # Explanation loss (same as dermoscopy)
        loss_dict_expl = self.explanation_loss(
            model=model,
            images=images,
            labels=labels,
            fgsm_attack=fgsm_attack,
            gradcam=gradcam,
            concept_bank=concept_bank,
        )

        loss_stability = loss_dict_expl["loss_stability"]
        loss_concept = loss_dict_expl["loss_concept"]

        # Total loss
        loss_total = (
            loss_task
            + self.lambda_rob * loss_robustness
            + self.lambda_expl * (loss_stability + loss_concept)
        )

        return {
            "loss_total": loss_total,
            "loss_task": loss_task,
            "loss_robustness": loss_robustness,
            "loss_stability": loss_stability,
            "loss_concept": loss_concept,
        }


class TriObjectiveCXRTrainer:
    """Trainer for tri-objective multi-label chest X-ray classification."""

    def __init__(
        self, config: Dict[str, Any], seed: int, device: torch.device, logger: Any
    ):
        """Initialize trainer."""
        self.config = config
        self.seed = seed
        self.device = device
        self.logger = logger

        # Training state
        self.current_epoch = 0
        self.global_step = 0
        self.best_metric = -np.inf
        self.patience_counter = 0

        # Initialize components
        self._init_data_loaders()
        self._init_model()
        self._init_loss()
        self._init_optimizer()
        self._init_scheduler()
        self._init_attacks()
        self._init_xai_components()

        # Threshold finder
        self.threshold_finder = OptimalThresholdFinder(
            num_classes=config["model"]["num_classes"]
        )

        # Mixed precision
        self.use_amp = config["training"]["mixed_precision"]
        self.scaler = GradScaler(enabled=self.use_amp)

        self.grad_accum_steps = config["training"]["gradient_accumulation_steps"]

        self.logger.info("✓ Tri-objective CXR trainer initialized")

    def _init_data_loaders(self) -> None:
        """Initialize data loaders for multi-label CXR."""
        self.logger.info("Initializing CXR data loaders...")

        dataset_config = self.config["dataset"]
        training_config = self.config["training"]

        # Training dataset
        train_dataset = ChestXRayDataset(
            root=dataset_config["root_dir"],
            split="train",
            csv_path=f"{dataset_config['root_dir']}/metadata.csv",
        )

        # Validation dataset
        val_dataset = ChestXRayDataset(
            root=dataset_config["root_dir"],
            split="val",
            csv_path=f"{dataset_config['root_dir']}/metadata.csv",
        )

        # Test dataset
        test_dataset = ChestXRayDataset(
            root=dataset_config["root_dir"],
            split="test",
            csv_path=f"{dataset_config['root_dir']}/metadata.csv",
        )

        # Data loaders
        self.train_loader = DataLoader(
            train_dataset,
            batch_size=training_config["batch_size"],
            shuffle=True,
            num_workers=training_config["num_workers"],
            pin_memory=training_config["pin_memory"],
            drop_last=True,
        )

        self.val_loader = DataLoader(
            val_dataset,
            batch_size=training_config["batch_size"],
            shuffle=False,
            num_workers=training_config["num_workers"],
            pin_memory=training_config["pin_memory"],
        )

        self.test_loader = DataLoader(
            test_dataset,
            batch_size=training_config["batch_size"],
            shuffle=False,
            num_workers=training_config["num_workers"],
            pin_memory=training_config["pin_memory"],
        )

        self.logger.info(
            f"✓ Data loaders ready: "
            f"Train={len(train_dataset)}, "
            f"Val={len(val_dataset)}, "
            f"Test={len(test_dataset)}"
        )

    def _init_model(self) -> None:
        """Initialize model for multi-label."""
        self.logger.info("Initializing ResNet-50 for multi-label...")

        model_config = self.config["model"]

        self.model = ResNet50Classifier(
            num_classes=model_config["num_classes"],
            pretrained=model_config["pretrained"],
            dropout=model_config["dropout"],
            multilabel=True,
        ).to(self.device)

        self.logger.info(f"✓ Model initialized: ResNet-50 (Multi-Label, 14 classes)")

    def _init_loss(self) -> None:
        """Initialize tri-objective loss for multi-label."""
        self.logger.info("Initializing tri-objective loss (multi-label)...")

        loss_config = self.config["loss"]

        # Task loss
        task_loss = MultiLabelTaskLoss(
            num_classes=self.config["model"]["num_classes"],
            positive_rates=self.config["dataset"]["positive_rates"],
            use_focal=loss_config["task"]["use_focal"],
            focal_gamma=loss_config["task"]["focal_gamma"],
            focal_alpha=loss_config["task"]["focal_alpha"],
        )

        # Robustness loss
        robust_loss = TRADESLoss(
            beta=loss_config["robustness"]["beta"],
            epsilon=loss_config["robustness"]["epsilon"],
            pgd_steps=loss_config["robustness"]["pgd_steps"],
        )

        # Explanation loss
        explanation_loss = ExplanationLoss(
            gamma=loss_config["explanation"]["gamma"],
            ssim_config=loss_config["explanation"]["stability"],
            concept_config=loss_config["explanation"]["concept"],
            device=self.device,
        )

        # Combined loss
        self.criterion = TriObjectiveMultiLabelLoss(
            task_loss=task_loss,
            robust_loss=robust_loss,
            explanation_loss=explanation_loss,
            lambda_rob=loss_config["lambda_rob"],
            lambda_expl=loss_config["lambda_expl"],
        )

        self.logger.info(
            f"✓ Tri-objective loss initialized (λ_rob={loss_config['lambda_rob']}, λ_expl={loss_config['lambda_expl']})"
        )

    def _init_optimizer(self) -> None:
        """Initialize optimizer."""
        opt_config = self.config["training"]["optimizer"]

        self.optimizer = optim.AdamW(
            self.model.parameters(),
            lr=opt_config["lr"],
            weight_decay=opt_config["weight_decay"],
            betas=opt_config["betas"],
        )

        self.logger.info(f"✓ Optimizer: AdamW (lr={opt_config['lr']})")

    def _init_scheduler(self) -> None:
        """Initialize LR scheduler."""
        sched_config = self.config["training"]["scheduler"]

        self.scheduler = optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=sched_config["T_max"], eta_min=sched_config["eta_min"]
        )

        self.logger.info(f"✓ LR Scheduler: CosineAnnealingLR")

    def _init_attacks(self) -> None:
        """Initialize adversarial attacks."""
        rob_config = self.config["loss"]["robustness"]
        expl_config = self.config["loss"]["explanation"]["stability"]

        self.pgd_attack = PGDAttack(
            epsilon=rob_config["epsilon"],
            step_size=rob_config["pgd_step_size"],
            num_steps=rob_config["pgd_steps"],
            random_start=rob_config["pgd_random_start"],
            norm=rob_config["norm"],
        )

        self.fgsm_attack = FGSMAttack(epsilon=expl_config["epsilon_adv"], norm="Linf")

        self.logger.info(f"✓ Attacks initialized")

    def _init_xai_components(self) -> None:
        """Initialize XAI components."""
        feature_layers = self.config["model"]["feature_layers"]
        concept_config = self.config["loss"]["explanation"]["concept"]

        self.gradcam = GradCAM(model=self.model, target_layer=feature_layers[-1])

        self.concept_bank = ConceptBank(
            precomputed_cavs_dir=concept_config["precomputed_cavs"],
            artifact_concepts=concept_config["artifact_concepts"],
            medical_concepts=concept_config["medical_concepts"],
            device=self.device,
        )

        self.logger.info(f"✓ XAI components initialized")

    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()

        metrics = {
            "loss_total": 0.0,
            "loss_task": 0.0,
            "loss_robustness": 0.0,
            "loss_stability": 0.0,
            "loss_concept": 0.0,
            "num_samples": 0,
        }

        pbar = tqdm(
            self.train_loader,
            desc=f"Epoch {epoch}/{self.config['training']['num_epochs']}",
            disable=not self.config["monitoring"]["progress_bar"],
        )

        for batch_idx, batch in enumerate(pbar):
            images = batch["image"].to(self.device)
            labels = batch["labels"].to(self.device)
            batch_size = images.size(0)

            with autocast(enabled=self.use_amp):
                loss_dict = self.criterion(
                    model=self.model,
                    images=images,
                    labels=labels,
                    pgd_attack=self.pgd_attack,
                    fgsm_attack=self.fgsm_attack,
                    gradcam=self.gradcam,
                    concept_bank=self.concept_bank,
                )

                loss = loss_dict["loss_total"] / self.grad_accum_steps

            self.scaler.scale(loss).backward()

            if (batch_idx + 1) % self.grad_accum_steps == 0:
                if self.config["training"]["gradient_clip_norm"] > 0:
                    self.scaler.unscale_(self.optimizer)
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(),
                        self.config["training"]["gradient_clip_norm"],
                    )

                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()

            # Update metrics
            metrics["loss_total"] += loss_dict["loss_total"].item() * batch_size
            metrics["loss_task"] += loss_dict["loss_task"].item() * batch_size
            metrics["loss_robustness"] += (
                loss_dict["loss_robustness"].item() * batch_size
            )
            metrics["loss_stability"] += loss_dict["loss_stability"].item() * batch_size
            metrics["loss_concept"] += loss_dict["loss_concept"].item() * batch_size
            metrics["num_samples"] += batch_size

            pbar.set_postfix(
                {
                    "loss": f"{loss_dict['loss_total'].item():.4f}",
                    "lr": f"{self.optimizer.param_groups[0]['lr']:.6f}",
                }
            )

            self.global_step += 1

        # Compute epoch metrics
        num_samples = metrics["num_samples"]
        return {
            "train_loss_total": metrics["loss_total"] / num_samples,
            "train_loss_task": metrics["loss_task"] / num_samples,
            "train_loss_robustness": metrics["loss_robustness"] / num_samples,
            "train_loss_stability": metrics["loss_stability"] / num_samples,
            "train_loss_concept": metrics["loss_concept"] / num_samples,
        }

    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model (multi-label)."""
        self.model.eval()

        all_probs = []
        all_labels = []

        with torch.no_grad():
            for batch in tqdm(
                self.val_loader,
                desc="Validation",
                disable=not self.config["monitoring"]["progress_bar"],
            ):
                images = batch["image"].to(self.device)
                labels = batch["labels"].to(self.device)

                outputs = self.model(images)
                probs = torch.sigmoid(outputs)

                all_probs.extend(probs.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

        all_probs = np.array(all_probs)
        all_labels = np.array(all_labels)

        # Optimize thresholds
        if self.config["validation"]["optimize_threshold"]:
            self.threshold_finder.find_optimal_thresholds(all_probs, all_labels)

        # Compute AUROC
        auroc_results = compute_multilabel_auroc(
            y_true=all_labels,
            y_prob=all_probs,
            class_names=self.config["dataset"]["class_names"],
        )

        val_metrics = {
            "val_macro_auroc": auroc_results["macro_auroc"],
            "val_micro_auroc": auroc_results["micro_auroc"],
        }

        # Combined score
        val_metrics["val_combined_score"] = 0.3 * val_metrics["val_macro_auroc"]

        return val_metrics

    def save_checkpoint(
        self, epoch: int, metrics: Dict[str, float], is_best: bool = False
    ) -> None:
        """Save checkpoint."""
        checkpoint_dir = Path(self.config["checkpointing"]["save_dir"])
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scheduler_state_dict": self.scheduler.state_dict(),
            "metrics": metrics,
            "config": self.config,
            "seed": self.seed,
            "class_thresholds": self.threshold_finder.get_thresholds(),
        }

        if epoch % self.config["checkpointing"]["save_every_n_epochs"] == 0:
            checkpoint_format = self.config["checkpointing"]["checkpoint_format"]
            checkpoint_path = checkpoint_dir / checkpoint_format.format(
                epoch=epoch, seed=self.seed
            )
            torch.save(checkpoint, checkpoint_path)
            self.logger.info(f"✓ Checkpoint saved: {checkpoint_path}")

        if is_best:
            best_format = self.config["checkpointing"]["best_checkpoint_format"]
            best_path = checkpoint_dir / best_format.format(seed=self.seed)
            torch.save(checkpoint, best_path)
            self.logger.info(f"✓ Best checkpoint saved: {best_path}")

    def train(self) -> Dict[str, Any]:
        """Main training loop."""
        self.logger.info("=" * 80)
        self.logger.info("STARTING TRI-OBJECTIVE TRAINING (MULTI-LABEL CXR)")
        self.logger.info("=" * 80)

        num_epochs = self.config["training"]["num_epochs"]
        history = {"train": [], "val": []}

        for epoch in range(1, num_epochs + 1):
            self.current_epoch = epoch

            # Training
            train_metrics = self.train_epoch(epoch)
            history["train"].append(train_metrics)

            # Validation
            if epoch % self.config["validation"]["eval_every_n_epochs"] == 0:
                val_metrics = self.validate(epoch)
                history["val"].append(val_metrics)

                all_metrics = {**train_metrics, **val_metrics}

                # MLflow logging
                if self.config["mlflow"]["log_metrics"]:
                    mlflow.log_metrics(all_metrics, step=epoch)

                # Check improvement
                current_metric = val_metrics["val_combined_score"]
                is_best = current_metric > self.best_metric

                if is_best:
                    self.best_metric = current_metric
                    self.patience_counter = 0
                    self.logger.info(
                        f"✨ New best model! Combined score: {current_metric:.4f}"
                    )
                else:
                    self.patience_counter += 1

                # Early stopping
                if self.config["training"]["early_stopping"]["enabled"]:
                    if (
                        self.patience_counter
                        >= self.config["training"]["early_stopping"]["patience"]
                    ):
                        self.logger.info(f"⚠️  Early stopping triggered")
                        break

                # Save checkpoint
                self.save_checkpoint(epoch, all_metrics, is_best=is_best)

                self.logger.info(
                    f"Epoch {epoch}/{num_epochs}: "
                    f"Loss={train_metrics['train_loss_total']:.4f}, "
                    f"Macro AUROC={val_metrics['val_macro_auroc']:.4f}"
                )

            # LR scheduling
            self.scheduler.step()

        self.logger.info("=" * 80)
        self.logger.info(f"TRAINING COMPLETED")
        self.logger.info(f"Best combined score: {self.best_metric:.4f}")
        self.logger.info("=" * 80)

        return history


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Tri-Objective Training for Multi-Label Chest X-Ray"
    )
    parser.add_argument("--config", type=str, required=True, help="Config YAML file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--gpu", type=int, default=0, help="GPU device ID")
    parser.add_argument("--debug", action="store_true", help="Debug mode")

    args = parser.parse_args()

    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    config["reproducibility"]["seed"] = args.seed

    if args.debug:
        config["training"]["num_epochs"] = 5
        config["training"]["batch_size"] = 8

    # Set up logging
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = Path("results/logs/training_cxr")
    log_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(
        name="tri_objective_cxr_training",
        log_file=log_dir / f"tri_obj_cxr_seed{args.seed}_{timestamp}.log",
    )

    # Set device
    if args.gpu >= 0 and torch.cuda.is_available():
        device = torch.device(f"cuda:{args.gpu}")
        logger.info(f"Using GPU: {torch.cuda.get_device_name(args.gpu)}")
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")

    # Set reproducibility
    set_seed(args.seed)
    logger.info(f"✓ Random seed set: {args.seed}")

    # Initialize MLflow
    mlflow.set_tracking_uri(config["mlflow"]["tracking_uri"])
    mlflow.set_experiment(config["mlflow"]["experiment_name"])

    run_name = config["mlflow"]["run_name_format"].format(seed=args.seed)

    with mlflow.start_run(run_name=run_name):
        # Log parameters
        if config["mlflow"]["log_params"]:
            mlflow.log_params(
                {
                    "seed": args.seed,
                    "model": config["model"]["architecture"],
                    "dataset": config["dataset"]["name"],
                    "task_type": "multi-label",
                    "batch_size": config["training"]["batch_size"],
                    "num_epochs": config["training"]["num_epochs"],
                    "learning_rate": config["training"]["optimizer"]["lr"],
                    "lambda_rob": config["loss"]["lambda_rob"],
                    "lambda_expl": config["loss"]["lambda_expl"],
                }
            )

        # Initialize trainer
        trainer = TriObjectiveCXRTrainer(
            config=config, seed=args.seed, device=device, logger=logger
        )

        # Train
        history = trainer.train()

        logger.info("✅ Training completed successfully!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\n❌ FATAL ERROR: {str(e)}")
        import traceback

        traceback.print_exc()
        sys.exit(1)
