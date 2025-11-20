#!/usr/bin/env python3
# coverage: ignore file
"""
Tri-Objective Training Script
==============================

Main entry point for training tri-objective models that simultaneously optimize:
1. Task performance (classification accuracy)
2. Adversarial robustness (via TRADES-style loss)
3. Explanation stability (via SSIM + TCAV, handled inside the loss + trainer)

This script is aligned with the dissertation tri-objective framework and is
compatible with:

- src/losses/tri_objective.py      (TriObjectiveLoss with lambda_rob / lambda_expl)
- src/train/triobj_training.py     (TriObjectiveTrainer, TrainingConfig)
- configs/experiments/tri_objective/debug.yaml  (CIFAR-10 debug pipeline)

Typical usage (from project root):

  # Debug mode (small CIFAR-10 subset, fast check)
  python scripts/training/train_tri_objective.py \
    --config configs/experiments/tri_objective/debug.yaml \
    --seed 42 \
    --debug

  # Full training (once you add a full config)
  python scripts/training/train_tri_objective.py \
    --config configs/experiments/tri_objective/full.yaml \
    --seed 42

  # Resume from checkpoint
  python scripts/training/train_tri_objective.py \
    --config configs/experiments/tri_objective/full.yaml \
    --seed 42 \
    --resume results/checkpoints/tri_objective/latest_model.pt

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import torch
import torch.nn as nn
import yaml
from torch.optim import AdamW, Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR, _LRScheduler
from torch.utils.data import DataLoader

# -------------------------------------------------------------------------
# Project import path - must be done before project imports
# -------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Now import project modules
from src.losses.tri_objective import TriObjectiveLoss  # noqa: E402
from src.train.triobj_training import TrainingConfig  # noqa: E402
from src.training.triobj_trainer import TriObjectiveTrainer  # noqa: E402
from src.utils.reproducibility import set_seed  # noqa: E402

# -------------------------------------------------------------------------
# Logging
# -------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


# =========================================================================
# ARGUMENTS
# =========================================================================


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Tri-Objective Adversarial Training (CIFAR-10 debug + medical-ready)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Debug mode (CIFAR-10, small subset)
  python scripts/training/train_tri_objective.py \\
    --config configs/experiments/tri_objective/debug.yaml \\
    --seed 42 \\
    --debug

  # Full training (once you have a medical-image config)
  python scripts/training/train_tri_objective.py \\
    --config configs/experiments/tri_objective/full.yaml \\
    --seed 42

  # Resume training
  python scripts/training/train_tri_objective.py \\
    --config configs/experiments/tri_objective/full.yaml \\
    --seed 42 \\
    --resume results/checkpoints/tri_objective/latest_model.pt
        """,
    )
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to config YAML file (e.g., configs/experiments/tri_objective/debug.yaml)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility (default: 42)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode: tiny subset, 2 epochs, frequent logging, MLflow off.",
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Overrides config if specified.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Override checkpoint output directory.",
    )
    return parser.parse_args()


# =========================================================================
# CONFIG LOADING & NORMALISATION
# =========================================================================


def load_raw_config(config_path: str) -> Dict[str, Any]:
    """Load raw configuration from YAML file."""
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    logger.info(f"Loading config from: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def normalize_config(raw_cfg: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalize various config styles into a unified schema:

    - raw_cfg['dataset'] or raw_cfg['data'] -> config['data']
    - raw_cfg['training'] (with num_epochs / lr) -> config['training'] (epochs / learning_rate)
    - raw_cfg['loss'] (lambda_rob, lambda_expl, trades_beta, epsilon, pgd_steps, pgd_alpha)
    - raw_cfg['model'] (optional; defaults provided if missing)
    - raw_cfg['experiment'] (name/output_dir) used for MLflow + checkpoints
    """
    experiment_cfg = raw_cfg.get("experiment", {}) or {}

    # ---- DATA / DATASET ---------------------------------------------------
    ds = raw_cfg.get("dataset", raw_cfg.get("data", {})) or {}
    training_raw = raw_cfg.get("training", {}) or {}
    model_raw = raw_cfg.get("model", {}) or {}

    num_classes = ds.get("num_classes", model_raw.get("num_classes", 10))
    batch_size = ds.get("batch_size", training_raw.get("batch_size", 32))
    num_workers = ds.get("num_workers", training_raw.get("num_workers", 2))
    pin_memory = ds.get("pin_memory", True)
    image_size = ds.get("image_size", 32)

    data_cfg = {
        "name": ds.get("name", "CIFAR10"),
        "data_root": ds.get("data_root", "./data/cifar10"),
        "num_classes": num_classes,
        "batch_size": batch_size,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "image_size": image_size,
        "max_train_batches_debug": ds.get("max_train_batches_debug"),
        "max_val_batches_debug": ds.get("max_val_batches_debug"),
    }

    # ---- MODEL ------------------------------------------------------------
    model_cfg = {
        "architecture": model_raw.get("architecture", "resnet50"),
        "num_classes": num_classes,
        "pretrained": model_raw.get("pretrained", False),
    }

    # ---- TRAINING ---------------------------------------------------------
    epochs = training_raw.get("epochs", training_raw.get("num_epochs", 2))
    lr = training_raw.get("learning_rate", training_raw.get("lr", 1e-4))
    wd = training_raw.get("weight_decay", 1e-4)
    device = training_raw.get("device", "cuda")

    training_cfg = {
        "epochs": epochs,
        "batch_size": batch_size,
        "learning_rate": lr,
        "weight_decay": wd,
        "grad_clip_norm": training_raw.get("grad_clip_norm", 1.0),
        "grad_accum_steps": training_raw.get("grad_accum_steps", 1),
        "mixed_precision": training_raw.get("mixed_precision", True),
        "device": device,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "early_stop_patience": training_raw.get("early_stop_patience", 10),
        "early_stop_metric": training_raw.get("early_stop_metric", "val_loss"),
        "early_stop_mode": training_raw.get("early_stop_mode", "min"),
        "save_freq": training_raw.get("save_freq", 0),
        "checkpoint_dir": training_raw.get(
            "checkpoint_dir",
            experiment_cfg.get("output_dir", "results/checkpoints/tri_objective"),
        ),
        "keep_n_checkpoints": training_raw.get("keep_n_checkpoints", 3),
        "use_mlflow": training_raw.get("use_mlflow", True),
        "mlflow_experiment": training_raw.get(
            "mlflow_experiment", experiment_cfg.get("name", "Tri-Objective-XAI")
        ),
        "mlflow_tracking_uri": training_raw.get("mlflow_tracking_uri"),
        "log_freq": training_raw.get("log_freq", 10),
    }

    # ---- LOSS -------------------------------------------------------------
    loss_raw = raw_cfg.get("loss", {}) or {}
    epsilon = loss_raw.get("epsilon", 8.0 / 255.0)
    pgd_alpha = loss_raw.get("pgd_alpha", epsilon / 4.0)
    loss_cfg = {
        "lambda_rob": loss_raw.get("lambda_rob", 0.3),
        "lambda_expl": loss_raw.get("lambda_expl", 0.1),
        "trades_beta": loss_raw.get("trades_beta", 6.0),
        "epsilon": epsilon,
        "pgd_steps": loss_raw.get("pgd_steps", 7),
        "pgd_alpha": pgd_alpha,
        "expl_freq": loss_raw.get("expl_freq", 1),
        "expl_subsample": loss_raw.get("expl_subsample", 1.0),
    }

    return {
        "experiment": experiment_cfg,
        "data": data_cfg,
        "model": model_cfg,
        "training": training_cfg,
        "loss": loss_cfg,
    }


def apply_debug_overrides(config: Dict[str, Any]) -> Dict[str, Any]:
    """
    Apply debug-mode overrides: tiny subset, minimal epochs, no MLflow, etc.
    Intended to keep runtime manageable while verifying E2E correctness.
    """
    logger.warning("🐛 DEBUG MODE ENABLED - configuration modified for quick testing")

    # Training overrides
    config["training"]["epochs"] = 2
    config["training"]["batch_size"] = min(config["training"]["batch_size"], 16)
    config["training"]["log_freq"] = 1
    config["training"]["save_freq"] = 1
    config["training"]["early_stop_patience"] = None  # disable early stopping
    config["training"]["use_mlflow"] = False  # no MLflow noise in debug

    # Data overrides for loader stability
    config["data"]["num_workers"] = 0
    config["data"]["max_train_batches_debug"] = config["data"].get(
        "max_train_batches_debug", 20
    )
    config["data"]["max_val_batches_debug"] = config["data"].get(
        "max_val_batches_debug", 5
    )

    logger.info("  - Epochs: 2")
    logger.info(f"  - Batch size: {config['training']['batch_size']}")
    logger.info("  - Num workers: 0")
    logger.info("  - MLflow: disabled")
    logger.info("  - Early stopping: disabled")
    logger.info(
        f"  - Debug train batches: {config['data']['max_train_batches_debug']} | "
        f"val batches: {config['data']['max_val_batches_debug']}"
    )

    return config


# =========================================================================
# MODEL / DATA / LOSS / OPTIM COMPONENTS
# =========================================================================


def create_model(config: Dict[str, Any]) -> nn.Module:
    """
    Create model from configuration.

    For the debug CIFAR-10 pipeline we use torchvision.models.resnet50 and
    replace the final classification layer. For medical imaging later, you can
    plug in domain-specific architectures while keeping the trainer unchanged.
    """
    from torchvision.models import resnet50

    model_cfg = config["model"]
    num_classes = model_cfg["num_classes"]

    model = resnet50(pretrained=model_cfg.get("pretrained", False))
    in_features = model.fc.in_features
    model.fc = nn.Linear(in_features, num_classes)

    logger.info(
        f"Created model: {model_cfg['architecture']} with num_classes={num_classes}"
    )
    return model


def _subset_for_debug(dataset, max_batches: Optional[int], batch_size: int):
    """Optionally wrap a dataset with a Subset to limit number of batches."""
    if max_batches is None:
        return dataset
    from torch.utils.data import Subset

    max_samples = batch_size * max_batches
    max_samples = min(max_samples, len(dataset))
    indices = list(range(max_samples))
    return Subset(dataset, indices)


def create_dataloaders(config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
    """
    Create train/val dataloaders.

    For Day 1 debug we use CIFAR-10 with standard augmentations. The same
    structure can later be swapped for medical datasets (NIH CXR, ISIC, etc.)
    without touching the trainer or loss code.
    """
    from torchvision import transforms
    from torchvision.datasets import CIFAR10

    data_cfg = config["data"]
    train_cfg = config["training"]

    data_root = Path(data_cfg["data_root"])
    batch_size = train_cfg["batch_size"]
    num_workers = train_cfg["num_workers"]
    pin_memory = train_cfg["pin_memory"]
    image_size = data_cfg.get("image_size", 32)

    # CIFAR-10 transforms
    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(image_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )
    val_transform = transforms.Compose(
        [
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=(0.4914, 0.4822, 0.4465),
                std=(0.2023, 0.1994, 0.2010),
            ),
        ]
    )

    logger.info(f"Loading CIFAR-10 at: {data_root}")
    train_ds = CIFAR10(
        root=str(data_root), train=True, download=True, transform=train_transform
    )
    val_ds = CIFAR10(
        root=str(data_root), train=False, download=True, transform=val_transform
    )

    # Debug subset limiting
    train_ds = _subset_for_debug(
        train_ds, data_cfg.get("max_train_batches_debug"), batch_size
    )
    val_ds = _subset_for_debug(
        val_ds, data_cfg.get("max_val_batches_debug"), batch_size
    )

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    logger.info("Created CIFAR-10 dataloaders:")
    logger.info(
        f"  - Train: {len(train_ds)} samples, {len(train_loader)} batches "
        f"(batch_size={batch_size})"
    )
    logger.info(
        f"  - Val:   {len(val_ds)} samples, {len(val_loader)} batches "
        f"(batch_size={batch_size})"
    )

    return train_loader, val_loader


def create_criterion(config: Dict[str, Any], device: torch.device) -> TriObjectiveLoss:
    """
    Create the TriObjectiveLoss consistent with the dissertation:

        L_total = L_task + lambda_rob * L_rob + lambda_expl * L_expl

    where:
      - L_task is calibrated cross-entropy
      - L_rob is TRADES-style robustness loss
      - L_expl is explanation stability loss (SSIM + optional TCAV)
    """
    loss_cfg = config["loss"]
    model_cfg = config["model"]

    criterion = TriObjectiveLoss(
        lambda_rob=loss_cfg["lambda_rob"],
        lambda_expl=loss_cfg["lambda_expl"],
        num_classes=model_cfg["num_classes"],
        class_weights=None,
        beta=loss_cfg["trades_beta"],
        epsilon=loss_cfg["epsilon"],
        pgd_steps=loss_cfg["pgd_steps"],
        # gamma, artifact_cavs, medical_cavs use defaults for now
        expl_freq=loss_cfg.get("expl_freq", 1),
        expl_subsample=loss_cfg.get("expl_subsample", 1.0),
    ).to(device)

    logger.info("Created TriObjectiveLoss with:")
    logger.info(f"  - lambda_rob:  {loss_cfg['lambda_rob']}")
    logger.info(f"  - lambda_expl: {loss_cfg['lambda_expl']}")
    logger.info(f"  - beta (TRADES): {loss_cfg['trades_beta']}")
    logger.info(f"  - epsilon: {loss_cfg['epsilon']}")
    logger.info(f"  - pgd_steps: {loss_cfg['pgd_steps']}")
    return criterion


def create_optimizer(model: nn.Module, config: Dict[str, Any]) -> Optimizer:
    """Create AdamW optimizer."""
    train_cfg = config["training"]
    optimizer = AdamW(
        model.parameters(),
        lr=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
    )
    logger.info("Created AdamW optimizer:")
    logger.info(f"  - lr: {train_cfg['learning_rate']:.2e}")
    logger.info(f"  - weight_decay: {train_cfg['weight_decay']:.2e}")
    return optimizer


def create_scheduler(
    optimizer: Optimizer, config: Dict[str, Any]
) -> Optional[_LRScheduler]:
    """Create LR scheduler (currently CosineAnnealingLR)."""
    train_cfg = config["training"]
    sched_cfg = train_cfg.get("scheduler", {}) or {}
    sched_type = sched_cfg.get("type", "cosine")

    if sched_type == "cosine":
        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=train_cfg["epochs"],
            eta_min=sched_cfg.get("min_lr", 1e-6),
        )
        logger.info(
            f"Created CosineAnnealingLR scheduler (T_max={train_cfg['epochs']}, "
            f"min_lr={sched_cfg.get('min_lr', 1e-6):.2e})"
        )
        return scheduler

    logger.info("No valid scheduler configured (or type not recognised); skipping.")
    return None


def create_training_config(
    config: Dict[str, Any],
    seed: int,
    device_str: str,
) -> TrainingConfig:
    """Create TrainingConfig dataclass from normalized config."""
    train_cfg = config["training"]
    loss_cfg = config["loss"]
    data_cfg = config["data"]

    return TrainingConfig(
        # Training hyperparameters
        epochs=train_cfg["epochs"],
        batch_size=train_cfg["batch_size"],
        learning_rate=train_cfg["learning_rate"],
        weight_decay=train_cfg["weight_decay"],
        grad_clip_norm=train_cfg.get("grad_clip_norm", 1.0),
        grad_accum_steps=train_cfg.get("grad_accum_steps", 1),
        mixed_precision=train_cfg.get("mixed_precision", True),
        seed=seed,
        device=device_str,
        num_workers=data_cfg["num_workers"],
        pin_memory=data_cfg["pin_memory"],
        # Loss weights (for logging)
        w_task=1.0,
        w_rob=loss_cfg["lambda_rob"],
        w_expl=loss_cfg["lambda_expl"],
        # Explanation settings (how often/what fraction)
        expl_freq=loss_cfg.get("expl_freq", 1),
        expl_subsample=loss_cfg.get("expl_subsample", 1.0),
        # Early stopping
        early_stop_patience=train_cfg.get("early_stop_patience"),
        early_stop_metric=train_cfg.get("early_stop_metric", "val_loss"),
        early_stop_mode=train_cfg.get("early_stop_mode", "min"),
        # Checkpointing
        save_freq=train_cfg.get("save_freq", 0),
        checkpoint_dir=train_cfg.get(
            "checkpoint_dir", "results/checkpoints/tri_objective"
        ),
        keep_n_checkpoints=train_cfg.get("keep_n_checkpoints", 3),
        # MLflow
        use_mlflow=train_cfg.get("use_mlflow", True),
        mlflow_experiment=train_cfg.get("mlflow_experiment", "Tri-Objective-XAI"),
        mlflow_run_name=f"seed_{seed}",
        mlflow_tracking_uri=train_cfg.get("mlflow_tracking_uri"),
        # Monitoring
        log_freq=train_cfg.get("log_freq", 10),
        verbose=True,
    )


# =========================================================================
# PRINT HELPERS
# =========================================================================


def print_banner(title: str) -> None:
    print("\n" + "=" * 70)
    print(title.center(70))
    print("=" * 70 + "\n")


def print_config_summary(config: Dict[str, Any], args: argparse.Namespace) -> None:
    print_banner("TRI-OBJECTIVE TRAINING CONFIGURATION")
    print(f"📄 Config File:        {args.config}")
    print(f"🎲 Random Seed:        {args.seed}")
    print(f"🐛 Debug Mode:         {'YES' if args.debug else 'NO'}")
    print(f"💾 Resume From:        {args.resume if args.resume else 'None (scratch)'}")
    print()
    print("🔧 Model:")
    print(f"   Architecture:       {config['model']['architecture']}")
    print(f"   Num Classes:        {config['model']['num_classes']}")
    print(f"   Pretrained:         {config['model'].get('pretrained', False)}")
    print()
    print("📊 Training:")
    print(f"   Epochs:             {config['training']['epochs']}")
    print(f"   Batch Size:         {config['training']['batch_size']}")
    print(f"   Learning Rate:      {config['training']['learning_rate']:.2e}")
    print(f"   Device:             {config['training']['device']}")
    print(f"   Mixed Precision:    {config['training'].get('mixed_precision', True)}")
    print()
    print("⚖️  Loss Weights:")
    print(f"   lambda_rob:         {config['loss']['lambda_rob']}")
    print(f"   lambda_expl:        {config['loss']['lambda_expl']}")
    print()
    print("💾 Output:")
    checkpoint_dir = config["training"].get(
        "checkpoint_dir", "results/checkpoints/tri_objective"
    )
    print(f"   Checkpoint Dir:     {checkpoint_dir}")
    print(f"   MLflow Enabled:     {config['training'].get('use_mlflow', True)}")
    print("\n" + "=" * 70 + "\n")


# =========================================================================
# MAIN
# =========================================================================


def main() -> None:
    args = parse_args()

    # Load and normalise configuration
    raw_cfg = load_raw_config(args.config)
    config = normalize_config(raw_cfg)

    # Debug overrides
    if args.debug:
        config = apply_debug_overrides(config)

    # Override device if passed on CLI
    if args.device is not None:
        config["training"]["device"] = args.device
        logger.info(f"Device overridden via CLI to: {args.device}")

    # Override checkpoint dir if passed
    if args.output_dir is not None:
        config["training"]["checkpoint_dir"] = args.output_dir
        logger.info(f"Checkpoint dir overridden via CLI to: {args.output_dir}")

    # Reproducibility
    set_seed(args.seed)
    logger.info(f"Set random seed to: {args.seed}")

    # Determine device
    device_str = config["training"]["device"]
    if device_str == "cuda" and not torch.cuda.is_available():
        logger.warning("CUDA requested but not available, falling back to CPU.")
        device_str = "cpu"
    device = torch.device(device_str)
    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(
            f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB"
        )

    # Summary
    print_config_summary(config, args)

    # Create model
    print_banner("CREATING MODEL")
    model = create_model(config).to(device)
    n_params = sum(p.numel() for p in model.parameters())
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model parameters: {n_params:,} total, {n_trainable:,} trainable")

    # Data
    print_banner("LOADING DATA")
    train_loader, val_loader = create_dataloaders(config)

    # Loss
    print_banner("CREATING TRI-OBJECTIVE LOSS")
    criterion = create_criterion(config, device)

    # Optimizer & scheduler
    print_banner("CREATING OPTIMIZER")
    optimizer = create_optimizer(model, config)

    print_banner("CREATING SCHEDULER")
    scheduler = create_scheduler(optimizer, config)

    # TrainingConfig
    training_config = create_training_config(config, args.seed, device_str)

    # Trainer
    print_banner("INITIALISING TRAINER")
    trainer = TriObjectiveTrainer(
        model=model,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        config=training_config,
    )

    # Resume checkpoint if requested
    if args.resume:
        print_banner("RESUMING FROM CHECKPOINT")
        logger.info(f"Loading checkpoint: {args.resume}")
        trainer.load_checkpoint(args.resume)

    # Train
    print_banner("TRAINING START")
    try:
        metrics = trainer.fit(train_loader, val_loader)

        print_banner("TRAINING COMPLETE")
        summary = metrics.get_summary()
        print("✅ Training completed successfully!\n")
        print("📊 Results Summary:")
        print(
            f"   Best Val Loss:       {summary['best_val_loss']:.4f} "
            f"(epoch {summary['best_epoch']})"
        )
        print(f"   Best Val Accuracy:   {summary['best_val_acc']:.4f}")
        print(f"   Final Train Loss:    {summary['final_train_loss']:.4f}")
        print(f"   Final Val Loss:      {summary['final_val_loss']:.4f}")
        print(f"   Total Epochs:        {summary['total_epochs']}")
        print(f"   Avg Epoch Time:      {summary['avg_epoch_time']:.2f}s")
        print()
        print(f"💾 Checkpoints saved to: {training_config.checkpoint_dir}")
        print(
            f"   - Best model:   {Path(training_config.checkpoint_dir) / 'best_model.pt'}"
        )
        print(
            f"   - Latest model: {Path(training_config.checkpoint_dir) / 'latest_model.pt'}"
        )
        if training_config.use_mlflow:
            print(f"\n📊 MLflow experiment: {training_config.mlflow_experiment}")
        print("\n" + "=" * 70 + "\n")

    except KeyboardInterrupt:
        logger.warning("Training interrupted by user.")
        print_banner("TRAINING INTERRUPTED")
        print("You can resume training with:")
        print("  python scripts/training/train_tri_objective.py \\")
        print(f"    --config {args.config} \\")
        print(f"    --seed {args.seed} \\")
        resume_path = Path(training_config.checkpoint_dir) / "latest_model.pt"
        print(f"    --resume {resume_path}")
        print()

    except Exception as e:
        logger.error(f"Training failed with error: {e}", exc_info=True)
        raise


if __name__ == "__main__":
    main()
