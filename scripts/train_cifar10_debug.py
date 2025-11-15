#!/usr/bin/env python3
"""
CIFAR-10 Debug Training Script with Full MLflow Integration
==========================================================

High-quality training pipeline to validate MLflow + PyTorch infrastructure.
Designed as an MSc A1-grade example:

- Strong engineering practices (logging, error handling, configuration)
- Reproducible training (seeds, deterministic flags)
- Rich MLflow logging (hyperparameters, metrics, artifacts)
- Clean model architecture suitable as a template for other datasets
- Uses standardized MLflow helper utilities from src.utils.mlflow_utils

It stays lightweight enough to run on CPU with small subsets by default,
so you can use it as a fast smoke test for your tri-objective project.

Usage:
    python scripts/train_cifar10_debug.py --epochs 5
    python scripts/train_cifar10_debug.py --subset-size 2048 --epochs 3 --batch-size 64
    python scripts/train_cifar10_debug.py --help

After training:
    mlflow ui --backend-store-uri "file:./mlruns"
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import socket
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, Optional, Tuple

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.metrics import classification_report, confusion_matrix
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Import standardized MLflow utilities
from src.utils.mlflow_utils import build_experiment_and_run_name, init_mlflow

# -----------------------------------------------------------------------------
# Logging setup
# -----------------------------------------------------------------------------

LOG_FILE_NAME = "train_cifar10_debug.log"


def configure_logging(log_dir: Path) -> logging.Logger:
    """Configure logger with stream + file handlers.

    Args:
        log_dir: Directory to store log files

    Returns:
        Configured logger instance
    """
    log_dir.mkdir(parents=True, exist_ok=True)
    log_file = log_dir / LOG_FILE_NAME

    logger = logging.getLogger(__name__)
    logger.setLevel(logging.INFO)

    # Avoid duplicate handlers if script is re-run
    if logger.handlers:
        return logger

    formatter = logging.Formatter(
        fmt="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    file_handler = logging.FileHandler(str(log_file))
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


# -----------------------------------------------------------------------------
# Configuration dataclass
# -----------------------------------------------------------------------------


@dataclass
class TrainingConfig:
    """Training configuration with sensible defaults for CIFAR-10 debug runs.

    Attributes:
        model: Model architecture name
        dropout: Dropout rate for regularization
        epochs: Number of training epochs
        batch_size: Batch size for training
        lr: Initial learning rate
        weight_decay: L2 regularization weight
        optimizer: Optimizer type (adam, sgd, adamw)
        scheduler: Learning rate scheduler (cosine, step, none)
        log_interval: Batches between training logs
        grad_clip_norm: Gradient clipping threshold (None to disable)
        data_root: Root directory for CIFAR-10 data
        subset_size: Training subset size for fast debug (None = full dataset)
        num_workers: DataLoader workers for parallel loading
        seed: Random seed for reproducibility
        checkpoint_dir: Directory for saving checkpoints
        results_artifact_dir: MLflow artifact path for results
        checkpoints_artifact_dir: MLflow artifact path for checkpoints
        dataset_name: Dataset identifier for MLflow naming
        objective: Training objective for MLflow naming
        run_tag: Optional tag for MLflow run name
        tracking_uri: MLflow tracking URI (None = use default)
        max_epochs_without_improvement: Early stopping patience
    """

    # Model / training
    model: str = "SimpleCIFARNet"
    dropout: float = 0.5
    epochs: int = 5
    batch_size: int = 128
    lr: float = 1e-3
    weight_decay: float = 1e-4
    optimizer: str = "adam"  # ["adam", "sgd", "adamw"]
    scheduler: str = "cosine"  # ["cosine", "step", "none"]
    log_interval: int = 20
    grad_clip_norm: Optional[float] = 5.0

    # Data
    data_root: str = "./data/cifar10"
    subset_size: Optional[int] = None  # e.g. 2048 for debug
    num_workers: int = 4

    # Reproducibility
    seed: int = 42

    # Checkpoints / artifacts
    checkpoint_dir: str = "./results/checkpoints/cifar10_debug"
    results_artifact_dir: str = "results"
    checkpoints_artifact_dir: str = "checkpoints"

    # MLflow (using standardized naming convention)
    dataset_name: str = "CIFAR10-debug"
    objective: str = "baseline"
    run_tag: Optional[str] = None  # e.g., "seed-42", "subset-2k"
    tracking_uri: Optional[str] = None  # None = use default from mlflow_utils

    # Misc
    max_epochs_without_improvement: int = 10  # early stopping patience


# -----------------------------------------------------------------------------
# Model
# -----------------------------------------------------------------------------


class SimpleCIFARNet(nn.Module):
    """
    Simple CNN architecture for CIFAR-10 classification.

    Architecture:
        - Conv2d(3, 32, 3x3) + BN + ReLU + MaxPool  [32x32 -> 16x16]
        - Conv2d(32, 64, 3x3) + BN + ReLU + MaxPool [16x16 -> 8x8]
        - Conv2d(64, 128, 3x3) + BN + ReLU + AdaptiveAvgPool [8x8 -> 1x1]
        - Linear(128, num_classes)

    Attributes:
        num_classes: Number of output classes
        dropout_rate: Dropout probability before final linear layer
    """

    def __init__(self, num_classes: int = 10, dropout_rate: float = 0.5):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: 3 -> 32
            nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32 -> 16x16
            # Block 2: 32 -> 64
            nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16 -> 8x8
            # Block 3: 64 -> 128
            nn.Conv2d(64, 128, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),  # 8x8 -> 1x1
        )

        self.classifier = nn.Sequential(
            nn.Dropout(p=dropout_rate),
            nn.Linear(128, num_classes),
        )

        self._initialize_weights()

    def _initialize_weights(self) -> None:
        """He-initialize conv layers, standard init for BN + Linear."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1.0)
                nn.init.constant_(m.bias, 0.0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)

        Returns:
            Logits of shape (batch_size, num_classes)
        """
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def get_feature_maps(self, x: torch.Tensor) -> torch.Tensor:
        """Expose feature maps for later XAI methods (e.g., Grad-CAM).

        Args:
            x: Input tensor of shape (batch_size, 3, 32, 32)

        Returns:
            Feature maps of shape (batch_size, 128, 1, 1)
        """
        return self.features(x)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def set_seed(seed: int, logger: logging.Logger) -> None:
    """Set seeds for reproducibility across PyTorch, NumPy, and Python.

    Args:
        seed: Random seed value
        logger: Logger instance for reporting
    """
    logger.info(f"Setting random seed to {seed}")
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)

    # Enable deterministic operations (may impact performance)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def get_device(logger: logging.Logger) -> torch.device:
    """Return CUDA device if available, otherwise CPU.

    Args:
        logger: Logger instance for reporting device info

    Returns:
        PyTorch device object
    """
    if torch.cuda.is_available():
        device = torch.device("cuda")
        props = torch.cuda.get_device_properties(0)
        logger.info(
            f"Using GPU: {torch.cuda.get_device_name(0)} "
            f"({props.total_memory / 1e9:.2f} GB)"
        )
    else:
        device = torch.device("cpu")
        logger.info("Using CPU")
    return device


def get_data_loaders(
    cfg: TrainingConfig,
    device: torch.device,
    logger: logging.Logger,
) -> Tuple[DataLoader, DataLoader]:
    """Create CIFAR-10 dataloaders with augmentation and optional subset.

    Args:
        cfg: Training configuration
        device: Device for pin_memory optimization
        logger: Logger instance

    Returns:
        Tuple of (train_loader, test_loader)
    """
    logger.info("Setting up CIFAR-10 datasets and dataloaders")

    # CIFAR-10 standard normalization values
    mean = [0.4914, 0.4822, 0.4465]
    std = [0.2470, 0.2435, 0.2616]

    train_transform = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )
    test_transform = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std),
        ]
    )

    train_dataset = datasets.CIFAR10(
        root=cfg.data_root,
        train=True,
        download=True,
        transform=train_transform,
    )
    test_dataset = datasets.CIFAR10(
        root=cfg.data_root,
        train=False,
        download=True,
        transform=test_transform,
    )

    # Create subset for fast debugging if specified
    if cfg.subset_size is not None:
        subset = min(cfg.subset_size, len(train_dataset))
        logger.info(f"Using training subset size: {subset}")
        train_indices = np.random.choice(len(train_dataset), subset, replace=False)
        train_dataset = Subset(train_dataset, train_indices)

        test_subset_size = max(512, subset // 4)
        test_subset_size = min(test_subset_size, len(test_dataset))
        logger.info(f"Using test subset size: {test_subset_size}")
        test_indices = np.random.choice(
            len(test_dataset), test_subset_size, replace=False
        )
        test_dataset = Subset(test_dataset, test_indices)

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.batch_size,
        shuffle=True,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=cfg.batch_size * 2,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=pin_memory,
        persistent_workers=cfg.num_workers > 0,
    )

    logger.info(f"Train set size: {len(train_dataset)}")
    logger.info(f"Test set size: {len(test_dataset)}")

    return train_loader, test_loader


def build_model(
    cfg: TrainingConfig, device: torch.device, logger: logging.Logger
) -> nn.Module:
    """Instantiate model and log parameter counts.

    Args:
        cfg: Training configuration
        device: Device to place model on
        logger: Logger instance

    Returns:
        Initialized model on specified device

    Raises:
        ValueError: If unsupported model type specified
    """
    if cfg.model != "SimpleCIFARNet":
        raise ValueError(f"Unsupported model: {cfg.model}")

    model = SimpleCIFARNet(num_classes=10, dropout_rate=cfg.dropout).to(device)

    num_params = sum(p.numel() for p in model.parameters())
    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Model: {cfg.model}")
    logger.info(f"Total parameters: {num_params:,}")
    logger.info(f"Trainable parameters: {num_trainable:,}")

    mlflow.log_param("num_parameters", num_params)
    mlflow.log_param("num_trainable_parameters", num_trainable)

    return model


def build_optimizer_and_scheduler(
    cfg: TrainingConfig,
    model: nn.Module,
) -> Tuple[optim.Optimizer, Optional[optim.lr_scheduler._LRScheduler]]:
    """Create optimizer and optional scheduler from config.

    Args:
        cfg: Training configuration
        model: Model to optimize

    Returns:
        Tuple of (optimizer, scheduler) where scheduler may be None

    Raises:
        ValueError: If unsupported optimizer or scheduler specified
    """
    if cfg.optimizer.lower() == "adam":
        optimizer = optim.Adam(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    elif cfg.optimizer.lower() == "sgd":
        optimizer = optim.SGD(
            model.parameters(),
            lr=cfg.lr,
            momentum=0.9,
            weight_decay=cfg.weight_decay,
        )
    elif cfg.optimizer.lower() == "adamw":
        optimizer = optim.AdamW(
            model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay
        )
    else:
        raise ValueError(f"Unsupported optimizer: {cfg.optimizer}")

    if cfg.scheduler == "cosine":
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.epochs)
    elif cfg.scheduler == "step":
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)
    elif cfg.scheduler == "none":
        scheduler = None
    else:
        raise ValueError(f"Unsupported scheduler: {cfg.scheduler}")

    return optimizer, scheduler


# -----------------------------------------------------------------------------
# Training / evaluation loops
# -----------------------------------------------------------------------------


def train_epoch(
    model: nn.Module,
    train_loader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: torch.device,
    epoch: int,
    cfg: TrainingConfig,
    logger: logging.Logger,
) -> Dict[str, float]:
    """Run one training epoch.

    Args:
        model: Neural network model
        train_loader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device for computation
        epoch: Current epoch number
        cfg: Training configuration
        logger: Logger instance

    Returns:
        Dictionary containing train_loss and train_acc
    """
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(inputs)
        loss = criterion(outputs, targets)

        loss.backward()

        if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)

        optimizer.step()

        running_loss += loss.item()
        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()

        if (batch_idx + 1) % cfg.log_interval == 0:
            avg_loss = running_loss / (batch_idx + 1)
            acc = 100.0 * correct / total
            logger.info(
                f"Epoch [{epoch}] Batch [{batch_idx + 1}/{len(train_loader)}] "
                f"Loss: {avg_loss:.4f} | Acc: {acc:.2f}%"
            )

    avg_loss = running_loss / len(train_loader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    return {"train_loss": avg_loss, "train_acc": accuracy}


@torch.no_grad()
def evaluate(
    model: nn.Module,
    test_loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
) -> Dict[str, float]:
    """Evaluate model and compute per-class accuracy + confusion matrix.

    Args:
        model: Neural network model
        test_loader: Test/validation data loader
        criterion: Loss function
        device: Device for computation

    Returns:
        Dictionary containing test metrics, confusion matrix, and classification report
    """
    model.eval()

    test_loss = 0.0
    correct = 0
    total = 0

    all_targets: list[int] = []
    all_preds: list[int] = []

    for inputs, targets in test_loader:
        inputs = inputs.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)

        outputs = model(inputs)
        loss = criterion(outputs, targets)

        test_loss += loss.item()
        _, preds = outputs.max(1)
        total += targets.size(0)
        correct += preds.eq(targets).sum().item()

        all_targets.extend(targets.cpu().tolist())
        all_preds.extend(preds.cpu().tolist())

    avg_loss = test_loss / len(test_loader)
    accuracy = 100.0 * correct / total if total > 0 else 0.0

    metrics: Dict[str, float] = {
        "test_loss": avg_loss,
        "test_acc": accuracy,
    }

    all_targets_np = np.array(all_targets)
    all_preds_np = np.array(all_preds)

    class_names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]

    # Compute per-class accuracy
    for idx, name in enumerate(class_names):
        mask = all_targets_np == idx
        if mask.sum() > 0:
            class_acc = 100.0 * (all_preds_np[mask] == idx).sum() / mask.sum()
            metrics[f"test_acc_{name}"] = float(class_acc)

    # Confusion matrix and classification report (for artifacts)
    cm = confusion_matrix(all_targets_np, all_preds_np, labels=list(range(10)))
    metrics["_confusion_matrix"] = cm.tolist()  # type: ignore[assignment]

    cls_report = classification_report(
        all_targets_np,
        all_preds_np,
        labels=list(range(10)),
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    metrics["_classification_report"] = cls_report  # type: ignore[assignment]

    return metrics


def save_checkpoint(
    model: nn.Module,
    optimizer: optim.Optimizer,
    epoch: int,
    metrics: Dict[str, float],
    checkpoint_dir: Path,
    filename: str,
    logger: logging.Logger,
) -> Path:
    """Save model checkpoint and return its path.

    Args:
        model: Model to save
        optimizer: Optimizer state to save
        epoch: Current epoch number
        metrics: Metrics dictionary to include
        checkpoint_dir: Directory to save checkpoint
        filename: Checkpoint filename
        logger: Logger instance

    Returns:
        Path to saved checkpoint
    """
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = checkpoint_dir / filename

    checkpoint = {
        "epoch": epoch,
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "metrics": metrics,
        "model_architecture": str(model),
    }

    torch.save(checkpoint, ckpt_path)
    logger.info(f"Checkpoint saved to {ckpt_path}")
    return ckpt_path


# -----------------------------------------------------------------------------
# MLflow helpers
# -----------------------------------------------------------------------------


def log_environment_to_mlflow(cfg: TrainingConfig, device: torch.device) -> None:
    """Log useful run metadata as MLflow tags.

    Args:
        cfg: Training configuration
        device: Compute device being used
    """
    mlflow.set_tags(
        {
            "host": socket.gethostname(),
            "device": str(device),
            "script": Path(__file__).name,
            "framework": "pytorch",
            "dataset": "CIFAR10",
            "model_architecture": cfg.model,
        }
    )

    # Log config as a single JSON param for reproducibility
    mlflow.log_param("config_json", json.dumps(asdict(cfg), indent=2))


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------


def parse_args() -> TrainingConfig:
    """Parse command line arguments and return training configuration.

    Returns:
        TrainingConfig instance populated from arguments
    """
    parser = argparse.ArgumentParser(
        description="CIFAR-10 debug training with MLflow (A1-grade pipeline)",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Model / training
    parser.add_argument("--model", type=str, default="SimpleCIFARNet")
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument(
        "--optimizer", type=str, default="adam", choices=["adam", "sgd", "adamw"]
    )
    parser.add_argument(
        "--scheduler", type=str, default="cosine", choices=["cosine", "step", "none"]
    )
    parser.add_argument("--log-interval", type=int, default=20)
    parser.add_argument("--grad-clip-norm", type=float, default=5.0)

    # Data
    parser.add_argument("--data-root", type=str, default="./data/cifar10")
    parser.add_argument(
        "--subset-size",
        type=int,
        default=None,
        help="Training subset size for fast debugging (None = full dataset)",
    )
    parser.add_argument("--num-workers", type=int, default=4)

    # Reproducibility
    parser.add_argument("--seed", type=int, default=42)

    # Checkpoints / artifacts
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="./results/checkpoints/cifar10_debug",
    )

    # MLflow (using standardized naming)
    parser.add_argument("--dataset-name", type=str, default="CIFAR10-debug")
    parser.add_argument("--objective", type=str, default="baseline")
    parser.add_argument(
        "--run-tag",
        type=str,
        default=None,
        help="Optional tag for MLflow run name (e.g., 'seed-42', 'subset-2k')",
    )
    parser.add_argument("--tracking-uri", type=str, default=None)

    # Misc
    parser.add_argument("--max-epochs-without-improvement", type=int, default=10)

    args = parser.parse_args()
    cfg = TrainingConfig(
        model=args.model,
        dropout=args.dropout,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        optimizer=args.optimizer,
        scheduler=args.scheduler,
        log_interval=args.log_interval,
        grad_clip_norm=args.grad_clip_norm,
        data_root=args.data_root,
        subset_size=args.subset_size,
        num_workers=args.num_workers,
        seed=args.seed,
        checkpoint_dir=args.checkpoint_dir,
        dataset_name=args.dataset_name,
        objective=args.objective,
        run_tag=args.run_tag,
        tracking_uri=args.tracking_uri,
        max_epochs_without_improvement=args.max_epochs_without_improvement,
    )
    return cfg


# -----------------------------------------------------------------------------
# Main training entrypoint
# -----------------------------------------------------------------------------


def main() -> Dict[str, float]:
    """Main training function with full MLflow integration.

    Returns:
        Dictionary containing training summary metrics

    Raises:
        Exception: Any error during training (with full traceback)
    """
    # Directories
    repo_root = Path(__file__).parent.parent.resolve()
    logs_dir = repo_root / "logs"
    checkpoints_root = repo_root / "results" / "checkpoints" / "cifar10_debug"

    logger = configure_logging(logs_dir)
    cfg = parse_args()

    # Resolve checkpoint directory
    cfg.checkpoint_dir = str(
        (
            checkpoints_root
            if cfg.checkpoint_dir == "./results/checkpoints/cifar10_debug"
            else Path(cfg.checkpoint_dir)
        ).resolve()
    )

    set_seed(cfg.seed, logger)
    device = get_device(logger)

    # Build experiment and run names using standardized helper
    experiment_name, run_name = build_experiment_and_run_name(
        dataset=cfg.dataset_name,
        model=cfg.model,
        objective=cfg.objective,
        extra_tag=cfg.run_tag or f"seed-{cfg.seed}",
    )

    logger.info(f"MLflow Experiment: {experiment_name}")
    logger.info(f"MLflow Run: {run_name}")

    # Initialize MLflow using standardized helper
    run = init_mlflow(
        experiment_name=experiment_name,
        run_name=run_name,
        tracking_uri=cfg.tracking_uri,
    )

    try:
        logger.info(f"Starting MLflow run: {run_name}")
        logger.info(f"Run ID: {run.info.run_id}")

        # Log basic hyperparameters
        mlflow.log_params(
            {
                "model": cfg.model,
                "dropout_rate": cfg.dropout,
                "num_epochs": cfg.epochs,
                "batch_size": cfg.batch_size,
                "learning_rate": cfg.lr,
                "weight_decay": cfg.weight_decay,
                "optimizer": cfg.optimizer,
                "scheduler": cfg.scheduler,
                "subset_size": cfg.subset_size,
                "seed": cfg.seed,
                "num_workers": cfg.num_workers,
                "grad_clip_norm": cfg.grad_clip_norm,
            }
        )

        log_environment_to_mlflow(cfg, device)

        train_loader, test_loader = get_data_loaders(cfg, device, logger)
        model = build_model(cfg, device, logger)

        criterion = nn.CrossEntropyLoss()
        optimizer, scheduler = build_optimizer_and_scheduler(cfg, model)

        best_test_acc = 0.0
        epochs_without_improvement = 0
        start_time = time.time()

        logger.info("=" * 80)
        logger.info("Starting training loop")
        logger.info("=" * 80)

        last_train_metrics: Dict[str, float] = {}
        last_test_metrics: Dict[str, float] = {}

        for epoch in range(1, cfg.epochs + 1):
            epoch_start = time.time()

            train_metrics = train_epoch(
                model=model,
                train_loader=train_loader,
                criterion=criterion,
                optimizer=optimizer,
                device=device,
                epoch=epoch,
                cfg=cfg,
                logger=logger,
            )

            test_metrics = evaluate(
                model=model,
                test_loader=test_loader,
                criterion=criterion,
                device=device,
            )

            # Extract confusion matrix and report from metrics dict
            cm = test_metrics.pop("_confusion_matrix")
            cls_report = test_metrics.pop("_classification_report")

            if scheduler is not None:
                scheduler.step()
                current_lr = scheduler.get_last_lr()[0]
            else:
                current_lr = cfg.lr

            epoch_time = time.time() - epoch_start

            metrics = {
                **train_metrics,
                **test_metrics,
                "learning_rate": current_lr,
                "epoch_time": epoch_time,
            }

            mlflow.log_metrics(metrics, step=epoch)

            logger.info("-" * 80)
            logger.info(
                f"Epoch [{epoch}/{cfg.epochs}] "
                f"Train Loss: {train_metrics['train_loss']:.4f} | "
                f"Train Acc: {train_metrics['train_acc']:.2f}% | "
                f"Test Loss: {test_metrics['test_loss']:.4f} | "
                f"Test Acc: {test_metrics['test_acc']:.2f}% | "
                f"LR: {current_lr:.6f} | "
                f"Epoch Time: {epoch_time:.2f}s"
            )
            logger.info("-" * 80)

            # Early stopping logic based on test accuracy
            current_acc = test_metrics["test_acc"]
            if current_acc > best_test_acc:
                best_test_acc = current_acc
                epochs_without_improvement = 0

                ckpt_path = save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    metrics=metrics,
                    checkpoint_dir=Path(cfg.checkpoint_dir),
                    filename="best_model.pt",
                    logger=logger,
                )
                mlflow.log_artifact(
                    str(ckpt_path), artifact_path=cfg.checkpoints_artifact_dir
                )
                mlflow.log_metric("best_test_acc", best_test_acc)
                logger.info(f"New best test accuracy: {best_test_acc:.2f}%")
            else:
                epochs_without_improvement += 1
                logger.info(
                    f"No improvement for {epochs_without_improvement} epoch(s) "
                    f"(best={best_test_acc:.2f}%)"
                )
                if epochs_without_improvement >= cfg.max_epochs_without_improvement:
                    logger.info(
                        f"Early stopping triggered after "
                        f"{epochs_without_improvement} epochs without improvement."
                    )
                    break

            last_train_metrics = train_metrics
            last_test_metrics = test_metrics

            # Save confusion matrix and classification report artifacts per epoch
            results_dir = Path(cfg.checkpoint_dir)
            results_dir.mkdir(parents=True, exist_ok=True)

            cm_path = results_dir / f"confusion_matrix_epoch_{epoch}.json"
            with cm_path.open("w") as f_cm:
                json.dump({"confusion_matrix": cm}, f_cm, indent=2)
            mlflow.log_artifact(str(cm_path), artifact_path=cfg.results_artifact_dir)

            report_path = results_dir / f"classification_report_epoch_{epoch}.txt"
            with report_path.open("w") as f_rep:
                f_rep.write(cls_report)
            mlflow.log_artifact(
                str(report_path), artifact_path=cfg.results_artifact_dir
            )

        # Final checkpoint
        final_metrics = {
            **last_train_metrics,
            **last_test_metrics,
            "best_test_acc": best_test_acc,
        }
        final_ckpt_path = save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=epoch,
            metrics=final_metrics,
            checkpoint_dir=Path(cfg.checkpoint_dir),
            filename="final_model.pt",
            logger=logger,
        )
        mlflow.log_artifact(
            str(final_ckpt_path), artifact_path=cfg.checkpoints_artifact_dir
        )

        # Log full model for MLflow model registry
        mlflow.pytorch.log_model(model, "model")

        total_time = time.time() - start_time
        mlflow.log_metric("total_training_time", total_time)

        logger.info("=" * 80)
        logger.info("Training complete")
        logger.info(f"Total time: {total_time:.2f}s")
        logger.info(f"Best test accuracy: {best_test_acc:.2f}%")
        logger.info(f"MLflow run id: {run.info.run_id}")
        logger.info("=" * 80)

        summary = {
            "best_test_acc": best_test_acc,
            "total_training_time": total_time,
            "final_train_loss": float(last_train_metrics.get("train_loss", 0.0)),
            "final_test_loss": float(last_test_metrics.get("test_loss", 0.0)),
            "run_id": run.info.run_id,
        }

        summary_path = Path(cfg.checkpoint_dir) / "summary.json"
        with summary_path.open("w") as f_sum:
            json.dump(summary, f_sum, indent=2)
        mlflow.log_artifact(str(summary_path), artifact_path=cfg.results_artifact_dir)

        logger.info("")  # Blank line for readability
        logger.info("To view results in MLflow UI, run:")
        logger.info('  mlflow ui --backend-store-uri "file:./mlruns"')

        return summary

    finally:
        # Ensure MLflow run is ended even if error occurs
        mlflow.end_run()


if __name__ == "__main__":
    try:
        main()
        sys.exit(0)
    except Exception as exc:
        # Defensive: surfaces full traceback for debugging
        print(f"Training failed with error: {exc}", file=sys.stderr)
        raise
