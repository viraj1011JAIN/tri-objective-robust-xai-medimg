"""
Retrain Model with Optimal Hyperparameters from HPO Study.

This script loads the best hyperparameters from a completed HPO study
and retrains the model with full training epochs for final deployment.

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Date: November 24, 2025
Version: 5.4.0
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

try:
    import optuna

    OPTUNA_AVAILABLE = True
except ImportError:
    OPTUNA_AVAILABLE = False

try:
    import mlflow

    MLFLOW_AVAILABLE = True
except ImportError:
    MLFLOW_AVAILABLE = False

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.training.hpo_config import HPOConfig

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/retrain_optimal.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


def parse_arguments() -> argparse.Namespace:
    """
    Parse command line arguments.

    Returns:
        Parsed arguments
    """
    parser = argparse.ArgumentParser(
        description="Retrain Model with Optimal Hyperparameters"
    )

    # Study configuration
    parser.add_argument(
        "--study-name", type=str, required=True, help="Name of the completed HPO study"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///hpo_study.db",
        help="Optuna storage URL",
    )
    parser.add_argument(
        "--trial-number",
        type=int,
        default=None,
        help="Specific trial number to use (default: best trial)",
    )

    # Training configuration
    parser.add_argument(
        "--n-epochs", type=int, default=200, help="Number of training epochs"
    )
    parser.add_argument(
        "--warmup-epochs", type=int, default=10, help="Number of warmup epochs"
    )
    parser.add_argument(
        "--use-scheduler", action="store_true", help="Use learning rate scheduler"
    )

    # Data configuration
    parser.add_argument(
        "--dataset",
        type=str,
        default="cifar10",
        choices=["cifar10", "cifar100", "svhn"],
        help="Dataset to use",
    )
    parser.add_argument("--data-dir", type=str, default="data", help="Data directory")
    parser.add_argument(
        "--batch-size", type=int, default=128, help="Batch size for training"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50", "wideresnet"],
        help="Model architecture",
    )

    # Device configuration
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Device to use",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use")

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase_5_4/final_model",
        help="Output directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/final_model",
        help="Checkpoint directory",
    )
    parser.add_argument(
        "--save-frequency", type=int, default=10, help="Save checkpoint every N epochs"
    )

    # MLflow configuration
    parser.add_argument("--use-mlflow", action="store_true", help="Log to MLflow")
    parser.add_argument(
        "--mlflow-tracking-uri", type=str, default="mlruns", help="MLflow tracking URI"
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="phase_5_4_final_training",
        help="MLflow experiment name",
    )

    return parser.parse_args()


def load_best_hyperparameters(
    study_name: str,
    storage: str,
    trial_number: Optional[int] = None,
) -> Dict[str, Any]:
    """
    Load best hyperparameters from HPO study.

    Args:
        study_name: Study name
        storage: Storage URL
        trial_number: Specific trial to load (default: best)

    Returns:
        Dictionary of hyperparameters
    """
    if not OPTUNA_AVAILABLE:
        raise ImportError("Optuna not available. Install with: pip install optuna")

    logger.info(f"Loading study '{study_name}' from {storage}")

    study = optuna.load_study(
        study_name=study_name,
        storage=storage,
    )

    if trial_number is not None:
        trial = study.trials[trial_number]
        logger.info(f"Using trial {trial_number} (value: {trial.value:.4f})")
    else:
        trial = study.best_trial
        logger.info(f"Using best trial {trial.number} (value: {trial.value:.4f})")

    hyperparams = trial.params.copy()

    logger.info("Loaded hyperparameters:")
    for key, value in hyperparams.items():
        logger.info(f"  {key}: {value}")

    return hyperparams


def setup_device(args: argparse.Namespace) -> torch.device:
    """
    Setup compute device.

    Args:
        args: Arguments

    Returns:
        PyTorch device
    """
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    elif args.device == "cuda":
        if not torch.cuda.is_available():
            logger.warning("CUDA not available, falling back to CPU")
            device = torch.device("cpu")
        else:
            device = torch.device(f"cuda:{args.gpu_id}")
    else:
        device = torch.device("cpu")

    logger.info(f"Using device: {device}")

    if device.type == "cuda":
        logger.info(f"GPU: {torch.cuda.get_device_name(device)}")
        logger.info(
            f"Memory: {torch.cuda.get_device_properties(device).total_memory / 1e9:.2f} GB"
        )

    return device


def load_dataset(args: argparse.Namespace) -> tuple:
    """
    Load dataset.

    Args:
        args: Arguments

    Returns:
        Tuple of (train_loader, test_loader, num_classes)
    """
    logger.info(f"Loading {args.dataset} dataset...")

    # Define transforms
    if args.dataset == "cifar10":
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
        num_classes = 10
    elif args.dataset == "cifar100":
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
        num_classes = 100
    else:  # SVHN
        mean = [0.4377, 0.4438, 0.4728]
        std = [0.1980, 0.2010, 0.1970]
        num_classes = 10

    transform_train = transforms.Compose(
        [
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    transform_test = transforms.Compose(
        [
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
    )

    # Load datasets
    data_dir = Path(args.data_dir)
    data_dir.mkdir(parents=True, exist_ok=True)

    if args.dataset == "cifar10":
        train_dataset = datasets.CIFAR10(
            root=str(data_dir), train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR10(
            root=str(data_dir), train=False, download=True, transform=transform_test
        )
    elif args.dataset == "cifar100":
        train_dataset = datasets.CIFAR100(
            root=str(data_dir), train=True, download=True, transform=transform_train
        )
        test_dataset = datasets.CIFAR100(
            root=str(data_dir), train=False, download=True, transform=transform_test
        )
    else:  # SVHN
        train_dataset = datasets.SVHN(
            root=str(data_dir), split="train", download=True, transform=transform_train
        )
        test_dataset = datasets.SVHN(
            root=str(data_dir), split="test", download=True, transform=transform_test
        )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    logger.info(
        f"Loaded dataset: {len(train_dataset)} train, "
        f"{len(test_dataset)} test samples"
    )

    return train_loader, test_loader, num_classes


def create_model(args: argparse.Namespace, num_classes: int) -> nn.Module:
    """
    Create model.

    Args:
        args: Arguments
        num_classes: Number of classes

    Returns:
        Model
    """
    if args.model == "resnet18":
        from torchvision.models import resnet18

        model = resnet18(num_classes=num_classes)
    elif args.model == "resnet34":
        from torchvision.models import resnet34

        model = resnet34(num_classes=num_classes)
    elif args.model == "resnet50":
        from torchvision.models import resnet50

        model = resnet50(num_classes=num_classes)
    else:  # wideresnet
        from torchvision.models import resnet18

        model = resnet18(num_classes=num_classes)
        logger.warning("WideResNet not implemented, using ResNet18")

    return model


def pgd_attack(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    epsilon: float,
    alpha: float = 2 / 255,
    num_steps: int = 10,
    device: torch.device = torch.device("cpu"),
) -> torch.Tensor:
    """
    PGD attack.

    Args:
        model: Model
        x: Input
        y: Labels
        epsilon: Perturbation budget
        alpha: Step size
        num_steps: Number of steps
        device: Device

    Returns:
        Adversarial examples
    """
    model.eval()
    x_adv = x.clone().detach()

    for _ in range(num_steps):
        x_adv.requires_grad = True

        with torch.enable_grad():
            logits = model(x_adv)
            loss = nn.CrossEntropyLoss()(logits, y)

        grad = torch.autograd.grad(loss, x_adv)[0]
        x_adv = x_adv.detach() + alpha * grad.sign()

        delta = torch.clamp(x_adv - x, -epsilon, epsilon)
        x_adv = torch.clamp(x + delta, 0, 1)

    return x_adv.detach()


def trades_loss(
    model: nn.Module,
    x: torch.Tensor,
    y: torch.Tensor,
    beta: float,
    epsilon: float,
    device: torch.device,
) -> tuple:
    """
    TRADES loss.

    Args:
        model: Model
        x: Input
        y: Labels
        beta: Beta parameter
        epsilon: Epsilon
        device: Device

    Returns:
        Tuple of (total_loss, natural_loss, robust_loss)
    """
    model.eval()
    x_adv = pgd_attack(model, x, y, epsilon, device=device)

    model.train()

    logits_natural = model(x)
    natural_loss = nn.CrossEntropyLoss()(logits_natural, y)

    logits_adv = model(x_adv)
    log_prob_natural = nn.functional.log_softmax(logits_natural, dim=1)
    prob_adv = nn.functional.softmax(logits_adv, dim=1)

    robust_loss = nn.functional.kl_div(
        log_prob_natural,
        prob_adv,
        reduction="batchmean",
    )

    total_loss = natural_loss + beta * robust_loss

    return total_loss, natural_loss, robust_loss


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    epsilon: float,
    device: torch.device,
) -> tuple:
    """
    Evaluate model.

    Args:
        model: Model
        loader: Data loader
        epsilon: Epsilon
        device: Device

    Returns:
        Tuple of (clean_acc, robust_acc)
    """
    model.eval()

    clean_correct = 0
    robust_correct = 0
    total = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        # Clean accuracy
        logits_clean = model(x)
        pred_clean = logits_clean.argmax(dim=1)
        clean_correct += (pred_clean == y).sum().item()

        # Robust accuracy
        x_adv = pgd_attack(model, x, y, epsilon, device=device)
        logits_adv = model(x_adv)
        pred_adv = logits_adv.argmax(dim=1)
        robust_correct += (pred_adv == y).sum().item()

        total += y.size(0)

    clean_acc = clean_correct / total
    robust_acc = robust_correct / total

    return clean_acc, robust_acc


def train_epoch(
    model: nn.Module,
    optimizer: optim.Optimizer,
    loader: DataLoader,
    beta: float,
    epsilon: float,
    device: torch.device,
) -> tuple:
    """
    Train one epoch.

    Args:
        model: Model
        optimizer: Optimizer
        loader: Data loader
        beta: Beta parameter
        epsilon: Epsilon
        device: Device

    Returns:
        Tuple of (avg_loss, avg_natural_loss, avg_robust_loss)
    """
    model.train()

    total_loss = 0.0
    total_natural_loss = 0.0
    total_robust_loss = 0.0
    n_batches = 0

    for x, y in loader:
        x, y = x.to(device), y.to(device)

        optimizer.zero_grad()

        loss, natural_loss, robust_loss = trades_loss(
            model, x, y, beta, epsilon, device
        )

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        total_natural_loss += natural_loss.item()
        total_robust_loss += robust_loss.item()
        n_batches += 1

    return (
        total_loss / n_batches,
        total_natural_loss / n_batches,
        total_robust_loss / n_batches,
    )


def main():
    """Main function."""
    args = parse_arguments()

    # Setup directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.checkpoint_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    logger.info("=" * 80)
    logger.info("RETRAIN WITH OPTIMAL HYPERPARAMETERS - PHASE 5.4")
    logger.info("=" * 80)

    # Load best hyperparameters
    hyperparams = load_best_hyperparameters(
        args.study_name,
        args.storage,
        args.trial_number,
    )

    # Setup MLflow
    if args.use_mlflow and MLFLOW_AVAILABLE:
        mlflow.set_tracking_uri(args.mlflow_tracking_uri)
        mlflow.set_experiment(args.mlflow_experiment)
        mlflow.start_run()
        mlflow.log_params(hyperparams)
        mlflow.log_params(vars(args))

    # Setup device
    device = setup_device(args)

    # Load dataset
    train_loader, test_loader, num_classes = load_dataset(args)

    # Create model
    model = create_model(args, num_classes)
    model = model.to(device)

    # Create optimizer
    learning_rate = hyperparams["learning_rate"]
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # Create scheduler
    if args.use_scheduler:
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.n_epochs)

    # Extract TRADES hyperparameters
    beta = hyperparams["beta"]
    epsilon = hyperparams["epsilon"]

    logger.info(
        f"Training with beta={beta:.2f}, epsilon={epsilon:.4f}, lr={learning_rate:.5f}"
    )
    logger.info(f"Training for {args.n_epochs} epochs")

    # Training loop
    best_robust_acc = 0.0
    best_epoch = 0

    for epoch in range(1, args.n_epochs + 1):
        start_time = time.time()

        # Train
        avg_loss, avg_natural_loss, avg_robust_loss = train_epoch(
            model, optimizer, train_loader, beta, epsilon, device
        )

        # Evaluate
        clean_acc, robust_acc = evaluate(model, test_loader, epsilon, device)

        epoch_time = time.time() - start_time

        # Update scheduler
        if args.use_scheduler:
            scheduler.step()

        # Log
        logger.info(
            f"Epoch {epoch}/{args.n_epochs} | "
            f"Loss: {avg_loss:.4f} | "
            f"Clean: {clean_acc:.4f} | "
            f"Robust: {robust_acc:.4f} | "
            f"Time: {epoch_time:.2f}s"
        )

        if args.use_mlflow and MLFLOW_AVAILABLE:
            mlflow.log_metrics(
                {
                    "loss": avg_loss,
                    "natural_loss": avg_natural_loss,
                    "robust_loss": avg_robust_loss,
                    "clean_accuracy": clean_acc,
                    "robust_accuracy": robust_acc,
                    "epoch_time": epoch_time,
                },
                step=epoch,
            )

        # Save checkpoint
        if robust_acc > best_robust_acc:
            best_robust_acc = robust_acc
            best_epoch = epoch

            checkpoint_path = Path(args.checkpoint_dir) / "best_model.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "hyperparameters": hyperparams,
                    "clean_accuracy": clean_acc,
                    "robust_accuracy": robust_acc,
                },
                checkpoint_path,
            )

            logger.info(f"✓ Saved best model (robust_acc: {robust_acc:.4f})")

        if epoch % args.save_frequency == 0:
            checkpoint_path = Path(args.checkpoint_dir) / f"checkpoint_epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "hyperparameters": hyperparams,
                    "clean_accuracy": clean_acc,
                    "robust_accuracy": robust_acc,
                },
                checkpoint_path,
            )

    # Save final model
    final_path = Path(args.checkpoint_dir) / "final_model.pt"
    torch.save(
        {
            "epoch": args.n_epochs,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "hyperparameters": hyperparams,
            "clean_accuracy": clean_acc,
            "robust_accuracy": robust_acc,
            "best_epoch": best_epoch,
            "best_robust_accuracy": best_robust_acc,
        },
        final_path,
    )

    logger.info("=" * 80)
    logger.info("TRAINING COMPLETED")
    logger.info("=" * 80)
    logger.info(f"Best robust accuracy: {best_robust_acc:.4f} (epoch {best_epoch})")
    logger.info(f"Final clean accuracy: {clean_acc:.4f}")
    logger.info(f"Final robust accuracy: {robust_acc:.4f}")
    logger.info(f"Model saved to {final_path}")
    logger.info("=" * 80)

    # Save summary
    summary = {
        "hyperparameters": hyperparams,
        "training_config": vars(args),
        "best_epoch": best_epoch,
        "best_robust_accuracy": float(best_robust_acc),
        "final_clean_accuracy": float(clean_acc),
        "final_robust_accuracy": float(robust_acc),
    }

    summary_path = Path(args.output_dir) / "training_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info(f"Summary saved to {summary_path}")

    if args.use_mlflow and MLFLOW_AVAILABLE:
        mlflow.log_artifact(str(summary_path))
        mlflow.end_run()


if __name__ == "__main__":
    main()
