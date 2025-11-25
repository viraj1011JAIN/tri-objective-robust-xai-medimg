"""
Main HPO Study Execution Script for TRADES Hyperparameter Optimization.

This script orchestrates the complete HPO workflow:
1. Data loading and preprocessing
2. Model factory creation
3. HPO study configuration and execution
4. Results analysis and visualization
5. Best trial checkpoint saving

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Date: November 24, 2025
Version: 5.4.0
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.hpo_analysis import analyze_study
from src.training.hpo_config import (
    HPOConfig,
    TRADESSearchSpace,
    create_default_hpo_config,
)
from src.training.hpo_objective import ObjectiveConfig, WeightedTriObjective
from src.training.hpo_trainer import TRADESHPOTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/hpo_study.log"),
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
        description="Run TRADES Hyperparameter Optimization Study"
    )

    # Study configuration
    parser.add_argument(
        "--study-name",
        type=str,
        default="trades_hpo_phase54",
        help="Name for the HPO study",
    )
    parser.add_argument(
        "--n-trials", type=int, default=50, help="Number of HPO trials to run"
    )
    parser.add_argument(
        "--n-epochs", type=int, default=10, help="Number of epochs per trial"
    )
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///hpo_study.db",
        help="Optuna storage URL",
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
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode (10 trials, 2 epochs, subset data)",
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
        help="Device to use for training",
    )
    parser.add_argument("--gpu-id", type=int, default=0, help="GPU ID to use")

    # Output configuration
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/phase_5_4",
        help="Output directory for results",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/hpo",
        help="Directory to save checkpoints",
    )

    # Objective configuration
    parser.add_argument(
        "--robust-weight", type=float, default=0.4, help="Weight for robust accuracy"
    )
    parser.add_argument(
        "--clean-weight", type=float, default=0.3, help="Weight for clean accuracy"
    )
    parser.add_argument(
        "--auroc-weight", type=float, default=0.3, help="Weight for cross-site AUROC"
    )

    # Analysis configuration
    parser.add_argument(
        "--skip-analysis", action="store_true", help="Skip analysis and visualization"
    )

    return parser.parse_args()


def setup_device(args: argparse.Namespace) -> torch.device:
    """
    Setup compute device.

    Args:
        args: Command line arguments

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
    Load and prepare dataset.

    Args:
        args: Command line arguments

    Returns:
        Tuple of (train_loader, val_loader, test_loader, num_classes)
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

    # Quick test mode: use subset
    if args.quick_test:
        logger.info("Quick test mode: using data subsets")
        train_indices = np.random.choice(len(train_dataset), 1000, replace=False)
        test_indices = np.random.choice(len(test_dataset), 500, replace=False)
        train_dataset = Subset(train_dataset, train_indices)
        test_dataset = Subset(test_dataset, test_indices)

    # Split train into train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )

    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
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
        f"{len(val_dataset)} val, {len(test_dataset)} test samples"
    )

    return train_loader, val_loader, test_loader, num_classes


def create_model_factory(args: argparse.Namespace, num_classes: int):
    """
    Create model factory function.

    Args:
        args: Command line arguments
        num_classes: Number of output classes

    Returns:
        Model factory function
    """

    def model_factory(hyperparams: Dict[str, Any]) -> nn.Module:
        """Create model from hyperparameters."""
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
            # Placeholder - implement WideResNet if needed
            from torchvision.models import resnet18

            model = resnet18(num_classes=num_classes)
            logger.warning("WideResNet not implemented, using ResNet18")

        return model

    return model_factory


def create_hpo_config(args: argparse.Namespace) -> HPOConfig:
    """
    Create HPO configuration from arguments.

    Args:
        args: Command line arguments

    Returns:
        HPO configuration
    """
    config = create_default_hpo_config()

    # Update from arguments
    config.study_name = args.study_name
    config.n_trials = args.n_trials if not args.quick_test else 10
    config.storage_url = args.storage

    # Use TRADES search space
    config.search_space = TRADESSearchSpace()

    logger.info(f"Created HPO config with {config.n_trials} trials")

    return config


def create_objective_function(args: argparse.Namespace) -> WeightedTriObjective:
    """
    Create objective function from arguments.

    Args:
        args: Command line arguments

    Returns:
        Objective function
    """
    obj_config = ObjectiveConfig(
        weights={
            "robust_accuracy": args.robust_weight,
            "clean_accuracy": args.clean_weight,
            "cross_site_auroc": args.auroc_weight,
        }
    )

    objective_fn = WeightedTriObjective(config=obj_config)

    logger.info(
        f"Created objective function with weights: "
        f"robust={args.robust_weight}, clean={args.clean_weight}, "
        f"auroc={args.auroc_weight}"
    )

    return objective_fn


def run_hpo_study(
    args: argparse.Namespace,
    hpo_config: HPOConfig,
    objective_fn: WeightedTriObjective,
    model_factory,
    train_loader: DataLoader,
    val_loader: DataLoader,
    test_loader: DataLoader,
    device: torch.device,
) -> TRADESHPOTrainer:
    """
    Run HPO study.

    Args:
        args: Command line arguments
        hpo_config: HPO configuration
        objective_fn: Objective function
        model_factory: Model factory
        train_loader: Training data loader
        val_loader: Validation data loader
        test_loader: Test data loader
        device: Device to use

    Returns:
        Completed HPO trainer
    """
    logger.info("Initializing HPO trainer...")

    # Create checkpoint directory
    checkpoint_dir = Path(args.checkpoint_dir)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Create trainer
    n_epochs = args.n_epochs if not args.quick_test else 2

    trainer = TRADESHPOTrainer(
        config=hpo_config,
        objective_fn=objective_fn,
        model_factory=model_factory,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        device=device,
        n_epochs=n_epochs,
        checkpoint_dir=checkpoint_dir,
    )

    # Create study
    logger.info("Creating Optuna study...")
    trainer.create_study(
        study_name=args.study_name,
        storage=args.storage,
        load_if_exists=True,
    )

    # Run study
    logger.info(f"Starting HPO study with {hpo_config.n_trials} trials...")
    start_time = time.time()

    trainer.run_study(n_trials=hpo_config.n_trials)

    elapsed_time = time.time() - start_time
    logger.info(f"HPO study completed in {elapsed_time / 3600:.2f} hours")

    # Print best trial
    logger.info("=" * 80)
    logger.info("BEST TRIAL RESULTS")
    logger.info("=" * 80)
    logger.info(f"Trial number: {trainer.study.best_trial.number}")
    logger.info(f"Objective value: {trainer.study.best_value:.4f}")
    logger.info("Best hyperparameters:")
    for key, value in trainer.study.best_params.items():
        logger.info(f"  {key}: {value}")

    if trainer.best_trial_metrics:
        logger.info("\nBest trial metrics:")
        logger.info(
            f"  Robust accuracy: {trainer.best_trial_metrics.robust_accuracy:.4f}"
        )
        logger.info(
            f"  Clean accuracy: {trainer.best_trial_metrics.clean_accuracy:.4f}"
        )
        logger.info(
            f"  Cross-site AUROC: {trainer.best_trial_metrics.cross_site_auroc:.4f}"
        )

    logger.info("=" * 80)

    return trainer


def main():
    """Main execution function."""
    # Parse arguments
    args = parse_arguments()

    # Setup directories
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path("logs").mkdir(exist_ok=True)

    logger.info("=" * 80)
    logger.info("TRADES HYPERPARAMETER OPTIMIZATION - PHASE 5.4")
    logger.info("=" * 80)
    logger.info(f"Study name: {args.study_name}")
    logger.info(f"Number of trials: {args.n_trials if not args.quick_test else 10}")
    logger.info(f"Epochs per trial: {args.n_epochs if not args.quick_test else 2}")
    logger.info(f"Dataset: {args.dataset}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Quick test mode: {args.quick_test}")
    logger.info("=" * 80)

    # Setup device
    device = setup_device(args)

    # Load dataset
    train_loader, val_loader, test_loader, num_classes = load_dataset(args)

    # Create model factory
    model_factory = create_model_factory(args, num_classes)

    # Create HPO config
    hpo_config = create_hpo_config(args)

    # Create objective function
    objective_fn = create_objective_function(args)

    # Run HPO study
    trainer = run_hpo_study(
        args,
        hpo_config,
        objective_fn,
        model_factory,
        train_loader,
        val_loader,
        test_loader,
        device,
    )

    # Analyze results
    if not args.skip_analysis:
        logger.info("Analyzing HPO results...")
        output_dir = Path(args.output_dir) / "analysis"
        analyze_study(
            trainer.study,
            hpo_config,
            output_dir,
        )
        logger.info(f"Analysis results saved to {output_dir}")

    logger.info("=" * 80)
    logger.info("HPO STUDY COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
