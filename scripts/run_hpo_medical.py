"""
HPO Study for Medical Imaging Datasets.

This script runs TRADES hyperparameter optimization on medical imaging datasets:
- ISIC 2018/2019/2020 (Dermatology)
- Derm7pt (Dermoscopic images)
- NIH ChestX-ray14 (Radiology)
- PadChest (Chest X-rays)

Adapts the HPO pipeline from run_hpo_study.py to work with medical datasets.

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Date: November 24, 2025
Version: 5.4.1 (Medical Datasets)
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Subset

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.datasets import ChestXRayDataset, Derm7ptDataset, ISICDataset
from src.datasets.transforms import (
    get_chest_xray_transforms,
    get_derm7pt_transforms,
    get_isic_transforms,
)
from src.training.hpo_analysis import analyze_study
from src.training.hpo_config import HPOConfig, create_default_hpo_config
from src.training.hpo_objective import ObjectiveConfig, WeightedTriObjective
from src.training.hpo_trainer import TRADESHPOTrainer

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("logs/hpo_medical.log"),
        logging.StreamHandler(sys.stdout),
    ],
)
logger = logging.getLogger(__name__)


# Medical dataset configurations
MEDICAL_DATASET_CONFIGS = {
    "isic2018": {
        "name": "ISIC 2018",
        "num_classes": 7,
        "img_size": 224,
        "dataset_class": ISICDataset,
        "dataset_kwargs": {
            "root": "data/processed/isic_2018",
            "csv_path": "data/processed/isic_2018/metadata.csv",
        },
        "transform_fn": get_isic_transforms,
    },
    "isic2019": {
        "name": "ISIC 2019",
        "num_classes": 8,
        "img_size": 224,
        "dataset_class": ISICDataset,
        "dataset_kwargs": {
            "root": "data/processed/isic_2019",
            "csv_path": "data/processed/isic_2019/metadata.csv",
        },
        "transform_fn": get_isic_transforms,
    },
    "isic2020": {
        "name": "ISIC 2020",
        "num_classes": 1,  # Binary classification
        "img_size": 224,
        "dataset_class": ISICDataset,
        "dataset_kwargs": {
            "root": "data/processed/isic_2020",
            "csv_path": "data/processed/isic_2020/metadata.csv",
        },
        "transform_fn": get_isic_transforms,
    },
    "derm7pt": {
        "name": "Derm7pt",
        "num_classes": 2,  # Binary classification
        "img_size": 224,
        "dataset_class": Derm7ptDataset,
        "dataset_kwargs": {
            "root": "data/processed/derm7pt",
            "csv_path": "data/processed/derm7pt/metadata.csv",
        },
        "transform_fn": get_derm7pt_transforms,
    },
    "nih_cxr": {
        "name": "NIH ChestX-ray14",
        "num_classes": 14,  # Multi-label
        "img_size": 224,
        "dataset_class": ChestXRayDataset,
        "dataset_kwargs": {
            "root": "data/processed/nih_cxr",
            "csv_path": "data/processed/nih_cxr/metadata.csv",
        },
        "transform_fn": get_chest_xray_transforms,
    },
    "padchest": {
        "name": "PadChest",
        "num_classes": 14,  # Multi-label
        "img_size": 224,
        "dataset_class": ChestXRayDataset,
        "dataset_kwargs": {
            "root": "data/processed/padchest",
            "csv_path": "data/processed/padchest/metadata.csv",
        },
        "transform_fn": get_chest_xray_transforms,
    },
}


def parse_arguments() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Run TRADES HPO on Medical Imaging Datasets"
    )

    # Dataset selection
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=list(MEDICAL_DATASET_CONFIGS.keys()),
        help="Medical dataset to use for HPO",
    )

    # Study configuration
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Name for the HPO study (auto-generated if not provided)",
    )
    parser.add_argument("--n-trials", type=int, default=50, help="Number of HPO trials")
    parser.add_argument("--n-epochs", type=int, default=10, help="Epochs per trial")
    parser.add_argument(
        "--storage",
        type=str,
        default="sqlite:///hpo_medical.db",
        help="Optuna storage URL",
    )

    # Training configuration
    parser.add_argument(
        "--batch-size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--num-workers", type=int, default=4, help="Number of data loading workers"
    )

    # Model configuration
    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["resnet18", "resnet34", "resnet50"],
        help="Model architecture",
    )
    parser.add_argument(
        "--pretrained", action="store_true", help="Use ImageNet pretrained weights"
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
        default="results/phase_5_4_medical",
        help="Output directory",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/hpo_medical",
        help="Checkpoint directory",
    )

    # Objective weights
    parser.add_argument(
        "--robust-weight", type=float, default=0.4, help="Weight for robust accuracy"
    )
    parser.add_argument(
        "--clean-weight", type=float, default=0.3, help="Weight for clean accuracy"
    )
    parser.add_argument(
        "--auroc-weight", type=float, default=0.3, help="Weight for cross-site AUROC"
    )

    # Special modes
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Quick test mode (10 trials, 2 epochs, subset data)",
    )
    parser.add_argument(
        "--skip-analysis", action="store_true", help="Skip analysis generation"
    )

    return parser.parse_args()


def setup_device(args: argparse.Namespace) -> str:
    """Setup compute device."""
    if args.device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    else:
        device = args.device

    if device == "cuda":
        torch.cuda.set_device(args.gpu_id)
        logger.info(f"Using GPU: {torch.cuda.get_device_name(args.gpu_id)}")
        memory_gb = torch.cuda.get_device_properties(args.gpu_id).total_memory / 1e9
        logger.info(f"Memory: {memory_gb:.2f} GB")
    else:
        logger.info("Using CPU")

    return device


def load_medical_dataset(dataset_name: str, args: argparse.Namespace) -> tuple:
    """
    Load medical imaging dataset.

    Returns:
        (train_loader, val_loader, test_loader, dataset_config)
    """
    config = MEDICAL_DATASET_CONFIGS[dataset_name]
    logger.info(f"Loading {config['name']} dataset...")

    dataset_class = config["dataset_class"]
    dataset_kwargs = config["dataset_kwargs"]
    transform_fn = config["transform_fn"]
    img_size = config["img_size"]

    try:
        # Create transforms for each split
        train_transforms = transform_fn("train", image_size=img_size)
        val_transforms = transform_fn("val", image_size=img_size)
        test_transforms = transform_fn("test", image_size=img_size)

        # Create datasets
        train_dataset = dataset_class(
            split="train", transforms=train_transforms, **dataset_kwargs
        )
        val_dataset = dataset_class(
            split="val", transforms=val_transforms, **dataset_kwargs
        )
        test_dataset = dataset_class(
            split="test", transforms=test_transforms, **dataset_kwargs
        )

        # Quick test mode: use subsets
        if args.quick_test:
            train_size = min(800, len(train_dataset))
            val_size = min(200, len(val_dataset))
            test_size = min(500, len(test_dataset))

            train_dataset = Subset(train_dataset, range(train_size))
            val_dataset = Subset(val_dataset, range(val_size))
            test_dataset = Subset(test_dataset, range(test_size))

            logger.info("Quick test mode: using data subsets")

        # Create dataloaders
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
            f"Loaded: {len(train_dataset)} train, "
            f"{len(val_dataset)} val, "
            f"{len(test_dataset)} test samples"
        )

        return train_loader, val_loader, test_loader, config

    except Exception as e:
        logger.error(f"Failed to load dataset: {e}")
        logger.error(
            f"Please ensure {config['name']} dataset is downloaded and processed"
        )
        raise


def create_model_factory(
    model_name: str, num_classes: int, pretrained: bool = False
) -> Callable[[], nn.Module]:
    """Create model factory function."""

    def factory():
        if model_name == "resnet18":
            model = models.resnet18(pretrained=pretrained)
        elif model_name == "resnet34":
            model = models.resnet34(pretrained=pretrained)
        elif model_name == "resnet50":
            model = models.resnet50(pretrained=pretrained)
        else:
            raise ValueError(f"Unknown model: {model_name}")

        # Replace final layer for medical imaging
        model.fc = nn.Linear(model.fc.in_features, num_classes)
        return model

    return factory


def create_hpo_config(args: argparse.Namespace) -> HPOConfig:
    """Create HPO configuration."""
    config = create_default_hpo_config()

    # Override with command line arguments
    if args.study_name:
        config.study_name = args.study_name
    else:
        config.study_name = f"trades_hpo_{args.dataset}"

    config.n_trials = args.n_trials if not args.quick_test else 10
    config.storage_url = args.storage

    logger.info(f"Created HPO config with {config.n_trials} trials")
    return config


def create_objective_function(args: argparse.Namespace) -> WeightedTriObjective:
    """Create objective function."""
    objective = WeightedTriObjective(
        robust_weight=args.robust_weight,
        clean_weight=args.clean_weight,
        auroc_weight=args.auroc_weight,
    )

    logger.info(
        f"Created objective: robust={args.robust_weight}, "
        f"clean={args.clean_weight}, auroc={args.auroc_weight}"
    )
    return objective


def run_hpo_study(
    train_loader: DataLoader,
    val_loader: DataLoader,
    model_factory: Callable,
    hpo_config: HPOConfig,
    objective_function: WeightedTriObjective,
    n_epochs: int,
    device: str,
    checkpoint_dir: Path,
) -> TRADESHPOTrainer:
    """Run HPO study."""
    logger.info("Initializing HPO trainer...")

    trainer = TRADESHPOTrainer(
        model_factory=model_factory,
        train_loader=train_loader,
        val_loader=val_loader,
        config=hpo_config,
        objective_function=objective_function,
        n_epochs=n_epochs,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    logger.info("Creating Optuna study...")
    trainer.create_study()

    logger.info(f"Starting HPO study with {hpo_config.n_trials} trials...")
    start_time = time.time()

    trainer.run_study(n_trials=hpo_config.n_trials)

    elapsed = (time.time() - start_time) / 3600
    logger.info(f"HPO study completed in {elapsed:.2f} hours")

    return trainer


def main():
    """Main execution function."""
    args = parse_arguments()

    # Create output directories
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = Path(args.checkpoint_dir) / args.dataset
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    # Log configuration
    dataset_config = MEDICAL_DATASET_CONFIGS[args.dataset]
    logger.info("=" * 80)
    logger.info("TRADES HPO - MEDICAL IMAGING")
    logger.info("=" * 80)
    logger.info(f"Dataset: {dataset_config['name']}")
    logger.info(f"Study name: {args.study_name or f'trades_hpo_{args.dataset}'}")
    logger.info(f"Trials: {args.n_trials if not args.quick_test else 10}")
    logger.info(f"Epochs per trial: {args.n_epochs if not args.quick_test else 2}")
    logger.info(f"Model: {args.model} (pretrained={args.pretrained})")
    logger.info(f"Quick test mode: {args.quick_test}")
    logger.info("=" * 80)

    # Setup device
    device = setup_device(args)

    # Load dataset
    train_loader, val_loader, test_loader, ds_config = load_medical_dataset(
        args.dataset, args
    )

    # Create model factory
    model_factory = create_model_factory(
        args.model, ds_config["num_classes"], args.pretrained
    )

    # Create HPO config
    hpo_config = create_hpo_config(args)

    # Create objective function
    objective_function = create_objective_function(args)

    # Run HPO study
    n_epochs = 2 if args.quick_test else args.n_epochs
    trainer = run_hpo_study(
        train_loader=train_loader,
        val_loader=val_loader,
        model_factory=model_factory,
        hpo_config=hpo_config,
        objective_function=objective_function,
        n_epochs=n_epochs,
        device=device,
        checkpoint_dir=checkpoint_dir,
    )

    # Display best trial
    logger.info("=" * 80)
    logger.info("BEST TRIAL RESULTS")
    logger.info("=" * 80)
    logger.info(f"Trial number: {trainer.study.best_trial.number}")
    logger.info(f"Objective value: {trainer.study.best_value:.4f}")
    logger.info("Best hyperparameters:")
    for param, value in trainer.study.best_params.items():
        logger.info(f"  {param}: {value}")
    logger.info("=" * 80)

    # Generate analysis
    if not args.skip_analysis:
        logger.info("Analyzing HPO results...")
        analysis_dir = output_dir / "analysis"
        analysis_dir.mkdir(exist_ok=True)

        analyze_study(
            study=trainer.study,
            output_dir=analysis_dir,
        )
        logger.info(f"Analysis saved to {analysis_dir}")

    logger.info("=" * 80)
    logger.info("HPO STUDY COMPLETED SUCCESSFULLY")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
