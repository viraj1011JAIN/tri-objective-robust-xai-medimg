"""Training script for baseline models."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Subset, TensorDataset

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# Import project modules
from src.datasets.isic import ISICDataset  # noqa: F401, E402
from src.models.build import build_model  # noqa: E402
from src.training.base_trainer import TrainingConfig  # noqa: E402
from src.training.baseline_trainer import BaselineTrainer  # noqa: E402
from src.utils.config import load_experiment_config  # noqa: E402
from src.utils.reproducibility import set_global_seed  # noqa: E402

# Optional MLflow import
try:
    import mlflow
except ImportError:
    mlflow = None

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def setup_logging(log_dir: Path) -> None:
    """
    Configure logging to both file and stdout.

    Parameters
    ----------
    log_dir:
        Directory where the ``train.log`` file will be written.
    """
    log_dir.mkdir(parents=True, exist_ok=True)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_dir / "train.log"),
            logging.StreamHandler(),
        ],
    )


def create_dataloaders(
    batch_size: int, dataset: str, use_real_data: bool = True
) -> Tuple[DataLoader, DataLoader, int]:
    """
    Create train/validation DataLoaders.

    ⚠️ CRITICAL FIX: This function now loads REAL medical imaging data by default.
    Previous version used synthetic random noise, causing 0% accuracy on real images.

    Parameters
    ----------
    batch_size:
        Batch size for both train and validation loaders.
    dataset:
        Dataset name. Supported (case-insensitive):
        - "isic2018" (and "isic", "isic_2018")
        - "nih_chestxray14" (and "chest_x_ray", "chestxray14")
    use_real_data:
        If True (DEFAULT), load REAL medical images from disk.
        If False, use synthetic data (ONLY for unit tests).

    Returns
    -------
    (train_loader, val_loader, num_classes)
    """
    from pathlib import Path

    from src.datasets.isic import ISICDataset
    from src.datasets.transforms import get_test_transforms, get_train_transforms

    name = dataset.lower()

    # ========================================================================
    # PRODUCTION MODE: Load REAL medical imaging data
    # ========================================================================
    if use_real_data and name in {"isic2018", "isic_2018", "isic"}:
        logger.info("=" * 80)
        logger.info("🩺 LOADING REAL ISIC2018 MEDICAL IMAGING DATA")
        logger.info("=" * 80)

        num_classes = 7
        data_root = Path("data/processed/isic2018")
        csv_path = data_root / "metadata_processed.csv"

        # Verify data exists
        if not data_root.exists():
            raise FileNotFoundError(
                f"\n❌ REAL DATA NOT FOUND\n"
                f"Expected data root: {data_root.absolute()}\n"
                f"Please run preprocessing first:\n"
                f"  python scripts/data/preprocess_isic2018.py\n"
                f"Or set use_real_data=False for testing with synthetic data."
            )

        if not csv_path.exists():
            raise FileNotFoundError(
                f"\n❌ METADATA CSV NOT FOUND\n"
                f"Expected: {csv_path.absolute()}\n"
                f"Please run preprocessing first."
            )

        # Load transforms
        train_transforms = get_train_transforms(dataset="isic", image_size=224)
        test_transforms = get_test_transforms(dataset="isic", image_size=224)

        # Load datasets
        train_dataset = ISICDataset(
            root=str(data_root),
            split="train",
            csv_path=str(csv_path),
            transforms=train_transforms,
        )

        val_dataset = ISICDataset(
            root=str(data_root),
            split="val",
            csv_path=str(csv_path),
            transforms=test_transforms,
        )

        # Create dataloaders
        train_loader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=0,  # Avoid Windows multiprocessing issues
            pin_memory=True,
        )
        val_loader = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=0,
            pin_memory=True,
        )

        logger.info(f"✅ LOADED REAL DATA:")
        logger.info(f"   Train: {len(train_dataset):,} samples")
        logger.info(f"   Val:   {len(val_dataset):,} samples")
        logger.info(f"   Classes: {num_classes}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info("=" * 80)

        return train_loader, val_loader, num_classes

    # ========================================================================
    # TEST MODE: Synthetic data (ONLY for unit tests)
    # ========================================================================
    logger.warning("⚠️  USING SYNTHETIC DATA (for testing only)")
    logger.warning("⚠️  This will NOT work for real evaluation!")

    if name in {"isic2018", "isic_2018", "isic"}:
        num_samples = 256
        num_classes = 7
        channels = 3
    elif name in {
        "nih_chestxray14",
        "chest_x_ray",
        "chest_xray",
        "chestxray14",
    }:
        num_samples = 512
        num_classes = 14
        channels = 1
    else:
        raise ValueError(f"Unknown dataset: {dataset!r}")

    images = torch.randn(num_samples, channels, 224, 224)
    labels = torch.randint(0, num_classes, (num_samples,))

    full_dataset = TensorDataset(images, labels)

    train_len = int(num_samples * 0.8)
    val_len = int(num_samples * 0.2)

    train_indices = list(range(train_len))
    val_indices = list(range(train_len, train_len + val_len))

    train_dataset = Subset(full_dataset, train_indices)
    val_dataset = Subset(full_dataset, val_indices)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader, num_classes


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def _cfg_from_experiment_object(cfg_obj: Any, device_fallback: str) -> Dict[str, Any]:
    """
    Convert a pydantic-style experiment object into a plain Python dict.

    This keeps ``train_baseline.py`` decoupled from the exact pydantic model
    definitions in ``src.utils.config`` and makes testing simpler.
    """
    experiment = cfg_obj.experiment.model_dump()
    model = cfg_obj.model.model_dump()
    dataset = cfg_obj.dataset.model_dump()
    training = cfg_obj.training.model_dump()

    # Ensure device key is present
    training.setdefault("device", device_fallback)

    return {
        "experiment": experiment,
        "model": model,
        "dataset": dataset,
        "training": training,
    }


def main(args):
    """Main training function."""
    set_global_seed(args.seed)
    setup_logging(Path(args.log_dir))

    logger.info("=" * 80)
    logger.info("Baseline Training | seed=%d | device=%s", args.seed, args.device)
    logger.info("=" * 80)

    # ------------------------------------------------------------------
    # Config
    # ------------------------------------------------------------------
    if args.config:
        cfg_obj = load_experiment_config(args.config)
        cfg: Dict[str, Any] = _cfg_from_experiment_object(cfg_obj, args.device)
    else:
        # Minimal default configuration for ad-hoc smoke tests
        cfg = {
            "experiment": {"name": "baseline"},
            "model": {"name": "resnet50", "num_classes": 7, "pretrained": True},
            "dataset": {"name": "isic2018", "batch_size": 32},
            "training": {
                "max_epochs": 10,
                "learning_rate": 1e-3,
                "weight_decay": 1e-5,
                "device": args.device,
                "early_stopping_patience": 5,
            },
        }

    exp_name = cfg.get("experiment", {}).get("name", "baseline")
    mlflow.set_experiment(exp_name)

    device = torch.device(cfg["training"].get("device", args.device))

    checkpoint_dir = Path(args.checkpoint_dir) / f"seed_{args.seed}"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    with mlflow.start_run(run_name=f"seed_{args.seed}"):
        # Log high-level hyperparameters
        mlflow.log_params(
            {
                "seed": args.seed,
                "model": cfg["model"]["name"],
                "dataset": cfg["dataset"]["name"],
                "lr": cfg["training"]["learning_rate"],
                "weight_decay": cfg["training"].get("weight_decay", 1e-5),
                "max_epochs": cfg["training"]["max_epochs"],
            }
        )

        # ------------------------------------------------------------------
        # Model & data
        # ------------------------------------------------------------------
        train_loader, val_loader, num_classes = create_dataloaders(
            batch_size=cfg["dataset"].get("batch_size", 32),
            dataset=cfg["dataset"]["name"],
        )

        model: nn.Module = build_model(
            name=cfg["model"]["name"],
            num_classes=num_classes,
            pretrained=cfg["model"].get("pretrained", True),
        )
        model.to(device)

        # ------------------------------------------------------------------
        # Optimiser & scheduler
        # ------------------------------------------------------------------
        optimizer = Adam(
            model.parameters(),
            lr=cfg["training"]["learning_rate"],
            weight_decay=cfg["training"].get("weight_decay", 1e-5),
        )

        scheduler = CosineAnnealingLR(
            optimizer,
            T_max=cfg["training"]["max_epochs"],
        )

        # ------------------------------------------------------------------
        # Trainer
        # ------------------------------------------------------------------
        trainer_cfg = TrainingConfig(
            max_epochs=cfg["training"]["max_epochs"],
            eval_every_n_epochs=cfg["training"].get("eval_every_n_epochs", 1),
            log_every_n_steps=cfg["training"].get("log_every_n_steps", 10),
            early_stopping_patience=cfg["training"]["early_stopping_patience"],
            gradient_clip_val=cfg["training"].get("gradient_clip_val", 1.0),
            checkpoint_dir=str(checkpoint_dir),
        )

        trainer = BaselineTrainer(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            optimizer=optimizer,
            config=trainer_cfg,
            num_classes=num_classes,
            scheduler=scheduler,
            device=device,
            checkpoint_dir=checkpoint_dir,
        )

        # ------------------------------------------------------------------
        # Training
        # ------------------------------------------------------------------
        logger.info("Starting training loop…")
        history = trainer.fit()

        # ------------------------------------------------------------------
        # Persist results
        # ------------------------------------------------------------------
        results = {
            "seed": args.seed,
            "model": cfg["model"]["name"],
            "dataset": cfg["dataset"]["name"],
            "best_epoch": trainer.best_epoch,
            "best_val_loss": float(trainer.best_val_loss),
            "history": {k: [float(v) for v in vals] for k, vals in history.items()},
        }

        results_dir = Path(args.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)

        results_file = (
            results_dir
            / f"{cfg['model']['name']}_{cfg['dataset']['name']}_seed{args.seed}.json"
        )

        with results_file.open("w", encoding="utf-8") as f:
            json.dump(results, f, indent=2)

        logger.info("Results saved to %s", results_file)

        # Log summary metrics to MLflow
        if history.get("val_loss"):  # True + False branches both tested
            mlflow.log_metric("final_val_loss", history["val_loss"][-1])
        mlflow.log_metric("best_val_loss", trainer.best_val_loss)
        mlflow.log_metric("best_epoch", trainer.best_epoch)

        logger.info("Training run finished.")


if __name__ == "__main__":  # pragma: no cover - CLI wrapper
    parser = argparse.ArgumentParser(
        description="Train baseline models on medical imaging datasets."
    )

    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to experiment config YAML file.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Compute device to use (e.g. 'cuda' or 'cpu').",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=str,
        default="checkpoints/baseline",
        help="Directory to save checkpoints.",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default="results/baseline",
        help="Directory to save JSON result summaries.",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default="logs/baseline",
        help="Directory to save training logs.",
    )

    main(parser.parse_args())
