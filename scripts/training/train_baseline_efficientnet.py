"""
Baseline EfficientNet Training for Transferability Study - Phase 4.4
=====================================================================

Train EfficientNet-B0 as a secondary baseline architecture to study
adversarial transferability across different model families.

Transferability Study Motivation:
----------------------------------
Understanding adversarial transferability is crucial because:
1. Black-box attacks rely on transferability
2. Ensemble defenses require understanding cross-model vulnerabilities
3. Medical imaging requires robustness across deployment architectures

Experimental Protocol:
----------------------
- Architecture: EfficientNet-B0 (different from ResNet-50)
- Dataset: ISIC 2018 (same as ResNet-50 baseline)
- Training: Same hyperparameters as ResNet-50 for fair comparison
- Seeds: 42, 123, 456 (statistical rigor)
- Checkpoint: Best validation accuracy model

Expected Results:
-----------------
- Clean accuracy: 80-87% (similar to ResNet-50)
- Different learned features due to architecture differences
- Enables transferability analysis in Phase 4.4

Reference Standards:
--------------------
- Tan & Le (2019): EfficientNet architecture
- Papernot et al. (2016): Transferability properties
- Demontis et al. (2019): Why Do Adversarial Attacks Transfer?

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Date: November 24, 2025
Version: 4.4.0
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from tqdm import tqdm

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.datasets.isic import ISICDataset
from src.datasets.transforms import get_test_transforms, get_train_transforms
from src.models.build import build_model
from src.utils.reproducibility import set_global_seed

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def train_epoch(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    optimizer: optim.Optimizer,
    device: str,
    epoch: int,
    scaler: Optional[GradScaler] = None,
) -> Dict[str, float]:
    """
    Train for one epoch with optional mixed precision.

    Args:
        model: Model to train
        dataloader: Training dataloader
        criterion: Loss function
        optimizer: Optimizer
        device: Device for computation
        epoch: Current epoch number
        scaler: Gradient scaler for AMP (None = no AMP)

    Returns:
        Dictionary with training metrics
    """
    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]", leave=False)
    for batch_idx, batch in enumerate(pbar):
        # Handle different batch formats
        if len(batch) == 3:
            images, labels, _ = batch
        else:
            images, labels = batch

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        # ============================================================
        # MIXED PRECISION TRAINING (CORRECT IMPLEMENTATION)
        # ============================================================
        if scaler is not None:
            # AMP enabled
            with autocast():
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backward with scaler
            scaler.scale(loss).backward()

            # Gradient clipping before unscaling
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

            # Optimizer step with scaler
            scaler.step(optimizer)
            scaler.update()
        else:
            # Standard training (no AMP)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()

            # Gradient clipping
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        # ============================================================

        # Metrics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

        # Update progress bar
        if batch_idx % 10 == 0:
            pbar.set_postfix(
                {
                    "loss": f"{running_loss / (batch_idx + 1):.4f}",
                    "acc": f"{100.0 * correct / total:.2f}%",
                }
            )

    metrics = {
        "loss": running_loss / len(dataloader),
        "accuracy": 100.0 * correct / total,
    }

    return metrics


def validate(
    model: nn.Module,
    dataloader: DataLoader,
    criterion: nn.Module,
    device: str,
) -> Dict[str, float]:
    """
    Validate model.

    Args:
        model: Model to validate
        dataloader: Validation dataloader
        criterion: Loss function
        device: Device for computation

    Returns:
        Dictionary with validation metrics
    """
    model.eval()

    running_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        pbar = tqdm(dataloader, desc="Validation", leave=False)
        for batch_idx, batch in enumerate(pbar):
            # Handle different batch formats
            if len(batch) == 3:
                images, labels, _ = batch
            else:
                images, labels = batch

            images = images.to(device)
            labels = labels.to(device)

            # Forward pass
            outputs = model(images)
            loss = criterion(outputs, labels)

            # Metrics
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()

            # Update progress bar
            pbar.set_postfix(
                {
                    "loss": f"{running_loss / (batch_idx + 1):.4f}",
                    "acc": f"{100.0 * correct / total:.2f}%",
                }
            )

    metrics = {
        "loss": running_loss / len(dataloader),
        "accuracy": 100.0 * correct / total,
    }

    return metrics


def train_efficientnet_baseline(
    data_root: Path,
    output_dir: Path,
    seed: int = 42,
    num_epochs: int = 50,
    batch_size: int = 32,
    learning_rate: float = 1e-4,
    weight_decay: float = 1e-4,
    device: str = "cuda",
) -> None:
    """
    Train EfficientNet-B0 baseline with error handling and AMP.

    Args:
        data_root: Root directory for ISIC dataset
        output_dir: Output directory for checkpoints
        seed: Random seed for reproducibility
        num_epochs: Number of training epochs
        batch_size: Batch size
        learning_rate: Learning rate
        weight_decay: Weight decay for regularization
        device: Device for computation
    """
    try:
        # Set seed for reproducibility
        set_global_seed(seed)

        # Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=" * 80)
        logger.info("EFFICIENTNET-B0 BASELINE TRAINING - PHASE 4.4")
        logger.info("=" * 80)
        logger.info(f"Data Root: {data_root}")
        logger.info(f"Output Directory: {output_dir}")
        logger.info(f"Seed: {seed}")
        logger.info(f"Epochs: {num_epochs}")
        logger.info(f"Batch Size: {batch_size}")
        logger.info(f"Learning Rate: {learning_rate}")
        logger.info(f"Device: {device}")
        logger.info("=" * 80)

        # Load datasets
        logger.info("\nLoading ISIC 2018 dataset...")
        try:
            train_transforms = get_train_transforms(dataset="isic", image_size=224)
            val_transforms = get_test_transforms(dataset="isic", image_size=224)

            train_dataset = ISICDataset(
                root=str(data_root),
                split="train",
                csv_path=str(data_root / "metadata_processed.csv"),
                transforms=train_transforms,
            )

            val_dataset = ISICDataset(
                root=str(data_root),
                split="val",
                csv_path=str(data_root / "metadata_processed.csv"),
                transforms=val_transforms,
            )

            train_loader = DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=True,
                num_workers=0,  # Avoid Windows multiprocessing issues
                pin_memory=True if torch.cuda.is_available() else False,
            )

            val_loader = DataLoader(
                val_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=0,
                pin_memory=True if torch.cuda.is_available() else False,
            )

            logger.info(
                f"✅ Dataset loaded: Train={len(train_dataset)}, Val={len(val_dataset)}"
            )
        except Exception as e:
            logger.error(f"❌ Failed to load dataset: {e}")
            raise

        # Build model
        logger.info("\nBuilding EfficientNet-B0 model...")
        try:
            num_classes = 7  # ISIC 2018
            model = build_model(
                "efficientnet_b0", num_classes=num_classes, pretrained=True
            )
            model.to(device)

            # VERIFY MODEL
            logger.info(f"Model class: {model.__class__.__name__}")
            dummy_input = torch.randn(2, 3, 224, 224).to(device)
            with torch.no_grad():
                dummy_output = model(dummy_input)
            assert dummy_output.shape == (
                2,
                num_classes,
            ), f"Wrong output shape: {dummy_output.shape}, expected (2, {num_classes})"
            logger.info(f"✅ Model verification passed: {dummy_output.shape}")

            # Count parameters
            total_params = sum(p.numel() for p in model.parameters())
            trainable_params = sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )
            logger.info(f"Total parameters: {total_params:,}")
            logger.info(f"Trainable parameters: {trainable_params:,}")
        except Exception as e:
            logger.error(f"❌ Failed to build model: {e}")
            raise

        # Loss and optimizer
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.AdamW(
            model.parameters(),
            lr=learning_rate,
            weight_decay=weight_decay,
        )

        # Learning rate scheduler
        scheduler = optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=num_epochs,
            eta_min=1e-6,
        )

        # Mixed precision scaler
        scaler = GradScaler() if device == "cuda" else None
        logger.info(
            f"Mixed Precision: {'Enabled (AMP)' if scaler else 'Disabled (CPU)'}"
        )

        # Training loop
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING")
        logger.info("=" * 80)

        best_val_acc = 0.0
        best_epoch = 0

        for epoch in range(1, num_epochs + 1):
            epoch_start = time.time()

            try:
                # Train
                train_metrics = train_epoch(
                    model, train_loader, criterion, optimizer, device, epoch, scaler
                )

                # Validate
                val_metrics = validate(model, val_loader, criterion, device)

                # Update scheduler
                scheduler.step()

                epoch_time = time.time() - epoch_start

                # Log metrics
                logger.info(
                    f"Epoch {epoch:02d}/{num_epochs} | "
                    f"Train Loss: {train_metrics['loss']:.4f} | "
                    f"Train Acc: {train_metrics['accuracy']:.2f}% | "
                    f"Val Loss: {val_metrics['loss']:.4f} | "
                    f"Val Acc: {val_metrics['accuracy']:.2f}% | "
                    f"Time: {epoch_time:.1f}s | "
                    f"LR: {optimizer.param_groups[0]['lr']:.2e}"
                )

                # Save best model
                if val_metrics["accuracy"] > best_val_acc:
                    best_val_acc = val_metrics["accuracy"]
                    best_epoch = epoch

                    checkpoint = {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "scheduler_state_dict": scheduler.state_dict(),
                        "train_metrics": train_metrics,
                        "val_metrics": val_metrics,
                        "seed": seed,
                        "architecture": "efficientnet_b0",
                    }

                    torch.save(checkpoint, output_dir / "best.pt")
                    logger.info(
                        f"   ✅ New best model saved (Val Acc: {best_val_acc:.2f}%)"
                    )

                # Save last checkpoint
                checkpoint = {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "scheduler_state_dict": scheduler.state_dict(),
                    "train_metrics": train_metrics,
                    "val_metrics": val_metrics,
                    "seed": seed,
                    "architecture": "efficientnet_b0",
                }
                torch.save(checkpoint, output_dir / "last.pt")

            except Exception as e:
                logger.error(f"❌ Training failed at epoch {epoch}: {e}")
                # Save emergency checkpoint
                emergency_path = output_dir / f"emergency_epoch_{epoch}.pt"
                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "error": str(e),
                    },
                    emergency_path,
                )
                logger.info(f"Emergency checkpoint saved: {emergency_path}")
                raise

        # Training complete
        logger.info("\n" + "=" * 80)
        logger.info("TRAINING COMPLETE")
        logger.info("=" * 80)
        logger.info(
            f"Best Validation Accuracy: {best_val_acc:.2f}% (Epoch {best_epoch})"
        )
        logger.info(f"Model saved to: {output_dir / 'best.pt'}")
        logger.info("=" * 80)

    except KeyboardInterrupt:
        logger.warning("\n⚠️ Training interrupted by user (Ctrl+C)")
        logger.info("Saving current state...")

        # Save interrupted checkpoint
        interrupted_path = output_dir / "interrupted.pt"
        torch.save(
            {
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "message": "Training interrupted by user",
            },
            interrupted_path,
        )
        logger.info(f"✅ Interrupted state saved: {interrupted_path}")

        logger.info("You can resume training later if needed.")
        sys.exit(0)

    except Exception as e:
        logger.error(f"\n❌ FATAL ERROR: {e}")
        logger.error("Training failed!")
        import traceback

        logger.error(f"Traceback:\n{traceback.format_exc()}")
        sys.exit(1)


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Train EfficientNet-B0 Baseline - Phase 4.4",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=PROJECT_ROOT / "data" / "processed" / "isic2018",
        help="Root directory for ISIC dataset",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=PROJECT_ROOT / "checkpoints" / "efficientnet_baseline",
        help="Output directory for checkpoints",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num-epochs",
        type=int,
        default=50,
        help="Number of training epochs",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for training",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        default=1e-4,
        help="Learning rate",
    )
    parser.add_argument(
        "--weight-decay",
        type=float,
        default=1e-4,
        help="Weight decay for regularization",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        choices=["cuda", "cpu"],
        help="Device for computation",
    )

    args = parser.parse_args()

    # Create seed-specific output directory
    output_dir = args.output_dir / f"seed_{args.seed}"

    # Train model
    train_efficientnet_baseline(
        data_root=args.data_root,
        output_dir=output_dir,
        seed=args.seed,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
        weight_decay=args.weight_decay,
        device=args.device,
    )


if __name__ == "__main__":
    main()
