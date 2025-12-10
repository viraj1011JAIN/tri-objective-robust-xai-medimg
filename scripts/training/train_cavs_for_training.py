#!/usr/bin/env python3
"""
Train Concept Activation Vectors (CAVs) for Tri-Objective Training
====================================================================

This script trains CAVs BEFORE the main tri-objective training begins.
The trained CAVs are then used in the explanation loss during training.

This is THE MISSING PIECE that makes H2.2-H2.4 work!

Usage:
    python scripts/training/train_cavs_for_training.py \\
        --data_dir data/processed/isic2018 \\
        --model_checkpoint checkpoints/baseline/seed_42/best.pt \\
        --output_dir checkpoints/cavs \\
        --n_concept_samples 50 \\
        --n_random_samples 100

Author: Viraj Pankaj Jain
Date: December 7, 2025
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

# Setup logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# ============================================================================
# Concept Transformations
# ============================================================================


def apply_concept_transform(image: torch.Tensor, concept_type: str) -> torch.Tensor:
    """Apply concept-specific transformation to image."""

    if concept_type == "border_irregularity":
        # Sobel edge enhancement
        sobel_x = torch.tensor(
            [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]],
            dtype=torch.float32,
            device=image.device,
        )
        sobel_y = torch.tensor(
            [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
            dtype=torch.float32,
            device=image.device,
        )
        sobel_x = sobel_x.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        sobel_y = sobel_y.view(1, 1, 3, 3).repeat(3, 1, 1, 1)

        edge_x = F.conv2d(image.unsqueeze(0), sobel_x, padding=1, groups=3)
        edge_y = F.conv2d(image.unsqueeze(0), sobel_y, padding=1, groups=3)
        edges = torch.sqrt(edge_x**2 + edge_y**2 + 1e-6)
        return torch.clamp(0.7 * image.unsqueeze(0) + 0.3 * edges, 0, 1).squeeze(0)

    elif concept_type == "color_variation":
        mean_color = image.mean(dim=0, keepdim=True)
        color_diff = image - mean_color
        enhanced = mean_color + 1.5 * color_diff
        return torch.clamp(enhanced, 0, 1)

    elif concept_type == "texture_heterogeneity":
        kernel = torch.tensor(
            [[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]],
            dtype=torch.float32,
            device=image.device,
        )
        kernel = kernel.view(1, 1, 3, 3).repeat(3, 1, 1, 1)
        texture = F.conv2d(image.unsqueeze(0), kernel, padding=1, groups=3)
        texture = (texture - texture.min()) / (texture.max() - texture.min() + 1e-6)
        return torch.clamp(0.6 * image.unsqueeze(0) + 0.4 * texture, 0, 1).squeeze(0)

    elif concept_type == "asymmetry":
        _, h, w = image.shape
        result = image.clone()
        result[:, :, w // 2 :] = torch.flip(image[:, :, : w // 2], dims=[2])[
            :, :, : (w - w // 2)
        ]
        return result

    elif concept_type == "background":
        # Heavy Gaussian blur
        kernel_size = 15
        sigma = 5.0
        x = (
            torch.arange(kernel_size, dtype=torch.float32, device=image.device)
            - kernel_size // 2
        )
        gaussian_1d = torch.exp(-(x**2) / (2 * sigma**2))
        gaussian_1d = gaussian_1d / gaussian_1d.sum()
        gaussian_2d = gaussian_1d.unsqueeze(0) * gaussian_1d.unsqueeze(1)
        gaussian_2d = gaussian_2d.view(1, 1, kernel_size, kernel_size).repeat(
            3, 1, 1, 1
        )
        padding = kernel_size // 2
        blurred = F.conv2d(image.unsqueeze(0), gaussian_2d, padding=padding, groups=3)
        return blurred.squeeze(0)

    elif concept_type == "ruler_marks":
        result = image.clone()
        _, h, w = image.shape
        for i in range(0, w, 20):
            if i < w:
                result[:, :, i : min(i + 2, w)] *= 0.3
        return result

    elif concept_type == "hair_artifacts":
        result = image.clone()
        _, h, w = image.shape
        np.random.seed(42)
        for _ in range(5):
            start_x = np.random.randint(0, w)
            start_y = np.random.randint(0, h)
            for j in range(min(50, h - start_y)):
                x = int(start_x + 10 * np.sin(j * 0.1))
                y = start_y + j
                if 0 <= x < w and 0 <= y < h:
                    result[:, y, max(0, x - 1) : min(w, x + 2)] *= 0.2
        return result

    else:
        return image


# ============================================================================
# Feature Extraction
# ============================================================================


def extract_features(
    model: nn.Module,
    images: torch.Tensor,
    device: torch.device,
    layer_name: str = "layer4",
) -> np.ndarray:
    """Extract features from specified layer."""
    model.eval()
    activations = []

    # Register hook - handle both direct ResNet and wrapped models
    if hasattr(model, "backbone"):
        # Model has backbone attribute (ResNet50Classifier)
        target_layer = getattr(model.backbone, layer_name)
    elif hasattr(model, layer_name):
        # Direct ResNet model
        target_layer = getattr(model, layer_name)
    else:
        raise ValueError(f"Cannot find layer '{layer_name}' in model")

    def hook_fn(module, input, output):
        activations.append(output)

    handle = target_layer.register_forward_hook(hook_fn)

    with torch.no_grad():
        _ = model(images.to(device))

    handle.remove()

    # Global average pooling
    features = F.adaptive_avg_pool2d(activations[0], (1, 1))
    features = features.view(features.size(0), -1)

    return features.cpu().numpy()


def generate_concept_activations(
    model: nn.Module,
    dataset: torch.utils.data.Dataset,
    concept_type: str,
    n_samples: int,
    device: torch.device,
    batch_size: int = 16,
) -> np.ndarray:
    """Generate concept-transformed activations."""

    logger.info(f"  Generating {n_samples} concept activations for '{concept_type}'...")

    # Sample random indices
    indices = np.random.choice(
        len(dataset), min(n_samples, len(dataset)), replace=False
    )

    all_features = []

    for i in tqdm(
        range(0, len(indices), batch_size), desc=f"    {concept_type}", leave=False
    ):
        batch_indices = indices[i : i + batch_size]
        batch_images = []

        for idx in batch_indices:
            image, _, _ = dataset[idx]  # Dataset returns (image, label, meta)
            # Apply concept transformation
            transformed = apply_concept_transform(image, concept_type)
            batch_images.append(transformed)

        batch_tensor = torch.stack(batch_images)
        features = extract_features(model, batch_tensor, device)
        all_features.append(features)

    return np.vstack(all_features)


# ============================================================================
# CAV Training
# ============================================================================


def train_cav(
    concept_acts: np.ndarray, random_acts: np.ndarray, concept_name: str
) -> Tuple[np.ndarray, StandardScaler, float]:
    """
    Train a Concept Activation Vector using Linear SVM.

    Returns:
        (cav_vector, scaler, accuracy)
    """
    # Prepare data
    X = np.vstack([concept_acts, random_acts])
    y = np.array([1] * len(concept_acts) + [0] * len(random_acts))

    # Train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )

    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # Train Linear SVM
    clf = LinearSVC(C=1.0, max_iter=10000, random_state=42)
    clf.fit(X_train_scaled, y_train)

    # Extract and normalize CAV
    cav_vector = clf.coef_[0]
    cav_vector = cav_vector / np.linalg.norm(cav_vector)

    # Evaluate
    accuracy = clf.score(X_test_scaled, y_test)

    logger.info(f"    ✓ CAV '{concept_name}' trained - Accuracy: {accuracy:.3f}")

    return cav_vector, scaler, accuracy


# ============================================================================
# Main Script
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Train CAVs for tri-objective training"
    )
    parser.add_argument(
        "--data_dir", type=str, required=True, help="Path to dataset directory"
    )
    parser.add_argument(
        "--model_checkpoint",
        type=str,
        required=True,
        help="Path to baseline model checkpoint",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints/cavs",
        help="Directory to save trained CAVs",
    )
    parser.add_argument(
        "--n_concept_samples",
        type=int,
        default=50,
        help="Number of samples per concept",
    )
    parser.add_argument(
        "--n_random_samples",
        type=int,
        default=100,
        help="Number of random baseline samples",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for feature extraction"
    )
    parser.add_argument("--seed", type=int, default=42, help="Random seed")

    args = parser.parse_args()

    # Set seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    logger.info("=" * 70)
    logger.info("TRAINING CONCEPT ACTIVATION VECTORS (CAVs)")
    logger.info("=" * 70)
    logger.info(f"Device: {device}")
    logger.info(f"Model: {args.model_checkpoint}")
    logger.info(f"Concept samples: {args.n_concept_samples}")
    logger.info(f"Random samples: {args.n_random_samples}")

    # Load model
    logger.info("\n📦 Loading model...")
    from src.models import build_model

    # Use the actual model builder from codebase
    model = build_model(name="resnet50", num_classes=7, pretrained=False)

    # PyTorch 2.6+ requires weights_only=False for numpy arrays
    checkpoint = torch.load(
        args.model_checkpoint, map_location=device, weights_only=False
    )
    if "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # Handle torch.compile wrapped models (_orig_mod prefix)
    if any(k.startswith("_orig_mod.") for k in state_dict.keys()):
        logger.info("Detected torch.compile wrapped model, unwrapping...")
        state_dict = {k.replace("_orig_mod.", ""): v for k, v in state_dict.items()}

    # Handle backbone prefix
    if any(k.startswith("backbone.") for k in state_dict.keys()):
        logger.info("Detected backbone prefix, removing...")
        state_dict = {k.replace("backbone.", ""): v for k, v in state_dict.items()}

    model.load_state_dict(state_dict, strict=False)

    model = model.to(device)
    model.eval()
    logger.info("✓ Model loaded")

    # Load dataset
    logger.info("\n📂 Loading dataset...")
    import albumentations as A
    from albumentations.pytorch import ToTensorV2

    from src.datasets.isic import ISICDataset

    dataset = ISICDataset(
        root=Path(args.data_dir),
        split="train",
        transforms=A.Compose(
            [
                A.Resize(224, 224),
                A.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
                ToTensorV2(),
            ]
        ),
    )
    logger.info(f"✓ Loaded {len(dataset)} images")

    # Define concepts
    concepts = {
        # Medical concepts
        "border_irregularity": "medical",
        "color_variation": "medical",
        "texture_heterogeneity": "medical",
        "asymmetry": "medical",
        # Artifact concepts
        "background": "artifact",
        "ruler_marks": "artifact",
        "hair_artifacts": "artifact",
    }

    logger.info(f"\n🧠 Training {len(concepts)} CAVs...")

    # Generate random activations (shared baseline)
    logger.info("\n  Generating random baseline activations...")
    random_indices = np.random.choice(
        len(dataset), args.n_random_samples, replace=False
    )
    random_images = []
    for idx in tqdm(random_indices, desc="    Random", leave=False):
        image, _, _ = dataset[idx]  # Dataset returns (image, label, meta)
        random_images.append(image)

    random_batch = torch.stack(random_images[: args.batch_size])
    random_acts = []
    for i in range(0, len(random_images), args.batch_size):
        batch = torch.stack(random_images[i : i + args.batch_size])
        features = extract_features(model, batch, device)
        random_acts.append(features)
    random_acts = np.vstack(random_acts)
    logger.info(f"  ✓ Generated {len(random_acts)} random activations")

    # Train CAVs
    cavs = {}
    cav_metadata = {}

    for concept_name, concept_type in concepts.items():
        logger.info(f"\n  Training CAV for '{concept_name}' ({concept_type})...")

        # Generate concept activations
        concept_acts = generate_concept_activations(
            model,
            dataset,
            concept_name,
            args.n_concept_samples,
            device,
            args.batch_size,
        )

        # Train CAV
        cav_vector, scaler, accuracy = train_cav(
            concept_acts, random_acts, concept_name
        )

        cavs[concept_name] = {
            "vector": cav_vector,
            "scaler": scaler,
            "accuracy": accuracy,
            "type": concept_type,
        }

        cav_metadata[concept_name] = {
            "accuracy": float(accuracy),
            "type": concept_type,
            "n_concept_samples": args.n_concept_samples,
            "n_random_samples": args.n_random_samples,
        }

    # Save CAVs
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save as PyTorch format (for use in training)
    output_path = output_dir / "trained_cavs.pt"
    torch.save(
        {
            "cavs": cavs,
            "metadata": cav_metadata,
            "model_checkpoint": args.model_checkpoint,
            "args": vars(args),
        },
        output_path,
    )

    logger.info(f"\n✅ CAVs saved to: {output_path}")

    # Print summary
    logger.info("\n" + "=" * 70)
    logger.info("CAV TRAINING SUMMARY")
    logger.info("=" * 70)

    medical_cavs = [name for name, data in cavs.items() if data["type"] == "medical"]
    artifact_cavs = [name for name, data in cavs.items() if data["type"] == "artifact"]

    logger.info(f"\nMedical Concepts ({len(medical_cavs)}):")
    for name in medical_cavs:
        acc = cavs[name]["accuracy"]
        logger.info(f"  {name:25s} - Accuracy: {acc:.3f}")

    logger.info(f"\nArtifact Concepts ({len(artifact_cavs)}):")
    for name in artifact_cavs:
        acc = cavs[name]["accuracy"]
        logger.info(f"  {name:25s} - Accuracy: {acc:.3f}")

    logger.info(f"\n✅ Training complete!")
    logger.info(f"\n📝 Next step: Use these CAVs in tri-objective training:")
    logger.info(f"   python scripts/training/train_tri_objective.py \\")
    logger.info(f"       --cavs_checkpoint {output_path}")


if __name__ == "__main__":
    main()
