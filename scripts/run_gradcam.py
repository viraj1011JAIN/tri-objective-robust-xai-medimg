"""
Phase 6.1 Grad-CAM Explanation Generator - CLI Wrapper.

Command-line interface for generating visual explanations using Grad-CAM
for medical image classification models.

Example Usage:
    # Basic Grad-CAM on single image
    python scripts/run_gradcam.py \\
        --checkpoint checkpoints/best.pt \\
        --image data/processed/isic2018/test/image1.jpg \\
        --output results/xai/gradcam

    # Grad-CAM++ with custom layer
    python scripts/run_gradcam.py \\
        --checkpoint checkpoints/best.pt \\
        --image data/processed/isic2018/test/image1.jpg \\
        --method gradcam++ \\
        --target-layer layer3 \\
        --output results/xai/gradcam++

    # Batch processing
    python scripts/run_gradcam.py \\
        --checkpoint checkpoints/best.pt \\
        --image-dir data/processed/isic2018/test \\
        --output results/xai/gradcam_batch \\
        --batch-size 16

Author: Viraj Pankaj Jain
Institution: University of Glasgow
"""

import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from tqdm import tqdm

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.model_registry import ModelRegistry
from src.xai.gradcam import GradCAM, GradCAMConfig, GradCAMPlusPlus, create_gradcam


def setup_logging(verbose: bool = False) -> None:
    """Setup logging configuration."""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )


def load_model_from_checkpoint(checkpoint_path: Path) -> nn.Module:
    """Load model from checkpoint using ModelRegistry.

    Args:
        checkpoint_path: Path to checkpoint file

    Returns:
        Loaded PyTorch model in eval mode
    """
    registry = ModelRegistry(checkpoint_dir=checkpoint_path.parent)

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu")

    # Extract model state
    if "model_state_dict" in checkpoint:
        model_state = checkpoint["model_state_dict"]
    else:
        model_state = checkpoint

    # Load model architecture from config
    if "config" in checkpoint:
        model_config = checkpoint["config"]
    else:
        raise ValueError("Checkpoint missing model config")

    # Create model (simplified - adjust for your architecture)
    from torchvision import models

    if "resnet" in str(model_config).lower():
        model = models.resnet50(pretrained=False)
    else:
        raise ValueError(f"Unsupported architecture: {model_config}")

    # Load weights
    model.load_state_dict(model_state)
    model.eval()

    return model


def load_image(
    image_path: Path, size: tuple = (224, 224), normalize: bool = True
) -> torch.Tensor:
    """Load and preprocess image for model input.

    Args:
        image_path: Path to image file
        size: Target size (H, W)
        normalize: Apply ImageNet normalization

    Returns:
        Preprocessed tensor (1, C, H, W)
    """
    img = Image.open(image_path).convert("RGB")
    img = img.resize(size[::-1])  # PIL uses (W, H)
    img_tensor = torch.from_numpy(np.array(img)).float() / 255.0

    # Transpose to (C, H, W)
    img_tensor = img_tensor.permute(2, 0, 1)

    # Normalize
    if normalize:
        mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
        std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
        img_tensor = (img_tensor - mean) / std

    return img_tensor.unsqueeze(0)


def save_results(
    image_path: Path,
    heatmap: np.ndarray,
    overlay: np.ndarray,
    output_dir: Path,
    class_idx: int,
    prediction: Dict,
) -> None:
    """Save heatmap, overlay, and metadata.

    Args:
        image_path: Original image path
        heatmap: Grad-CAM heatmap
        overlay: Overlay visualization
        output_dir: Output directory
        class_idx: Target class index
        prediction: Model prediction info
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    stem = image_path.stem

    # Save heatmap
    heatmap_path = output_dir / f"{stem}_heatmap.png"
    heatmap_colored = cv2.applyColorMap(
        (heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET
    )
    cv2.imwrite(str(heatmap_path), heatmap_colored)

    # Save overlay
    overlay_path = output_dir / f"{stem}_overlay.png"
    cv2.imwrite(str(overlay_path), cv2.cvtColor(overlay, cv2.COLOR_RGB2BGR))

    # Save metadata
    metadata = {
        "image": str(image_path),
        "target_class": class_idx,
        "prediction": prediction,
        "heatmap": str(heatmap_path),
        "overlay": str(overlay_path),
    }

    metadata_path = output_dir / f"{stem}_metadata.json"
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=2)


def process_single_image(
    model: nn.Module,
    gradcam: GradCAM,
    image_path: Path,
    output_dir: Path,
    class_idx: Optional[int] = None,
) -> None:
    """Process single image with Grad-CAM.

    Args:
        model: PyTorch model
        gradcam: GradCAM instance
        image_path: Path to image
        output_dir: Output directory
        class_idx: Target class (uses predicted if None)
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Processing: {image_path}")

    # Load image
    img_tensor = load_image(image_path)

    # Get prediction
    with torch.no_grad():
        output = model(img_tensor)
        probs = torch.softmax(output, dim=1)
        pred_class = output.argmax(dim=1).item()
        pred_prob = probs[0, pred_class].item()

    # Use predicted class if not specified
    target_class = class_idx if class_idx is not None else pred_class

    # Generate heatmap
    heatmap = gradcam.generate_heatmap(img_tensor, class_idx=target_class)

    # Create overlay
    overlay = gradcam.visualize(img_tensor, heatmap, alpha=0.5)

    # Save results
    prediction_info = {
        "predicted_class": pred_class,
        "confidence": pred_prob,
        "top_k_classes": probs[0].topk(3).indices.tolist(),
        "top_k_probs": probs[0].topk(3).values.tolist(),
    }

    save_results(
        image_path, heatmap, overlay, output_dir, target_class, prediction_info
    )

    logger.info(
        f"  Predicted: class {pred_class} (conf={pred_prob:.3f}), "
        f"Explained: class {target_class}"
    )


def process_batch(
    model: nn.Module,
    gradcam: GradCAM,
    image_dir: Path,
    output_dir: Path,
    batch_size: int = 16,
) -> None:
    """Process directory of images in batches.

    Args:
        model: PyTorch model
        gradcam: GradCAM instance
        image_dir: Directory containing images
        output_dir: Output directory
        batch_size: Batch size for processing
    """
    logger = logging.getLogger(__name__)

    # Find all images
    image_paths = []
    for ext in ["*.jpg", "*.jpeg", "*.png"]:
        image_paths.extend(list(image_dir.glob(ext)))
        image_paths.extend(list(image_dir.glob(ext.upper())))

    if not image_paths:
        logger.warning(f"No images found in {image_dir}")
        return

    logger.info(f"Found {len(image_paths)} images to process")

    # Process in batches
    for image_path in tqdm(image_paths, desc="Processing images"):
        try:
            process_single_image(model, gradcam, image_path, output_dir)
        except Exception as e:
            logger.error(f"Error processing {image_path}: {e}")


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Phase 6.1: Grad-CAM Visual Explanations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )

    # Model
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )

    # Input
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--image",
        type=Path,
        help="Path to single image",
    )
    group.add_argument(
        "--image-dir",
        type=Path,
        help="Directory containing images for batch processing",
    )

    # Output
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/xai/gradcam"),
        help="Output directory (default: results/xai/gradcam)",
    )

    # Grad-CAM options
    parser.add_argument(
        "--method",
        type=str,
        default="gradcam",
        choices=["gradcam", "gradcam++"],
        help="Method to use (default: gradcam)",
    )

    parser.add_argument(
        "--target-layer",
        type=str,
        help="Target layer name (auto-detects if not specified)",
    )

    parser.add_argument(
        "--class-idx",
        type=int,
        help="Target class index (uses predicted class if not specified)",
    )

    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Overlay blending factor (default: 0.5)",
    )

    # Processing options
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for batch processing (default: 16)",
    )

    parser.add_argument(
        "--no-cuda",
        action="store_true",
        help="Disable CUDA",
    )

    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Enable verbose logging",
    )

    return parser.parse_args()


def main() -> None:
    """Main entry point."""
    args = parse_args()
    setup_logging(args.verbose)

    logger = logging.getLogger(__name__)
    logger.info("=" * 80)
    logger.info("Phase 6.1: Grad-CAM Visual Explanations")
    logger.info("=" * 80)

    # Load model
    logger.info(f"Loading model from {args.checkpoint}")
    try:
        model = load_model_from_checkpoint(args.checkpoint)
        logger.info(f"Model loaded: {model.__class__.__name__}")
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        sys.exit(1)

    # Create Grad-CAM
    target_layers = [args.target_layer] if args.target_layer else None
    use_cuda = not args.no_cuda

    logger.info(f"Creating {args.method} explainer")
    gradcam = create_gradcam(
        model, target_layers=target_layers, method=args.method, use_cuda=use_cuda
    )

    logger.info(f"Target layers: {gradcam.config.target_layers}")
    logger.info(f"Device: {gradcam.device}")

    # Process images
    if args.image:
        # Single image
        process_single_image(
            model, gradcam, args.image, args.output, class_idx=args.class_idx
        )
    else:
        # Batch processing
        process_batch(model, gradcam, args.image_dir, args.output, args.batch_size)

    # Cleanup
    gradcam.remove_hooks()

    logger.info("=" * 80)
    logger.info(f"✓ Complete. Results saved to: {args.output}")
    logger.info("=" * 80)


if __name__ == "__main__":
    main()
