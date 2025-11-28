"""
Generate Stability Scores for Phase 8.4 Threshold Tuning

This script evaluates trained models and computes:
1. Confidence scores (softmax max, entropy)
2. Stability scores (SSIM, rank correlation, L2 distance)

Output: CSV files in results/metrics/rq3_selective/
Format: conf_softmax, conf_entropy, stab_ssim, stab_rank_corr, stab_l2, correct
"""

import sys
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

# Add project root
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.datasets.isic import ISICDataset
from src.datasets.transforms import get_isic_transforms
from src.models.resnet import ResNet50Classifier


def compute_confidence_scores(logits: torch.Tensor) -> dict:
    """Compute confidence scores from logits."""
    probs = F.softmax(logits, dim=-1)

    # Softmax maximum (highest class probability)
    conf_softmax = probs.max(dim=-1)[0].item()

    # Entropy (lower = more confident)
    entropy = -(probs * torch.log(probs + 1e-10)).sum(dim=-1).item()
    # Normalize to [0, 1] for binary classification
    max_entropy = -np.log(1.0 / logits.shape[-1])
    conf_entropy = 1.0 - (entropy / max_entropy)

    return {
        "conf_softmax": conf_softmax,
        "conf_entropy": conf_entropy,
    }


def compute_stability_scores(
    model: nn.Module,
    image: torch.Tensor,
    device: torch.device,
    n_perturbations: int = 5,
) -> dict:
    """
    Compute stability scores using perturbations.

    Stability measures how consistent predictions are under small input changes:
    - SSIM: Structural similarity of attention/activations
    - Rank Correlation: Spearman correlation of logit rankings
    - L2 Distance: Euclidean distance between predictions
    """
    model.eval()

    # Original prediction
    with torch.no_grad():
        outputs_orig = model(image)
        if isinstance(outputs_orig, dict):
            logits_orig = outputs_orig["logits"]
        else:
            logits_orig = outputs_orig
        probs_orig = F.softmax(logits_orig, dim=-1).cpu().numpy().flatten()

    # Generate perturbations (small Gaussian noise)
    perturbations = []
    all_probs = [probs_orig]

    for _ in range(n_perturbations):
        # Add small random noise
        noise = torch.randn_like(image) * 0.01  # Small noise
        perturbed = torch.clamp(image + noise, 0, 1)

        with torch.no_grad():
            outputs_pert = model(perturbed)
            if isinstance(outputs_pert, dict):
                logits_pert = outputs_pert["logits"]
            else:
                logits_pert = outputs_pert
            probs_pert = F.softmax(logits_pert, dim=-1).cpu().numpy().flatten()

        all_probs.append(probs_pert)

    # Compute SSIM (treat probability distributions as signals)
    ssim_scores = []
    for probs_pert in all_probs[1:]:
        # Reshape to 2D for SSIM (required by skimage)
        orig_2d = probs_orig.reshape(-1, 1)
        pert_2d = probs_pert.reshape(-1, 1)

        # Compute SSIM with data_range for probability values
        ssim_val = ssim(
            orig_2d,
            pert_2d,
            data_range=1.0,
            win_size=min(3, orig_2d.shape[0]),
        )
        ssim_scores.append(ssim_val)

    # Compute Rank Correlation
    rank_corrs = []
    for probs_pert in all_probs[1:]:
        corr, _ = spearmanr(probs_orig, probs_pert)
        rank_corrs.append(corr if not np.isnan(corr) else 1.0)

    # Compute L2 Distance
    l2_distances = []
    for probs_pert in all_probs[1:]:
        l2_dist = np.linalg.norm(probs_orig - probs_pert)
        l2_distances.append(l2_dist)

    # Aggregate (mean across perturbations)
    stab_ssim = np.mean(ssim_scores)
    stab_rank_corr = np.mean(rank_corrs)
    stab_l2_raw = np.mean(l2_distances)

    # Normalize L2 to [0, 1] (invert: lower distance = higher stability)
    # Typical L2 distance range is [0, sqrt(2)] for probability distributions
    stab_l2 = 1.0 - np.clip(stab_l2_raw / np.sqrt(2), 0, 1)

    return {
        "stab_ssim": stab_ssim,
        "stab_rank_corr": stab_rank_corr,
        "stab_l2": stab_l2,
    }


def evaluate_model(
    model_path: Path,
    dataset: ISICDataset,
    device: torch.device,
    n_samples: int = 500,
    batch_size: int = 1,  # Process one at a time for stability
) -> pd.DataFrame:
    """
    Evaluate model and generate confidence + stability scores.

    Returns DataFrame with columns:
    - conf_softmax, conf_entropy
    - stab_ssim, stab_rank_corr, stab_l2
    - correct (boolean)
    """
    # Load model
    print(f"Loading model from {model_path}")
    checkpoint = torch.load(model_path, map_location=device)

    model = ResNet50Classifier(num_classes=8)  # ISIC has 8 classes

    if "model_state_dict" in checkpoint:
        model.load_state_dict(checkpoint["model_state_dict"])
    elif "state_dict" in checkpoint:
        model.load_state_dict(checkpoint["state_dict"])
    else:
        model.load_state_dict(checkpoint)

    model = model.to(device)
    model.eval()

    # Create dataloader
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    results = []

    print(f"Evaluating {n_samples} samples...")
    with torch.no_grad():
        for i, batch in enumerate(tqdm(dataloader, total=n_samples)):
            if i >= n_samples:
                break

            images = batch["image"].to(device)
            labels = batch["label"].to(device)

            # Get predictions
            outputs = model(images)
            if isinstance(outputs, dict):
                logits = outputs["logits"]
            else:
                logits = outputs

            preds = logits.argmax(dim=-1)
            correct = (preds == labels).item()

            # Compute confidence scores
            conf_scores = compute_confidence_scores(logits[0])

            # Compute stability scores
            stab_scores = compute_stability_scores(
                model, images, device, n_perturbations=5
            )

            # Combine
            result = {
                **conf_scores,
                **stab_scores,
                "correct": correct,
            }
            results.append(result)

    df = pd.DataFrame(results)
    return df


def main():
    """Generate stability scores for all models."""
    print("=" * 80)
    print("GENERATING STABILITY SCORES FOR PHASE 8.4")
    print("=" * 80)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    # Output directory
    output_dir = PROJECT_ROOT / "results" / "metrics" / "rq3_selective"
    output_dir.mkdir(parents=True, exist_ok=True)

    # Dataset paths
    data_root = PROJECT_ROOT / "data" / "isic"

    # Models to evaluate
    models_config = [
        {
            "name": "baseline",
            "path": PROJECT_ROOT
            / "results"
            / "checkpoints"
            / "baseline_isic2018_resnet50"
            / "seed_42"
            / "best.pt",
        },
        # Add more models as they become available:
        # {
        #     'name': 'trades',
        #     'path': PROJECT_ROOT / "results" / "checkpoints" / "trades_isic2018_resnet50" / "seed_42" / "best.pt",
        # },
        # {
        #     'name': 'tri_objective',
        #     'path': PROJECT_ROOT / "results" / "checkpoints" / "tri_objective_isic2018_resnet50" / "seed_42" / "best.pt",
        # },
    ]

    # Datasets to evaluate on
    datasets_config = [
        {
            "name": "isic2018_test",
            "split": "test",
        },
        # Add more datasets as available:
        # {
        #     'name': 'isic2019',
        #     'split': 'test',
        # },
    ]

    # Evaluation loop
    for model_cfg in models_config:
        model_name = model_cfg["name"]
        model_path = model_cfg["path"]

        if not model_path.exists():
            print(f"\n⚠️  Skipping {model_name}: checkpoint not found at {model_path}")
            continue

        for dataset_cfg in datasets_config:
            dataset_name = dataset_cfg["name"]

            print(f"\n{'=' * 80}")
            print(f"Model: {model_name} | Dataset: {dataset_name}")
            print(f"{'=' * 80}")

            # Load dataset
            try:
                val_transforms = get_isic_transforms(split="test", image_size=224)
                dataset = ISICDataset(
                    root=data_root,
                    split=dataset_cfg["split"],
                    transforms=val_transforms,
                )
                print(f"Dataset loaded: {len(dataset)} samples")
            except Exception as e:
                print(f"⚠️  Error loading dataset: {e}")
                print(f"⚠️  Skipping {model_name}_{dataset_name}")
                continue

            # Evaluate
            try:
                df_scores = evaluate_model(
                    model_path=model_path,
                    dataset=dataset,
                    device=device,
                    n_samples=min(500, len(dataset)),  # Limit to 500 samples
                    batch_size=1,
                )

                # Save results
                output_path = output_dir / f"{model_name}_{dataset_name}_scores.csv"
                df_scores.to_csv(output_path, index=False)

                print(f"\n✅ Results saved: {output_path}")
                print(f"   • Total samples: {len(df_scores)}")
                print(f"   • Baseline accuracy: {df_scores['correct'].mean():.4f}")
                print(f"   • Mean conf_softmax: {df_scores['conf_softmax'].mean():.3f}")
                print(f"   • Mean stab_ssim: {df_scores['stab_ssim'].mean():.3f}")

            except Exception as e:
                print(f"⚠️  Error during evaluation: {e}")
                import traceback

                traceback.print_exc()
                continue

    print("\n" + "=" * 80)
    print("✅ STABILITY SCORE GENERATION COMPLETE")
    print("=" * 80)
    print(f"Output directory: {output_dir}")
    print(
        "\nNext step: Run Phase 8.4 notebook to load real data and perform threshold tuning"
    )


if __name__ == "__main__":
    main()
