"""
RQ2 Complete Evaluation: Explanation Stability and Concept Grounding.

Evaluates all models on all RQ2 metrics and generates tables/figures.
"""

import argparse
import sys
from pathlib import Path
from typing import Dict

import numpy as np
import pandas as pd
import torch
from tqdm import tqdm

sys.path.append(".")

from src.datasets.isic_dataset import ISICDataset
from src.models.resnet import ResNet50Classifier
from src.xai.gradcam_production import GradCAM
from src.xai.tcav_production import TCAV, ConceptBank

# SSIM library
try:
    from pytorch_msssim import ssim

    SSIM_AVAILABLE = True
except ImportError:
    print("Warning: pytorch_msssim not installed. Using manual SSIM computation.")
    SSIM_AVAILABLE = False

from scipy.ndimage import gaussian_filter
from scipy.stats import spearmanr


def fgsm_attack(model, images, labels, epsilon=2 / 255, device="cuda"):
    """Simple FGSM attack for adversarial heatmaps."""
    images = images.to(device)
    labels = labels.to(device)
    images.requires_grad_(True)

    outputs = model(images)
    loss = torch.nn.functional.cross_entropy(outputs, labels)
    model.zero_grad()
    loss.backward()

    # Generate adversarial examples
    sign_data_grad = images.grad.sign()
    perturbed_images = images + epsilon * sign_data_grad
    perturbed_images = torch.clamp(perturbed_images, 0, 1)

    return perturbed_images


def compute_ssim_manual(
    img1: np.ndarray, img2: np.ndarray, K1=0.01, K2=0.03, window_size=11
):
    """Compute SSIM manually (fallback)."""

    C1 = (K1 * 1.0) ** 2
    C2 = (K2 * 1.0) ** 2

    # Mean
    mu1 = gaussian_filter(img1, sigma=window_size / 6)
    mu2 = gaussian_filter(img2, sigma=window_size / 6)

    mu1_sq = mu1**2
    mu2_sq = mu2**2
    mu1_mu2 = mu1 * mu2

    # Variance and covariance
    sigma1_sq = gaussian_filter(img1**2, sigma=window_size / 6) - mu1_sq
    sigma2_sq = gaussian_filter(img2**2, sigma=window_size / 6) - mu2_sq
    sigma12 = gaussian_filter(img1 * img2, sigma=window_size / 6) - mu1_mu2

    # SSIM formula
    numerator = (2 * mu1_mu2 + C1) * (2 * sigma12 + C2)
    denominator = (mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2)

    ssim_map = numerator / denominator

    return ssim_map.mean()


def evaluate_model_rq2(
    model_name: str,
    seed: int,
    checkpoint_dir: Path,
    test_loader,
    device: torch.device,
    concept_bank_path: str,
) -> Dict[str, float]:
    """
    Evaluate one model on RQ2 metrics.

    Returns:
        results: Dict with all RQ2 metrics
    """

    print(f"\nEvaluating {model_name} (seed {seed})...")

    # Load model
    checkpoint_path = checkpoint_dir / f"{model_name}_seed{seed}_best.pth"
    model = ResNet50Classifier(num_classes=7)
    model.load_state_dict(torch.load(checkpoint_path, map_location=device))
    model = model.to(device)
    model.eval()

    results = {"model": model_name, "seed": seed}

    # 1. Explanation Stability Metrics
    print("  Computing explanation stability...")

    gradcam = GradCAM(model, target_layer="layer4", device=device)

    all_ssim = []
    all_rank_corr = []
    all_l2_dist = []

    # Limit evaluation to first 200 samples for efficiency
    sample_count = 0
    max_samples = 200

    for images, labels in tqdm(test_loader, desc="  Stability", leave=False):
        if sample_count >= max_samples:
            break

        images = images.to(device)
        labels = labels.to(device)

        # Generate clean heatmaps
        heatmaps_clean = gradcam.generate_heatmap(images, target_class=labels)

        # Generate adversarial examples
        images_adv = fgsm_attack(model, images, labels, epsilon=2 / 255, device=device)

        # Generate adversarial heatmaps
        heatmaps_adv = gradcam.generate_heatmap(images_adv, target_class=labels)

        # Compute SSIM
        if SSIM_AVAILABLE:
            # Use pytorch_msssim
            h_clean = heatmaps_clean.unsqueeze(1)  # (B, 1, H, W)
            h_adv = heatmaps_adv.unsqueeze(1)
            ssim_scores = ssim(h_clean, h_adv, data_range=1.0, size_average=False)
            ssim_scores = ssim_scores.cpu().numpy()
        else:
            # Manual SSIM
            ssim_scores = []
            for i in range(len(images)):
                score = compute_ssim_manual(
                    heatmaps_clean[i].cpu().numpy(), heatmaps_adv[i].cpu().numpy()
                )
                ssim_scores.append(score)
            ssim_scores = np.array(ssim_scores)

        all_ssim.extend(ssim_scores.tolist())

        # Rank correlation
        for i in range(len(images)):
            h_clean = heatmaps_clean[i].cpu().numpy().flatten()
            h_adv = heatmaps_adv[i].cpu().numpy().flatten()
            corr, _ = spearmanr(h_clean, h_adv)
            all_rank_corr.append(corr)

        # L2 distance
        diff = heatmaps_clean - heatmaps_adv
        l2_dist = torch.norm(diff.view(len(diff), -1), p=2, dim=1)
        # Normalize by heatmap size
        h, w = heatmaps_clean.shape[1:]
        l2_dist_normalized = l2_dist / np.sqrt(h * w)
        all_l2_dist.extend(l2_dist_normalized.cpu().numpy().tolist())

        sample_count += len(images)

    results["ssim"] = np.mean(all_ssim)
    results["rank_corr"] = np.mean(all_rank_corr)
    results["l2_distance"] = np.mean(all_l2_dist)

    gradcam.cleanup()

    # 2. Concept Reliance Metrics (TCAV)
    print("  Computing TCAV scores...")

    # Load concept bank
    concept_bank = ConceptBank(cav_save_path=concept_bank_path)
    concept_bank.load_cavs()

    tcav = TCAV(model, target_layer="layer4", device=device)

    # Compute TCAV scores for artifacts
    artifact_scores = []
    for concept in concept_bank.artifact_concepts:
        cav = concept_bank.get_cav(concept)

        # Average over all classes
        scores_per_class = []
        for cls in range(7):  # 7 classes in ISIC
            score = tcav.compute_tcav_score(test_loader, cav, cls)
            scores_per_class.append(score)

        score = np.mean(scores_per_class)
        results[concept] = score
        artifact_scores.append(score)

    # Compute TCAV scores for medical concepts
    medical_scores = []
    for concept in concept_bank.medical_concepts:
        cav = concept_bank.get_cav(concept)

        scores_per_class = []
        for cls in range(7):
            score = tcav.compute_tcav_score(test_loader, cav, cls)
            scores_per_class.append(score)

        score = np.mean(scores_per_class)
        results[concept] = score
        medical_scores.append(score)

    # Aggregate
    results["artifact_mean"] = np.mean(artifact_scores)
    results["medical_mean"] = np.mean(medical_scores)
    results["tcav_ratio"] = results["medical_mean"] / (results["artifact_mean"] + 1e-8)

    tcav.cleanup()

    print(f"  ✓ SSIM: {results['ssim']:.3f}")
    print(f"  ✓ Artifact TCAV: {results['artifact_mean']:.3f}")
    print(f"  ✓ Medical TCAV: {results['medical_mean']:.3f}")
    print(f"  ✓ TCAV Ratio: {results['tcav_ratio']:.2f}")

    return results


def main(args):
    """Main evaluation."""

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # Load test data
    test_dataset = ISICDataset(root=args.data_root, split="test")
    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=2
    )

    print(f"Test set size: {len(test_dataset)}")

    # Evaluate all models
    all_results = []

    checkpoint_dir = Path(args.checkpoint_dir)

    for model_name in args.models:
        for seed in args.seeds:
            try:
                results = evaluate_model_rq2(
                    model_name,
                    seed,
                    checkpoint_dir,
                    test_loader,
                    device,
                    concept_bank_path=args.concept_bank_path,
                )
                all_results.append(results)
            except FileNotFoundError as e:
                print(f"Warning: {e}")
                continue
            except Exception as e:
                print(f"Error evaluating {model_name} seed {seed}: {e}")
                continue

    # Save results
    results_df = pd.DataFrame(all_results)

    output_dir = Path("results/rq2_complete")
    output_dir.mkdir(parents=True, exist_ok=True)

    results_df.to_csv(output_dir / "rq2_all_results.csv", index=False)

    print("\n" + "=" * 60)
    print("RQ2 RESULTS SUMMARY")
    print("=" * 60)

    # Aggregate by model
    if len(results_df) > 0:
        summary = results_df.groupby("model").agg(
            {
                "ssim": ["mean", "std"],
                "rank_corr": ["mean", "std"],
                "l2_distance": ["mean", "std"],
                "artifact_mean": ["mean", "std"],
                "medical_mean": ["mean", "std"],
                "tcav_ratio": ["mean", "std"],
            }
        )

        print("\n", summary)

        summary.to_csv(output_dir / "rq2_summary.csv")

        print(f"\n✓ Results saved to {output_dir}")

        # Check hypotheses
        print("\n" + "=" * 60)
        print("HYPOTHESIS TESTING")
        print("=" * 60)

        baseline = results_df[results_df["model"] == "baseline"]
        triobj = results_df[results_df["model"] == "tri_objective"]

        if len(baseline) > 0:
            print("\nBaseline Results:")
            print(
                f"  SSIM: {baseline['ssim'].mean():.3f} ± {baseline['ssim'].std():.3f}"
            )
            print(
                f"  Artifact TCAV: {baseline['artifact_mean'].mean():.3f} ± {baseline['artifact_mean'].std():.3f}"
            )
            print(
                f"  Medical TCAV: {baseline['medical_mean'].mean():.3f} ± {baseline['medical_mean'].std():.3f}"
            )
            print(
                f"  TCAV Ratio: {baseline['tcav_ratio'].mean():.2f} ± {baseline['tcav_ratio'].std():.2f}"
            )

        if len(triobj) > 0:
            print(f"\nTri-objective Results:")
            print(f"  SSIM: {triobj['ssim'].mean():.3f} ± {triobj['ssim'].std():.3f}")
            print(
                f"  Artifact TCAV: {triobj['artifact_mean'].mean():.3f} ± {triobj['artifact_mean'].std():.3f}"
            )
            print(
                f"  Medical TCAV: {triobj['medical_mean'].mean():.3f} ± {triobj['medical_mean'].std():.3f}"
            )
            print(
                f"  TCAV Ratio: {triobj['tcav_ratio'].mean():.2f} ± {triobj['tcav_ratio'].std():.2f}"
            )

            print("\nHypothesis Status:")
            print(
                f"  H2.1 (SSIM ≥0.75): {'✓' if triobj['ssim'].mean() >= 0.75 else '✗'}"
            )
            print(
                f"  H2.2 (Artifact ≤0.20): {'✓' if triobj['artifact_mean'].mean() <= 0.20 else '✗'}"
            )
            print(
                f"  H2.3 (Medical ≥0.65): {'✓' if triobj['medical_mean'].mean() >= 0.65 else '✗'}"
            )
            print(
                f"  H2.4 (Ratio ≥3.0): {'✓' if triobj['tcav_ratio'].mean() >= 3.0 else '✗'}"
            )
    else:
        print("No results generated!")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_root", default="data/ISIC2018")
    parser.add_argument("--checkpoint_dir", default="results/checkpoints")
    parser.add_argument(
        "--concept_bank_path", default="data/concepts/dermoscopy_cavs.pth"
    )
    parser.add_argument("--models", nargs="+", default=["baseline", "tri_objective"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 123, 456])
    parser.add_argument("--batch_size", type=int, default=8)

    args = parser.parse_args()
    main(args)
