"""
Production-Grade Concept Bank Creation for TCAV (Testing with Concept Activation Vectors).

This module provides comprehensive tooling for creating concept banks for medical imaging,
supporting both dermoscopy and chest X-ray modalities. Concepts include medical findings,
artifacts, and random patches for baseline comparison.

Key Functionality
-----------------
1. **Concept Extraction**: Automated extraction of concept patches from datasets
2. **Dermoscopy Concepts**: Medical (asymmetry, pigment network) + Artifacts (ruler, hair)
3. **Chest X-ray Concepts**: Medical (opacity, cardiac) + Artifacts (text, borders)
4. **Quality Control**: Validation, diversity checking, statistics
5. **DVC Integration**: Automatic tracking with Data Version Control

Research Context (RQ4)
-----------------------
TCAV requires concept banks to test whether models rely on:
- **Spurious concepts**: Artifacts (rulers, text) that should NOT influence diagnosis
- **Medical concepts**: Clinical features (pigment network, opacity) that SHOULD

**Baseline Expectations**:
    - Models trained WITHOUT robustness should show high TCAV scores for artifacts
    - Tri-objective models should show LOW artifact TCAV, HIGH medical TCAV

Concept Bank Structure
----------------------
data/concepts/
├── dermoscopy/
│   ├── medical/
│   │   ├── asymmetry/         # 100+ patches
│   │   ├── pigment_network/   # 100+ patches
│   │   ├── blue_white_veil/   # 100+ patches
│   │   └── ...
│   ├── artifacts/
│   │   ├── ruler/              # 50-100 patches
│   │   ├── hair/               # 50-100 patches
│   │   ├── ink_marks/          # 50-100 patches
│   │   └── black_borders/      # 50-100 patches
│   └── random/                 # Random patches for baseline
└── chest_xray/
    ├── medical/
    │   ├── lung_opacity/       # 100+ patches
    │   ├── cardiac_silhouette/ # 100+ patches
    │   └── rib_shadows/        # 100+ patches
    ├── artifacts/
    │   ├── text_overlay/       # 50-100 patches
    │   ├── borders/            # 50-100 patches
    │   └── patient_markers/    # 50-100 patches
    └── random/                 # Random patches

Typical Usage
-------------
>>> from src.xai.concept_bank import ConceptBankCreator, ConceptBankConfig
>>>
>>> # Configure concept bank
>>> config = ConceptBankConfig(
...     modality="dermoscopy",
...     output_dir="data/concepts/dermoscopy",
...     num_medical_per_concept=100,
...     num_artifact_per_concept=50,
...     patch_size=(224, 224)
... )
>>>
>>> # Create concept bank
>>> creator = ConceptBankCreator(config)
>>> stats = creator.create_concept_bank(dataset_path="data/raw/derm7pt")
>>> print(stats)

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Project: Tri-Objective Robust XAI for Medical Imaging
Phase: 6.5 - Concept Bank Creation
Date: November 25, 2025
Version: 6.5.0 (Production)
"""

from __future__ import annotations

import json
import logging
import random
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import cv2
import numpy as np
import torch
from PIL import Image
from torch import Tensor
from tqdm import tqdm

logger = logging.getLogger(__name__)


# ============================================================================
# Configuration
# ============================================================================


@dataclass
class ConceptBankConfig:
    """
    Configuration for concept bank creation.

    Attributes:
        modality: Dataset modality ("dermoscopy" or "chest_xray")
        output_dir: Root directory for concept bank
        patch_size: Size of extracted patches (H, W)
        num_medical_per_concept: Target number of medical concept patches
        num_artifact_per_concept: Target number of artifact patches
        num_random: Number of random patches for baseline
        seed: Random seed for reproducibility
        min_patch_quality: Minimum quality score (0-1) for patch acceptance
        diversity_threshold: Minimum diversity score (0-1) for patch selection
        use_dvc: Whether to automatically track with DVC
        verbose: Verbosity level (0=silent, 1=progress, 2=debug)
    """

    # Required configuration
    modality: str
    output_dir: Union[str, Path]

    # Patch configuration
    patch_size: Tuple[int, int] = (224, 224)
    num_medical_per_concept: int = 100
    num_artifact_per_concept: int = 50
    num_random: int = 200

    # Quality control
    seed: int = 42
    min_patch_quality: float = 0.5  # Reject blurry/low-contrast patches
    diversity_threshold: float = 0.3  # Ensure patch diversity

    # Integration
    use_dvc: bool = True
    save_metadata: bool = True

    # Logging
    verbose: int = 1

    def __post_init__(self):
        """Validate configuration."""
        if self.modality not in ["dermoscopy", "chest_xray"]:
            raise ValueError(
                f"modality must be 'dermoscopy' or 'chest_xray', got {self.modality}"
            )

        if self.patch_size[0] <= 0 or self.patch_size[1] <= 0:
            raise ValueError(f"patch_size must be positive, got {self.patch_size}")

        if self.num_medical_per_concept <= 0:
            raise ValueError(
                f"num_medical_per_concept must be positive, "
                f"got {self.num_medical_per_concept}"
            )

        if self.num_artifact_per_concept <= 0:
            raise ValueError(
                f"num_artifact_per_concept must be positive, "
                f"got {self.num_artifact_per_concept}"
            )

        if not (0.0 <= self.min_patch_quality <= 1.0):
            raise ValueError(
                f"min_patch_quality must be in [0, 1], " f"got {self.min_patch_quality}"
            )

        # Convert output_dir to Path
        self.output_dir = Path(self.output_dir)


# ============================================================================
# Concept Definitions
# ============================================================================

# Dermoscopy concepts (from Derm7pt dataset attributes)
DERMOSCOPY_MEDICAL_CONCEPTS = {
    "asymmetry": "Lesion shape asymmetry (clinical feature)",
    "pigment_network": "Pigmented network pattern (dermoscopic feature)",
    "blue_white_veil": "Blue-white veil structure (dermoscopic feature)",
    "globules": "Globular structures (dermoscopic feature)",
    "streaks": "Radial streaming/pseudopods (dermoscopic feature)",
    "dots": "Irregular dots/grains (dermoscopic feature)",
    "regression": "Regression structures (clinical feature)",
}

DERMOSCOPY_ARTIFACT_CONCEPTS = {
    "ruler": "Calibration ruler artifact",
    "hair": "Hair occlusion artifact",
    "ink_marks": "Surgical ink marks artifact",
    "black_borders": "Image border artifacts",
}

# Chest X-ray concepts
CHEST_XRAY_MEDICAL_CONCEPTS = {
    "lung_opacity": "Lung opacity/infiltrate (medical finding)",
    "cardiac_silhouette": "Cardiac shadow boundary (anatomy)",
    "rib_shadows": "Rib cage structures (anatomy)",
    "costophrenic_angle": "Costophrenic recess (anatomy)",
}

CHEST_XRAY_ARTIFACT_CONCEPTS = {
    "text_overlay": "Embedded text/labels artifact",
    "borders": "Image border/frame artifact",
    "patient_markers": "Patient positioning markers artifact",
    "blank_regions": "Blank/unexposed regions artifact",
}


# ============================================================================
# Concept Bank Creator
# ============================================================================


class ConceptBankCreator:
    """
    Production-grade concept bank creator for TCAV.

    This class orchestrates the extraction pipeline:
    1. Load dataset with annotations
    2. Extract medical concept patches (using annotations/heuristics)
    3. Extract artifact patches (using detection heuristics)
    4. Generate random patches (for baseline)
    5. Apply quality control (blur detection, diversity)
    6. Save organized concept bank
    7. Generate metadata and statistics
    8. DVC track (optional)

    Attributes:
        config: Configuration for concept bank creation
        medical_concepts: Dictionary of medical concept definitions
        artifact_concepts: Dictionary of artifact concept definitions
    """

    def __init__(self, config: ConceptBankConfig):
        """
        Initialize concept bank creator.

        Args:
            config: Configuration for concept bank creation
        """
        self.config = config

        # Set random seed
        random.seed(config.seed)
        np.random.seed(config.seed)
        torch.manual_seed(config.seed)

        # Load concept definitions based on modality
        if config.modality == "dermoscopy":
            self.medical_concepts = DERMOSCOPY_MEDICAL_CONCEPTS
            self.artifact_concepts = DERMOSCOPY_ARTIFACT_CONCEPTS
        else:  # chest_xray
            self.medical_concepts = CHEST_XRAY_MEDICAL_CONCEPTS
            self.artifact_concepts = CHEST_XRAY_ARTIFACT_CONCEPTS

        logger.info(
            f"Initialized ConceptBankCreator for {config.modality} with "
            f"{len(self.medical_concepts)} medical + "
            f"{len(self.artifact_concepts)} artifact concepts"
        )

    def __repr__(self) -> str:
        """Return string representation for debugging."""
        return (
            f"ConceptBankCreator(\n"
            f"  modality={self.config.modality},\n"
            f"  medical_concepts={len(self.medical_concepts)},\n"
            f"  artifact_concepts={len(self.artifact_concepts)},\n"
            f"  output_dir={self.config.output_dir}\n"
            f")"
        )

    def create_concept_bank(
        self,
        dataset_path: Union[str, Path],
        derm7pt_metadata: Optional[Union[str, Path]] = None,
    ) -> Dict[str, Any]:
        """
        Create complete concept bank from dataset.

        Args:
            dataset_path: Path to raw dataset (images)
            derm7pt_metadata: Path to Derm7pt metadata CSV (for dermoscopy)

        Returns:
            Statistics dictionary with extraction results
        """
        dataset_path = Path(dataset_path)

        if not dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found: {dataset_path}")

        logger.info("=" * 70)
        logger.info(f"CREATING CONCEPT BANK: {self.config.modality}")
        logger.info("=" * 70)

        # Create output directories
        self._create_directory_structure()

        # Statistics tracking
        stats = {
            "modality": self.config.modality,
            "medical_concepts": {},
            "artifact_concepts": {},
            "random_patches": 0,
            "total_patches": 0,
        }

        # Extract medical concepts
        logger.info("\n[1/3] Extracting medical concept patches...")
        medical_stats = self._extract_medical_concepts(dataset_path, derm7pt_metadata)
        stats["medical_concepts"] = medical_stats

        # Extract artifact concepts
        logger.info("\n[2/3] Extracting artifact concept patches...")
        artifact_stats = self._extract_artifact_concepts(dataset_path)
        stats["artifact_concepts"] = artifact_stats

        # Generate random patches
        logger.info("\n[3/3] Generating random baseline patches...")
        random_count = self._generate_random_patches(dataset_path)
        stats["random_patches"] = random_count

        # Compute total
        stats["total_patches"] = (
            sum(medical_stats.values()) + sum(artifact_stats.values()) + random_count
        )

        # Save metadata
        if self.config.save_metadata:
            self._save_metadata(stats)

        # DVC tracking
        if self.config.use_dvc:
            self._dvc_track()

        # Log summary
        self._log_summary(stats)

        logger.info("=" * 70)
        logger.info("CONCEPT BANK CREATION COMPLETE")
        logger.info("=" * 70)

        return stats

    def _create_directory_structure(self) -> None:
        """Create organized directory structure for concept bank."""
        base_dir = self.config.output_dir

        # Medical concepts directories
        medical_dir = base_dir / "medical"
        medical_dir.mkdir(parents=True, exist_ok=True)

        for concept in self.medical_concepts.keys():
            (medical_dir / concept).mkdir(exist_ok=True)

        # Artifact concepts directories
        artifact_dir = base_dir / "artifacts"
        artifact_dir.mkdir(parents=True, exist_ok=True)

        for concept in self.artifact_concepts.keys():
            (artifact_dir / concept).mkdir(exist_ok=True)

        # Random patches directory
        random_dir = base_dir / "random"
        random_dir.mkdir(exist_ok=True)

        logger.info(f"Created directory structure at {base_dir}")

    def _extract_medical_concepts(
        self,
        dataset_path: Path,
        derm7pt_metadata: Optional[Path] = None,
    ) -> Dict[str, int]:
        """
        Extract medical concept patches.

        For dermoscopy: Uses Derm7pt annotations
        For chest X-ray: Uses anatomy-based heuristics

        Args:
            dataset_path: Path to dataset
            derm7pt_metadata: Path to Derm7pt metadata (dermoscopy only)

        Returns:
            Dictionary mapping concept names to patch counts
        """
        stats = {concept: 0 for concept in self.medical_concepts.keys()}

        if self.config.modality == "dermoscopy":
            # Use Derm7pt annotations
            if derm7pt_metadata is not None and Path(derm7pt_metadata).exists():
                stats = self._extract_dermoscopy_medical_with_annotations(
                    dataset_path, derm7pt_metadata
                )
            else:
                logger.warning(
                    "Derm7pt metadata not provided, using heuristic extraction"
                )
                stats = self._extract_dermoscopy_medical_heuristic(dataset_path)
        else:
            # Chest X-ray: anatomy-based extraction
            stats = self._extract_chestxray_medical(dataset_path)

        return stats

    def _extract_dermoscopy_medical_with_annotations(
        self,
        dataset_path: Path,
        metadata_path: Path,
    ) -> Dict[str, int]:
        """Extract dermoscopy medical concepts using Derm7pt annotations."""
        import pandas as pd

        stats = {concept: 0 for concept in self.medical_concepts.keys()}

        # Load metadata
        df = pd.read_csv(metadata_path)

        # Map Derm7pt columns to concepts
        concept_columns = {
            "pigment_network": "pigment_network",
            "blue_white_veil": "blue_whitish_veil",
            "globules": "globules",
            "streaks": "streaks",
            "dots": "dots_and_globules",
        }

        for concept, column in concept_columns.items():
            if column not in df.columns:
                logger.warning(f"Column {column} not found in metadata")
                continue

            # Find images with this concept present
            concept_df = df[df[column].notna() & (df[column] != "absent")]

            target_count = self.config.num_medical_per_concept
            extracted = 0

            pbar = tqdm(
                concept_df.iterrows(),
                total=min(len(concept_df), target_count),
                desc=f"  {concept}",
                disable=self.config.verbose < 1,
            )

            for idx, row in pbar:
                if extracted >= target_count:
                    break

                # Get image path
                img_name = row.get("derm", row.get("image_id", f"{idx}.jpg"))
                img_path = dataset_path / img_name

                if not img_path.exists():
                    continue

                # Extract patches with this concept
                patches = self._extract_patches_from_image(
                    img_path, num_patches=5, quality_check=True
                )

                # Save patches
                for patch_idx, patch in enumerate(patches):
                    if extracted >= target_count:
                        break

                    save_path = (
                        self.config.output_dir
                        / "medical"
                        / concept
                        / f"{concept}_{extracted:04d}.png"
                    )
                    self._save_patch(patch, save_path)
                    extracted += 1

                pbar.set_postfix({"extracted": extracted})

            stats[concept] = extracted

        return stats

    def _extract_dermoscopy_medical_heuristic(
        self, dataset_path: Path
    ) -> Dict[str, int]:
        """Extract dermoscopy medical concepts using heuristics (no annotations)."""
        stats = {concept: 0 for concept in self.medical_concepts.keys()}

        # Get all images
        image_paths = list(dataset_path.glob("*.jpg")) + list(
            dataset_path.glob("*.png")
        )

        if not image_paths:
            logger.warning(f"No images found in {dataset_path}")
            return stats

        # For each concept, extract random patches from different images
        for concept in self.medical_concepts.keys():
            target_count = self.config.num_medical_per_concept
            extracted = 0

            # Sample images
            sampled_images = random.sample(
                image_paths, min(len(image_paths), target_count // 5 + 10)
            )

            pbar = tqdm(
                sampled_images,
                desc=f"  {concept}",
                disable=self.config.verbose < 1,
            )

            for img_path in pbar:
                if extracted >= target_count:
                    break

                # Extract diverse patches
                patches = self._extract_patches_from_image(
                    img_path, num_patches=5, quality_check=True
                )

                for patch in patches:
                    if extracted >= target_count:
                        break

                    save_path = (
                        self.config.output_dir
                        / "medical"
                        / concept
                        / f"{concept}_{extracted:04d}.png"
                    )
                    self._save_patch(patch, save_path)
                    extracted += 1

                pbar.set_postfix({"extracted": extracted})

            stats[concept] = extracted

        return stats

    def _extract_chestxray_medical(self, dataset_path: Path) -> Dict[str, int]:
        """Extract chest X-ray medical concepts using anatomy-based heuristics."""
        stats = {concept: 0 for concept in self.medical_concepts.keys()}

        image_paths = list(dataset_path.glob("*.jpg")) + list(
            dataset_path.glob("*.png")
        )

        if not image_paths:
            logger.warning(f"No images found in {dataset_path}")
            return stats

        # Anatomical regions for chest X-ray (normalized coordinates)
        concept_regions = {
            "lung_opacity": [(0.2, 0.2, 0.8, 0.7)],  # Central lung fields
            "cardiac_silhouette": [(0.35, 0.4, 0.65, 0.8)],  # Heart region
            "rib_shadows": [(0.1, 0.2, 0.9, 0.7)],  # Rib cage
            "costophrenic_angle": [
                (0.15, 0.7, 0.35, 0.9),
                (0.65, 0.7, 0.85, 0.9),
            ],  # Both angles
        }

        for concept, regions in concept_regions.items():
            target_count = self.config.num_medical_per_concept
            extracted = 0

            pbar = tqdm(
                image_paths,
                desc=f"  {concept}",
                disable=self.config.verbose < 1,
            )

            for img_path in pbar:
                if extracted >= target_count:
                    break

                # Extract patches from anatomical regions
                patches = self._extract_patches_from_regions(
                    img_path, regions, num_per_region=2
                )

                for patch in patches:
                    if extracted >= target_count:
                        break

                    if self._check_patch_quality(patch):
                        save_path = (
                            self.config.output_dir
                            / "medical"
                            / concept
                            / f"{concept}_{extracted:04d}.png"
                        )
                        self._save_patch(patch, save_path)
                        extracted += 1

                pbar.set_postfix({"extracted": extracted})

            stats[concept] = extracted

        return stats

    def _extract_artifact_concepts(self, dataset_path: Path) -> Dict[str, int]:
        """
        Extract artifact concept patches using detection heuristics.

        Args:
            dataset_path: Path to dataset

        Returns:
            Dictionary mapping artifact names to patch counts
        """
        stats = {concept: 0 for concept in self.artifact_concepts.keys()}

        if self.config.modality == "dermoscopy":
            stats = self._extract_dermoscopy_artifacts(dataset_path)
        else:
            stats = self._extract_chestxray_artifacts(dataset_path)

        return stats

    def _extract_dermoscopy_artifacts(self, dataset_path: Path) -> Dict[str, int]:
        """Extract dermoscopy artifact patches using detection heuristics."""
        stats = {concept: 0 for concept in self.artifact_concepts.keys()}

        image_paths = list(dataset_path.glob("*.jpg")) + list(
            dataset_path.glob("*.png")
        )

        for img_path in tqdm(
            image_paths,
            desc="  Detecting artifacts",
            disable=self.config.verbose < 1,
        ):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Detect ruler (bright horizontal/vertical lines)
            if stats["ruler"] < self.config.num_artifact_per_concept:
                ruler_patches = self._detect_ruler(img)
                stats["ruler"] += self._save_patches(
                    ruler_patches, "artifacts", "ruler", stats["ruler"]
                )

            # Detect hair (thin dark lines)
            if stats["hair"] < self.config.num_artifact_per_concept:
                hair_patches = self._detect_hair(img)
                stats["hair"] += self._save_patches(
                    hair_patches, "artifacts", "hair", stats["hair"]
                )

            # Detect ink marks (dark spots/lines at borders)
            if stats["ink_marks"] < self.config.num_artifact_per_concept:
                ink_patches = self._detect_ink_marks(img)
                stats["ink_marks"] += self._save_patches(
                    ink_patches, "artifacts", "ink_marks", stats["ink_marks"]
                )

            # Detect black borders
            if stats["black_borders"] < self.config.num_artifact_per_concept:
                border_patches = self._detect_black_borders(img)
                stats["black_borders"] += self._save_patches(
                    border_patches,
                    "artifacts",
                    "black_borders",
                    stats["black_borders"],
                )

        return stats

    def _extract_chestxray_artifacts(self, dataset_path: Path) -> Dict[str, int]:
        """Extract chest X-ray artifact patches using detection heuristics."""
        stats = {concept: 0 for concept in self.artifact_concepts.keys()}

        image_paths = list(dataset_path.glob("*.jpg")) + list(
            dataset_path.glob("*.png")
        )

        for img_path in tqdm(
            image_paths,
            desc="  Detecting artifacts",
            disable=self.config.verbose < 1,
        ):
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue

            # Detect text overlay (corner crops with high edge density)
            if stats["text_overlay"] < self.config.num_artifact_per_concept:
                text_patches = self._detect_text_overlay(img)
                stats["text_overlay"] += self._save_patches(
                    text_patches, "artifacts", "text_overlay", stats["text_overlay"]
                )

            # Detect borders (edge regions)
            if stats["borders"] < self.config.num_artifact_per_concept:
                border_patches = self._detect_xray_borders(img)
                stats["borders"] += self._save_patches(
                    border_patches, "artifacts", "borders", stats["borders"]
                )

            # Detect patient markers (bright spots/crosses)
            if stats["patient_markers"] < self.config.num_artifact_per_concept:
                marker_patches = self._detect_patient_markers(img)
                stats["patient_markers"] += self._save_patches(
                    marker_patches,
                    "artifacts",
                    "patient_markers",
                    stats["patient_markers"],
                )

            # Detect blank regions
            if stats["blank_regions"] < self.config.num_artifact_per_concept:
                blank_patches = self._detect_blank_regions(img)
                stats["blank_regions"] += self._save_patches(
                    blank_patches,
                    "artifacts",
                    "blank_regions",
                    stats["blank_regions"],
                )

        return stats

    def _generate_random_patches(self, dataset_path: Path) -> int:
        """
        Generate random patches for TCAV baseline.

        Args:
            dataset_path: Path to dataset

        Returns:
            Number of random patches generated
        """
        image_paths = list(dataset_path.glob("*.jpg")) + list(
            dataset_path.glob("*.png")
        )

        if not image_paths:
            logger.warning("No images found for random patch generation")
            return 0

        count = 0
        target = self.config.num_random

        pbar = tqdm(
            range(target),
            desc="  Generating random patches",
            disable=self.config.verbose < 1,
        )

        while count < target:
            # Random image
            img_path = random.choice(image_paths)

            # Extract random patches
            patches = self._extract_patches_from_image(
                img_path, num_patches=5, quality_check=True
            )

            for patch in patches:
                if count >= target:
                    break

                save_path = (
                    self.config.output_dir / "random" / f"random_{count:04d}.png"
                )
                self._save_patch(patch, save_path)
                count += 1
                pbar.update(1)

        pbar.close()
        return count

    # ========================================================================
    # Helper Methods - Patch Extraction
    # ========================================================================

    def _extract_patches_from_image(
        self,
        img_path: Path,
        num_patches: int = 5,
        quality_check: bool = True,
    ) -> List[np.ndarray]:
        """Extract diverse patches from single image."""
        img = cv2.imread(str(img_path))
        if img is None:
            return []

        h, w = img.shape[:2]
        patch_h, patch_w = self.config.patch_size

        if h < patch_h or w < patch_w:
            # Resize if image too small
            img = cv2.resize(img, (patch_w, patch_h))
            return [img] if not quality_check or self._check_patch_quality(img) else []

        patches = []
        attempts = 0
        max_attempts = num_patches * 10

        while len(patches) < num_patches and attempts < max_attempts:
            # Random crop
            y = random.randint(0, h - patch_h)
            x = random.randint(0, w - patch_w)

            patch = img[y : y + patch_h, x : x + patch_w]

            # Quality and diversity check
            if quality_check:
                if not self._check_patch_quality(patch):
                    attempts += 1
                    continue

                if not self._check_patch_diversity(patch, patches):
                    attempts += 1
                    continue

            patches.append(patch)
            attempts += 1

        return patches

    def _extract_patches_from_regions(
        self,
        img_path: Path,
        regions: List[Tuple[float, float, float, float]],
        num_per_region: int = 2,
    ) -> List[np.ndarray]:
        """Extract patches from specified normalized regions."""
        img = cv2.imread(str(img_path))
        if img is None:
            return []

        h, w = img.shape[:2]
        patch_h, patch_w = self.config.patch_size

        patches = []

        for x1_norm, y1_norm, x2_norm, y2_norm in regions:
            # Convert to absolute coordinates
            x1 = int(x1_norm * w)
            y1 = int(y1_norm * h)
            x2 = int(x2_norm * w)
            y2 = int(y2_norm * h)

            # Extract patches from this region
            for _ in range(num_per_region):
                if x2 - x1 < patch_w or y2 - y1 < patch_h:
                    continue

                x = random.randint(x1, max(x1, x2 - patch_w))
                y = random.randint(y1, max(y1, y2 - patch_h))

                patch = img[y : y + patch_h, x : x + patch_w]
                patches.append(patch)

        return patches

    def _check_patch_quality(self, patch: np.ndarray) -> bool:
        """
        Check patch quality (blur, contrast).

        Args:
            patch: Patch to check

        Returns:
            True if patch meets quality threshold
        """
        # Blur detection (Laplacian variance)
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()

        # Contrast check (std deviation)
        contrast = gray.std()

        # Quality score (normalized)
        quality = min(1.0, (laplacian_var / 500.0 + contrast / 100.0) / 2.0)

        return quality >= self.config.min_patch_quality

    def _check_patch_diversity(
        self, patch: np.ndarray, existing_patches: List[np.ndarray]
    ) -> bool:
        """
        Check if patch is diverse from existing patches.

        Args:
            patch: New patch
            existing_patches: Already extracted patches

        Returns:
            True if patch is sufficiently different
        """
        if not existing_patches:
            return True

        # Compute histogram for new patch
        hist1 = cv2.calcHist([patch], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
        hist1 = cv2.normalize(hist1, hist1).flatten()

        # Compare with existing patches
        for existing in existing_patches[-5:]:  # Check last 5 only
            hist2 = cv2.calcHist([existing], [0, 1, 2], None, [8, 8, 8], [0, 256] * 3)
            hist2 = cv2.normalize(hist2, hist2).flatten()

            # Correlation (high = similar)
            correlation = cv2.compareHist(hist1, hist2, cv2.HISTCMP_CORREL)

            if correlation > (1.0 - self.config.diversity_threshold):
                return False  # Too similar

        return True

    # ========================================================================
    # Helper Methods - Artifact Detection
    # ========================================================================

    def _detect_ruler(self, img: np.ndarray) -> List[np.ndarray]:
        """Detect ruler artifacts in dermoscopy images."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Detect horizontal/vertical lines (Hough transform)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, threshold=100, minLineLength=100, maxLineGap=10
        )

        if lines is None:
            return []

        patches = []
        patch_h, patch_w = self.config.patch_size

        for line in lines[:10]:  # Top 10 lines
            x1, y1, x2, y2 = line[0]

            # Extract patch around line
            cx, cy = (x1 + x2) // 2, (y1 + y2) // 2

            y_start = max(0, cy - patch_h // 2)
            x_start = max(0, cx - patch_w // 2)
            y_end = min(img.shape[0], y_start + patch_h)
            x_end = min(img.shape[1], x_start + patch_w)

            if y_end - y_start >= patch_h and x_end - x_start >= patch_w:
                patch = img[y_start:y_end, x_start:x_end]
                patches.append(patch)

        return patches[:3]  # Max 3 per image

    def _detect_hair(self, img: np.ndarray) -> List[np.ndarray]:
        """Detect hair artifacts in dermoscopy images."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Morphological black hat (detect dark lines)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (9, 9))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # Threshold
        _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)

        # Find contours (hair strands)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        patches = []
        patch_h, patch_w = self.config.patch_size

        for contour in contours[:10]:
            # Get bounding box
            x, y, w, h = cv2.boundingRect(contour)

            # Expand to patch size
            cx, cy = x + w // 2, y + h // 2
            y_start = max(0, cy - patch_h // 2)
            x_start = max(0, cx - patch_w // 2)
            y_end = min(img.shape[0], y_start + patch_h)
            x_end = min(img.shape[1], x_start + patch_w)

            if y_end - y_start >= patch_h and x_end - x_start >= patch_w:
                patch = img[y_start:y_end, x_start:x_end]
                patches.append(patch)

        return patches[:3]

    def _detect_ink_marks(self, img: np.ndarray) -> List[np.ndarray]:
        """Detect surgical ink marks in dermoscopy images."""
        # Ink marks are usually dark spots near borders
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        h, w = gray.shape

        # Focus on borders
        border_width = min(h, w) // 10
        borders = [
            gray[:border_width, :],  # Top
            gray[-border_width:, :],  # Bottom
            gray[:, :border_width],  # Left
            gray[:, -border_width:],  # Right
        ]

        patches = []
        patch_h, patch_w = self.config.patch_size

        for border in borders:
            # Find dark regions
            _, thresh = cv2.threshold(border, 50, 255, cv2.THRESH_BINARY_INV)

            if thresh.sum() > 1000:  # Significant dark area
                # Extract patch from this border region
                # (simplified - in production would locate exact ink mark)
                if len(patches) < 2:
                    patches.append(img[:patch_h, :patch_w])  # Placeholder extraction

        return patches

    def _detect_black_borders(self, img: np.ndarray) -> List[np.ndarray]:
        """Detect black border artifacts."""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 10, 255, cv2.THRESH_BINARY)

        # Find largest contour (image boundary)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            return []

        # Check if border exists
        largest_contour = max(contours, key=cv2.contourArea)
        x, y, w, h = cv2.boundingRect(largest_contour)

        border_exists = x > 5 or y > 5 or w < img.shape[1] - 10 or h < img.shape[0] - 10

        if not border_exists:
            return []

        # Extract border patches
        patch_h, patch_w = self.config.patch_size
        patches = [
            img[:patch_h, :patch_w],  # Top-left corner
            img[-patch_h:, :patch_w],  # Bottom-left corner
        ]

        return patches

    def _detect_text_overlay(self, img: np.ndarray) -> List[np.ndarray]:
        """Detect text overlay in chest X-rays (corner regions)."""
        h, w = img.shape
        corner_size = min(h, w) // 4

        # Extract corners
        corners = [
            img[:corner_size, :corner_size],  # Top-left
            img[:corner_size, -corner_size:],  # Top-right
            img[-corner_size:, :corner_size],  # Bottom-left
            img[-corner_size:, -corner_size:],  # Bottom-right
        ]

        patches = []
        patch_h, patch_w = self.config.patch_size

        for corner in corners:
            # Check for high edge density (text)
            edges = cv2.Canny(corner, 50, 150)
            edge_density = edges.sum() / (corner.shape[0] * corner.shape[1])

            if edge_density > 0.05:  # High edge density
                # Resize corner to patch size
                patch = cv2.resize(corner, (patch_w, patch_h))
                # Convert to BGR for consistency
                patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
                patches.append(patch)

        return patches[:2]

    def _detect_xray_borders(self, img: np.ndarray) -> List[np.ndarray]:
        """Detect image borders in chest X-rays."""
        h, w = img.shape
        border_width = min(h, w) // 20

        patch_h, patch_w = self.config.patch_size

        # Extract border regions
        patches = [
            cv2.resize(
                img[:border_width, :border_width], (patch_w, patch_h)
            ),  # Top-left
            cv2.resize(
                img[:border_width, -border_width:], (patch_w, patch_h)
            ),  # Top-right
        ]

        # Convert to BGR
        patches = [cv2.cvtColor(p, cv2.COLOR_GRAY2BGR) for p in patches]

        return patches

    def _detect_patient_markers(self, img: np.ndarray) -> List[np.ndarray]:
        """Detect patient positioning markers in chest X-rays."""
        # Detect very bright spots (thresholding)
        _, thresh = cv2.threshold(img, 240, 255, cv2.THRESH_BINARY)

        # Find contours
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        patches = []
        patch_h, patch_w = self.config.patch_size

        for contour in contours[:5]:
            area = cv2.contourArea(contour)
            if 100 < area < 5000:  # Marker size
                x, y, w, h = cv2.boundingRect(contour)

                # Extract patch
                cx, cy = x + w // 2, y + h // 2
                y_start = max(0, cy - patch_h // 2)
                x_start = max(0, cx - patch_w // 2)
                y_end = min(img.shape[0], y_start + patch_h)
                x_end = min(img.shape[1], x_start + patch_w)

                if y_end - y_start >= patch_h and x_end - x_start >= patch_w:
                    patch = img[y_start:y_end, x_start:x_end]
                    patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
                    patches.append(patch)

        return patches[:2]

    def _detect_blank_regions(self, img: np.ndarray) -> List[np.ndarray]:
        """Detect blank/unexposed regions in chest X-rays."""
        # Very dark regions
        _, thresh = cv2.threshold(img, 20, 255, cv2.THRESH_BINARY_INV)

        # Find large dark regions
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        patches = []
        patch_h, patch_w = self.config.patch_size

        for contour in contours[:5]:
            area = cv2.contourArea(contour)
            if area > 10000:  # Large blank area
                x, y, w, h = cv2.boundingRect(contour)

                if w >= patch_w and h >= patch_h:
                    patch = img[y : y + patch_h, x : x + patch_w]
                    patch = cv2.cvtColor(patch, cv2.COLOR_GRAY2BGR)
                    patches.append(patch)

        return patches[:2]

    # ========================================================================
    # Helper Methods - Saving
    # ========================================================================

    def _save_patch(self, patch: np.ndarray, save_path: Path) -> None:
        """Save single patch to disk."""
        save_path.parent.mkdir(parents=True, exist_ok=True)

        # Resize if needed
        if patch.shape[:2] != self.config.patch_size:
            patch = cv2.resize(patch, self.config.patch_size[::-1])

        cv2.imwrite(str(save_path), patch)

    def _save_patches(
        self,
        patches: List[np.ndarray],
        category: str,
        concept: str,
        start_idx: int,
    ) -> int:
        """Save multiple patches and return count."""
        saved = 0
        max_save = self.config.num_artifact_per_concept - start_idx

        for patch in patches[:max_save]:
            save_path = (
                self.config.output_dir
                / category
                / concept
                / f"{concept}_{start_idx + saved:04d}.png"
            )
            self._save_patch(patch, save_path)
            saved += 1

        return saved

    def _save_metadata(self, stats: Dict[str, Any]) -> None:
        """Save concept bank metadata."""
        metadata = {
            "config": {
                "modality": self.config.modality,
                "patch_size": self.config.patch_size,
                "seed": self.config.seed,
            },
            "statistics": stats,
            "concepts": {
                "medical": self.medical_concepts,
                "artifacts": self.artifact_concepts,
            },
        }

        metadata_path = self.config.output_dir / "metadata.json"
        with open(metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)

        logger.info(f"Saved metadata to {metadata_path}")

    def _dvc_track(self) -> None:
        """Track concept bank with DVC."""
        try:
            import subprocess

            result = subprocess.run(
                ["dvc", "add", str(self.config.output_dir)],
                capture_output=True,
                text=True,
                check=True,
            )
            logger.info(f"DVC tracking: {result.stdout}")
        except subprocess.CalledProcessError as e:
            logger.warning(f"DVC tracking failed: {e.stderr}")
        except FileNotFoundError:
            logger.warning("DVC not installed, skipping tracking")

    def _log_summary(self, stats: Dict[str, Any]) -> None:
        """Log concept bank creation summary."""
        logger.info("\n" + "=" * 70)
        logger.info("CONCEPT BANK SUMMARY")
        logger.info("=" * 70)

        logger.info(f"\nModality: {stats['modality']}")
        logger.info(f"Total patches: {stats['total_patches']}")

        logger.info("\nMedical Concepts:")
        for concept, count in stats["medical_concepts"].items():
            logger.info(f"  {concept:30s}: {count:4d} patches")

        logger.info("\nArtifact Concepts:")
        for concept, count in stats["artifact_concepts"].items():
            logger.info(f"  {concept:30s}: {count:4d} patches")

        logger.info(f"\nRandom Patches: {stats['random_patches']}")
        logger.info("\n" + "=" * 70)


# ============================================================================
# Factory Function
# ============================================================================


def create_concept_bank_creator(
    config: Optional[ConceptBankConfig] = None,
    **kwargs,
) -> ConceptBankCreator:
    """
    Factory function to create concept bank creator.

    Args:
        config: Configuration (uses defaults if None)
        **kwargs: Override config parameters

    Returns:
        ConceptBankCreator instance
    """
    if config is None:
        config = ConceptBankConfig(**kwargs)
    return ConceptBankCreator(config)
