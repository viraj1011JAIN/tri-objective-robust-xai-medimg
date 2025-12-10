"""
Create concept bank for TCAV analysis.
Curate artifact and medical concept examples from ISIC dataset.
"""

import os
import shutil
from pathlib import Path
from typing import List, Tuple

import cv2
import numpy as np
import pandas as pd
from tqdm import tqdm


class ConceptBankCreator:
    """Curate concept examples from dermoscopy images."""

    def __init__(
        self,
        isic_root: str = "data/ISIC2018",
        output_root: str = "data/concepts/dermoscopy",
        target_per_concept: int = 100,
    ):
        self.isic_root = Path(isic_root)
        self.output_root = Path(output_root)
        self.target_per_concept = target_per_concept

        # Create output directories
        self.output_root.mkdir(parents=True, exist_ok=True)

    def extract_artifact_concepts(self):
        """
        Extract artifact concept examples.

        Artifacts to detect:
        1. Ruler markings (high-contrast edges in corners)
        2. Hair (thin dark lines)
        3. Ink marks (dark circular regions)
        4. Black borders (uniform dark regions at edges)
        """

        print("\n" + "=" * 60)
        print("EXTRACTING ARTIFACT CONCEPTS")
        print("=" * 60)

        images_dir = self.isic_root / "ISIC2018_Task3_Training_Input"
        all_images = list(images_dir.glob("*.jpg"))

        # Storage for detected concepts
        concepts = {"ruler": [], "hair": [], "ink_marks": [], "black_borders": []}

        for img_path in tqdm(all_images, desc="Scanning for artifacts"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            # Detect each artifact type
            detections = self._detect_artifacts(img, img_path.stem)

            for concept, patches in detections.items():
                concepts[concept].extend(patches)

        # Save top examples for each concept
        for concept, patches in concepts.items():
            self._save_concept_patches(
                patches, concept, "artifacts", self.target_per_concept
            )

    def _detect_artifacts(self, img: np.ndarray, img_id: str) -> dict:
        """
        Detect various artifacts in image.

        Returns:
            dict: {concept_name: [(patch, score), ...]}
        """
        h, w = img.shape[:2]
        detections = {"ruler": [], "hair": [], "ink_marks": [], "black_borders": []}

        # Convert to grayscale for detection
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # 1. Ruler detection (Hough lines in corners)
        edges = cv2.Canny(gray, 50, 150)
        lines = cv2.HoughLinesP(
            edges, 1, np.pi / 180, 100, minLineLength=50, maxLineGap=10
        )

        if lines is not None:
            # Check for straight lines in corners (ruler indicator)
            for line in lines[:10]:  # Top 10 lines
                x1, y1, x2, y2 = line[0]
                # Extract patch around line
                cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
                if self._is_in_corner(cx, cy, w, h):
                    patch = self._extract_patch(img, cx, cy, 128)
                    if patch is not None:
                        score = self._compute_ruler_score(patch)
                        detections["ruler"].append((patch, score, img_id))

        # 2. Hair detection (thin dark lines)
        # Use morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 15))
        blackhat = cv2.morphologyEx(gray, cv2.MORPH_BLACKHAT, kernel)

        # Find contours (potential hairs)
        _, thresh = cv2.threshold(blackhat, 10, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(
            thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        for contour in contours[:20]:  # Top 20 candidates
            # Get bounding box
            x, y, w_c, h_c = cv2.boundingRect(contour)
            aspect_ratio = h_c / (w_c + 1e-6)

            # Hair is thin and long
            if aspect_ratio > 5 and h_c > 30:
                cx, cy = x + w_c // 2, y + h_c // 2
                patch = self._extract_patch(img, cx, cy, 128)
                if patch is not None:
                    score = self._compute_hair_score(patch)
                    detections["hair"].append((patch, score, img_id))

        # 3. Ink marks detection (dark circular regions)
        # Find dark circles
        circles = cv2.HoughCircles(
            gray,
            cv2.HOUGH_GRADIENT,
            1,
            20,
            param1=50,
            param2=30,
            minRadius=10,
            maxRadius=50,
        )

        if circles is not None:
            circles = np.uint16(np.around(circles))
            for circle in circles[0, :10]:  # Top 10 circles
                cx, cy, r = circle
                # Check if dark enough to be ink
                roi = gray[max(0, cy - r) : cy + r, max(0, cx - r) : cx + r]
                if roi.mean() < 80:  # Dark region
                    patch = self._extract_patch(img, cx, cy, 128)
                    if patch is not None:
                        score = self._compute_ink_score(patch)
                        detections["ink_marks"].append((patch, score, img_id))

        # 4. Black borders detection (uniform dark edges)
        # Check all four borders
        border_width = 20
        borders = {
            "top": gray[:border_width, :],
            "bottom": gray[-border_width:, :],
            "left": gray[:, :border_width],
            "right": gray[:, -border_width:],
        }

        for border_name, border_region in borders.items():
            if border_region.mean() < 30:  # Very dark
                # Extract corner patch
                if border_name == "top":
                    patch = img[:128, :128]
                elif border_name == "bottom":
                    patch = img[-128:, :128]
                elif border_name == "left":
                    patch = img[:128, :128]
                else:  # right
                    patch = img[:128, -128:]

                if patch.shape[:2] == (128, 128):
                    score = self._compute_border_score(patch)
                    detections["black_borders"].append((patch, score, img_id))

        return detections

    def _is_in_corner(self, x: int, y: int, w: int, h: int) -> bool:
        """Check if point is in corner region (20% of image)."""
        corner_threshold = 0.2
        return (
            x < w * corner_threshold
            or x > w * (1 - corner_threshold)
            or y < h * corner_threshold
            or y > h * (1 - corner_threshold)
        )

    def _extract_patch(
        self, img: np.ndarray, cx: int, cy: int, size: int
    ) -> np.ndarray:
        """Extract centered patch of given size."""
        h, w = img.shape[:2]
        half_size = size // 2

        y1 = max(0, cy - half_size)
        y2 = min(h, cy + half_size)
        x1 = max(0, cx - half_size)
        x2 = min(w, cx + half_size)

        patch = img[y1:y2, x1:x2]

        # Resize to exact size if needed
        if patch.shape[:2] != (size, size):
            patch = cv2.resize(patch, (size, size))

        return patch if patch.shape[:2] == (size, size) else None

    def _compute_ruler_score(self, patch: np.ndarray) -> float:
        """Score patch for ruler-like characteristics."""
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)
        edges = cv2.Canny(gray, 50, 150)

        # High edge density + straight lines = ruler
        edge_density = edges.sum() / edges.size

        # Detect straight lines
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50)
        n_lines = len(lines) if lines is not None else 0

        return edge_density * 0.7 + min(n_lines / 10, 1.0) * 0.3

    def _compute_hair_score(self, patch: np.ndarray) -> float:
        """Score patch for hair-like characteristics."""
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        # Thin dark structure score
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)

        # Morphological operations to detect thin structures
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 3))
        thin_structures = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel)

        thinness = thin_structures.sum() / (binary.sum() + 1e-6)
        darkness = 1.0 - gray.mean() / 255

        return thinness * 0.6 + darkness * 0.4

    def _compute_ink_score(self, patch: np.ndarray) -> float:
        """Score patch for ink mark characteristics."""
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        # Dark + circular
        darkness = 1.0 - gray.mean() / 255

        # Detect circularity via contours
        _, binary = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY_INV)
        contours, _ = cv2.findContours(
            binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )

        circularity = 0
        if contours:
            # Find largest contour
            c = max(contours, key=cv2.contourArea)
            area = cv2.contourArea(c)
            perimeter = cv2.arcLength(c, True)
            if perimeter > 0:
                circularity = 4 * np.pi * area / (perimeter**2)

        return darkness * 0.5 + circularity * 0.5

    def _compute_border_score(self, patch: np.ndarray) -> float:
        """Score patch for black border characteristics."""
        gray = cv2.cvtColor(patch, cv2.COLOR_BGR2GRAY)

        # Uniformly dark
        darkness = 1.0 - gray.mean() / 255
        uniformity = 1.0 - (gray.std() / 128)  # Low variance

        return darkness * 0.7 + uniformity * 0.3

    def _save_concept_patches(
        self, patches: List[Tuple], concept: str, category: str, max_count: int
    ):
        """Save top-scoring patches for a concept."""

        if not patches:
            print(f"Warning: No patches found for {concept}")
            return

        # Sort by score (descending)
        patches = sorted(patches, key=lambda x: x[1], reverse=True)

        # Take top N
        patches = patches[:max_count]

        # Create output directory
        output_dir = self.output_root / category / concept
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save patches
        for idx, (patch, score, img_id) in enumerate(patches):
            output_path = output_dir / f"{img_id}_{idx:03d}_score{score:.2f}.jpg"
            cv2.imwrite(str(output_path), patch)

        print(
            f"✓ Saved {len(patches)} patches for {concept} "
            f"(avg score: {np.mean([s for _, s, _ in patches]):.3f})"
        )

    def extract_medical_concepts(self, metadata_path: str = None):
        """
        Extract medical concept examples using heuristics.

        Medical concepts:
        1. Asymmetry
        2. Pigment network
        3. Blue-white veil
        """

        print("\n" + "=" * 60)
        print("EXTRACTING MEDICAL CONCEPTS")
        print("=" * 60)

        print("Note: Using ISIC images with heuristic detection.")
        print(
            "For publication, consider using Derm7pt dataset with ground-truth annotations."
        )

        images_dir = self.isic_root / "ISIC2018_Task3_Training_Input"
        all_images = list(images_dir.glob("*.jpg"))

        concepts = {"asymmetry": [], "pigment_network": [], "blue_white_veil": []}

        for img_path in tqdm(all_images, desc="Detecting medical concepts"):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            detections = self._detect_medical_concepts(img, img_path.stem)

            for concept, patches in detections.items():
                concepts[concept].extend(patches)

        # Save concepts
        for concept, patches in concepts.items():
            self._save_concept_patches(
                patches, concept, "medical", self.target_per_concept
            )

    def _detect_medical_concepts(self, img: np.ndarray, img_id: str) -> dict:
        """
        Detect medical concepts using image analysis heuristics.

        Note: This is simplified. Real implementation should use:
        - Derm7pt dataset annotations
        - Dermatologist-labeled examples
        - Or segmentation masks
        """

        h, w = img.shape[:2]
        detections = {"asymmetry": [], "pigment_network": [], "blue_white_veil": []}

        # 1. Asymmetry detection
        # Split image and compare halves
        left_half = img[:, : w // 2]
        right_half = img[:, w // 2 :]

        # Flip right half
        right_flipped = cv2.flip(right_half, 1)

        # Compute difference
        if left_half.shape == right_flipped.shape:
            diff = cv2.absdiff(left_half, right_flipped)
            asymmetry_score = diff.mean() / 255

            if asymmetry_score > 0.15:  # Significant asymmetry
                # Extract full lesion as concept example
                patch = cv2.resize(img, (224, 224))
                detections["asymmetry"].append((patch, asymmetry_score, img_id))

        # 2. Pigment network detection
        # Look for net-like dark patterns
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        # Apply morphological operations to detect network
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        tophat = cv2.morphologyEx(gray, cv2.MORPH_TOPHAT, kernel)

        network_score = tophat.std() / 128  # High variance = network pattern

        if network_score > 0.15:
            patch = cv2.resize(img, (224, 224))
            detections["pigment_network"].append((patch, network_score, img_id))

        # 3. Blue-white veil detection
        # Look for blue-ish regions
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

        # Blue hue range
        lower_blue = np.array([90, 30, 30])
        upper_blue = np.array([130, 255, 255])

        blue_mask = cv2.inRange(hsv, lower_blue, upper_blue)
        blue_ratio = blue_mask.sum() / blue_mask.size

        if blue_ratio > 0.05:  # Some blue present
            # White regions (high value)
            _, white_mask = cv2.threshold(hsv[:, :, 2], 200, 255, cv2.THRESH_BINARY)

            # Blue + White = blue-white veil
            veil_mask = cv2.bitwise_and(blue_mask, white_mask)
            veil_score = veil_mask.sum() / veil_mask.size

            if veil_score > 0.02:
                patch = cv2.resize(img, (224, 224))
                detections["blue_white_veil"].append((patch, veil_score, img_id))

        return detections

    def create_random_concepts(self, n_random: int = 500):
        """
        Create random concept patches for CAV training baseline.

        Args:
            n_random: Number of random patches to extract
        """

        print("\n" + "=" * 60)
        print("EXTRACTING RANDOM CONCEPTS")
        print("=" * 60)

        images_dir = self.isic_root / "ISIC2018_Task3_Training_Input"
        all_images = list(images_dir.glob("*.jpg"))

        output_dir = self.output_root / "random"
        output_dir.mkdir(parents=True, exist_ok=True)

        random_patches = []

        for img_path in tqdm(
            np.random.choice(all_images, min(n_random, len(all_images)), replace=False),
            desc="Extracting random patches",
        ):
            img = cv2.imread(str(img_path))
            if img is None:
                continue

            h, w = img.shape[:2]

            # Random location
            cx = np.random.randint(64, w - 64)
            cy = np.random.randint(64, h - 64)

            # Extract patch
            patch = self._extract_patch(img, cx, cy, 128)

            if patch is not None:
                random_patches.append((patch, img_path.stem))

                if len(random_patches) >= n_random:
                    break

        # Save random patches
        for idx, (patch, img_id) in enumerate(random_patches):
            output_path = output_dir / f"{img_id}_random_{idx:03d}.jpg"
            cv2.imwrite(str(output_path), patch)

        print(f"✓ Saved {len(random_patches)} random patches")

    def verify_concept_bank(self):
        """Verify concept bank is complete and print statistics."""

        print("\n" + "=" * 60)
        print("CONCEPT BANK VERIFICATION")
        print("=" * 60)

        # Check all concept directories
        concepts_to_check = {
            "Artifacts": ["ruler", "hair", "ink_marks", "black_borders"],
            "Medical": ["asymmetry", "pigment_network", "blue_white_veil"],
            "Random": [""],
        }

        for category, concepts in concepts_to_check.items():
            print(f"\n{category}:")

            if category == "Random":
                path = self.output_root / "random"
                count = len(list(path.glob("*.jpg"))) if path.exists() else 0
                print(f"  Random patches: {count}")
            else:
                for concept in concepts:
                    path = self.output_root / category.lower() / concept
                    count = len(list(path.glob("*.jpg"))) if path.exists() else 0
                    status = "✓" if count >= 50 else "⚠"
                    print(f"  {status} {concept}: {count} images")

        print("\n" + "=" * 60)


def main():
    """Main execution."""

    creator = ConceptBankCreator(
        isic_root="data/ISIC2018",
        output_root="data/concepts/dermoscopy",
        target_per_concept=100,
    )

    # Extract all concepts
    print("Starting concept bank creation...")

    creator.extract_artifact_concepts()
    creator.extract_medical_concepts(metadata_path=None)  # Use heuristics
    creator.create_random_concepts(n_random=500)

    # Verify
    creator.verify_concept_bank()

    print("\n✓ Concept bank creation complete!")
    print("Next step: Train CAVs using scripts/evaluation/train_cavs.py")


if __name__ == "__main__":
    main()
