"""
TCAV (Testing with Concept Activation Vectors) implementation.

Reference:
    Been Kim et al. "Interpretability Beyond Feature Attribution:
    Quantitative Testing with Concept Activation Vectors (TCAV)"
    ICML 2018.
"""

from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import numpy as np
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from torch.utils.data import DataLoader, Dataset


class ConceptDataset(Dataset):
    """Dataset for concept examples."""

    def __init__(self, concept_dir: Path, transform=None):
        self.images = list(concept_dir.glob("*.jpg"))
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        else:
            # Default: resize and normalize
            img = cv2.resize(img, (224, 224))
            img = torch.from_numpy(img).permute(2, 0, 1).float() / 255.0
            # ImageNet normalization
            mean = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
            std = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)
            img = (img - mean) / std

        return img


class TCAV:
    """
    Testing with Concept Activation Vectors.

    Measures directional sensitivity of model predictions to concepts.
    """

    def __init__(self, model: nn.Module, target_layer: str, device: torch.device):
        """
        Initialize TCAV.

        Args:
            model: PyTorch model
            target_layer: Layer name to extract activations from (e.g., 'layer4')
            device: Computation device
        """
        self.model = model
        self.target_layer = target_layer
        self.device = device

        # Storage for activations
        self.activations = None
        self.hook_handle = None

        # Attach hook
        self._register_hook()

    def _register_hook(self):
        """Register forward hook to capture activations."""

        def hook_fn(module, input, output):
            # Global average pooling
            self.activations = output.mean(dim=[2, 3])  # (B, C)

        # Get target layer
        layer = dict(self.model.named_modules())[self.target_layer]
        self.hook_handle = layer.register_forward_hook(hook_fn)

    def extract_activations(self, images: torch.Tensor) -> np.ndarray:
        """
        Extract activations for images.

        Args:
            images: (B, C, H, W) input images

        Returns:
            activations: (B, D) activation vectors
        """
        self.model.eval()

        with torch.no_grad():
            images = images.to(self.device)
            _ = self.model(images)

            activations = self.activations.cpu().numpy()

        return activations

    def train_cav(
        self,
        concept_activations: np.ndarray,
        random_activations: np.ndarray,
        test_size: float = 0.3,
    ) -> Tuple[np.ndarray, float]:
        """
        Train Concept Activation Vector using linear SVM.

        Args:
            concept_activations: (N_concept, D) activations for concept examples
            random_activations: (N_random, D) activations for random examples
            test_size: Fraction for validation

        Returns:
            cav: (D,) unit vector representing concept direction
            accuracy: Classification accuracy on validation set
        """

        # Combine data
        X = np.vstack([concept_activations, random_activations])
        y = np.array([1] * len(concept_activations) + [0] * len(random_activations))

        # Split for validation
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42, stratify=y
        )

        # Train linear SVM
        svm = LinearSVC(C=1.0, max_iter=5000, random_state=42)
        svm.fit(X_train, y_train)

        # Get CAV (normal vector of decision boundary)
        cav = svm.coef_[0]
        cav = cav / np.linalg.norm(cav)  # Normalize to unit vector

        # Validation accuracy
        accuracy = svm.score(X_test, y_test)

        return cav, accuracy

    def compute_tcav_score(
        self, test_loader: DataLoader, cav: np.ndarray, target_class: int
    ) -> float:
        """
        Compute TCAV score: fraction of examples with positive directional derivative.

        TCAV score = (1/N) * Σ I[∇_a S_c(x) · v > 0]

        Where:
            ∇_a S_c(x): Gradient of class logit w.r.t. activations
            v: CAV (concept direction)

        Args:
            test_loader: DataLoader for test examples
            cav: (D,) Concept Activation Vector
            target_class: Class to compute sensitivity for

        Returns:
            tcav_score: Fraction in range [0, 1]
        """

        self.model.eval()
        cav_tensor = torch.from_numpy(cav).float().to(self.device)

        positive_count = 0
        total_count = 0

        for images, labels in test_loader:
            images = images.to(self.device)
            labels = labels.to(self.device)

            # Only compute for images of target class
            mask = labels == target_class
            if not mask.any():
                continue

            images = images[mask]
            batch_size = images.size(0)

            # Forward pass to get activations
            images.requires_grad_(True)
            _ = self.model(images)
            activations = self.activations  # (B, D)

            # Get logit for target class
            logits = self.model(images)
            target_logits = logits[:, target_class]

            # Compute gradient of logit w.r.t. activations
            # ∇_a S_c(x) for each example
            for i in range(batch_size):
                self.model.zero_grad()

                target_logits[i].backward(retain_graph=True)

                # Get gradient
                grad = activations.grad[i]  # (D,)

                # Directional derivative: grad · cav
                directional_deriv = torch.dot(grad, cav_tensor)

                # Check if positive
                if directional_deriv > 0:
                    positive_count += 1

                total_count += 1

                # Clear gradients
                images.grad = None
                activations.grad = None

        tcav_score = positive_count / total_count if total_count > 0 else 0.0

        return tcav_score

    def cleanup(self):
        """Remove forward hook."""
        if self.hook_handle is not None:
            self.hook_handle.remove()


class ConceptBank:
    """
    Manage concept examples and CAVs.
    """

    def __init__(
        self,
        concept_root: str = "data/concepts/dermoscopy",
        cav_save_path: str = "data/concepts/dermoscopy_cavs.pth",
    ):
        self.concept_root = Path(concept_root)
        self.cav_save_path = Path(cav_save_path)

        # Concept definitions
        self.artifact_concepts = ["ruler", "hair", "ink_marks", "black_borders"]
        self.medical_concepts = ["asymmetry", "pigment_network", "blue_white_veil"]

        # Storage
        self.cavs: Dict[str, np.ndarray] = {}
        self.cav_accuracies: Dict[str, float] = {}

    def get_concept_loader(self, concept_name: str, batch_size: int = 32) -> DataLoader:
        """Get DataLoader for concept examples."""

        if concept_name == "random":
            concept_dir = self.concept_root / "random"
        else:
            # Search in artifacts and medical
            artifact_path = self.concept_root / "artifacts" / concept_name
            medical_path = self.concept_root / "medical" / concept_name

            if artifact_path.exists():
                concept_dir = artifact_path
            elif medical_path.exists():
                concept_dir = medical_path
            else:
                raise ValueError(f"Concept {concept_name} not found")

        dataset = ConceptDataset(concept_dir)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        return loader

    def train_all_cavs(self, model: nn.Module, target_layer: str, device: torch.device):
        """
        Train CAVs for all concepts.

        Args:
            model: PyTorch model
            target_layer: Layer to extract activations from
            device: Computation device
        """

        print("\n" + "=" * 60)
        print("TRAINING CONCEPT ACTIVATION VECTORS (CAVs)")
        print("=" * 60)

        tcav = TCAV(model, target_layer, device)

        # Get random activations (baseline)
        print("\nExtracting random activations...")
        random_loader = self.get_concept_loader("random", batch_size=64)

        random_acts = []
        for images in random_loader:
            acts = tcav.extract_activations(images)
            random_acts.append(acts)

        random_acts = np.vstack(random_acts)
        print(f"  Random activations shape: {random_acts.shape}")

        # Train CAV for each concept
        all_concepts = self.artifact_concepts + self.medical_concepts

        for concept in all_concepts:
            print(f"\nTraining CAV for: {concept}")

            # Get concept activations
            concept_loader = self.get_concept_loader(concept, batch_size=64)

            concept_acts = []
            for images in concept_loader:
                acts = tcav.extract_activations(images)
                concept_acts.append(acts)

            concept_acts = np.vstack(concept_acts)
            print(f"  Concept activations shape: {concept_acts.shape}")

            # Train CAV
            cav, accuracy = tcav.train_cav(concept_acts, random_acts)

            self.cavs[concept] = cav
            self.cav_accuracies[concept] = accuracy

            print(f"  ✓ CAV trained with {accuracy*100:.1f}% accuracy")

        # Cleanup
        tcav.cleanup()

        # Save CAVs
        self.save_cavs()

        print("\n" + "=" * 60)
        print("CAV TRAINING SUMMARY")
        print("=" * 60)

        print("\nArtifacts:")
        for concept in self.artifact_concepts:
            acc = self.cav_accuracies[concept]
            status = "✓" if acc > 0.7 else "⚠"
            print(f"  {status} {concept}: {acc*100:.1f}%")

        print("\nMedical:")
        for concept in self.medical_concepts:
            acc = self.cav_accuracies[concept]
            status = "✓" if acc > 0.7 else "⚠"
            print(f"  {status} {concept}: {acc*100:.1f}%")

        print("\n✓ All CAVs saved to:", self.cav_save_path)

    def save_cavs(self):
        """Save CAVs to disk."""
        save_dict = {
            "cavs": self.cavs,
            "accuracies": self.cav_accuracies,
            "artifact_concepts": self.artifact_concepts,
            "medical_concepts": self.medical_concepts,
        }

        self.cav_save_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(save_dict, self.cav_save_path)

    def load_cavs(self):
        """Load CAVs from disk."""
        if not self.cav_save_path.exists():
            raise FileNotFoundError(f"CAV file not found: {self.cav_save_path}")

        save_dict = torch.load(self.cav_save_path)

        self.cavs = save_dict["cavs"]
        self.cav_accuracies = save_dict["accuracies"]
        self.artifact_concepts = save_dict["artifact_concepts"]
        self.medical_concepts = save_dict["medical_concepts"]

        print(f"✓ Loaded {len(self.cavs)} CAVs from {self.cav_save_path}")

    def get_cav(self, concept_name: str) -> np.ndarray:
        """Get CAV for a concept."""
        if concept_name not in self.cavs:
            raise ValueError(f"CAV not found for concept: {concept_name}")
        return self.cavs[concept_name]
