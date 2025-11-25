"""Testing with Concept Activation Vectors (TCAV) Implementation.

This module implements the TCAV methodology introduced by Kim et al. (2018) for
interpreting neural networks using human-understandable concepts. TCAV provides
quantitative measures of the importance of user-defined concepts to a neural
network's predictions.

Key Components:
    - TCAVConfig: Configuration dataclass for TCAV parameters
    - ConceptDataset: PyTorch Dataset for concept examples
    - ActivationExtractor: Extract activations from specific layers
    - CAVTrainer: Train Concept Activation Vectors (CAVs)
    - TCAV: Main class implementing the TCAV methodology

Algorithm Overview:
    1. Extract activations from target layer for concept examples
    2. Train linear classifier to distinguish concept vs random examples
    3. Extract CAV (concept activation vector) as classifier's normal vector
    4. Compute TCAV score: percentage of samples where gradient aligns with CAV
    5. Statistical testing with multiple random concepts

Research Context:
    TCAV enables testing hypotheses about which concepts a model uses for predictions,
    providing human-interpretable explanations beyond pixel-level attribution methods.
    In medical imaging, TCAV can validate if models use clinically relevant features
    (e.g., "pigment network" for melanoma) rather than spurious correlations.

Reference:
    Kim, B., Wattenberg, M., Gilmer, J., Cai, C., Wexler, J., Viegas, F., & Sayres, R. (2018).
    Interpretability beyond feature attribution: Quantitative testing with concept activation
    vectors (tcav). In International Conference on Machine Learning (pp. 2668-2677). PMLR.

Example:
    >>> from src.xai.tcav import create_tcav
    >>> import torch.nn as nn
    >>>
    >>> # Create TCAV instance
    >>> tcav = create_tcav(
    ...     model=my_resnet,
    ...     target_layers=["layer3", "layer4"],
    ...     concept_data_dir="data/concepts",
    ...     cav_dir="checkpoints/cavs"
    ... )
    >>>
    >>> # Train CAV for a concept
    >>> cav, metrics = tcav.train_cav(
    ...     concept="pigment_network",
    ...     layer="layer3",
    ...     random_concept="random_0"
    ... )
    >>>
    >>> # Compute TCAV score
    >>> images = torch.randn(32, 3, 224, 224)
    >>> score = tcav.compute_tcav_score(
    ...     inputs=images,
    ...     target_class=1,  # melanoma
    ...     concept="pigment_network",
    ...     layer="layer3"
    ... )
    >>> print(f"TCAV score: {score:.2%}")
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import torch
import torch.nn as nn
from PIL import Image
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class TCAVConfig:
    """Configuration for TCAV.

    Attributes:
        model: PyTorch model for explanation
        target_layers: List of layer names to extract CAVs from
        concept_data_dir: Directory containing concept datasets
        cav_dir: Directory to save/load CAVs
        batch_size: Batch size for data loading
        num_random_concepts: Number of random concepts for statistical testing
        alpha: Significance level for statistical testing
        min_cav_accuracy: Minimum CAV accuracy threshold
        device: Device to run computations on
        seed: Random seed for reproducibility
        verbose: Verbosity level (0=silent, 1=info, 2=debug)
    """

    model: nn.Module
    target_layers: List[str]
    concept_data_dir: Union[str, Path]
    cav_dir: Union[str, Path]
    batch_size: int = 32
    num_random_concepts: int = 10
    alpha: float = 0.05
    min_cav_accuracy: float = 0.7
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    verbose: int = 1

    def __post_init__(self):
        """Validate configuration parameters."""
        if not self.target_layers:
            raise ValueError("target_layers cannot be empty")

        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")

        if not 0 < self.alpha < 1:
            raise ValueError(f"alpha must be in (0, 1), got {self.alpha}")

        if not 0 < self.min_cav_accuracy <= 1:
            raise ValueError(
                f"min_cav_accuracy must be in (0, 1], got {self.min_cav_accuracy}"
            )

        # Convert paths
        self.concept_data_dir = Path(self.concept_data_dir)
        self.cav_dir = Path(self.cav_dir)

        # Create directories if needed
        self.cav_dir.mkdir(parents=True, exist_ok=True)


class ConceptDataset(Dataset):
    """PyTorch Dataset for concept examples.

    Args:
        images: Image tensors (N, C, H, W)
        labels: Labels for images (N,)
        transform: Optional transforms to apply
    """

    def __init__(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        transform: Optional[transforms.Compose] = None,
    ):
        """Initialize dataset."""
        self.images = images
        self.labels = labels
        self.transform = transform

    def __len__(self) -> int:
        """Return dataset size."""
        return len(self.images)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get item by index."""
        image = self.images[idx]
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label


class ActivationExtractor:
    """Extract activations from specific layers of a model.

    This class uses forward hooks to capture activations from target layers
    during model forward pass. Supports global average pooling to reduce
    spatial dimensions.

    Args:
        model: PyTorch model
        target_layers: List of layer names to extract from
        device: Device to run on
    """

    def __init__(self, model: nn.Module, target_layers: List[str], device: str = "cpu"):
        """Initialize activation extractor."""
        self.model = model
        self.target_layers = target_layers
        self.device = device
        self.activations: Dict[str, torch.Tensor] = {}
        self.hooks = []

        self._register_hooks()

    def _register_hooks(self):
        """Register forward hooks on target layers."""

        def get_activation(name):
            def hook(module, input, output):
                # Apply global average pooling if spatial dimensions exist
                if output.dim() == 4:  # (B, C, H, W)
                    output = output.mean(dim=[2, 3])  # (B, C)
                # Don't detach - we need gradients for TCAV score computation
                self.activations[name] = output

            return hook

        # Find and register hooks
        for name, module in self.model.named_modules():
            if name in self.target_layers:
                handle = module.register_forward_hook(get_activation(name))
                self.hooks.append(handle)

                if logger.level <= logging.DEBUG:
                    logger.debug(f"Registered hook on layer: {name}")

        if len(self.hooks) != len(self.target_layers):
            found_layers = [
                name
                for name, _ in self.model.named_modules()
                if name in self.target_layers
            ]
            missing = set(self.target_layers) - set(found_layers)
            raise ValueError(f"Could not find layers: {missing}")

    def extract(
        self, dataloader: DataLoader, layer_name: str, detach: bool = True
    ) -> torch.Tensor:
        """Extract activations for all samples in dataloader.

        Args:
            dataloader: DataLoader with samples
            layer_name: Name of layer to extract from
            detach: Whether to detach activations (for CAV training)

        Returns:
            Activations tensor (N, C) where N is number of samples
        """
        self.model.eval()
        all_activations = []

        with torch.no_grad():
            for batch, _ in dataloader:
                batch = batch.to(self.device)

                # Forward pass (triggers hooks)
                _ = self.model(batch)

                if layer_name not in self.activations:
                    raise ValueError(f"Layer {layer_name} not in target_layers")

                acts = self.activations[layer_name]
                if detach:
                    acts = acts.detach()
                all_activations.append(acts.cpu())

        return torch.cat(all_activations, dim=0)

    def remove_hooks(self):
        """Remove all registered hooks."""
        for handle in self.hooks:
            handle.remove()
        self.hooks = []

    def __del__(self):
        """Cleanup hooks on deletion."""
        self.remove_hooks()


class CAVTrainer:
    """Train Concept Activation Vectors (CAVs).

    CAVs are learned by training a linear classifier to separate concept
    examples from random examples in the activation space. The classifier's
    normal vector becomes the CAV.

    Args:
        device: Device to run on
        seed: Random seed
    """

    def __init__(self, device: str = "cpu", seed: int = 42):
        """Initialize CAV trainer."""
        self.device = device
        self.seed = seed

    def train(
        self,
        concept_acts: torch.Tensor,
        random_acts: torch.Tensor,
        val_split: float = 0.2,
    ) -> Tuple[torch.Tensor, float, Dict]:
        """Train CAV using linear SVM.

        Args:
            concept_acts: Activations for concept examples (N, C)
            random_acts: Activations for random examples (M, C)
            val_split: Fraction of data for validation

        Returns:
            Tuple of (cav, accuracy, metrics)
                - cav: Concept activation vector (C,)
                - accuracy: Validation accuracy
                - metrics: Dictionary with training metrics
        """
        # Prepare data
        X_concept = concept_acts.cpu().numpy()
        X_random = random_acts.cpu().numpy()

        X = np.vstack([X_concept, X_random])
        y = np.concatenate([np.ones(len(X_concept)), np.zeros(len(X_random))])

        # Train/val split
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, random_state=self.seed, stratify=y
        )

        # Train linear SVM using SGD
        clf = SGDClassifier(
            loss="hinge",  # Linear SVM
            penalty="l2",
            alpha=0.01,
            max_iter=1000,
            tol=1e-3,
            random_state=self.seed,
        )
        clf.fit(X_train, y_train)

        # Extract CAV (normal vector to decision boundary)
        cav = torch.from_numpy(clf.coef_[0]).float()

        # Normalize to unit vector
        cav = cav / torch.norm(cav)

        # Compute metrics
        train_acc = clf.score(X_train, y_train)
        val_acc = clf.score(X_val, y_val)

        metrics = {
            "train_accuracy": train_acc,
            "val_accuracy": val_acc,
            "n_concept": len(X_concept),
            "n_random": len(X_random),
        }

        return cav, val_acc, metrics


class TCAV:
    """Testing with Concept Activation Vectors.

    Main class implementing the TCAV methodology for concept-based neural
    network interpretability.

    Args:
        config: TCAV configuration
    """

    def __init__(self, config: TCAVConfig):
        """Initialize TCAV."""
        self.config = config
        self.model = config.model.to(config.device)
        self.model.eval()

        # Initialize components
        self.extractor = ActivationExtractor(
            self.model, config.target_layers, config.device
        )
        self.trainer = CAVTrainer(config.device, config.seed)

        # Storage for CAVs and metrics
        self.cavs: Dict[str, Dict[str, torch.Tensor]] = {}
        self.cav_metrics: Dict[str, Dict[str, Dict]] = {}

        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"TCAV(target_layers={self.config.target_layers}, "
            f"num_cavs={sum(len(v) for v in self.cavs.values())})"
        )

    def load_concept_data(
        self, concept_name: str, transform: Optional[transforms.Compose] = None
    ) -> Tuple[torch.Tensor, int]:
        """Load concept data from directory.

        Args:
            concept_name: Name of concept subdirectory
            transform: Optional transforms

        Returns:
            Tuple of (images, num_images)
        """
        concept_dir = self.config.concept_data_dir / concept_name

        if not concept_dir.exists():
            raise ValueError(f"Concept directory not found: {concept_dir}")

        # Load images
        image_files = list(concept_dir.glob("*.png")) + list(concept_dir.glob("*.jpg"))

        if len(image_files) == 0:
            raise ValueError(f"No images found in {concept_dir}")

        images = []
        default_transform = transforms.Compose(
            [
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
            ]
        )

        for img_path in tqdm(
            image_files,
            desc=f"Loading {concept_name}",
            disable=self.config.verbose < 1,
        ):
            img = Image.open(img_path).convert("RGB")
            if transform:
                img = transform(img)
            else:
                img = default_transform(img)
            images.append(img)

        images = torch.stack(images)

        logger.info(f"Loaded {len(images)} images for concept '{concept_name}'")
        return images, len(images)

    def train_cav(
        self,
        concept: str,
        layer: str,
        random_concept: str,
        save: bool = True,
    ) -> Tuple[torch.Tensor, Dict]:
        """Train CAV for a concept at specific layer.

        Args:
            concept: Concept name
            layer: Target layer name
            random_concept: Random concept for negative examples
            save: Whether to save trained CAV

        Returns:
            Tuple of (cav, metrics)
        """
        # Check if already trained
        cav_path = self.config.cav_dir / f"{concept}_{layer}.pt"
        if cav_path.exists():
            logger.info(f"Loading existing CAV from {cav_path}")
            state = torch.load(cav_path)
            cav = state["cav"]
            metrics = state["metrics"]

            # Store in memory
            if layer not in self.cavs:
                self.cavs[layer] = {}
                self.cav_metrics[layer] = {}
            self.cavs[layer][concept] = cav
            self.cav_metrics[layer][concept] = metrics

            return cav, metrics

        # Load concept data
        concept_imgs, _ = self.load_concept_data(concept)
        random_imgs, _ = self.load_concept_data(random_concept)

        # Create dataloaders
        concept_dataset = ConceptDataset(
            concept_imgs,
            torch.ones(len(concept_imgs)),
        )
        random_dataset = ConceptDataset(
            random_imgs,
            torch.zeros(len(random_imgs)),
        )

        concept_loader = DataLoader(
            concept_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )
        random_loader = DataLoader(
            random_dataset,
            batch_size=self.config.batch_size,
            shuffle=False,
        )

        # Extract activations
        logger.info(f"Extracting activations for {concept} at {layer}")
        concept_acts = self.extractor.extract(concept_loader, layer)
        random_acts = self.extractor.extract(random_loader, layer)

        # Train CAV
        logger.info(f"Training CAV for {concept} at {layer}")
        cav, accuracy, metrics = self.trainer.train(concept_acts, random_acts)

        # Validate accuracy
        if accuracy < self.config.min_cav_accuracy:
            logger.warning(
                f"CAV accuracy ({accuracy:.3f}) below threshold "
                f"({self.config.min_cav_accuracy})"
            )

        # Store
        if layer not in self.cavs:
            self.cavs[layer] = {}
            self.cav_metrics[layer] = {}

        self.cavs[layer][concept] = cav
        self.cav_metrics[layer][concept] = metrics

        # Save to disk
        if save:
            torch.save({"cav": cav, "metrics": metrics}, cav_path)
            logger.info(f"Saved CAV to {cav_path}")

        return cav, metrics

    def compute_tcav_score(
        self,
        inputs: torch.Tensor,
        target_class: int,
        concept: str,
        layer: str,
    ) -> float:
        """Compute TCAV score for concept.

        TCAV score measures the percentage of samples where the gradient
        of the target class logit w.r.t. the layer activations has a
        positive directional derivative in the CAV direction.

        Args:
            inputs: Input images (N, C, H, W)
            target_class: Target class index
            concept: Concept name
            layer: Layer name

        Returns:
            TCAV score (percentage in [0, 100])
        """
        # Check CAV exists
        if layer not in self.cavs or concept not in self.cavs[layer]:
            raise ValueError(
                f"CAV not trained for concept '{concept}' at layer '{layer}'"
            )

        cav = self.cavs[layer][concept].to(self.config.device)

        # Set model to train mode temporarily to enable gradients
        was_training = self.model.training
        self.model.train()

        inputs = inputs.to(self.config.device)

        #  Store activation gradients
        activation_grads = []

        def backward_hook(module, grad_input, grad_output):
            """Hook to capture gradients flowing through the layer."""
            # grad_output[0] contains gradients w.r.t. layer outputs
            if grad_output[0].dim() == 4:  # (B, C, H, W)
                grad = grad_output[0].mean(dim=[2, 3])  # Apply GAP
            else:
                grad = grad_output[0]
            activation_grads.append(grad.detach())

        # Register backward hook
        target_module = None
        for name, module in self.model.named_modules():
            if name == layer:
                target_module = module
                break

        if target_module is None:
            raise ValueError(f"Layer {layer} not found in model")

        hook_handle = target_module.register_full_backward_hook(backward_hook)

        # Forward + backward for each sample
        for i in range(len(inputs)):
            sample = inputs[i : i + 1]
            sample.requires_grad_(True)

            outputs = self.model(sample)
            target_logit = outputs[0, target_class]

            # Backward to capture gradient
            self.model.zero_grad()
            target_logit.backward()

        # Remove hook
        hook_handle.remove()

        # Stack gradients
        gradients = torch.stack(activation_grads)

        # Restore model mode
        self.model.train(was_training)

        # Compute directional derivatives: grad · cav
        directional_derivs = torch.matmul(gradients, cav)

        # Count positive derivatives
        num_positive = (directional_derivs > 0).sum().item()
        tcav_score = 100.0 * num_positive / len(inputs)

        return tcav_score

    def compute_multilayer_tcav(
        self,
        inputs: torch.Tensor,
        target_class: int,
        concept: str,
    ) -> Dict[str, float]:
        """Compute TCAV scores across multiple layers.

        Args:
            inputs: Input images (N, C, H, W)
            target_class: Target class index
            concept: Concept name

        Returns:
            Dictionary mapping layer names to TCAV scores
        """
        scores = {}

        for layer in self.config.target_layers:
            # Skip if CAV not trained
            if layer not in self.cavs or concept not in self.cavs[layer]:
                logger.warning(f"Skipping layer {layer} - CAV not trained")
                continue

            score = self.compute_tcav_score(inputs, target_class, concept, layer)
            scores[layer] = score

        return scores

    def precompute_all_cavs(self, concepts: List[str]):
        """Precompute CAVs for all concepts and layers.

        Args:
            concepts: List of concept names to train CAVs for
        """
        logger.info(f"Precomputing CAVs for {len(concepts)} concepts")

        # Generate random concepts
        random_concepts = [
            f"random_{i}" for i in range(self.config.num_random_concepts)
        ]

        total = len(concepts) * len(self.config.target_layers) * len(random_concepts)
        pbar = tqdm(total=total, desc="Training CAVs", disable=self.config.verbose < 1)

        for concept in concepts:
            for layer in self.config.target_layers:
                for random_concept in random_concepts:
                    try:
                        self.train_cav(concept, layer, random_concept, save=True)
                    except Exception as e:
                        logger.error(
                            f"Failed to train CAV for {concept} at {layer}: {e}"
                        )

                    pbar.update(1)

        pbar.close()
        logger.info(f"Precomputed {sum(len(v) for v in self.cavs.values())} CAVs")

    def save_state(self, path: Union[str, Path]):
        """Save TCAV state (all CAVs and metrics).

        Note: Model is not saved to avoid pickling issues.

        Args:
            path: Path to save state
        """
        path = Path(path)
        torch.save(
            {
                "cavs": self.cavs,
                "cav_metrics": self.cav_metrics,
                "target_layers": self.config.target_layers,
                "min_cav_accuracy": self.config.min_cav_accuracy,
            },
            path,
        )

        logger.info(f"TCAV state saved to {path}")

    def load_state(self, path: Union[str, Path]):
        """Load TCAV state from disk.

        Args:
            path: Path to load from
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"State file not found: {path}")

        state = torch.load(path)
        self.cavs = state["cavs"]
        self.cav_metrics = state["cav_metrics"]

        logger.info(f"Loaded TCAV state from {path}")
        logger.info(f"Loaded {sum(len(v) for v in self.cavs.values())} CAVs")


def create_tcav(
    model: nn.Module,
    target_layers: List[str],
    concept_data_dir: Union[str, Path],
    cav_dir: Union[str, Path],
    **kwargs,
) -> TCAV:
    """Factory function to create TCAV instance.

    Args:
        model: PyTorch model
        target_layers: List of layer names
        concept_data_dir: Path to concept datasets
        cav_dir: Path to save CAVs
        **kwargs: Additional config parameters

    Returns:
        TCAV instance
    """
    config = TCAVConfig(
        model=model,
        target_layers=target_layers,
        concept_data_dir=concept_data_dir,
        cav_dir=cav_dir,
        **kwargs,
    )

    return TCAV(config)
