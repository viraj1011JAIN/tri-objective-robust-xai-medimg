"""Representation Analysis for Domain Gap and Feature Similarity.

This module implements representation analysis techniques for measuring similarity
between neural network feature representations. Primary use case is domain gap
analysis between in-domain and cross-site medical imaging datasets.

Key Components:
    - RepresentationConfig: Configuration for representation analysis
    - CKAAnalyzer: Centered Kernel Alignment (CKA) implementation
    - SVCCAAnalyzer: Singular Vector Canonical Correlation Analysis (optional)
    - DomainGapAnalyzer: Domain gap measurement using CKA
    - Visualization utilities for representation similarity

Algorithms:
    1. Centered Kernel Alignment (CKA):
       - Linear CKA: Measures similarity using linear kernels
       - RBF CKA: Measures similarity using RBF (Gaussian) kernels
       - Invariant to orthogonal transformations and isotropic scaling

    2. SVCCA (Singular Vector CCA):
       - Dimensionality reduction via SVD
       - Canonical Correlation Analysis on reduced representations
       - Identifies shared directions of variation

Research Context:
    Representation analysis reveals domain gaps between training and deployment
    datasets in medical imaging. Large domain gaps (low CKA similarity) indicate
    distribution shift that may harm model performance and robustness.

    Expected Results:
    - In-domain similarity: CKA > 0.85 (high similarity)
    - Cross-site similarity: CKA < 0.60 (large domain gap)
    - Layer-wise analysis shows where domain gap emerges

References:
    - Kornblith et al. (2019). Similarity of Neural Network Representations Revisited.
      In ICML 2019.
    - Raghu et al. (2017). SVCCA: Singular Vector Canonical Correlation Analysis
      for Deep Learning Dynamics and Interpretability. NeurIPS 2017.

Example:
    >>> from src.xai.representation_analysis import create_cka_analyzer
    >>> import torch
    >>>
    >>> # Create CKA analyzer
    >>> analyzer = create_cka_analyzer(
    ...     model=my_resnet50,
    ...     layers=["layer2", "layer3", "layer4"]
    ... )
    >>>
    >>> # Extract features
    >>> in_domain_features = analyzer.extract_features(in_domain_loader, "layer3")
    >>> cross_site_features = analyzer.extract_features(cross_site_loader, "layer3")
    >>>
    >>> # Compute CKA similarity
    >>> similarity = analyzer.compute_cka(in_domain_features, cross_site_features)
    >>> print(f"CKA similarity: {similarity:.3f}")
    >>>
    >>> # Domain gap analysis
    >>> from src.xai.representation_analysis import DomainGapAnalyzer
    >>> gap_analyzer = DomainGapAnalyzer(model, layers=["layer3", "layer4"])
    >>> results = gap_analyzer.analyze_domain_gap(
    ...     source_loader=in_domain_loader,
    ...     target_loader=cross_site_loader
    ... )
    >>> print(f"Domain gap (layer3): {1 - results['layer3']:.3f}")
"""

import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from scipy.linalg import svd
from sklearn.cross_decomposition import CCA
from torch.utils.data import DataLoader
from tqdm import tqdm

logger = logging.getLogger(__name__)


@dataclass
class RepresentationConfig:
    """Configuration for representation analysis.

    Attributes:
        model: PyTorch model to analyze
        layers: List of layer names to extract features from
        kernel_type: Kernel type for CKA ('linear' or 'rbf')
        rbf_sigma: Sigma parameter for RBF kernel (if kernel_type='rbf')
        device: Device to run analysis on
        batch_size: Batch size for feature extraction
        num_workers: Number of workers for data loading
        verbose: Logging verbosity (0=silent, 1=info, 2=debug)
        seed: Random seed for reproducibility
    """

    model: nn.Module
    layers: List[str]
    kernel_type: str = "linear"
    rbf_sigma: Optional[float] = None
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    batch_size: int = 64
    num_workers: int = 4
    verbose: int = 1
    seed: int = 42

    def __post_init__(self):
        """Validate configuration."""
        if not self.layers:
            raise ValueError("layers cannot be empty")

        if self.kernel_type not in ["linear", "rbf"]:
            raise ValueError(
                f"kernel_type must be 'linear' or 'rbf', got {self.kernel_type}"
            )

        if self.kernel_type == "rbf" and self.rbf_sigma is None:
            logger.warning(
                "RBF kernel selected but rbf_sigma not specified. Using default sigma=1.0"
            )
            self.rbf_sigma = 1.0

        if self.batch_size <= 0:
            raise ValueError("batch_size must be positive")

        if self.num_workers < 0:
            raise ValueError("num_workers must be non-negative")


class CKAAnalyzer:
    """Centered Kernel Alignment (CKA) analyzer.

    Implements CKA for measuring similarity between feature representations.
    CKA is invariant to orthogonal transformations and isotropic scaling,
    making it suitable for comparing representations across different layers
    or models.

    Attributes:
        config: RepresentationConfig instance
        model: Neural network model
        hooks: Dictionary of forward hooks for feature extraction
        features: Dictionary storing extracted features
    """

    def __init__(self, config: RepresentationConfig):
        """Initialize CKA analyzer.

        Args:
            config: RepresentationConfig instance
        """
        self.config = config
        self.model = config.model.to(config.device)
        self.model.eval()
        self.hooks = {}
        self.features = {}

        # Set random seed
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)

        if config.verbose >= 1:
            logger.info(f"Initialized CKA analyzer with {len(config.layers)} layers")
            logger.info(f"Kernel type: {config.kernel_type}")

    def _register_hooks(self, layer_name: str):
        """Register forward hook for a layer.

        Args:
            layer_name: Name of layer to register hook for
        """

        def hook_fn(module, input, output):
            """Hook function to capture activations."""
            # Store output as numpy array
            self.features[layer_name] = output.detach().cpu().numpy()

        # Find module by name
        module = dict(self.model.named_modules())[layer_name]
        hook = module.register_forward_hook(hook_fn)
        self.hooks[layer_name] = hook

    def _remove_hooks(self):
        """Remove all registered hooks."""
        for hook in self.hooks.values():
            hook.remove()
        self.hooks.clear()

    def extract_features(self, dataloader: DataLoader, layer: str) -> np.ndarray:
        """Extract features from a specific layer.

        Args:
            dataloader: PyTorch DataLoader providing input data
            layer: Layer name to extract features from

        Returns:
            Extracted features as numpy array of shape (N, D)
            where N is number of samples and D is feature dimension
        """
        if layer not in self.config.layers:
            raise ValueError(
                f"Layer {layer} not in configured layers: {self.config.layers}"
            )

        self.features.clear()
        self._register_hooks(layer)

        all_features = []

        with torch.no_grad():
            for batch in tqdm(
                dataloader, desc=f"Extracting {layer}", disable=self.config.verbose == 0
            ):
                if isinstance(batch, (tuple, list)):
                    inputs = batch[0]
                else:
                    inputs = batch

                inputs = inputs.to(self.config.device)
                _ = self.model(inputs)

                # Get features and flatten
                feats = self.features[layer]
                # Shape: (batch, channels, height, width) -> (batch, features)
                feats = feats.reshape(feats.shape[0], -1)
                all_features.append(feats)

        self._remove_hooks()

        # Concatenate all batches
        features = np.concatenate(all_features, axis=0)

        if self.config.verbose >= 1:
            logger.info(f"Extracted features from {layer}: shape {features.shape}")

        return features

    @staticmethod
    def _centering_matrix(n: int) -> np.ndarray:
        """Compute centering matrix H = I - (1/n)11^T.

        Args:
            n: Matrix dimension

        Returns:
            Centering matrix of shape (n, n)
        """
        return np.eye(n) - np.ones((n, n)) / n

    @staticmethod
    def _linear_kernel(X: np.ndarray) -> np.ndarray:
        """Compute linear kernel K = XX^T.

        Args:
            X: Feature matrix of shape (n, d)

        Returns:
            Kernel matrix of shape (n, n)
        """
        return X @ X.T

    def _rbf_kernel(self, X: np.ndarray, sigma: Optional[float] = None) -> np.ndarray:
        """Compute RBF (Gaussian) kernel.

        Args:
            X: Feature matrix of shape (n, d)
            sigma: Kernel bandwidth (uses config.rbf_sigma if None)

        Returns:
            Kernel matrix of shape (n, n)
        """
        if sigma is None:
            sigma = self.config.rbf_sigma

        if sigma is None:
            sigma = 1.0  # Default fallback

        # Compute pairwise squared distances
        n = X.shape[0]
        K = np.zeros((n, n))

        for i in range(n):
            dists = np.sum((X - X[i : i + 1]) ** 2, axis=1)
            K[i] = np.exp(-dists / (2 * sigma**2))

        return K

    def _compute_kernel(self, X: np.ndarray) -> np.ndarray:
        """Compute kernel matrix based on configured kernel type.

        Args:
            X: Feature matrix of shape (n, d)

        Returns:
            Kernel matrix of shape (n, n)
        """
        if self.config.kernel_type == "linear":
            return self._linear_kernel(X)
        else:  # rbf
            return self._rbf_kernel(X)

    def compute_cka(
        self,
        features_x: np.ndarray,
        features_y: np.ndarray,
        kernel_type: Optional[str] = None,
    ) -> float:
        """Compute Centered Kernel Alignment (CKA) between two feature sets.

        CKA measures similarity between representations X and Y:
            CKA(X, Y) = HSIC(X, Y) / sqrt(HSIC(X, X) * HSIC(Y, Y))

        where HSIC is the Hilbert-Schmidt Independence Criterion.

        Args:
            features_x: First feature matrix of shape (n, d1)
            features_y: Second feature matrix of shape (n, d2)
            kernel_type: Kernel type override ('linear' or 'rbf')

        Returns:
            CKA similarity score in [0, 1]
        """
        n = features_x.shape[0]
        if features_y.shape[0] != n:
            raise ValueError(
                f"Feature matrices must have same number of samples: "
                f"{features_x.shape[0]} vs {features_y.shape[0]}"
            )

        # Temporarily override kernel type if specified
        original_kernel = self.config.kernel_type
        if kernel_type is not None:
            self.config.kernel_type = kernel_type

        # Compute kernel matrices
        K = self._compute_kernel(features_x)
        L = self._compute_kernel(features_y)

        # Restore original kernel type
        self.config.kernel_type = original_kernel

        # Center kernels: HKH and HLH
        H = self._centering_matrix(n)
        K_centered = H @ K @ H
        L_centered = H @ L @ H

        # Compute HSIC terms
        hsic_xy = np.trace(K_centered @ L_centered)
        hsic_xx = np.trace(K_centered @ K_centered)
        hsic_yy = np.trace(L_centered @ L_centered)

        # CKA = HSIC(X,Y) / sqrt(HSIC(X,X) * HSIC(Y,Y))
        cka = hsic_xy / np.sqrt(hsic_xx * hsic_yy + 1e-10)

        if self.config.verbose >= 2:
            logger.debug(
                f"CKA computation: HSIC(X,Y)={hsic_xy:.4f}, "
                f"HSIC(X,X)={hsic_xx:.4f}, HSIC(Y,Y)={hsic_yy:.4f}"
            )

        return float(cka)

    def compute_cka_matrix(self, features_list: List[np.ndarray]) -> np.ndarray:
        """Compute pairwise CKA similarity matrix.

        Args:
            features_list: List of feature matrices

        Returns:
            CKA similarity matrix of shape (len(features_list), len(features_list))
        """
        n_features = len(features_list)
        cka_matrix = np.zeros((n_features, n_features))

        for i in range(n_features):
            for j in range(i, n_features):
                cka = self.compute_cka(features_list[i], features_list[j])
                cka_matrix[i, j] = cka
                cka_matrix[j, i] = cka

        return cka_matrix

    def __repr__(self) -> str:
        """String representation."""
        return (
            f"CKAAnalyzer(layers={len(self.config.layers)}, "
            f"kernel={self.config.kernel_type})"
        )


class SVCCAAnalyzer:
    """Singular Vector Canonical Correlation Analysis (SVCCA) analyzer.

    Implements SVCCA for measuring similarity between neural network
    representations. SVCCA first performs SVD for dimensionality reduction,
    then applies CCA to find maximally correlated subspaces.

    Attributes:
        config: RepresentationConfig instance
        model: Neural network model
        threshold: Variance threshold for SVD (default 0.99)
    """

    def __init__(self, config: RepresentationConfig, threshold: float = 0.99):
        """Initialize SVCCA analyzer.

        Args:
            config: RepresentationConfig instance
            threshold: Cumulative variance threshold for SVD (0-1)
        """
        self.config = config
        self.model = config.model.to(config.device)
        self.model.eval()
        self.threshold = threshold

        if not 0 < threshold <= 1:
            raise ValueError(f"threshold must be in (0, 1], got {threshold}")

        # Use CKA analyzer for feature extraction
        self.cka_analyzer = CKAAnalyzer(config)

        if config.verbose >= 1:
            logger.info(f"Initialized SVCCA analyzer with threshold={threshold}")

    def extract_features(self, dataloader: DataLoader, layer: str) -> np.ndarray:
        """Extract features using CKA analyzer.

        Args:
            dataloader: PyTorch DataLoader
            layer: Layer name

        Returns:
            Extracted features
        """
        return self.cka_analyzer.extract_features(dataloader, layer)

    def _perform_svd(self, X: np.ndarray) -> np.ndarray:
        """Perform SVD and select components based on variance threshold.

        Args:
            X: Feature matrix of shape (n, d)

        Returns:
            Reduced features of shape (n, k) where k <= d
        """
        # Center features
        X_centered = X - X.mean(axis=0, keepdims=True)

        # SVD: X = U * S * Vt
        U, S, Vt = svd(X_centered, full_matrices=False)

        # Compute explained variance
        explained_variance = (S**2) / (X.shape[0] - 1)
        total_variance = explained_variance.sum()
        explained_variance_ratio = explained_variance / total_variance

        # Select components based on threshold
        cumsum = np.cumsum(explained_variance_ratio)
        n_components = np.searchsorted(cumsum, self.threshold) + 1

        if self.config.verbose >= 2:
            logger.debug(
                f"SVD: selected {n_components}/{len(S)} components "
                f"explaining {cumsum[n_components-1]:.2%} variance"
            )

        # Project to reduced space
        X_reduced = U[:, :n_components] * S[:n_components]

        return X_reduced

    def compute_svcca(
        self, features_x: np.ndarray, features_y: np.ndarray
    ) -> Tuple[float, np.ndarray]:
        """Compute SVCCA similarity between two feature sets.

        SVCCA pipeline:
        1. Apply SVD to both feature sets for dimensionality reduction
        2. Apply CCA to find maximally correlated directions
        3. Average canonical correlations as similarity score

        Args:
            features_x: First feature matrix of shape (n, d1)
            features_y: Second feature matrix of shape (n, d2)

        Returns:
            Tuple of (mean_correlation, canonical_correlations)
        """
        n = features_x.shape[0]
        if features_y.shape[0] != n:
            raise ValueError(
                f"Feature matrices must have same number of samples: "
                f"{features_x.shape[0]} vs {features_y.shape[0]}"
            )

        # Step 1: SVD dimensionality reduction
        X_reduced = self._perform_svd(features_x)
        Y_reduced = self._perform_svd(features_y)

        # Step 2: CCA using sklearn
        n_components = min(X_reduced.shape[1], Y_reduced.shape[1])
        cca = CCA(n_components=n_components, max_iter=1000)

        try:
            cca.fit(X_reduced, Y_reduced)
            X_c, Y_c = cca.transform(X_reduced, Y_reduced)

            # Compute correlations
            correlations = np.array(
                [np.corrcoef(X_c[:, i], Y_c[:, i])[0, 1] for i in range(n_components)]
            )
            # Clamp to [0, 1] for numerical stability
            correlations = np.clip(np.abs(correlations), 0, 1)
        except Exception as e:
            logger.warning(f"CCA computation failed: {e}")
            correlations = np.zeros(n_components)

        # Step 3: Mean correlation
        mean_corr = float(np.mean(correlations))

        if self.config.verbose >= 1:
            logger.info(f"SVCCA: mean correlation = {mean_corr:.4f}")

        return mean_corr, correlations

    def __repr__(self) -> str:
        """String representation."""
        return f"SVCCAAnalyzer(layers={len(self.config.layers)}, threshold={self.threshold})"


class DomainGapAnalyzer:
    """Analyzer for measuring domain gap between source and target datasets.

    Uses CKA to measure similarity between feature representations extracted
    from in-domain (source) and cross-site (target) datasets. Lower similarity
    indicates larger domain gap.

    Attributes:
        cka_analyzer: CKAAnalyzer instance
        layers: List of layers to analyze
        results: Dictionary storing analysis results
    """

    def __init__(
        self,
        model: nn.Module,
        layers: List[str],
        kernel_type: str = "linear",
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
        verbose: int = 1,
    ):
        """Initialize domain gap analyzer.

        Args:
            model: PyTorch model to analyze
            layers: List of layer names to analyze
            kernel_type: Kernel type for CKA ('linear' or 'rbf')
            device: Device to run on
            verbose: Logging verbosity
        """
        config = RepresentationConfig(
            model=model,
            layers=layers,
            kernel_type=kernel_type,
            device=device,
            verbose=verbose,
        )
        self.cka_analyzer = CKAAnalyzer(config)
        self.layers = layers
        self.results = {}

        if verbose >= 1:
            logger.info(f"Initialized DomainGapAnalyzer for {len(layers)} layers")

    def analyze_domain_gap(
        self,
        source_loader: DataLoader,
        target_loader: DataLoader,
        layers: Optional[List[str]] = None,
    ) -> Dict[str, float]:
        """Analyze domain gap between source and target datasets.

        Args:
            source_loader: DataLoader for source (in-domain) data
            target_loader: DataLoader for target (cross-site) data
            layers: Subset of layers to analyze (uses all if None)

        Returns:
            Dictionary mapping layer names to CKA similarity scores
            Lower scores indicate larger domain gaps
        """
        if layers is None:
            layers = self.layers

        results = {}

        for layer in layers:
            if self.cka_analyzer.config.verbose >= 1:
                logger.info(f"Analyzing domain gap for {layer}...")

            # Extract features
            source_features = self.cka_analyzer.extract_features(source_loader, layer)
            target_features = self.cka_analyzer.extract_features(target_loader, layer)

            # Compute CKA similarity
            similarity = self.cka_analyzer.compute_cka(source_features, target_features)
            results[layer] = similarity

            # Compute domain gap (1 - similarity)
            gap = 1.0 - similarity

            if self.cka_analyzer.config.verbose >= 1:
                logger.info(f"{layer}: CKA={similarity:.3f}, Gap={gap:.3f}")

        self.results = results
        return results

    def visualize_domain_gap(
        self,
        results: Optional[Dict[str, float]] = None,
        save_path: Optional[Union[str, Path]] = None,
        figsize: Tuple[int, int] = (10, 6),
    ) -> plt.Figure:
        """Visualize domain gap across layers.

        Args:
            results: Analysis results (uses self.results if None)
            save_path: Path to save figure
            figsize: Figure size

        Returns:
            Matplotlib figure
        """
        if results is None:
            if not self.results:
                raise ValueError("No results available. Run analyze_domain_gap first.")
            results = self.results

        # Create figure
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)

        layers = list(results.keys())
        similarities = [results[layer] for layer in layers]
        gaps = [1.0 - s for s in similarities]

        # Plot 1: CKA similarity
        colors_sim = [
            "green" if s > 0.7 else "orange" if s > 0.5 else "red" for s in similarities
        ]
        bars1 = ax1.bar(range(len(layers)), similarities, color=colors_sim, alpha=0.7)
        ax1.set_xticks(range(len(layers)))
        ax1.set_xticklabels(layers, rotation=45, ha="right")
        ax1.set_ylabel("CKA Similarity", fontsize=12)
        ax1.set_title(
            "Domain Similarity (Higher = Better)", fontsize=14, fontweight="bold"
        )
        ax1.axhline(
            y=0.7, color="green", linestyle="--", alpha=0.5, label="Good (>0.7)"
        )
        ax1.axhline(
            y=0.5, color="orange", linestyle="--", alpha=0.5, label="Moderate (>0.5)"
        )
        ax1.set_ylim(0, 1.0)
        ax1.legend()
        ax1.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, sim in zip(bars1, similarities):
            height = bar.get_height()
            ax1.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{sim:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        # Plot 2: Domain gap
        colors_gap = [
            "red" if g > 0.5 else "orange" if g > 0.3 else "green" for g in gaps
        ]
        bars2 = ax2.bar(range(len(layers)), gaps, color=colors_gap, alpha=0.7)
        ax2.set_xticks(range(len(layers)))
        ax2.set_xticklabels(layers, rotation=45, ha="right")
        ax2.set_ylabel("Domain Gap", fontsize=12)
        ax2.set_title("Domain Gap (Lower = Better)", fontsize=14, fontweight="bold")
        ax2.axhline(
            y=0.3, color="orange", linestyle="--", alpha=0.5, label="Moderate (<0.3)"
        )
        ax2.axhline(y=0.5, color="red", linestyle="--", alpha=0.5, label="Large (>0.5)")
        ax2.set_ylim(0, 1.0)
        ax2.legend()
        ax2.grid(axis="y", alpha=0.3)

        # Add value labels
        for bar, gap in zip(bars2, gaps):
            height = bar.get_height()
            ax2.text(
                bar.get_x() + bar.get_width() / 2.0,
                height,
                f"{gap:.3f}",
                ha="center",
                va="bottom",
                fontsize=10,
            )

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches="tight")
            if self.cka_analyzer.config.verbose >= 1:
                logger.info(f"Saved domain gap visualization to {save_path}")

        return fig

    def compute_summary_statistics(
        self, results: Optional[Dict[str, float]] = None
    ) -> Dict[str, float]:
        """Compute summary statistics for domain gap analysis.

        Args:
            results: Analysis results (uses self.results if None)

        Returns:
            Dictionary with summary statistics
        """
        if results is None:
            if not self.results:
                raise ValueError("No results available. Run analyze_domain_gap first.")
            results = self.results

        similarities = list(results.values())
        gaps = [1.0 - s for s in similarities]

        summary = {
            "mean_similarity": float(np.mean(similarities)),
            "std_similarity": float(np.std(similarities)),
            "min_similarity": float(np.min(similarities)),
            "max_similarity": float(np.max(similarities)),
            "mean_gap": float(np.mean(gaps)),
            "std_gap": float(np.std(gaps)),
            "max_gap": float(np.max(gaps)),
            "max_gap_layer": list(results.keys())[np.argmax(gaps)],
        }

        return summary

    def __repr__(self) -> str:
        """String representation."""
        return f"DomainGapAnalyzer(layers={len(self.layers)})"


def create_cka_analyzer(
    model: nn.Module,
    layers: List[str],
    kernel_type: str = "linear",
    rbf_sigma: Optional[float] = None,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: int = 1,
) -> CKAAnalyzer:
    """Factory function to create CKA analyzer.

    Args:
        model: PyTorch model to analyze
        layers: List of layer names to extract features from
        kernel_type: Kernel type ('linear' or 'rbf')
        rbf_sigma: Sigma parameter for RBF kernel
        device: Device to run on
        verbose: Logging verbosity

    Returns:
        Configured CKAAnalyzer instance
    """
    config = RepresentationConfig(
        model=model,
        layers=layers,
        kernel_type=kernel_type,
        rbf_sigma=rbf_sigma,
        device=device,
        verbose=verbose,
    )
    return CKAAnalyzer(config)


def create_svcca_analyzer(
    model: nn.Module,
    layers: List[str],
    threshold: float = 0.99,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: int = 1,
) -> SVCCAAnalyzer:
    """Factory function to create SVCCA analyzer.

    Args:
        model: PyTorch model to analyze
        layers: List of layer names to extract features from
        threshold: Variance threshold for SVD (0-1)
        device: Device to run on
        verbose: Logging verbosity

    Returns:
        Configured SVCCAAnalyzer instance
    """
    config = RepresentationConfig(
        model=model, layers=layers, device=device, verbose=verbose
    )
    return SVCCAAnalyzer(config, threshold=threshold)


def create_domain_gap_analyzer(
    model: nn.Module,
    layers: List[str],
    kernel_type: str = "linear",
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    verbose: int = 1,
) -> DomainGapAnalyzer:
    """Factory function to create domain gap analyzer.

    Args:
        model: PyTorch model to analyze
        layers: List of layer names to analyze
        kernel_type: Kernel type for CKA
        device: Device to run on
        verbose: Logging verbosity

    Returns:
        Configured DomainGapAnalyzer instance
    """
    return DomainGapAnalyzer(
        model=model,
        layers=layers,
        kernel_type=kernel_type,
        device=device,
        verbose=verbose,
    )
