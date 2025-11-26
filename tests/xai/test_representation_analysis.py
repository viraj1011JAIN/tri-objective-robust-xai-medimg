"""Tests for representation analysis module."""

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from torch.utils.data import DataLoader, TensorDataset  # noqa: E402

from src.xai.representation_analysis import (  # noqa: E402
    CKAAnalyzer,
    DomainGapAnalyzer,
    RepresentationConfig,
    SVCCAAnalyzer,
    create_cka_analyzer,
    create_domain_gap_analyzer,
    create_svcca_analyzer,
)


# Simple model for testing
class SimpleModel(nn.Module):
    """Simple CNN for testing."""

    def __init__(self):
        """Initialize model."""
        super().__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.fc = nn.Linear(64 * 28 * 28, 10)

    def forward(self, x):
        """Forward pass."""
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


@pytest.fixture
def simple_model():
    """Create simple model for testing."""
    return SimpleModel()


@pytest.fixture
def mock_dataloader():
    """Create mock dataloader."""
    data = torch.randn(32, 3, 224, 224)
    labels = torch.randint(0, 10, (32,))
    dataset = TensorDataset(data, labels)
    return DataLoader(dataset, batch_size=8, shuffle=False)


@pytest.fixture
def representation_config(simple_model):
    """Create representation config for testing."""
    return RepresentationConfig(
        model=simple_model,
        layers=["layer1", "layer2", "layer3"],
        kernel_type="linear",
        batch_size=8,
        verbose=0,
    )


class TestRepresentationConfig:
    """Test RepresentationConfig."""

    def test_valid_config(self, simple_model):
        """Test valid configuration."""
        config = RepresentationConfig(
            model=simple_model,
            layers=["layer1", "layer2"],
            kernel_type="linear",
        )

        assert config.batch_size == 64
        assert config.num_workers == 4
        assert config.verbose == 1
        assert config.kernel_type == "linear"

    def test_invalid_empty_layers(self, simple_model):
        """Test empty layers error."""
        with pytest.raises(ValueError, match="layers cannot be empty"):
            RepresentationConfig(
                model=simple_model,
                layers=[],
            )

    def test_invalid_kernel_type(self, simple_model):
        """Test invalid kernel type."""
        with pytest.raises(ValueError, match="kernel_type must be"):
            RepresentationConfig(
                model=simple_model,
                layers=["layer1"],
                kernel_type="invalid",
            )

    def test_rbf_sigma_warning(self, simple_model, caplog):
        """Test RBF sigma warning."""
        config = RepresentationConfig(
            model=simple_model,
            layers=["layer1"],
            kernel_type="rbf",
            rbf_sigma=None,
        )
        assert config.rbf_sigma == 1.0

    def test_invalid_batch_size(self, simple_model):
        """Test invalid batch size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            RepresentationConfig(
                model=simple_model,
                layers=["layer1"],
                batch_size=0,
            )

    def test_invalid_num_workers(self, simple_model):
        """Test invalid num_workers."""
        with pytest.raises(ValueError, match="num_workers must be non-negative"):
            RepresentationConfig(
                model=simple_model,
                layers=["layer1"],
                num_workers=-1,
            )


class TestCKAAnalyzer:
    """Test CKAAnalyzer."""

    def test_analyzer_initialization(self, representation_config):
        """Test analyzer initialization."""
        analyzer = CKAAnalyzer(representation_config)

        assert analyzer.model is not None
        assert len(analyzer.config.layers) == 3
        assert analyzer.config.kernel_type == "linear"

    def test_analyzer_repr(self, representation_config):
        """Test string representation."""
        analyzer = CKAAnalyzer(representation_config)
        repr_str = repr(analyzer)

        assert "CKAAnalyzer" in repr_str
        assert "layers=3" in repr_str
        assert "kernel=linear" in repr_str

    def test_extract_features(self, representation_config, mock_dataloader):
        """Test feature extraction."""
        analyzer = CKAAnalyzer(representation_config)
        features = analyzer.extract_features(mock_dataloader, "layer2")

        assert features.shape[0] == 32  # Number of samples
        assert len(features.shape) == 2  # (N, D)
        assert features.shape[1] > 0  # Has features

    def test_extract_features_invalid_layer(
        self, representation_config, mock_dataloader
    ):
        """Test feature extraction with invalid layer."""
        analyzer = CKAAnalyzer(representation_config)

        with pytest.raises(ValueError, match="Layer .* not in configured layers"):
            analyzer.extract_features(mock_dataloader, "invalid_layer")

    def test_centering_matrix(self):
        """Test centering matrix computation."""
        H = CKAAnalyzer._centering_matrix(5)

        assert H.shape == (5, 5)
        # H should be symmetric
        assert np.allclose(H, H.T)
        # H @ 1 = 0 (centering property)
        assert np.allclose(H @ np.ones(5), 0)

    def test_linear_kernel(self):
        """Test linear kernel computation."""
        X = np.random.randn(10, 5)
        K = CKAAnalyzer._linear_kernel(X)

        assert K.shape == (10, 10)
        # Kernel should be symmetric
        assert np.allclose(K, K.T)
        # Kernel should be PSD (all eigenvalues >= 0)
        eigvals = np.linalg.eigvalsh(K)
        assert np.all(eigvals >= -1e-10)

    def test_rbf_kernel(self, representation_config):
        """Test RBF kernel computation."""
        config = RepresentationConfig(
            model=representation_config.model,
            layers=["layer1"],
            kernel_type="rbf",
            rbf_sigma=1.0,
            verbose=0,
        )
        analyzer = CKAAnalyzer(config)

        X = np.random.randn(10, 5)
        K = analyzer._rbf_kernel(X, sigma=1.0)

        assert K.shape == (10, 10)
        # Kernel should be symmetric
        assert np.allclose(K, K.T)
        # Diagonal should be 1 (self-similarity)
        assert np.allclose(np.diag(K), 1.0)
        # All values should be in [0, 1]
        assert np.all(K >= 0) and np.all(K <= 1)

    def test_compute_cka_identical(self, representation_config, mock_dataloader):
        """Test CKA for identical features."""
        analyzer = CKAAnalyzer(representation_config)
        features = analyzer.extract_features(mock_dataloader, "layer2")

        cka = analyzer.compute_cka(features, features)

        # CKA of identical features should be 1
        assert np.isclose(cka, 1.0, atol=1e-4)

    def test_compute_cka_different(self, representation_config, mock_dataloader):
        """Test CKA for different features."""
        analyzer = CKAAnalyzer(representation_config)
        features1 = analyzer.extract_features(mock_dataloader, "layer1")
        features2 = analyzer.extract_features(mock_dataloader, "layer3")

        cka = analyzer.compute_cka(features1, features2)

        # CKA should be in [0, 1]
        assert 0 <= cka <= 1

    def test_compute_cka_unequal_samples(self, representation_config):
        """Test CKA with unequal number of samples."""
        analyzer = CKAAnalyzer(representation_config)

        features1 = np.random.randn(20, 10)
        features2 = np.random.randn(30, 10)

        with pytest.raises(ValueError, match="must have same number of samples"):
            analyzer.compute_cka(features1, features2)

    def test_compute_cka_rbf_kernel(self, representation_config):
        """Test CKA with RBF kernel."""
        config = RepresentationConfig(
            model=representation_config.model,
            layers=["layer1"],
            kernel_type="rbf",
            rbf_sigma=1.0,
            verbose=0,
        )
        analyzer = CKAAnalyzer(config)

        features1 = np.random.randn(20, 10)
        features2 = np.random.randn(20, 15)

        cka = analyzer.compute_cka(features1, features2)

        assert 0 <= cka <= 1

    def test_compute_cka_matrix(self, representation_config):
        """Test pairwise CKA matrix computation."""
        analyzer = CKAAnalyzer(representation_config)

        features_list = [
            np.random.randn(20, 10),
            np.random.randn(20, 15),
            np.random.randn(20, 20),
        ]

        cka_matrix = analyzer.compute_cka_matrix(features_list)

        assert cka_matrix.shape == (3, 3)
        # Diagonal should be 1
        assert np.allclose(np.diag(cka_matrix), 1.0, atol=1e-4)
        # Matrix should be symmetric
        assert np.allclose(cka_matrix, cka_matrix.T)
        # All values in [0, 1]
        assert np.all(cka_matrix >= 0) and np.all(cka_matrix <= 1)

    def test_kernel_type_override(self, representation_config):
        """Test kernel type override in compute_cka."""
        analyzer = CKAAnalyzer(representation_config)

        features1 = np.random.randn(20, 10)
        features2 = np.random.randn(20, 10)

        # Original kernel is linear
        cka_linear = analyzer.compute_cka(features1, features2, kernel_type="linear")

        # Temporarily override to RBF
        cka_rbf = analyzer.compute_cka(features1, features2, kernel_type="rbf")

        # Kernel should be restored to linear
        assert analyzer.config.kernel_type == "linear"

        # Results should differ
        assert not np.isclose(cka_linear, cka_rbf)


class TestSVCCAAnalyzer:
    """Test SVCCAAnalyzer."""

    def test_analyzer_initialization(self, representation_config):
        """Test SVCCA analyzer initialization."""
        analyzer = SVCCAAnalyzer(representation_config, threshold=0.95)

        assert analyzer.threshold == 0.95
        assert analyzer.cka_analyzer is not None

    def test_invalid_threshold(self, representation_config):
        """Test invalid threshold."""
        with pytest.raises(ValueError, match="threshold must be in"):
            SVCCAAnalyzer(representation_config, threshold=1.5)

    def test_analyzer_repr(self, representation_config):
        """Test string representation."""
        analyzer = SVCCAAnalyzer(representation_config, threshold=0.95)
        repr_str = repr(analyzer)

        assert "SVCCAAnalyzer" in repr_str
        assert "threshold=0.95" in repr_str

    def test_extract_features(self, representation_config, mock_dataloader):
        """Test feature extraction."""
        analyzer = SVCCAAnalyzer(representation_config)
        features = analyzer.extract_features(mock_dataloader, "layer2")

        assert features.shape[0] == 32
        assert len(features.shape) == 2

    def test_perform_svd(self, representation_config):
        """Test SVD dimensionality reduction."""
        analyzer = SVCCAAnalyzer(representation_config, threshold=0.90)

        X = np.random.randn(50, 20)
        X_reduced = analyzer._perform_svd(X)

        assert X_reduced.shape[0] == 50
        assert X_reduced.shape[1] <= 20

    def test_compute_svcca_identical(self, representation_config):
        """Test SVCCA for identical features."""
        analyzer = SVCCAAnalyzer(representation_config, threshold=0.99)

        features = np.random.randn(50, 20)
        mean_corr, correlations = analyzer.compute_svcca(features, features)

        # Should have high correlation
        assert mean_corr > 0.95
        assert len(correlations) > 0

    def test_compute_svcca_different(self, representation_config):
        """Test SVCCA for different features."""
        analyzer = SVCCAAnalyzer(representation_config)

        features1 = np.random.randn(50, 20)
        features2 = np.random.randn(50, 30)

        mean_corr, correlations = analyzer.compute_svcca(features1, features2)

        assert 0 <= mean_corr <= 1
        assert len(correlations) > 0

    def test_compute_svcca_unequal_samples(self, representation_config):
        """Test SVCCA with unequal samples."""
        analyzer = SVCCAAnalyzer(representation_config)

        features1 = np.random.randn(40, 20)
        features2 = np.random.randn(50, 20)

        with pytest.raises(ValueError, match="must have same number of samples"):
            analyzer.compute_svcca(features1, features2)


class TestDomainGapAnalyzer:
    """Test DomainGapAnalyzer."""

    def test_analyzer_initialization(self, simple_model):
        """Test domain gap analyzer initialization."""
        analyzer = DomainGapAnalyzer(
            model=simple_model,
            layers=["layer1", "layer2"],
            verbose=0,
        )

        assert len(analyzer.layers) == 2
        assert analyzer.cka_analyzer is not None

    def test_analyzer_repr(self, simple_model):
        """Test string representation."""
        analyzer = DomainGapAnalyzer(
            model=simple_model,
            layers=["layer1", "layer2"],
            verbose=0,
        )
        repr_str = repr(analyzer)

        assert "DomainGapAnalyzer" in repr_str
        assert "layers=2" in repr_str

    def test_analyze_domain_gap(self, simple_model, mock_dataloader):
        """Test domain gap analysis."""
        analyzer = DomainGapAnalyzer(
            model=simple_model,
            layers=["layer1", "layer2"],
            verbose=0,
        )

        # Use same dataloader as both source and target
        results = analyzer.analyze_domain_gap(
            source_loader=mock_dataloader,
            target_loader=mock_dataloader,
        )

        assert len(results) == 2
        assert "layer1" in results
        assert "layer2" in results

        # Same data should have high similarity
        for similarity in results.values():
            assert similarity > 0.95

    def test_analyze_domain_gap_subset_layers(self, simple_model, mock_dataloader):
        """Test domain gap analysis with subset of layers."""
        analyzer = DomainGapAnalyzer(
            model=simple_model,
            layers=["layer1", "layer2", "layer3"],
            verbose=0,
        )

        results = analyzer.analyze_domain_gap(
            source_loader=mock_dataloader,
            target_loader=mock_dataloader,
            layers=["layer1", "layer3"],  # Only analyze subset
        )

        assert len(results) == 2
        assert "layer1" in results
        assert "layer3" in results
        assert "layer2" not in results

    def test_visualize_domain_gap(self, simple_model, mock_dataloader, tmp_path):
        """Test domain gap visualization."""
        analyzer = DomainGapAnalyzer(
            model=simple_model,
            layers=["layer1", "layer2"],
            verbose=0,
        )

        results = analyzer.analyze_domain_gap(
            source_loader=mock_dataloader,
            target_loader=mock_dataloader,
        )

        # Test without saving
        fig = analyzer.visualize_domain_gap(results)
        assert fig is not None

        # Test with saving
        save_path = tmp_path / "domain_gap.png"
        fig = analyzer.visualize_domain_gap(results, save_path=save_path)
        assert save_path.exists()

    def test_visualize_without_results(self, simple_model):
        """Test visualization without results."""
        analyzer = DomainGapAnalyzer(
            model=simple_model,
            layers=["layer1"],
            verbose=0,
        )

        with pytest.raises(ValueError, match="No results available"):
            analyzer.visualize_domain_gap()

    def test_compute_summary_statistics(self, simple_model, mock_dataloader):
        """Test summary statistics computation."""
        analyzer = DomainGapAnalyzer(
            model=simple_model,
            layers=["layer1", "layer2"],
            verbose=0,
        )

        results = analyzer.analyze_domain_gap(
            source_loader=mock_dataloader,
            target_loader=mock_dataloader,
        )

        summary = analyzer.compute_summary_statistics(results)

        assert "mean_similarity" in summary
        assert "std_similarity" in summary
        assert "min_similarity" in summary
        assert "max_similarity" in summary
        assert "mean_gap" in summary
        assert "std_gap" in summary
        assert "max_gap" in summary
        assert "max_gap_layer" in summary

        # Verify values
        assert 0 <= summary["mean_similarity"] <= 1
        assert 0 <= summary["mean_gap"] <= 1
        assert summary["max_gap_layer"] in ["layer1", "layer2"]

    def test_summary_without_results(self, simple_model):
        """Test summary statistics without results."""
        analyzer = DomainGapAnalyzer(
            model=simple_model,
            layers=["layer1"],
            verbose=0,
        )

        with pytest.raises(ValueError, match="No results available"):
            analyzer.compute_summary_statistics()


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_cka_analyzer(self, simple_model):
        """Test CKA analyzer factory."""
        analyzer = create_cka_analyzer(
            model=simple_model,
            layers=["layer1", "layer2"],
            kernel_type="linear",
            verbose=0,
        )

        assert isinstance(analyzer, CKAAnalyzer)
        assert len(analyzer.config.layers) == 2

    def test_create_cka_analyzer_rbf(self, simple_model):
        """Test CKA analyzer factory with RBF kernel."""
        analyzer = create_cka_analyzer(
            model=simple_model,
            layers=["layer1"],
            kernel_type="rbf",
            rbf_sigma=2.0,
            verbose=0,
        )

        assert analyzer.config.kernel_type == "rbf"
        assert analyzer.config.rbf_sigma == 2.0

    def test_create_svcca_analyzer(self, simple_model):
        """Test SVCCA analyzer factory."""
        analyzer = create_svcca_analyzer(
            model=simple_model,
            layers=["layer1", "layer2"],
            threshold=0.95,
            verbose=0,
        )

        assert isinstance(analyzer, SVCCAAnalyzer)
        assert analyzer.threshold == 0.95

    def test_create_domain_gap_analyzer(self, simple_model):
        """Test domain gap analyzer factory."""
        analyzer = create_domain_gap_analyzer(
            model=simple_model,
            layers=["layer1", "layer2"],
            kernel_type="linear",
            verbose=0,
        )

        assert isinstance(analyzer, DomainGapAnalyzer)
        assert len(analyzer.layers) == 2


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_cka(self, simple_model, mock_dataloader):
        """Test complete CKA workflow."""
        # Create analyzer
        analyzer = create_cka_analyzer(
            model=simple_model,
            layers=["layer1", "layer2", "layer3"],
            verbose=0,
        )

        # Extract features from multiple layers
        features_layer1 = analyzer.extract_features(mock_dataloader, "layer1")
        features_layer2 = analyzer.extract_features(mock_dataloader, "layer2")
        features_layer3 = analyzer.extract_features(mock_dataloader, "layer3")

        # Compute CKA matrix
        features_list = [features_layer1, features_layer2, features_layer3]
        cka_matrix = analyzer.compute_cka_matrix(features_list)

        assert cka_matrix.shape == (3, 3)
        assert np.allclose(np.diag(cka_matrix), 1.0, atol=1e-4)

    def test_end_to_end_domain_gap(self, simple_model, mock_dataloader, tmp_path):
        """Test complete domain gap analysis workflow."""
        # Create analyzer
        analyzer = create_domain_gap_analyzer(
            model=simple_model,
            layers=["layer1", "layer2", "layer3"],
            verbose=0,
        )

        # Analyze domain gap
        results = analyzer.analyze_domain_gap(
            source_loader=mock_dataloader,
            target_loader=mock_dataloader,
        )

        # Compute summary
        summary = analyzer.compute_summary_statistics(results)
        assert summary["mean_similarity"] > 0.9

        # Visualize
        save_path = tmp_path / "domain_gap.png"
        _ = analyzer.visualize_domain_gap(results, save_path=save_path)
        assert save_path.exists()

    def test_end_to_end_svcca(self, simple_model, mock_dataloader):
        """Test complete SVCCA workflow."""
        # Create analyzer
        analyzer = create_svcca_analyzer(
            model=simple_model,
            layers=["layer1", "layer2"],
            threshold=0.95,
            verbose=0,
        )

        # Extract features
        features1 = analyzer.extract_features(mock_dataloader, "layer1")
        features2 = analyzer.extract_features(mock_dataloader, "layer2")

        # Compute SVCCA
        mean_corr, correlations = analyzer.compute_svcca(features1, features2)

        assert 0 <= mean_corr <= 1
        assert len(correlations) > 0


class TestEdgeCases:
    """Test edge cases."""

    def test_small_batch_size(self, simple_model):
        """Test with very small batch size."""
        data = torch.randn(4, 3, 224, 224)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=2)

        analyzer = create_cka_analyzer(
            model=simple_model,
            layers=["layer1"],
            verbose=0,
        )

        features = analyzer.extract_features(dataloader, "layer1")
        assert features.shape[0] == 4

    def test_single_sample(self, simple_model):
        """Test with single sample."""
        data = torch.randn(1, 3, 224, 224)
        dataset = TensorDataset(data)
        dataloader = DataLoader(dataset, batch_size=1)

        analyzer = create_cka_analyzer(
            model=simple_model,
            layers=["layer1"],
            verbose=0,
        )

        features = analyzer.extract_features(dataloader, "layer1")
        assert features.shape[0] == 1

    def test_high_dimensional_features(self, representation_config):
        """Test CKA with high-dimensional features."""
        analyzer = CKAAnalyzer(representation_config)

        # Create high-dimensional features
        features1 = np.random.randn(20, 1000)
        features2 = np.random.randn(20, 1000)

        cka = analyzer.compute_cka(features1, features2)
        assert 0 <= cka <= 1

    def test_low_dimensional_features(self, representation_config):
        """Test CKA with low-dimensional features."""
        analyzer = CKAAnalyzer(representation_config)

        # Create low-dimensional features
        features1 = np.random.randn(50, 2)
        features2 = np.random.randn(50, 2)

        cka = analyzer.compute_cka(features1, features2)
        assert 0 <= cka <= 1
