"""Tests for baseline TCAV evaluation."""

from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend  # noqa: E402

import numpy as np  # noqa: E402
import pytest  # noqa: E402
import torch  # noqa: E402
import torch.nn as nn  # noqa: E402
from PIL import Image  # noqa: E402

from src.xai.baseline_tcav_evaluation import (  # noqa: E402
    BaselineTCAVConfig,
    BaselineTCAVEvaluator,
    ConceptCategory,
    create_baseline_tcav_evaluator,
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
def mock_concept_dir(tmp_path):
    """Create mock concept directory with medical and artifact concepts."""
    concept_dir = tmp_path / "concepts"
    concept_dir.mkdir()

    # Medical concepts
    medical = ["pigment_network", "atypical_network"]
    for concept in medical:
        concept_path = concept_dir / concept
        concept_path.mkdir()
        for i in range(10):
            img = Image.new("RGB", (224, 224), color=(i * 20, i * 20, i * 20))
            img.save(concept_path / f"img_{i}.png")

    # Artifact concepts
    artifacts = ["ruler", "hair"]
    for concept in artifacts:
        concept_path = concept_dir / concept
        concept_path.mkdir()
        for i in range(10):
            img = Image.new(
                "RGB", (224, 224), color=(100 + i * 10, 100 + i * 10, 100 + i * 10)
            )
            img.save(concept_path / f"img_{i}.png")

    # Random concepts
    for i in range(3):
        random_dir = concept_dir / f"random_{i}"
        random_dir.mkdir()
        for j in range(10):
            img = Image.new("RGB", (224, 224), color=(j * 20, j * 20, j * 20))
            img.save(random_dir / f"img_{j}.png")

    return concept_dir


@pytest.fixture
def baseline_config(simple_model, mock_concept_dir, tmp_path):
    """Create baseline TCAV config for testing."""
    return BaselineTCAVConfig(
        model=simple_model,
        target_layers=["layer2", "layer3"],
        concept_data_dir=mock_concept_dir,
        medical_concepts=["pigment_network", "atypical_network"],
        artifact_concepts=["ruler", "hair"],
        cav_dir=tmp_path / "cavs",
        batch_size=4,
        num_random_concepts=2,
        min_cav_accuracy=0.5,  # Lower for test data
        verbose=0,
    )


class TestConceptCategory:
    """Test ConceptCategory enum."""

    def test_concept_category_values(self):
        """Test enum values."""
        assert ConceptCategory.MEDICAL.value == "medical"
        assert ConceptCategory.ARTIFACT.value == "artifact"
        assert ConceptCategory.RANDOM.value == "random"

    def test_concept_category_members(self):
        """Test enum members."""
        categories = list(ConceptCategory)
        assert len(categories) == 3
        assert ConceptCategory.MEDICAL in categories
        assert ConceptCategory.ARTIFACT in categories
        assert ConceptCategory.RANDOM in categories


class TestBaselineTCAVConfig:
    """Test BaselineTCAVConfig."""

    def test_valid_config(self, simple_model, mock_concept_dir, tmp_path):
        """Test valid configuration."""
        config = BaselineTCAVConfig(
            model=simple_model,
            target_layers=["layer2"],
            concept_data_dir=mock_concept_dir,
            medical_concepts=["pigment_network"],
            artifact_concepts=["ruler"],
            cav_dir=tmp_path / "cavs",
        )

        assert config.batch_size == 32
        assert config.num_random_concepts == 10
        assert config.min_cav_accuracy == 0.7
        assert isinstance(config.concept_data_dir, Path)
        assert isinstance(config.cav_dir, Path)

    def test_invalid_empty_target_layers(
        self, simple_model, mock_concept_dir, tmp_path
    ):
        """Test empty target layers."""
        with pytest.raises(ValueError, match="target_layers cannot be empty"):
            BaselineTCAVConfig(
                model=simple_model,
                target_layers=[],
                concept_data_dir=mock_concept_dir,
                medical_concepts=["pigment_network"],
                artifact_concepts=["ruler"],
                cav_dir=tmp_path / "cavs",
            )

    def test_invalid_empty_medical_concepts(
        self, simple_model, mock_concept_dir, tmp_path
    ):
        """Test empty medical concepts."""
        with pytest.raises(ValueError, match="medical_concepts cannot be empty"):
            BaselineTCAVConfig(
                model=simple_model,
                target_layers=["layer2"],
                concept_data_dir=mock_concept_dir,
                medical_concepts=[],
                artifact_concepts=["ruler"],
                cav_dir=tmp_path / "cavs",
            )

    def test_invalid_empty_artifact_concepts(
        self, simple_model, mock_concept_dir, tmp_path
    ):
        """Test empty artifact concepts."""
        with pytest.raises(ValueError, match="artifact_concepts cannot be empty"):
            BaselineTCAVConfig(
                model=simple_model,
                target_layers=["layer2"],
                concept_data_dir=mock_concept_dir,
                medical_concepts=["pigment_network"],
                artifact_concepts=[],
                cav_dir=tmp_path / "cavs",
            )

    def test_invalid_batch_size(self, simple_model, mock_concept_dir, tmp_path):
        """Test invalid batch size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            BaselineTCAVConfig(
                model=simple_model,
                target_layers=["layer2"],
                concept_data_dir=mock_concept_dir,
                medical_concepts=["pigment_network"],
                artifact_concepts=["ruler"],
                cav_dir=tmp_path / "cavs",
                batch_size=0,
            )

    def test_invalid_num_random_concepts(
        self, simple_model, mock_concept_dir, tmp_path
    ):
        """Test invalid num_random_concepts."""
        with pytest.raises(ValueError, match="num_random_concepts must be positive"):
            BaselineTCAVConfig(
                model=simple_model,
                target_layers=["layer2"],
                concept_data_dir=mock_concept_dir,
                medical_concepts=["pigment_network"],
                artifact_concepts=["ruler"],
                cav_dir=tmp_path / "cavs",
                num_random_concepts=-1,
            )

    def test_invalid_min_cav_accuracy(self, simple_model, mock_concept_dir, tmp_path):
        """Test invalid min_cav_accuracy."""
        with pytest.raises(ValueError, match="min_cav_accuracy must be in"):
            BaselineTCAVConfig(
                model=simple_model,
                target_layers=["layer2"],
                concept_data_dir=mock_concept_dir,
                medical_concepts=["pigment_network"],
                artifact_concepts=["ruler"],
                cav_dir=tmp_path / "cavs",
                min_cav_accuracy=1.5,
            )

    def test_directory_creation(self, simple_model, mock_concept_dir, tmp_path):
        """Test automatic CAV directory creation."""
        cav_dir = tmp_path / "new_cavs"
        assert not cav_dir.exists()

        BaselineTCAVConfig(
            model=simple_model,
            target_layers=["layer2"],
            concept_data_dir=mock_concept_dir,
            medical_concepts=["pigment_network"],
            artifact_concepts=["ruler"],
            cav_dir=cav_dir,
        )

        assert cav_dir.exists()


class TestBaselineTCAVEvaluator:
    """Test BaselineTCAVEvaluator."""

    def test_evaluator_initialization(self, baseline_config):
        """Test evaluator initialization."""
        evaluator = BaselineTCAVEvaluator(baseline_config)

        assert evaluator.model is not None
        assert evaluator.tcav is not None
        assert isinstance(evaluator.results, dict)
        assert len(evaluator.results) == 0

    def test_evaluator_repr(self, baseline_config):
        """Test string representation."""
        evaluator = BaselineTCAVEvaluator(baseline_config)
        repr_str = repr(evaluator)

        assert "BaselineTCAVEvaluator" in repr_str
        assert "medical_concepts=2" in repr_str
        assert "artifact_concepts=2" in repr_str

    def test_precompute_cavs(self, baseline_config):
        """Test CAV precomputation."""
        evaluator = BaselineTCAVEvaluator(baseline_config)

        # Should not raise
        evaluator.precompute_cavs()

        # Check that some CAVs were trained
        assert len(evaluator.tcav.cavs) > 0

    def test_evaluate_baseline(self, baseline_config):
        """Test baseline evaluation."""
        evaluator = BaselineTCAVEvaluator(baseline_config)

        images = torch.randn(8, 3, 224, 224)
        results = evaluator.evaluate_baseline(
            images=images, target_class=1, precompute=True
        )

        # Check result structure
        assert "medical_scores" in results
        assert "artifact_scores" in results
        assert "medical_mean" in results
        assert "medical_std" in results
        assert "artifact_mean" in results
        assert "artifact_std" in results
        assert "medical_layer_means" in results
        assert "artifact_layer_means" in results
        assert "statistical_comparison" in results
        assert "num_images" in results
        assert "target_class" in results

        # Check values are reasonable
        assert 0 <= results["medical_mean"] <= 1
        assert 0 <= results["artifact_mean"] <= 1
        assert results["num_images"] == 8
        assert results["target_class"] == 1

        # Check scores structure
        assert isinstance(results["medical_scores"], dict)
        assert "pigment_network" in results["medical_scores"]
        assert "atypical_network" in results["medical_scores"]

        assert isinstance(results["artifact_scores"], dict)
        assert "ruler" in results["artifact_scores"]
        assert "hair" in results["artifact_scores"]

        # Check layer means
        assert "layer2" in results["medical_layer_means"]
        assert "layer3" in results["medical_layer_means"]

        # Check statistical comparison
        stat = results["statistical_comparison"]
        assert "t_statistic" in stat
        assert "p_value" in stat
        assert "cohens_d" in stat
        assert "significant" in stat

    def test_evaluate_without_precompute(self, baseline_config):
        """Test evaluation without precomputation."""
        evaluator = BaselineTCAVEvaluator(baseline_config)

        # Precompute first
        evaluator.precompute_cavs()

        images = torch.randn(4, 3, 224, 224)
        results = evaluator.evaluate_baseline(
            images=images, target_class=0, precompute=False
        )

        assert results is not None
        assert results["num_images"] == 4

    def test_analyze_multilayer_activation(self, baseline_config):
        """Test multi-layer activation analysis."""
        evaluator = BaselineTCAVEvaluator(baseline_config)

        images = torch.randn(4, 3, 224, 224)
        evaluator.evaluate_baseline(images=images, target_class=1)

        analysis = evaluator.analyze_multilayer_activation()

        assert "medical" in analysis
        assert "artifact" in analysis
        assert "layer_differences" in analysis

        assert "layer2" in analysis["medical"]
        assert "layer3" in analysis["medical"]
        assert "layer2" in analysis["artifact"]
        assert "layer3" in analysis["artifact"]

    def test_analyze_multilayer_without_evaluation(self, baseline_config):
        """Test multi-layer analysis without prior evaluation."""
        evaluator = BaselineTCAVEvaluator(baseline_config)

        with pytest.raises(ValueError, match="No results available"):
            evaluator.analyze_multilayer_activation()

    def test_visualize_concept_scores(self, baseline_config, tmp_path):
        """Test concept score visualization."""
        evaluator = BaselineTCAVEvaluator(baseline_config)

        images = torch.randn(4, 3, 224, 224)
        evaluator.evaluate_baseline(images=images, target_class=1)

        # Test without saving
        fig = evaluator.visualize_concept_scores()
        assert fig is not None

        # Test with saving
        save_path = tmp_path / "baseline_tcav.png"
        fig = evaluator.visualize_concept_scores(save_path=save_path)
        assert save_path.exists()

    def test_visualize_without_evaluation(self, baseline_config):
        """Test visualization without prior evaluation."""
        evaluator = BaselineTCAVEvaluator(baseline_config)

        with pytest.raises(ValueError, match="No results available"):
            evaluator.visualize_concept_scores()

    def test_save_results(self, baseline_config, tmp_path):
        """Test saving results."""
        evaluator = BaselineTCAVEvaluator(baseline_config)

        images = torch.randn(4, 3, 224, 224)
        evaluator.evaluate_baseline(images=images, target_class=1)

        save_path = tmp_path / "results.npz"
        evaluator.save_results(save_path)

        assert save_path.exists()

        # Load and verify
        loaded = np.load(save_path)
        assert "medical_mean" in loaded
        assert "artifact_mean" in loaded
        assert "stat_t_statistic" in loaded

    def test_save_results_without_evaluation(self, baseline_config, tmp_path):
        """Test saving results without evaluation."""
        evaluator = BaselineTCAVEvaluator(baseline_config)

        with pytest.raises(ValueError, match="No results to save"):
            evaluator.save_results(tmp_path / "results.npz")

    def test_flatten_scores(self, baseline_config):
        """Test score flattening."""
        evaluator = BaselineTCAVEvaluator(baseline_config)

        scores = {
            "concept1": {"layer1": 50.0, "layer2": 60.0},
            "concept2": {"layer1": 55.0, "layer2": 65.0},
        }

        flattened = evaluator._flatten_scores(scores)

        assert len(flattened) == 4
        assert np.allclose(flattened, [0.50, 0.60, 0.55, 0.65])

    def test_flatten_scores_with_nan(self, baseline_config):
        """Test score flattening with NaN values."""
        evaluator = BaselineTCAVEvaluator(baseline_config)

        scores = {
            "concept1": {"layer1": 50.0, "layer2": float("nan")},
            "concept2": {"layer1": 55.0, "layer2": 65.0},
        }

        flattened = evaluator._flatten_scores(scores)

        # Should exclude NaN values
        assert len(flattened) == 3
        assert not np.any(np.isnan(flattened))

    def test_compute_layer_means(self, baseline_config):
        """Test layer mean computation."""
        evaluator = BaselineTCAVEvaluator(baseline_config)

        scores = {
            "concept1": {"layer2": 50.0, "layer3": 60.0},
            "concept2": {"layer2": 60.0, "layer3": 70.0},
        }

        layer_means = evaluator._compute_layer_means(scores)

        assert "layer2" in layer_means
        assert "layer3" in layer_means
        assert np.isclose(layer_means["layer2"], 0.55)  # (50+60)/200
        assert np.isclose(layer_means["layer3"], 0.65)  # (60+70)/200

    def test_statistical_comparison(self, baseline_config):
        """Test statistical comparison."""
        evaluator = BaselineTCAVEvaluator(baseline_config)

        medical_scores = np.array([0.6, 0.62, 0.58, 0.61])
        artifact_scores = np.array([0.45, 0.47, 0.43, 0.46])

        stat = evaluator._statistical_comparison(medical_scores, artifact_scores)

        assert "t_statistic" in stat
        assert "p_value" in stat
        assert "cohens_d" in stat
        assert "significant" in stat

        # Medical should be significantly higher
        assert stat["t_statistic"] > 0
        assert stat["p_value"] < 0.05
        assert stat["significant"] is True


class TestFactoryFunction:
    """Test factory function."""

    def test_create_baseline_tcav_evaluator(
        self, simple_model, mock_concept_dir, tmp_path
    ):
        """Test factory function."""
        evaluator = create_baseline_tcav_evaluator(
            model=simple_model,
            target_layers=["layer2", "layer3"],
            concept_data_dir=mock_concept_dir,
            medical_concepts=["pigment_network"],
            artifact_concepts=["ruler"],
            cav_dir=tmp_path / "cavs",
            batch_size=16,
        )

        assert isinstance(evaluator, BaselineTCAVEvaluator)
        assert evaluator.config.batch_size == 16


class TestIntegration:
    """Integration tests."""

    def test_end_to_end_evaluation(self, baseline_config, tmp_path):
        """Test complete evaluation workflow."""
        evaluator = BaselineTCAVEvaluator(baseline_config)

        # 1. Precompute CAVs
        evaluator.precompute_cavs()

        # 2. Evaluate baseline
        images = torch.randn(4, 3, 224, 224)
        results = evaluator.evaluate_baseline(
            images=images, target_class=1, precompute=False
        )

        assert results is not None
        assert "medical_mean" in results
        assert "artifact_mean" in results

        # 3. Multi-layer analysis
        analysis = evaluator.analyze_multilayer_activation(results)
        assert analysis is not None

        # 4. Visualization
        fig = evaluator.visualize_concept_scores(results)
        assert fig is not None

        # 5. Save results
        evaluator.save_results(tmp_path / "results.npz")
        assert (tmp_path / "results.npz").exists()


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_single_image_evaluation(self, baseline_config):
        """Test evaluation with single image."""
        evaluator = BaselineTCAVEvaluator(baseline_config)

        images = torch.randn(1, 3, 224, 224)
        results = evaluator.evaluate_baseline(images=images, target_class=0)

        assert results["num_images"] == 1
        assert not np.isnan(results["medical_mean"])

    def test_large_batch_evaluation(self, baseline_config):
        """Test evaluation with large batch."""
        evaluator = BaselineTCAVEvaluator(baseline_config)

        images = torch.randn(32, 3, 224, 224)
        results = evaluator.evaluate_baseline(images=images, target_class=1)

        assert results["num_images"] == 32

    def test_custom_figsize_visualization(self, baseline_config):
        """Test visualization with custom figure size."""
        evaluator = BaselineTCAVEvaluator(baseline_config)

        images = torch.randn(4, 3, 224, 224)
        evaluator.evaluate_baseline(images=images, target_class=1)

        fig = evaluator.visualize_concept_scores(figsize=(20, 15))
        assert fig.get_size_inches()[0] == 20
        assert fig.get_size_inches()[1] == 15
