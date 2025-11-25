"""Tests for TCAV implementation."""

import pytest
import torch
import torch.nn as nn
from PIL import Image

from src.xai.tcav import (
    TCAV,
    ActivationExtractor,
    CAVTrainer,
    ConceptDataset,
    TCAVConfig,
    create_tcav,
)


# Define SimpleModel at module level to avoid pickle issues
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
    """Create mock concept directory with images."""
    concept_dir = tmp_path / "concepts"
    concept_dir.mkdir()

    # Create three concepts
    for concept in ["pigment_network", "ruler", "random"]:
        concept_path = concept_dir / concept
        concept_path.mkdir()

        # Create 10 test images for each concept
        for i in range(10):
            img = Image.new("RGB", (224, 224), color=(i * 20, i * 20, i * 20))
            img.save(concept_path / f"img_{i}.png")

    return concept_dir


@pytest.fixture
def tcav_config(simple_model, mock_concept_dir, tmp_path):
    """Create TCAV config for testing."""
    return TCAVConfig(
        model=simple_model,
        target_layers=["layer2", "layer3"],
        concept_data_dir=mock_concept_dir,
        cav_dir=tmp_path / "cavs",
        batch_size=4,
        num_random_concepts=2,
        min_cav_accuracy=0.5,  # Lower threshold for test data
        verbose=0,
    )


class TestTCAVConfig:
    """Test TCAVConfig."""

    def test_valid_config(self, simple_model, tmp_path):
        """Test valid configuration."""
        config = TCAVConfig(
            model=simple_model,
            target_layers=["layer1"],
            concept_data_dir=tmp_path / "concepts",
            cav_dir=tmp_path / "cavs",
        )

        assert config.batch_size == 32
        assert config.num_random_concepts == 10
        assert config.alpha == 0.05
        assert config.min_cav_accuracy == 0.7

    def test_invalid_target_layers_empty(self, simple_model, tmp_path):
        """Test empty target layers."""
        with pytest.raises(ValueError, match="target_layers cannot be empty"):
            TCAVConfig(
                model=simple_model,
                target_layers=[],
                concept_data_dir=tmp_path / "concepts",
                cav_dir=tmp_path / "cavs",
            )

    def test_invalid_batch_size(self, simple_model, tmp_path):
        """Test invalid batch size."""
        with pytest.raises(ValueError, match="batch_size must be positive"):
            TCAVConfig(
                model=simple_model,
                target_layers=["layer1"],
                concept_data_dir=tmp_path / "concepts",
                cav_dir=tmp_path / "cavs",
                batch_size=-1,
            )

    def test_invalid_alpha(self, simple_model, tmp_path):
        """Test invalid alpha."""
        with pytest.raises(ValueError, match="alpha must be in"):
            TCAVConfig(
                model=simple_model,
                target_layers=["layer1"],
                concept_data_dir=tmp_path / "concepts",
                cav_dir=tmp_path / "cavs",
                alpha=1.5,
            )

    def test_invalid_min_cav_accuracy(self, simple_model, tmp_path):
        """Test invalid min_cav_accuracy."""
        with pytest.raises(ValueError, match="min_cav_accuracy must be in"):
            TCAVConfig(
                model=simple_model,
                target_layers=["layer1"],
                concept_data_dir=tmp_path / "concepts",
                cav_dir=tmp_path / "cavs",
                min_cav_accuracy=1.5,
            )

    def test_directory_creation(self, simple_model, tmp_path):
        """Test automatic directory creation."""
        cav_dir = tmp_path / "new_cavs"
        assert not cav_dir.exists()

        _ = TCAVConfig(
            model=simple_model,
            target_layers=["layer1"],
            concept_data_dir=tmp_path / "concepts",
            cav_dir=cav_dir,
        )

        assert cav_dir.exists()


class TestConceptDataset:
    """Test ConceptDataset."""

    def test_dataset_creation(self):
        """Test dataset creation."""
        images = torch.randn(10, 3, 224, 224)
        labels = torch.ones(10)

        dataset = ConceptDataset(images, labels)

        assert len(dataset) == 10
        img, label = dataset[0]
        assert img.shape == (3, 224, 224)
        assert label == 1.0

    def test_dataset_with_transform(self):
        """Test dataset with transforms."""
        from torchvision import transforms

        images = torch.randn(5, 3, 224, 224)
        labels = torch.zeros(5)

        transform = transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        dataset = ConceptDataset(images, labels, transform=transform)

        img, label = dataset[0]
        assert img.shape == (3, 224, 224)
        assert label == 0.0


class TestActivationExtractor:
    """Test ActivationExtractor."""

    def test_extractor_initialization(self, simple_model):
        """Test extractor initialization."""
        extractor = ActivationExtractor(
            simple_model, target_layers=["layer1", "layer2"]
        )

        assert len(extractor.hooks) == 2
        assert extractor.target_layers == ["layer1", "layer2"]

    def test_activation_extraction(self, simple_model):
        """Test activation extraction."""
        from torch.utils.data import DataLoader, TensorDataset

        extractor = ActivationExtractor(simple_model, target_layers=["layer2"])

        # Create dummy data
        images = torch.randn(8, 3, 224, 224)
        labels = torch.zeros(8)
        dataset = TensorDataset(images, labels)
        loader = DataLoader(dataset, batch_size=4)

        # Extract activations
        acts = extractor.extract(loader, "layer2")

        assert acts.shape[0] == 8  # Number of samples
        assert acts.dim() == 2  # (N, C) after global average pooling

    def test_hook_cleanup(self, simple_model):
        """Test hook cleanup."""
        extractor = ActivationExtractor(simple_model, target_layers=["layer1"])

        assert len(extractor.hooks) == 1

        extractor.remove_hooks()
        assert len(extractor.hooks) == 0

    def test_invalid_layer(self, simple_model):
        """Test invalid layer name."""
        with pytest.raises(ValueError, match="Could not find layers"):
            ActivationExtractor(simple_model, target_layers=["nonexistent_layer"])


class TestCAVTrainer:
    """Test CAVTrainer."""

    def test_cav_training(self):
        """Test CAV training."""
        trainer = CAVTrainer()

        # Create dummy activations
        concept_acts = torch.randn(50, 32)
        random_acts = torch.randn(50, 32)

        cav, accuracy, metrics = trainer.train(concept_acts, random_acts)

        assert cav.shape == (32,)
        assert torch.allclose(torch.norm(cav), torch.tensor(1.0), atol=1e-5)
        assert 0 <= accuracy <= 1
        assert "train_accuracy" in metrics
        assert "val_accuracy" in metrics

    def test_cav_training_with_difficult_data(self):
        """Test CAV training with overlapping distributions."""
        trainer = CAVTrainer()

        # Create overlapping distributions (harder to separate)
        base = torch.randn(100, 32)
        concept_acts = base + torch.randn(100, 32) * 0.1
        random_acts = base + torch.randn(100, 32) * 0.1

        cav, accuracy, metrics = trainer.train(concept_acts, random_acts)

        assert cav.shape == (32,)
        assert torch.allclose(torch.norm(cav), torch.tensor(1.0), atol=1e-5)
        # Accuracy might be lower for difficult data
        assert 0 <= accuracy <= 1


class TestTCAV:
    """Test TCAV main class."""

    def test_tcav_initialization(self, tcav_config):
        """Test TCAV initialization."""
        tcav = TCAV(tcav_config)

        assert tcav.model is not None
        assert tcav.extractor is not None
        assert tcav.trainer is not None
        assert len(tcav.cavs) == 0

    def test_tcav_repr(self, tcav_config):
        """Test TCAV string representation."""
        tcav = TCAV(tcav_config)
        repr_str = repr(tcav)

        assert "TCAV" in repr_str
        assert "layer2" in repr_str or "layer3" in repr_str

    def test_load_concept_data(self, tcav_config):
        """Test loading concept data."""
        tcav = TCAV(tcav_config)

        images, num_images = tcav.load_concept_data("pigment_network")

        assert num_images == 10
        assert images.shape == (10, 3, 224, 224)

    def test_load_nonexistent_concept(self, tcav_config):
        """Test loading nonexistent concept."""
        tcav = TCAV(tcav_config)

        with pytest.raises(ValueError, match="Concept directory not found"):
            tcav.load_concept_data("nonexistent_concept")

    def test_train_cav(self, tcav_config):
        """Test training a CAV."""
        tcav = TCAV(tcav_config)

        cav, metrics = tcav.train_cav(
            concept="pigment_network", layer="layer2", random_concept="random"
        )

        assert cav.shape[0] > 0  # Has features
        assert torch.allclose(torch.norm(cav), torch.tensor(1.0), atol=1e-5)
        assert "train_accuracy" in metrics
        assert "val_accuracy" in metrics

        # Check stored
        assert "layer2" in tcav.cavs
        assert "pigment_network" in tcav.cavs["layer2"]

    def test_cav_save_and_load(self, tcav_config):
        """Test CAV persistence."""
        tcav = TCAV(tcav_config)

        # Train and save
        cav1, _ = tcav.train_cav(
            concept="pigment_network",
            layer="layer2",
            random_concept="random",
            save=True,
        )

        # Clear memory
        tcav.cavs.clear()
        tcav.cav_metrics.clear()

        # Load again (should load from disk)
        cav2, _ = tcav.train_cav(
            concept="pigment_network", layer="layer2", random_concept="random"
        )

        assert torch.allclose(cav1, cav2)

    def test_compute_tcav_score(self, tcav_config):
        """Test TCAV score computation."""
        tcav = TCAV(tcav_config)

        # Train CAV first
        tcav.train_cav(
            concept="pigment_network", layer="layer2", random_concept="random"
        )

        # Compute TCAV score
        inputs = torch.randn(8, 3, 224, 224)
        score = tcav.compute_tcav_score(
            inputs=inputs,
            target_class=1,
            concept="pigment_network",
            layer="layer2",
        )

        assert 0 <= score <= 100

    def test_compute_tcav_without_training(self, tcav_config):
        """Test TCAV score without training CAV first."""
        tcav = TCAV(tcav_config)

        inputs = torch.randn(4, 3, 224, 224)

        with pytest.raises(ValueError, match="CAV not trained"):
            tcav.compute_tcav_score(
                inputs=inputs,
                target_class=0,
                concept="pigment_network",
                layer="layer2",
            )

    def test_compute_multilayer_tcav(self, tcav_config):
        """Test multi-layer TCAV."""
        tcav = TCAV(tcav_config)

        # Train CAVs for multiple layers
        for layer in ["layer2", "layer3"]:
            tcav.train_cav(
                concept="pigment_network", layer=layer, random_concept="random"
            )

        # Compute multi-layer scores
        inputs = torch.randn(4, 3, 224, 224)
        scores = tcav.compute_multilayer_tcav(
            inputs=inputs, target_class=1, concept="pigment_network"
        )

        assert "layer2" in scores
        assert "layer3" in scores
        assert all(0 <= s <= 100 for s in scores.values())

    def test_save_and_load_state(self, tcav_config, tmp_path):
        """Test saving and loading TCAV state."""
        tcav1 = TCAV(tcav_config)

        # Train some CAVs
        tcav1.train_cav("pigment_network", "layer2", "random")
        tcav1.train_cav("ruler", "layer3", "random")

        # Save state
        state_path = tmp_path / "tcav_state.pt"
        tcav1.save_state(state_path)

        assert state_path.exists()

        # Load state in new instance
        tcav2 = TCAV(tcav_config)
        tcav2.load_state(state_path)

        assert len(tcav2.cavs) == len(tcav1.cavs)
        assert "layer2" in tcav2.cavs
        assert "pigment_network" in tcav2.cavs["layer2"]

        # Verify CAVs are identical
        assert torch.allclose(
            tcav1.cavs["layer2"]["pigment_network"],
            tcav2.cavs["layer2"]["pigment_network"],
        )


class TestTCAVIntegration:
    """Integration tests for TCAV."""

    def test_end_to_end_workflow(self, tcav_config):
        """Test complete TCAV workflow."""
        tcav = TCAV(tcav_config)

        # 1. Train CAV
        cav, metrics = tcav.train_cav(
            concept="pigment_network", layer="layer2", random_concept="random"
        )

        assert cav is not None
        assert metrics["val_accuracy"] > 0

        # 2. Compute TCAV score
        inputs = torch.randn(4, 3, 224, 224)
        score = tcav.compute_tcav_score(
            inputs=inputs,
            target_class=1,
            concept="pigment_network",
            layer="layer2",
        )

        assert 0 <= score <= 100

        # 3. Multi-layer analysis
        tcav.train_cav(
            concept="pigment_network", layer="layer3", random_concept="random"
        )
        multi_scores = tcav.compute_multilayer_tcav(
            inputs=inputs, target_class=1, concept="pigment_network"
        )

        assert len(multi_scores) == 2


class TestFactoryFunction:
    """Test factory function."""

    def test_create_tcav(self, simple_model, tmp_path):
        """Test TCAV creation via factory."""
        concept_dir = tmp_path / "concepts"
        concept_dir.mkdir()

        tcav = create_tcav(
            model=simple_model,
            target_layers=["layer1", "layer2"],
            concept_data_dir=concept_dir,
            cav_dir=tmp_path / "cavs",
            batch_size=16,
        )

        assert isinstance(tcav, TCAV)
        assert tcav.config.batch_size == 16


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_concept_directory(self, tcav_config, tmp_path):
        """Test loading from empty directory."""
        empty_dir = tmp_path / "empty_concept"
        empty_dir.mkdir()

        tcav = TCAV(tcav_config)
        tcav.config.concept_data_dir = tmp_path

        with pytest.raises(ValueError, match="No images found"):
            tcav.load_concept_data("empty_concept")

    def test_few_samples(self, tcav_config, tmp_path):
        """Test with very few samples."""
        # Create concept with only 2 images
        few_concept_dir = tmp_path / "few_samples"
        few_concept_dir.mkdir()

        for i in range(2):
            img = Image.new("RGB", (224, 224), color=(100, 100, 100))
            img.save(few_concept_dir / f"img_{i}.png")

        tcav = TCAV(tcav_config)
        tcav.config.concept_data_dir = tmp_path

        images, count = tcav.load_concept_data("few_samples")
        assert count == 2

    def test_low_accuracy_cav_warning(self, tcav_config, caplog):
        """Test warning when CAV accuracy is low."""
        import logging

        # Set strict threshold
        tcav_config.min_cav_accuracy = 0.99  # Nearly impossible to achieve
        tcav = TCAV(tcav_config)

        with caplog.at_level(logging.WARNING):
            cav, metrics = tcav.train_cav(
                concept="pigment_network", layer="layer2", random_concept="random"
            )

        # Check warning was logged
        assert any("below threshold" in record.message for record in caplog.records)
        assert cav is not None  # CAV still returned

    def test_precompute_all_cavs(self, tcav_config, mock_concept_dir):
        """Test precomputing all CAVs."""
        tcav = TCAV(tcav_config)

        # Create random concept directories
        for i in range(2):
            random_dir = mock_concept_dir / f"random_{i}"
            random_dir.mkdir()
            for j in range(10):
                img = Image.new("RGB", (224, 224), color=(j * 20, j * 20, j * 20))
                img.save(random_dir / f"img_{j}.png")

        concepts = ["pigment_network", "ruler"]

        # Precompute with 2 random concepts
        tcav.config.num_random_concepts = 2
        tcav.precompute_all_cavs(concepts)

        # Should have CAVs for both layers and concepts
        assert "layer2" in tcav.cavs
        assert "layer3" in tcav.cavs
        assert "pigment_network" in tcav.cavs["layer2"]
        assert "ruler" in tcav.cavs["layer2"]

    def test_precompute_with_error(self, tcav_config, mock_concept_dir, caplog):
        """Test precompute handles errors gracefully."""
        import logging

        tcav = TCAV(tcav_config)

        # Create random concept directories
        for i in range(2):
            random_dir = mock_concept_dir / f"random_{i}"
            random_dir.mkdir()
            for j in range(10):
                img = Image.new("RGB", (224, 224), color=(j * 20, j * 20, j * 20))
                img.save(random_dir / f"img_{j}.png")

        # Include a non-existent concept
        concepts = ["pigment_network", "nonexistent_concept"]

        with caplog.at_level(logging.ERROR):
            tcav.precompute_all_cavs(concepts)

        # Check error was logged
        assert any("Failed to train CAV" in record.message for record in caplog.records)

        # Should still have CAV for valid concept
        assert "pigment_network" in tcav.cavs.get("layer2", {})

    def test_multilayer_tcav_with_missing_cavs(self, tcav_config, caplog):
        """Test multilayer TCAV when some CAVs are missing."""
        import logging

        tcav = TCAV(tcav_config)

        # Only train CAV for one layer
        tcav.train_cav("pigment_network", "layer2", "random")
        # Don't train for layer3

        inputs = torch.randn(4, 3, 224, 224)

        with caplog.at_level(logging.WARNING):
            scores = tcav.compute_multilayer_tcav(
                inputs=inputs, target_class=1, concept="pigment_network"
            )

        # Should have score for layer2 only
        assert "layer2" in scores
        assert "layer3" not in scores

        # Check warning was logged
        assert any(
            "Skipping layer layer3" in record.message for record in caplog.records
        )

    def test_backward_hook_with_4d_activation(self, tcav_config):
        """Test compute_tcav_score with 4D activations (spatial)."""
        tcav = TCAV(tcav_config)

        # Train CAV
        tcav.train_cav("pigment_network", "layer2", "random")

        # layer2 outputs 4D tensors (B, C, H, W)
        inputs = torch.randn(4, 3, 224, 224)
        score = tcav.compute_tcav_score(
            inputs=inputs,
            target_class=1,
            concept="pigment_network",
            layer="layer2",  # This layer has spatial dims
        )

        assert 0 <= score <= 100

    def test_invalid_layer_in_compute_tcav(self, tcav_config):
        """Test compute_tcav_score with invalid layer."""
        tcav = TCAV(tcav_config)

        # Manually add a CAV for a non-existent layer to bypass the CAV check
        tcav.cavs["nonexistent_layer"] = {}
        tcav.cavs["nonexistent_layer"]["pigment_network"] = torch.randn(32)

        inputs = torch.randn(4, 3, 224, 224)

        # Try to compute - should fail when looking for layer in model
        with pytest.raises(ValueError, match="not found"):
            tcav.compute_tcav_score(
                inputs=inputs,
                target_class=1,
                concept="pigment_network",
                layer="nonexistent_layer",
            )

    def test_info_logging(self, tcav_config, caplog):
        """Test info logging is properly called."""
        import logging

        tcav = TCAV(tcav_config)

        # Test load_concept_data logging
        with caplog.at_level(logging.INFO):
            images, count = tcav.load_concept_data("pigment_network")

        # Check info log for loaded images
        assert any(
            "Loaded" in record.message and "pigment_network" in record.message
            for record in caplog.records
        )

        caplog.clear()

        # Test load_state logging
        tcav.train_cav("pigment_network", "layer2", "random")
        state_path = tcav_config.cav_dir / "test_state.pt"
        tcav.save_state(state_path)

        tcav2 = TCAV(tcav_config)
        with caplog.at_level(logging.INFO):
            tcav2.load_state(state_path)

        # Check info logs for loaded state
        assert any("Loaded TCAV state" in record.message for record in caplog.records)
        assert any(
            "Loaded" in record.message and "CAVs" in record.message
            for record in caplog.records
        )

    def test_backward_hook_with_3d_activation(self, tcav_config):
        """Test compute_tcav_score with 3D (already pooled) activations."""
        tcav = TCAV(tcav_config)

        # Train CAV for layer3 (which might have different dims)
        tcav.train_cav("pigment_network", "layer3", "random")

        # Compute TCAV score - tests the else branch in backward hook
        inputs = torch.randn(4, 3, 224, 224)
        score = tcav.compute_tcav_score(
            inputs=inputs,
            target_class=1,
            concept="pigment_network",
            layer="layer3",
        )

        assert 0 <= score <= 100
