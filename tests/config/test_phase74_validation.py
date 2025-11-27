"""
Phase 7.4 HPO Module Validation Tests.

Simple validation tests to verify Phase 7.4 HPO infrastructure is working.

Author: Viraj Pankaj Jain
"""

from pathlib import Path

import pytest


class TestPhase74Imports:
    """Test that all Phase 7.4 modules can be imported."""

    def test_hyperparameters_import(self):
        """Test hyperparameters module imports."""
        from src.config.hpo import hyperparameters

        assert hasattr(hyperparameters, "HyperparameterConfig")

    def test_search_spaces_import(self):
        """Test search_spaces module imports."""
        from src.config.hpo import search_spaces

        assert hasattr(search_spaces, "SearchSpaceFactory")

    def test_objectives_import(self):
        """Test objectives module imports."""
        from src.config.hpo import objectives

        assert hasattr(objectives, "ObjectiveMetrics")
        assert hasattr(objectives, "WeightedSumObjective")

    def test_pruners_import(self):
        """Test pruners module imports."""
        from src.config.hpo import pruners

        assert hasattr(pruners, "create_pruner")

    def test_hpo_trainer_import(self):
        """Test hpo_trainer module imports."""
        from src.config.hpo import hpo_trainer

        assert hasattr(hpo_trainer, "HPOTrainer")


class TestPhase74Configuration:
    """Test Phase 7.4 configuration files."""

    def test_default_config_exists(self):
        """Test default HPO config file exists."""
        config_path = Path("configs/hpo/default_hpo_config.yaml")
        assert config_path.exists(), "Default HPO config file should exist"

    def test_default_config_readable(self):
        """Test default config is valid YAML."""
        import yaml

        config_path = Path("configs/hpo/default_hpo_config.yaml")

        with open(config_path) as f:
            config = yaml.safe_load(f)

        assert config is not None
        assert "hpo" in config
        assert "sampler" in config
        assert "pruner" in config
        assert "search_space" in config
        assert "objective" in config


class TestPhase74BasicFunctionality:
    """Test basic functionality of Phase 7.4 components."""

    def test_hyperparameter_config_creation(self):
        """Test creating hyperparameter configuration."""
        from src.config.hpo.hyperparameters import HyperparameterConfig

        config = HyperparameterConfig()
        assert config is not None
        # Config has nested structure with optimizer containing learning_rate
        assert hasattr(config, "optimizer")
        assert hasattr(config.optimizer, "learning_rate")

    def test_objective_metrics_creation(self):
        """Test creating objective metrics."""
        from src.config.hpo.objectives import ObjectiveMetrics

        metrics = ObjectiveMetrics(
            accuracy=0.85,
            robustness=0.75,
            explainability=0.80,
        )

        assert metrics.accuracy == 0.85
        assert metrics.robustness == 0.75
        assert metrics.explainability == 0.80

    def test_search_space_factory(self):
        """Test search space factory functionality."""
        from src.config.hpo.search_spaces import SearchSpaceFactory

        # Factory has static methods for creating search spaces
        assert hasattr(SearchSpaceFactory, "create_quick_search_space")
        assert hasattr(SearchSpaceFactory, "create_full_search_space")
        assert hasattr(SearchSpaceFactory, "create_balanced_search_space")

        # Can create search space function
        space_fn = SearchSpaceFactory.create_quick_search_space()
        assert callable(space_fn)

    def test_weighted_sum_objective(self):
        """Test weighted sum objective creation."""
        from src.config.hpo.objectives import WeightedSumObjective

        # Create with proper constructor
        objective = WeightedSumObjective(
            accuracy_weight=0.5, robustness_weight=0.3, explainability_weight=0.2
        )

        assert objective is not None
        assert hasattr(objective, "accuracy_weight")


class TestPhase74FileStructure:
    """Test Phase 7.4 file structure."""

    def test_hpo_package_structure(self):
        """Test HPO package has required files."""
        hpo_dir = Path("src/config/hpo")

        required_files = [
            "__init__.py",
            "hyperparameters.py",
            "search_spaces.py",
            "objectives.py",
            "pruners.py",
            "hpo_trainer.py",
        ]

        for filename in required_files:
            file_path = hpo_dir / filename
            assert file_path.exists(), f"{filename} should exist in HPO package"

    def test_config_directory_structure(self):
        """Test configs/hpo directory exists."""
        config_dir = Path("configs/hpo")
        assert config_dir.exists(), "configs/hpo directory should exist"


class TestPhase74Documentation:
    """Test Phase 7.4 has documentation."""

    def test_module_docstrings(self):
        """Test modules have docstrings."""
        from src.config.hpo import hyperparameters, objectives, search_spaces

        assert hyperparameters.__doc__ is not None
        assert search_spaces.__doc__ is not None
        assert objectives.__doc__ is not None

    def test_config_has_comments(self):
        """Test YAML config has comments."""
        config_path = Path("configs/hpo/default_hpo_config.yaml")

        with open(config_path) as f:
            content = f.read()

        # Should have comment lines
        assert "#" in content
        assert "Blueprint specification" in content


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
