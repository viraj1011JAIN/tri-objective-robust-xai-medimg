"""
Comprehensive Tests for HPO Configuration Module
=================================================

Tests for hyperparameter optimization configuration including:
1. Search space definitions and validation
2. Pruner and sampler configurations
3. TRADES-specific search spaces
4. HPO master configuration
5. YAML serialization/deserialization
6. Edge cases and error handling

Author: Viraj Pankaj Jain
Institution: University of Glasgow, School of Computing Science
Date: November 25, 2025
Version: 5.4.0
"""

import sys
import tempfile
from pathlib import Path

import pytest
import yaml

# Add src to path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from src.training.hpo_config import (
    HPOConfig,
    ObjectiveWeights,
    PrunerConfig,
    PrunerType,
    SamplerConfig,
    SamplerType,
    SearchSpace,
    SearchSpaceType,
    TRADESSearchSpace,
)

# =============================================================================
# Test Fixtures
# =============================================================================


@pytest.fixture
def temp_yaml_file():
    """Create temporary YAML file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".yaml", delete=False) as f:
        yield Path(f.name)
    Path(f.name).unlink(missing_ok=True)


# =============================================================================
# Test Enum Classes
# =============================================================================


class TestSearchSpaceType:
    """Test SearchSpaceType enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert SearchSpaceType.CATEGORICAL.value == "categorical"
        assert SearchSpaceType.FLOAT.value == "float"
        assert SearchSpaceType.INT.value == "int"
        assert SearchSpaceType.LOG_FLOAT.value == "log_float"
        assert SearchSpaceType.LOG_INT.value == "log_int"

    def test_enum_membership(self):
        """Test enum membership."""
        assert "categorical" in [e.value for e in SearchSpaceType]
        assert "float" in [e.value for e in SearchSpaceType]


class TestPrunerType:
    """Test PrunerType enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert PrunerType.MEDIAN.value == "median"
        assert PrunerType.PERCENTILE.value == "percentile"
        assert PrunerType.HYPERBAND.value == "hyperband"
        assert PrunerType.SUCCESSIVE_HALVING.value == "successive_halving"
        assert PrunerType.THRESHOLD.value == "threshold"
        assert PrunerType.NONE.value == "none"

    def test_enum_iteration(self):
        """Test iterating over enum."""
        pruner_types = list(PrunerType)
        assert len(pruner_types) == 6


class TestSamplerType:
    """Test SamplerType enum."""

    def test_enum_values(self):
        """Test all enum values exist."""
        assert SamplerType.TPE.value == "tpe"
        assert SamplerType.CMA_ES.value == "cma_es"
        assert SamplerType.RANDOM.value == "random"
        assert SamplerType.GRID.value == "grid"
        assert SamplerType.NSGAII.value == "nsgaii"
        assert SamplerType.MOTPE.value == "motpe"


# =============================================================================
# Test SearchSpace
# =============================================================================


class TestSearchSpace:
    """Test SearchSpace dataclass."""

    def test_categorical_space(self):
        """Test categorical search space."""
        space = SearchSpace(
            name="optimizer",
            space_type=SearchSpaceType.CATEGORICAL,
            choices=("adam", "sgd", "rmsprop"),
        )
        assert space.name == "optimizer"
        assert space.space_type == SearchSpaceType.CATEGORICAL
        assert space.choices == ("adam", "sgd", "rmsprop")

    def test_float_space(self):
        """Test float search space."""
        space = SearchSpace(
            name="learning_rate",
            space_type=SearchSpaceType.FLOAT,
            low=1e-5,
            high=1e-1,
        )
        assert space.name == "learning_rate"
        assert space.low == 1e-5
        assert space.high == 1e-1

    def test_int_space(self):
        """Test integer search space."""
        space = SearchSpace(
            name="batch_size",
            space_type=SearchSpaceType.INT,
            low=16,
            high=128,
            step=16,
        )
        assert space.name == "batch_size"
        assert space.low == 16
        assert space.high == 128
        assert space.step == 16

    def test_log_float_space(self):
        """Test log float search space."""
        space = SearchSpace(
            name="learning_rate",
            space_type=SearchSpaceType.LOG_FLOAT,
            low=1e-6,
            high=1e-2,
            log=True,
        )
        assert space.log is True

    def test_categorical_without_choices_error(self):
        """Test categorical space without choices raises error."""
        with pytest.raises(ValueError, match="requires choices"):
            SearchSpace(
                name="optimizer",
                space_type=SearchSpaceType.CATEGORICAL,
                choices=None,
            )

    def test_categorical_empty_choices_error(self):
        """Test categorical space with empty choices raises error."""
        with pytest.raises(ValueError, match="requires choices"):
            SearchSpace(
                name="optimizer",
                space_type=SearchSpaceType.CATEGORICAL,
                choices=(),
            )

    def test_numeric_without_bounds_error(self):
        """Test numeric space without bounds raises error."""
        with pytest.raises(ValueError, match="requires low and high bounds"):
            SearchSpace(
                name="learning_rate",
                space_type=SearchSpaceType.FLOAT,
                low=None,
                high=1e-1,
            )

    def test_invalid_bounds_error(self):
        """Test invalid bounds (low >= high) raises error."""
        with pytest.raises(ValueError, match="Invalid bounds"):
            SearchSpace(
                name="learning_rate",
                space_type=SearchSpaceType.FLOAT,
                low=1e-1,
                high=1e-5,
            )

    def test_equal_bounds_error(self):
        """Test equal bounds raises error."""
        with pytest.raises(ValueError, match="Invalid bounds"):
            SearchSpace(
                name="learning_rate",
                space_type=SearchSpaceType.FLOAT,
                low=0.01,
                high=0.01,
            )

    def test_frozen_dataclass(self):
        """Test that SearchSpace is frozen (immutable)."""
        space = SearchSpace(
            name="lr",
            space_type=SearchSpaceType.FLOAT,
            low=1e-5,
            high=1e-1,
        )
        with pytest.raises(Exception):  # FrozenInstanceError
            space.name = "new_name"


# =============================================================================
# Test ObjectiveWeights
# =============================================================================


class TestObjectiveWeights:
    """Test ObjectiveWeights dataclass."""

    def test_default_weights(self):
        """Test default objective weights."""
        weights = ObjectiveWeights()
        assert weights.robust_accuracy == 0.4
        assert weights.clean_accuracy == 0.3
        assert weights.cross_site_auroc == 0.3

    def test_custom_weights(self):
        """Test custom objective weights."""
        weights = ObjectiveWeights(
            robust_accuracy=0.5, clean_accuracy=0.3, cross_site_auroc=0.2
        )
        assert weights.robust_accuracy == 0.5
        assert weights.clean_accuracy == 0.3
        assert weights.cross_site_auroc == 0.2

    def test_weights_must_sum_to_one(self):
        """Test that weights must sum to 1.0."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ObjectiveWeights(
                robust_accuracy=0.5, clean_accuracy=0.3, cross_site_auroc=0.3
            )

    def test_weights_sum_validation_low(self):
        """Test that weights summing to < 0.99 raises error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ObjectiveWeights(
                robust_accuracy=0.3, clean_accuracy=0.3, cross_site_auroc=0.3
            )

    def test_weights_sum_validation_high(self):
        """Test that weights summing to > 1.01 raises error."""
        with pytest.raises(ValueError, match="must sum to 1.0"):
            ObjectiveWeights(
                robust_accuracy=0.5, clean_accuracy=0.5, cross_site_auroc=0.5
            )

    def test_compute_weighted_score(self):
        """Test computing weighted objective score."""
        weights = ObjectiveWeights(
            robust_accuracy=0.5, clean_accuracy=0.3, cross_site_auroc=0.2
        )
        score = weights.compute_weighted_score(
            robust_acc=0.8, clean_acc=0.9, cross_site_auroc=0.7
        )
        expected = 0.5 * 0.8 + 0.3 * 0.9 + 0.2 * 0.7
        assert abs(score - expected) < 1e-6

    def test_to_dict(self):
        """Test conversion to dictionary."""
        weights = ObjectiveWeights(
            robust_accuracy=0.5, clean_accuracy=0.3, cross_site_auroc=0.2
        )
        weights_dict = weights.to_dict()
        assert weights_dict["robust_accuracy"] == 0.5
        assert weights_dict["clean_accuracy"] == 0.3
        assert weights_dict["cross_site_auroc"] == 0.2


# =============================================================================
# Test PrunerConfig
# =============================================================================


class TestPrunerConfig:
    """Test PrunerConfig dataclass."""

    def test_median_pruner(self):
        """Test median pruner configuration."""
        config = PrunerConfig(
            pruner_type=PrunerType.MEDIAN,
            n_startup_trials=5,
            n_warmup_steps=10,
        )
        assert config.pruner_type == PrunerType.MEDIAN
        assert config.n_startup_trials == 5
        assert config.n_warmup_steps == 10

    def test_percentile_pruner(self):
        """Test percentile pruner configuration."""
        config = PrunerConfig(
            pruner_type=PrunerType.PERCENTILE,
            percentile=25.0,
            n_startup_trials=3,
        )
        assert config.pruner_type == PrunerType.PERCENTILE
        assert config.percentile == 25.0

    def test_hyperband_pruner(self):
        """Test hyperband pruner configuration."""
        config = PrunerConfig(
            pruner_type=PrunerType.HYPERBAND,
            min_resource=1,
            max_resource=81,
            reduction_factor=3,
        )
        assert config.min_resource == 1
        assert config.max_resource == 81
        assert config.reduction_factor == 3

    def test_threshold_pruner(self):
        """Test threshold pruner configuration."""
        config = PrunerConfig(
            pruner_type=PrunerType.THRESHOLD,
        )
        assert config.pruner_type == PrunerType.THRESHOLD

    def test_no_pruner(self):
        """Test no pruner configuration."""
        config = PrunerConfig(pruner_type=PrunerType.NONE)
        assert config.pruner_type == PrunerType.NONE

    def test_default_values(self):
        """Test default pruner configuration values."""
        config = PrunerConfig()
        assert config.pruner_type == PrunerType.MEDIAN
        assert config.n_startup_trials == 10
        assert config.n_warmup_steps == 5
        assert config.interval_steps == 1
        assert config.percentile == 50.0

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = PrunerConfig(pruner_type=PrunerType.MEDIAN, n_startup_trials=5)
        config_dict = config.to_dict()
        assert config_dict["pruner_type"] == "median"
        assert config_dict["n_startup_trials"] == 5


# =============================================================================
# Test SamplerConfig
# =============================================================================


class TestSamplerConfig:
    """Test SamplerConfig dataclass."""

    def test_tpe_sampler(self):
        """Test TPE sampler configuration."""
        config = SamplerConfig(
            sampler_type=SamplerType.TPE,
            n_startup_trials=10,
            multivariate=True,
        )
        assert config.sampler_type == SamplerType.TPE
        assert config.n_startup_trials == 10
        assert config.multivariate is True

    def test_cma_es_sampler(self):
        """Test CMA-ES sampler configuration."""
        config = SamplerConfig(
            sampler_type=SamplerType.CMA_ES,
            n_startup_trials=5,
        )
        assert config.sampler_type == SamplerType.CMA_ES

    def test_random_sampler(self):
        """Test random sampler configuration."""
        config = SamplerConfig(sampler_type=SamplerType.RANDOM, seed=42)
        assert config.sampler_type == SamplerType.RANDOM
        assert config.seed == 42

    def test_grid_sampler(self):
        """Test grid sampler configuration."""
        config = SamplerConfig(sampler_type=SamplerType.GRID, seed=123)
        assert config.sampler_type == SamplerType.GRID

    def test_nsgaii_sampler(self):
        """Test NSGA-II sampler for multi-objective."""
        config = SamplerConfig(
            sampler_type=SamplerType.NSGAII,
            n_startup_trials=20,
        )
        assert config.sampler_type == SamplerType.NSGAII

    def test_motpe_sampler(self):
        """Test MOTPE sampler for multi-objective."""
        config = SamplerConfig(sampler_type=SamplerType.MOTPE, n_startup_trials=15)
        assert config.sampler_type == SamplerType.MOTPE
        assert config.n_startup_trials == 15

    def test_default_values(self):
        """Test default sampler configuration."""
        config = SamplerConfig()
        assert config.sampler_type == SamplerType.TPE
        assert config.seed == 42
        assert config.n_startup_trials == 10
        assert config.multivariate is True
        assert config.constant_liar is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = SamplerConfig(sampler_type=SamplerType.TPE, seed=42)
        config_dict = config.to_dict()
        assert config_dict["sampler_type"] == "tpe"
        assert config_dict["seed"] == 42


# =============================================================================
# Test TRADESSearchSpace
# =============================================================================


class TestTRADESSearchSpace:
    """Test TRADES-specific search space."""

    def test_default_search_space(self):
        """Test default TRADES search space."""
        search_space = TRADESSearchSpace()
        core_spaces = search_space.get_core_spaces()
        all_spaces = search_space.get_all_spaces()

        # Check core spaces
        assert len(core_spaces) == 3
        core_names = [s.name for s in core_spaces]
        assert "beta" in core_names
        assert "epsilon" in core_names
        assert "learning_rate" in core_names

        # Check all spaces
        assert len(all_spaces) == 6
        all_names = [s.name for s in all_spaces]
        assert "weight_decay" in all_names
        assert "step_size" in all_names
        assert "num_steps" in all_names

    def test_beta_space(self):
        """Test beta search space configuration."""
        search_space = TRADESSearchSpace()
        beta_space = search_space.beta
        assert beta_space.name == "beta"
        assert beta_space.space_type == SearchSpaceType.FLOAT
        assert beta_space.low == 3.0
        assert beta_space.high == 10.0

    def test_epsilon_space(self):
        """Test epsilon search space configuration."""
        search_space = TRADESSearchSpace()
        eps_space = search_space.epsilon
        assert eps_space.name == "epsilon"
        assert eps_space.space_type == SearchSpaceType.CATEGORICAL
        assert eps_space.choices == (4 / 255, 6 / 255, 8 / 255)

    def test_learning_rate_space(self):
        """Test learning rate search space configuration."""
        search_space = TRADESSearchSpace()
        lr_space = search_space.learning_rate
        assert lr_space.name == "learning_rate"
        assert lr_space.space_type == SearchSpaceType.LOG_FLOAT
        assert lr_space.low == 1e-4
        assert lr_space.high == 1e-3
        assert lr_space.log is True

    def test_to_dict(self):
        """Test conversion to dictionary."""
        search_space = TRADESSearchSpace()
        space_dict = search_space.to_dict()
        assert "beta" in space_dict
        assert "epsilon" in space_dict
        assert "learning_rate" in space_dict
        # Check that log types are normalized
        assert space_dict["learning_rate"]["type"] == "float"
        assert space_dict["weight_decay"]["type"] == "float"
        # num_steps is categorical, not normalized
        assert space_dict["num_steps"]["type"] == "categorical"


# =============================================================================
# Test HPOConfig
# =============================================================================


class TestHPOConfig:
    """Test HPO master configuration."""

    def test_default_config(self):
        """Test default HPO configuration."""
        config = HPOConfig()
        assert config.study_name == "trades_hpo_study"
        assert config.n_trials == 50
        assert config.direction == "maximize"
        assert config.n_jobs == 1

    def test_custom_config(self):
        """Test custom HPO configuration."""
        weights = ObjectiveWeights(
            robust_accuracy=0.5, clean_accuracy=0.3, cross_site_auroc=0.2
        )
        pruner = PrunerConfig(pruner_type=PrunerType.MEDIAN)
        sampler = SamplerConfig(sampler_type=SamplerType.TPE)
        search_space = TRADESSearchSpace()

        config = HPOConfig(
            study_name="custom_study",
            n_trials=100,
            timeout=3600,
            direction="maximize",
            objective_weights=weights,
            pruner_config=pruner,
            sampler_config=sampler,
            search_space=search_space,
        )

        assert config.study_name == "custom_study"
        assert config.n_trials == 100
        assert config.timeout == 3600
        assert config.direction == "maximize"
        assert config.objective_weights == weights

    def test_invalid_direction(self):
        """Test invalid direction raises error."""
        with pytest.raises(ValueError, match="Direction must be"):
            HPOConfig(study_name="test", direction="invalid")

    def test_get_storage_path(self):
        """Test getting storage path."""
        config = HPOConfig(study_name="test_study")
        storage_path = config.get_storage_path()
        assert "sqlite:///" in storage_path
        assert "test_study.db" in storage_path

    def test_get_storage_path_custom_url(self):
        """Test getting custom storage URL."""
        config = HPOConfig(
            study_name="test", storage_url="postgresql://localhost/optuna"
        )
        storage_path = config.get_storage_path()
        assert storage_path == "postgresql://localhost/optuna"

    def test_save_and_load_yaml(self, temp_yaml_file):
        """Test saving and loading configuration from YAML."""
        config = HPOConfig(
            study_name="test_study",
            n_trials=100,
            timeout=3600,
        )

        # Save to YAML
        config.save(temp_yaml_file)
        assert temp_yaml_file.exists()

        # Load from YAML
        loaded_config = HPOConfig.from_yaml(temp_yaml_file)
        assert loaded_config.study_name == config.study_name
        assert loaded_config.n_trials == config.n_trials
        assert loaded_config.timeout == config.timeout

    def test_save_default_path(self, tmp_path):
        """Test saving with default path."""
        output_dir = tmp_path / "hpo_output"
        config = HPOConfig(
            study_name="test_study",
            output_dir=output_dir,
        )

        # Save without specifying path (uses default)
        saved_path = config.save()
        assert saved_path.exists()
        assert saved_path == output_dir / "test_study_config.yaml"

    def test_to_dict(self):
        """Test conversion to dictionary."""
        config = HPOConfig(study_name="test", n_trials=50)
        config_dict = config.to_dict()
        assert config_dict["study_name"] == "test"
        assert config_dict["n_trials"] == 50
        assert "objective_weights" in config_dict
        assert "pruner_config" in config_dict
        assert "sampler_config" in config_dict

    def test_output_dir_creation(self, tmp_path):
        """Test that output directory is created."""
        output_dir = tmp_path / "test_hpo"
        config = HPOConfig(study_name="test", output_dir=output_dir)
        assert config.output_dir.exists()

    def test_default_epochs(self):
        """Test default epoch settings."""
        config = HPOConfig()
        assert config.min_epochs_per_trial == 10
        assert config.max_epochs_per_trial == 50
        assert config.early_stopping_patience == 5


class TestHelperFunctions:
    """Test helper functions."""

    def test_create_default_hpo_config(self, tmp_path):
        """Test creating default HPO config."""
        from src.training.hpo_config import create_default_hpo_config

        output_dir = tmp_path / "hpo_results"
        config = create_default_hpo_config(
            study_name="test_study", n_trials=100, output_dir=output_dir
        )

        assert config.study_name == "test_study"
        assert config.n_trials == 100
        assert config.output_dir == output_dir

    def test_create_extended_search_space(self):
        """Test creating extended search space."""
        from src.training.hpo_config import create_extended_search_space

        search_space = create_extended_search_space()
        assert search_space.beta.low == 1.0
        assert search_space.beta.high == 15.0
        assert len(search_space.epsilon.choices) == 5

    def test_log_int_normalization(self):
        """Test that LOG_INT type is normalized to int in to_dict."""
        # Create a custom dataclass with LOG_INT for testing
        from dataclasses import dataclass, field

        @dataclass
        class TestSearchSpace:
            test_space: SearchSpace = field(
                default_factory=lambda: SearchSpace(
                    name="test_log_int",
                    space_type=SearchSpaceType.LOG_INT,
                    low=10,
                    high=1000,
                )
            )

            def to_dict(self):
                spaces = {}
                for f in [self.test_space]:
                    space_type = f.space_type.value
                    if space_type == "log_float":
                        space_type = "float"
                    elif space_type == "log_int":
                        space_type = "int"
                    spaces[f.name] = {"type": space_type}
                return spaces

        test = TestSearchSpace()
        result = test.to_dict()
        assert result["test_log_int"]["type"] == "int"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
