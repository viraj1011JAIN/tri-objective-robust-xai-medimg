"""
Comprehensive Test Suite for Threshold Tuner (Phase 8.4).

Tests for production-grade threshold optimization for selective prediction
with multi-signal gating.

Coverage Goals:
    - ThresholdConfig validation and configuration
    - TuningResult creation, validation, and serialization
    - ThresholdTuner initialization and validation
    - Grid search and optimization
    - Bootstrap confidence intervals
    - Save/load functionality
    - Utility functions (tune_thresholds_for_dataset, compare_strategies)

Author: GitHub Copilot
Date: 2025
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.validation.threshold_tuner import (
    DEFAULT_BOOTSTRAP_SAMPLES,
    DEFAULT_CONF_MAX,
    DEFAULT_CONF_MIN,
    DEFAULT_CONF_STEP,
    DEFAULT_CONFIDENCE_LEVEL,
    DEFAULT_STAB_MAX,
    DEFAULT_STAB_MIN,
    DEFAULT_STAB_STEP,
    DEFAULT_TARGET_ACCURACY,
    DEFAULT_TARGET_COVERAGE,
    ThresholdConfig,
    ThresholdTuner,
    TuningObjective,
    TuningResult,
    compare_strategies,
    tune_thresholds_for_dataset,
)

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def simple_data():
    """Create simple test data with 100 samples."""
    np.random.seed(42)
    n = 100
    confidence = np.random.rand(n)
    stability = np.random.rand(n)
    # Make top 70% correct, bottom 30% incorrect
    is_correct = np.concatenate([np.ones(70, dtype=bool), np.zeros(30, dtype=bool)])
    np.random.shuffle(is_correct)
    return confidence, stability, is_correct


@pytest.fixture
def perfect_data():
    """Create perfect separation data for testing."""
    n = 100
    # High conf/stab → correct, low conf/stab → incorrect
    confidence = np.concatenate([np.full(50, 0.9), np.full(50, 0.3)])
    stability = np.concatenate([np.full(50, 0.8), np.full(50, 0.2)])
    is_correct = np.concatenate([np.ones(50, dtype=bool), np.zeros(50, dtype=bool)])
    return confidence, stability, is_correct


@pytest.fixture
def default_config():
    """Create default ThresholdConfig."""
    return ThresholdConfig()


@pytest.fixture
def simple_tuner(simple_data):
    """Create simple tuner with default settings."""
    conf, stab, correct = simple_data
    return ThresholdTuner(
        confidence_scores=conf,
        stability_scores=stab,
        is_correct=correct,
        target_coverage=0.9,
    )


@pytest.fixture
def temp_result_file(tmp_path):
    """Create temporary file path for results."""
    return tmp_path / "tuning_result.json"


# ============================================================================
# Test ThresholdConfig
# ============================================================================


class TestThresholdConfig:
    """Test ThresholdConfig dataclass."""

    def test_default_initialization(self):
        """Test ThresholdConfig with default values."""
        config = ThresholdConfig()
        assert config.conf_min == DEFAULT_CONF_MIN
        assert config.conf_max == DEFAULT_CONF_MAX
        assert config.conf_step == DEFAULT_CONF_STEP
        assert config.stab_min == DEFAULT_STAB_MIN
        assert config.stab_max == DEFAULT_STAB_MAX
        assert config.stab_step == DEFAULT_STAB_STEP
        assert config.target_coverage == DEFAULT_TARGET_COVERAGE
        assert config.target_accuracy == DEFAULT_TARGET_ACCURACY
        assert config.bootstrap_samples == DEFAULT_BOOTSTRAP_SAMPLES
        assert config.confidence_level == DEFAULT_CONFIDENCE_LEVEL

    def test_custom_initialization(self):
        """Test ThresholdConfig with custom values."""
        config = ThresholdConfig(
            conf_min=0.6,
            conf_max=0.9,
            conf_step=0.1,
            stab_min=0.5,
            stab_max=0.8,
            stab_step=0.1,
            target_coverage=0.85,
            target_accuracy=0.90,
            bootstrap_samples=500,
            confidence_level=0.90,
        )
        assert config.conf_min == 0.6
        assert config.conf_max == 0.9
        assert config.target_coverage == 0.85
        assert config.bootstrap_samples == 500

    def test_invalid_confidence_range(self):
        """Test validation of invalid confidence range."""
        with pytest.raises(ValueError, match="Invalid confidence range"):
            ThresholdConfig(conf_min=0.8, conf_max=0.6)

    def test_invalid_stability_range(self):
        """Test validation of invalid stability range."""
        with pytest.raises(ValueError, match="Invalid stability range"):
            ThresholdConfig(stab_min=0.9, stab_max=0.4)

    def test_invalid_confidence_step(self):
        """Test validation of invalid confidence step."""
        with pytest.raises(ValueError, match="Invalid confidence step size"):
            ThresholdConfig(conf_step=0.0)

        with pytest.raises(ValueError, match="Invalid confidence step size"):
            ThresholdConfig(conf_min=0.5, conf_max=0.6, conf_step=0.5)

    def test_invalid_stability_step(self):
        """Test validation of invalid stability step."""
        with pytest.raises(ValueError, match="Invalid stability step size"):
            ThresholdConfig(stab_step=-0.1)

    def test_invalid_target_coverage(self):
        """Test validation of invalid target coverage."""
        with pytest.raises(ValueError, match="Invalid target coverage"):
            ThresholdConfig(target_coverage=0.0)

        with pytest.raises(ValueError, match="Invalid target coverage"):
            ThresholdConfig(target_coverage=1.5)

    def test_invalid_target_accuracy(self):
        """Test validation of invalid target accuracy."""
        with pytest.raises(ValueError, match="Invalid target accuracy"):
            ThresholdConfig(target_accuracy=-0.1)

    def test_low_bootstrap_samples_warning(self):
        """Test warning for low bootstrap samples."""
        with pytest.warns(UserWarning, match="Bootstrap samples"):
            ThresholdConfig(bootstrap_samples=50)

    def test_invalid_confidence_level(self):
        """Test validation of invalid confidence level."""
        with pytest.raises(ValueError, match="Invalid confidence level"):
            ThresholdConfig(confidence_level=0.4)

        with pytest.raises(ValueError, match="Invalid confidence level"):
            ThresholdConfig(confidence_level=1.0)

    def test_get_conf_thresholds(self, default_config):
        """Test confidence threshold grid generation."""
        thresholds = default_config.get_conf_thresholds()
        assert len(thresholds) == 10  # 0.5 to 0.95 step 0.05
        assert thresholds[0] == pytest.approx(0.5)
        assert thresholds[-1] == pytest.approx(0.95)

    def test_get_stab_thresholds(self, default_config):
        """Test stability threshold grid generation."""
        thresholds = default_config.get_stab_thresholds()
        assert len(thresholds) == 11  # 0.4 to 0.9 step 0.05
        assert thresholds[0] == pytest.approx(0.4)
        assert thresholds[-1] == pytest.approx(0.9)

    def test_get_search_space_size(self, default_config):
        """Test search space size calculation."""
        size = default_config.get_search_space_size()
        assert size == 110  # 10 conf * 11 stab


# ============================================================================
# Test TuningResult
# ============================================================================


class TestTuningResult:
    """Test TuningResult dataclass."""

    @pytest.fixture
    def valid_result_params(self, default_config):
        """Create valid parameters for TuningResult."""
        grid_df = pd.DataFrame(
            {
                "conf_threshold": [0.5, 0.6],
                "stab_threshold": [0.4, 0.5],
                "accuracy": [0.8, 0.85],
                "coverage": [0.9, 0.88],
            }
        )
        return {
            "optimal_conf_threshold": 0.75,
            "optimal_stab_threshold": 0.65,
            "accuracy": 0.85,
            "coverage": 0.90,
            "precision": 0.87,
            "recall": 0.83,
            "f1_score": 0.85,
            "n_selected": 90,
            "n_total": 100,
            "objective_value": 0.85,
            "confidence_interval": (0.82, 0.88),
            "grid_search_results": grid_df,
            "objective": TuningObjective.MAX_ACCURACY_AT_COVERAGE,
            "config": default_config,
        }

    def test_valid_initialization(self, valid_result_params):
        """Test TuningResult with valid parameters."""
        result = TuningResult(**valid_result_params)
        assert result.optimal_conf_threshold == 0.75
        assert result.accuracy == 0.85
        assert result.n_selected == 90

    def test_invalid_optimal_conf_threshold(self, valid_result_params):
        """Test validation of invalid optimal confidence threshold."""
        valid_result_params["optimal_conf_threshold"] = 1.5
        with pytest.raises(ValueError, match="Invalid optimal confidence"):
            TuningResult(**valid_result_params)

    def test_invalid_optimal_stab_threshold(self, valid_result_params):
        """Test validation of invalid optimal stability threshold."""
        valid_result_params["optimal_stab_threshold"] = -0.1
        with pytest.raises(ValueError, match="Invalid optimal stability"):
            TuningResult(**valid_result_params)

    def test_invalid_accuracy(self, valid_result_params):
        """Test validation of invalid accuracy."""
        valid_result_params["accuracy"] = 1.2
        with pytest.raises(ValueError, match="Invalid accuracy"):
            TuningResult(**valid_result_params)

    def test_invalid_coverage(self, valid_result_params):
        """Test validation of invalid coverage."""
        valid_result_params["coverage"] = -0.1
        with pytest.raises(ValueError, match="Invalid coverage"):
            TuningResult(**valid_result_params)

    def test_invalid_n_selected(self, valid_result_params):
        """Test validation of n_selected > n_total."""
        valid_result_params["n_selected"] = 150
        with pytest.raises(ValueError, match="n_selected.*n_total"):
            TuningResult(**valid_result_params)

    def test_to_dict(self, valid_result_params):
        """Test conversion to dictionary."""
        result = TuningResult(**valid_result_params)
        result_dict = result.to_dict()

        assert isinstance(result_dict, dict)
        assert result_dict["optimal_conf_threshold"] == 0.75
        assert result_dict["objective"] == "max_accuracy_at_coverage"
        assert isinstance(result_dict["grid_search_results"], list)
        assert isinstance(result_dict["config"], dict)

    def test_save_and_load(self, valid_result_params, temp_result_file):
        """Test saving and loading TuningResult."""
        result = TuningResult(**valid_result_params)
        result.save(temp_result_file)

        assert temp_result_file.exists()

        loaded_result = TuningResult.load(temp_result_file)
        assert loaded_result.optimal_conf_threshold == result.optimal_conf_threshold
        assert loaded_result.accuracy == result.accuracy
        assert loaded_result.coverage == result.coverage

    def test_summary(self, valid_result_params):
        """Test summary string generation."""
        result = TuningResult(**valid_result_params)
        summary = result.summary()

        assert "Optimal Thresholds" in summary
        assert "0.75" in summary  # conf threshold
        assert "0.65" in summary  # stab threshold
        assert "85.0%" in summary or "0.850" in summary  # accuracy


# ============================================================================
# Test ThresholdTuner
# ============================================================================


class TestThresholdTuner:
    """Test ThresholdTuner class."""

    def test_initialization(self, simple_data):
        """Test ThresholdTuner initialization."""
        conf, stab, correct = simple_data
        tuner = ThresholdTuner(
            confidence_scores=conf,
            stability_scores=stab,
            is_correct=correct,
            target_coverage=0.9,
        )
        assert tuner.n_samples == 100
        assert tuner.config.target_coverage == 0.9

    def test_initialization_with_custom_config(self, simple_data):
        """Test initialization with custom ThresholdConfig."""
        conf, stab, correct = simple_data
        config = ThresholdConfig(conf_step=0.1, stab_step=0.1)
        tuner = ThresholdTuner(
            confidence_scores=conf,
            stability_scores=stab,
            is_correct=correct,
            target_coverage=0.9,
            config=config,
        )
        assert tuner.config.conf_step == 0.1

    def test_invalid_array_lengths(self, simple_data):
        """Test validation of mismatched array lengths."""
        conf, stab, correct = simple_data
        with pytest.raises(ValueError, match="Length mismatch"):
            ThresholdTuner(
                confidence_scores=conf[:50],
                stability_scores=stab,
                is_correct=correct,
            )

    def test_invalid_confidence_range(self, simple_data):
        """Test validation of invalid confidence score range."""
        conf, stab, correct = simple_data
        invalid_conf = np.concatenate([conf[:90], np.array([1.5] * 10)])
        with pytest.raises(ValueError, match="confidence_scores must be in"):
            ThresholdTuner(
                confidence_scores=invalid_conf,
                stability_scores=stab,
                is_correct=correct,
            )

    def test_invalid_stability_range(self, simple_data):
        """Test validation of invalid stability score range."""
        conf, stab, correct = simple_data
        invalid_stab = np.concatenate([stab[:90], np.array([-0.1] * 10)])
        with pytest.raises(ValueError, match="stability_scores must be in"):
            ThresholdTuner(
                confidence_scores=conf,
                stability_scores=invalid_stab,
                is_correct=correct,
            )

    def test_evaluate_thresholds(self, simple_tuner):
        """Test threshold evaluation."""
        metrics = simple_tuner.evaluate_thresholds(
            conf_threshold=0.5, stab_threshold=0.5
        )
        assert "accuracy" in metrics
        assert "coverage" in metrics
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1" in metrics
        assert "n_selected" in metrics
        assert 0 <= metrics["accuracy"] <= 1
        assert 0 <= metrics["coverage"] <= 1

    def test_evaluate_thresholds_no_selection(self, simple_tuner):
        """Test evaluation when thresholds select no samples."""
        metrics = simple_tuner.evaluate_thresholds(
            conf_threshold=1.0, stab_threshold=1.0
        )
        assert metrics["coverage"] == 0.0
        assert metrics["n_selected"] == 0
        # When no samples selected, accuracy should be 0
        assert metrics["accuracy"] == 0.0

    def test_grid_search(self, simple_tuner):
        """Test grid search execution."""
        conf_thresholds = np.arange(0.5, 0.71, 0.1)
        stab_thresholds = np.arange(0.4, 0.61, 0.1)
        results_df = simple_tuner.grid_search(
            conf_thresholds=conf_thresholds,
            stab_thresholds=stab_thresholds,
        )
        assert isinstance(results_df, pd.DataFrame)
        assert len(results_df) > 0
        assert "conf_threshold" in results_df.columns
        assert "stab_threshold" in results_df.columns
        assert "accuracy" in results_df.columns
        assert "coverage" in results_df.columns

    def test_find_optimal_max_accuracy(self, perfect_data):
        """Test finding optimal thresholds for MAX_ACCURACY_AT_COVERAGE."""
        conf, stab, correct = perfect_data
        tuner = ThresholdTuner(
            confidence_scores=conf,
            stability_scores=stab,
            is_correct=correct,
            target_coverage=0.6,
        )

        conf_thresholds = np.arange(0.3, 0.91, 0.1)
        stab_thresholds = np.arange(0.2, 0.81, 0.1)
        grid_df = tuner.grid_search(
            conf_thresholds=conf_thresholds,
            stab_thresholds=stab_thresholds,
        )

        optimal_conf, optimal_stab, _ = tuner.find_optimal_thresholds(
            objective=TuningObjective.MAX_ACCURACY_AT_COVERAGE,
            grid_results=grid_df,
        )

        assert optimal_conf >= 0.3
        assert optimal_stab >= 0.2

    def test_find_optimal_max_coverage(self, perfect_data):
        """Test finding optimal thresholds for MAX_COVERAGE_AT_ACCURACY."""
        conf, stab, correct = perfect_data
        tuner = ThresholdTuner(
            confidence_scores=conf,
            stability_scores=stab,
            is_correct=correct,
            target_accuracy=0.6,
        )

        conf_thresholds = np.arange(0.3, 0.91, 0.1)
        stab_thresholds = np.arange(0.2, 0.81, 0.1)
        grid_df = tuner.grid_search(
            conf_thresholds=conf_thresholds,
            stab_thresholds=stab_thresholds,
        )

        optimal_conf, optimal_stab, _ = tuner.find_optimal_thresholds(
            objective=TuningObjective.MAX_COVERAGE_AT_ACCURACY,
            grid_results=grid_df,
        )

        # Verify thresholds are in expected range
        assert 0 <= optimal_conf <= 1
        assert 0 <= optimal_stab <= 1

    def test_find_optimal_balanced(self, perfect_data):
        """Test finding optimal thresholds for BALANCED objective."""
        conf, stab, correct = perfect_data
        tuner = ThresholdTuner(
            confidence_scores=conf,
            stability_scores=stab,
            is_correct=correct,
        )

        conf_thresholds = np.arange(0.3, 0.91, 0.1)
        stab_thresholds = np.arange(0.2, 0.81, 0.1)
        grid_df = tuner.grid_search(
            conf_thresholds=conf_thresholds,
            stab_thresholds=stab_thresholds,
        )

        optimal_conf, optimal_stab, _ = tuner.find_optimal_thresholds(
            objective=TuningObjective.BALANCED,
            grid_results=grid_df,
        )

        assert 0 <= optimal_conf <= 1
        assert 0 <= optimal_stab <= 1

    def test_compute_confidence_interval(self, simple_tuner):
        """Test bootstrap confidence interval computation."""
        ci_lower, ci_upper = simple_tuner.compute_confidence_interval(
            conf_threshold=0.5,
            stab_threshold=0.5,
            n_bootstrap=100,  # Small for speed
            confidence_level=0.95,
        )
        assert ci_lower <= ci_upper
        assert 0 <= ci_lower <= 1
        assert 0 <= ci_upper <= 1

    def test_tune_max_accuracy(self, perfect_data):
        """Test full tuning pipeline for MAX_ACCURACY_AT_COVERAGE."""
        conf, stab, correct = perfect_data
        tuner = ThresholdTuner(
            confidence_scores=conf,
            stability_scores=stab,
            is_correct=correct,
            target_coverage=0.6,
        )

        # Update config for smaller bootstrap samples (for speed)
        tuner.config.bootstrap_samples = 50

        result = tuner.tune(
            objective=TuningObjective.MAX_ACCURACY_AT_COVERAGE,
            conf_range=(0.3, 0.9),
            stab_range=(0.2, 0.8),
            conf_step=0.2,
            stab_step=0.2,
            compute_ci=True,
        )

        assert isinstance(result, TuningResult)
        assert result.coverage >= 0.6
        assert result.accuracy > 0
        assert len(result.confidence_interval) == 2

    def test_tune_max_coverage(self, perfect_data):
        """Test full tuning pipeline for MAX_COVERAGE_AT_ACCURACY."""
        conf, stab, correct = perfect_data
        tuner = ThresholdTuner(
            confidence_scores=conf,
            stability_scores=stab,
            is_correct=correct,
            target_accuracy=0.6,
        )

        tuner.config.bootstrap_samples = 50

        result = tuner.tune(
            objective=TuningObjective.MAX_COVERAGE_AT_ACCURACY,
            conf_range=(0.3, 0.9),
            stab_range=(0.2, 0.8),
            conf_step=0.2,
            stab_step=0.2,
            compute_ci=True,
        )

        assert result.accuracy >= 0.6
        assert result.coverage > 0

    def test_tune_balanced(self, perfect_data):
        """Test full tuning pipeline for BALANCED objective."""
        conf, stab, correct = perfect_data
        tuner = ThresholdTuner(
            confidence_scores=conf,
            stability_scores=stab,
            is_correct=correct,
        )

        tuner.config.bootstrap_samples = 50

        result = tuner.tune(
            objective=TuningObjective.BALANCED,
            conf_range=(0.3, 0.9),
            stab_range=(0.2, 0.8),
            conf_step=0.2,
            stab_step=0.2,
            compute_ci=True,
        )

        assert result.accuracy > 0
        assert result.coverage > 0


# ============================================================================
# Test Utility Functions
# ============================================================================


class TestUtilityFunctions:
    """Test utility functions."""

    def test_tune_thresholds_for_dataset(self, simple_data, tmp_path):
        """Test tune_thresholds_for_dataset function."""
        conf, stab, correct = simple_data
        df = pd.DataFrame(
            {
                "confidence": conf,
                "stability": stab,
                "is_correct": correct,
            }
        )

        result = tune_thresholds_for_dataset(
            df=df,
            conf_column="confidence",
            stab_column="stability",
            correct_column="is_correct",
            target_coverage=0.8,
            objective=TuningObjective.MAX_ACCURACY_AT_COVERAGE,
            save_path=tmp_path / "result.json",
        )

        assert isinstance(result, TuningResult)
        assert (tmp_path / "result.json").exists()

    def test_compare_strategies(self, simple_data, tmp_path):
        """Test compare_strategies function."""
        conf, stab, correct = simple_data
        df = pd.DataFrame(
            {
                "confidence": conf,
                "stability": stab,
                "is_correct": correct,
            }
        )

        strategies = {
            "Confidence-Only": ("confidence", "confidence"),
            "Dual-Signal": ("confidence", "stability"),
        }

        results_dict = compare_strategies(
            df=df,
            strategies=strategies,
            correct_column="is_correct",
            target_coverage=0.8,
            save_dir=tmp_path,
        )

        assert isinstance(results_dict, dict)
        assert "Confidence-Only" in results_dict
        assert "Dual-Signal" in results_dict
        assert all(isinstance(r, TuningResult) for r in results_dict.values())

    def test_tune_thresholds_for_dataset_missing_column(self, simple_data):
        """Test error handling for missing column."""
        conf, stab, correct = simple_data
        df = pd.DataFrame(
            {
                "confidence": conf,
                "stability": stab,
            }
        )

        with pytest.raises(KeyError):
            tune_thresholds_for_dataset(
                df=df,
                conf_column="confidence",
                stab_column="stability",
                correct_column="is_correct",  # This column doesn't exist
            )
