"""
Unit Tests for RQ1 Evaluation Module
=====================================

Tests for Phase 9.2 RQ1 evaluation infrastructure.

Phase 9.2: RQ1 Evaluation
Author: Viraj Jain
Date: November 2024
"""

import json
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import matplotlib

matplotlib.use("Agg")  # Use non-interactive backend for tests

import numpy as np
import pytest
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.evaluation.rq1_evaluator import (
    CalibrationResults,
    CrossSiteResults,
    EvaluationConfig,
    HypothesisTestResults,
    ModelCheckpoint,
    RobustnessResults,
    RQ1Evaluator,
    TaskPerformanceResults,
    create_rq1_evaluator,
)
from src.evaluation.rq1_report_generator import (
    RQ1ReportGenerator,
    create_rq1_report_generator,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def dummy_model():
    """Create a dummy model for testing."""
    model = nn.Sequential(
        nn.Conv2d(3, 16, 3, padding=1),
        nn.ReLU(),
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(16, 8),
    )
    return model


@pytest.fixture
def dummy_dataset():
    """Create a dummy dataset."""
    images = torch.randn(50, 3, 64, 64)
    labels = torch.randint(0, 8, (50,))
    dataset = TensorDataset(images, labels)
    return DataLoader(dataset, batch_size=10, shuffle=False)


@pytest.fixture
def model_checkpoints(tmp_path):
    """Create dummy model checkpoints."""
    checkpoints = []

    for model_type in ["baseline", "pgd-at", "tri-objective"]:
        for seed in [42, 43, 44]:
            checkpoint_path = tmp_path / f"{model_type}_seed{seed}.pt"

            # Save dummy state dict
            dummy_state = {"conv.weight": torch.randn(16, 3, 3, 3)}
            torch.save(dummy_state, checkpoint_path)

            checkpoint = ModelCheckpoint(
                name=f"{model_type}_seed{seed}",
                path=checkpoint_path,
                seed=seed,
                model_type=model_type,
            )
            checkpoints.append(checkpoint)

    return checkpoints


@pytest.fixture
def evaluation_config(model_checkpoints, dummy_dataset, tmp_path):
    """Create evaluation configuration."""
    config = EvaluationConfig(
        models=model_checkpoints[:3],  # Use first 3 checkpoints
        datasets={"isic2018_test": dummy_dataset, "isic2019": dummy_dataset},
        source_dataset_name="isic2018_test",
        target_dataset_names=["isic2019"],
        num_classes=8,
        device="cpu",
        batch_size=10,
        output_dir=tmp_path / "results",
        save_intermediate=True,
        verbose=False,
    )
    return config


@pytest.fixture
def dummy_results():
    """Create dummy evaluation results."""
    results = {
        "task_performance": [
            TaskPerformanceResults(
                model_name="baseline_seed42",
                seed=42,
                dataset_name="isic2018_test",
                accuracy=0.85,
                auroc_macro=0.88,
                auroc_weighted=0.89,
                f1_macro=0.84,
                f1_weighted=0.85,
                mcc=0.82,
                auroc_per_class={"mel": 0.90, "nv": 0.88},
                accuracy_ci=(0.83, 0.87),
                auroc_macro_ci=(0.86, 0.90),
            ),
            TaskPerformanceResults(
                model_name="tri-objective_seed42",
                seed=42,
                dataset_name="isic2018_test",
                accuracy=0.82,
                auroc_macro=0.86,
                auroc_weighted=0.87,
                f1_macro=0.81,
                f1_weighted=0.82,
                mcc=0.79,
                auroc_per_class={"mel": 0.88, "nv": 0.85},
            ),
        ],
        "robustness": [
            RobustnessResults(
                model_name="baseline_seed42",
                seed=42,
                attack_name="pgd",
                attack_params={"epsilon": 8 / 255, "num_steps": 40},
                clean_accuracy=0.85,
                robust_accuracy=0.42,
                clean_auroc=0.88,
                robust_auroc=0.75,
                attack_success_rate=0.51,
                num_samples=100,
                time_elapsed=5.2,
            ),
            RobustnessResults(
                model_name="tri-objective_seed42",
                seed=42,
                attack_name="pgd",
                attack_params={"epsilon": 8 / 255, "num_steps": 40},
                clean_accuracy=0.82,
                robust_accuracy=0.74,
                clean_auroc=0.86,
                robust_auroc=0.82,
                attack_success_rate=0.10,
                num_samples=100,
                time_elapsed=5.5,
            ),
        ],
        "cross_site": [
            CrossSiteResults(
                model_name="baseline_seed42",
                seed=42,
                source_dataset="isic2018_test",
                target_dataset="isic2019",
                source_auroc=0.88,
                target_auroc=0.73,
                auroc_drop=15.0,
            ),
            CrossSiteResults(
                model_name="tri-objective_seed42",
                seed=42,
                source_dataset="isic2018_test",
                target_dataset="isic2019",
                source_auroc=0.86,
                target_auroc=0.80,
                auroc_drop=6.0,
            ),
        ],
        "calibration": [
            CalibrationResults(
                model_name="baseline_seed42",
                seed=42,
                dataset_name="isic2018_test",
                condition="clean",
                ece=0.0523,
                mce=0.0987,
                brier_score=0.1245,
                bin_confidences=[0.1, 0.3, 0.5, 0.7, 0.9],
                bin_accuracies=[0.12, 0.32, 0.48, 0.68, 0.87],
                bin_counts=[10, 15, 20, 25, 30],
            ),
        ],
        "hypothesis_tests": [
            HypothesisTestResults(
                hypothesis_name="H1a",
                hypothesis_description="Tri-objective robust accuracy ≥ Baseline + 35pp",
                group1_name="Tri-Objective",
                group2_name="Baseline",
                group1_values=[74.0, 75.0, 74.5],
                group2_values=[42.0, 41.5, 42.5],
                t_statistic=25.4,
                p_value=0.0001,
                significant=True,
                alpha=0.01,
                cohens_d=2.85,
                effect_interpretation="Large",
                ci_lower=30.5,
                ci_upper=34.5,
                ci_contains_zero=False,
                improvement=32.5,
                threshold=35.0,
                hypothesis_supported=False,  # Below threshold
            ),
        ],
    }
    return results


# ============================================================================
# DATA CLASS TESTS
# ============================================================================


class TestModelCheckpoint:
    """Test ModelCheckpoint data class."""

    def test_creation_valid(self, tmp_path):
        """Test creating valid checkpoint."""
        checkpoint_path = tmp_path / "model.pt"
        checkpoint_path.touch()

        checkpoint = ModelCheckpoint(
            name="baseline_seed42",
            path=checkpoint_path,
            seed=42,
            model_type="baseline",
        )

        assert checkpoint.name == "baseline_seed42"
        assert checkpoint.seed == 42
        assert checkpoint.model_type == "baseline"

    def test_creation_invalid_path(self, tmp_path):
        """Test creating checkpoint with invalid path."""
        with pytest.raises(FileNotFoundError):
            ModelCheckpoint(
                name="test",
                path=tmp_path / "nonexistent.pt",
                seed=42,
                model_type="baseline",
            )


class TestEvaluationConfig:
    """Test EvaluationConfig data class."""

    def test_creation_valid(self, model_checkpoints, dummy_dataset, tmp_path):
        """Test creating valid configuration."""
        config = EvaluationConfig(
            models=model_checkpoints[:2],
            datasets={"test": dummy_dataset},
            source_dataset_name="test",
            target_dataset_names=[],
            output_dir=tmp_path,
        )

        assert len(config.models) == 2
        assert "test" in config.datasets
        assert config.output_dir.exists()

    def test_validation_no_models(self, dummy_dataset):
        """Test validation with no models."""
        with pytest.raises(ValueError, match="No models specified"):
            EvaluationConfig(
                models=[],
                datasets={"test": dummy_dataset},
                source_dataset_name="test",
            )

    def test_validation_no_datasets(self, model_checkpoints):
        """Test validation with no datasets."""
        with pytest.raises(ValueError, match="No datasets specified"):
            EvaluationConfig(
                models=model_checkpoints[:1],
                datasets={},
                source_dataset_name="test",
            )

    def test_validation_invalid_source(self, model_checkpoints, dummy_dataset):
        """Test validation with invalid source dataset."""
        with pytest.raises(ValueError, match="Source dataset"):
            EvaluationConfig(
                models=model_checkpoints[:1],
                datasets={"test": dummy_dataset},
                source_dataset_name="nonexistent",
            )


class TestTaskPerformanceResults:
    """Test TaskPerformanceResults data class."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = TaskPerformanceResults(
            model_name="test_model",
            seed=42,
            dataset_name="test_dataset",
            accuracy=0.85,
            auroc_macro=0.88,
            auroc_weighted=0.89,
            f1_macro=0.84,
            f1_weighted=0.85,
            mcc=0.82,
            auroc_per_class={"class1": 0.90},
            accuracy_ci=(0.83, 0.87),
        )

        result_dict = result.to_dict()

        assert result_dict["model_name"] == "test_model"
        assert result_dict["accuracy"] == 0.85
        assert result_dict["accuracy_ci"] == (0.83, 0.87)
        assert "class1" in result_dict["auroc_per_class"]


class TestRobustnessResults:
    """Test RobustnessResults data class."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = RobustnessResults(
            model_name="test_model",
            seed=42,
            attack_name="pgd",
            attack_params={"epsilon": 8 / 255},
            clean_accuracy=0.85,
            robust_accuracy=0.72,
            clean_auroc=0.88,
            robust_auroc=0.80,
            attack_success_rate=0.15,
            num_samples=100,
            time_elapsed=5.2,
        )

        result_dict = result.to_dict()

        assert result_dict["attack_name"] == "pgd"
        assert result_dict["robust_accuracy"] == 0.72
        assert "epsilon" in result_dict["attack_params"]


class TestCrossSiteResults:
    """Test CrossSiteResults data class."""

    def test_to_dict(self):
        """Test converting to dictionary."""
        result = CrossSiteResults(
            model_name="test_model",
            seed=42,
            source_dataset="source",
            target_dataset="target",
            source_auroc=0.88,
            target_auroc=0.75,
            auroc_drop=13.0,
            cka_similarity={"layer1": 0.85, "layer2": 0.72},
            mean_cka_similarity=0.785,
            domain_gap=0.215,
        )

        result_dict = result.to_dict()

        assert result_dict["auroc_drop"] == 13.0
        assert result_dict["mean_cka_similarity"] == 0.785
        assert "layer1" in result_dict["cka_similarity"]


# ============================================================================
# RQ1 EVALUATOR TESTS
# ============================================================================


class TestRQ1Evaluator:
    """Test RQ1Evaluator main class."""

    def test_initialization(self, evaluation_config):
        """Test evaluator initialization."""
        evaluator = RQ1Evaluator(evaluation_config)

        assert evaluator.config == evaluation_config
        assert "task_performance" in evaluator.results
        assert "robustness" in evaluator.results
        assert "cross_site" in evaluator.results

    @patch("src.evaluation.rq1_evaluator.RQ1Evaluator._load_model")
    def test_evaluate_task_performance(
        self, mock_load_model, evaluation_config, dummy_model
    ):
        """Test task performance evaluation."""
        mock_load_model.return_value = dummy_model

        evaluator = RQ1Evaluator(evaluation_config)
        evaluator._evaluate_task_performance()

        # Should have results for each model-dataset combination
        assert len(evaluator.results["task_performance"]) > 0

        result = evaluator.results["task_performance"][0]
        assert hasattr(result, "accuracy")
        assert hasattr(result, "auroc_macro")
        assert 0 <= result.accuracy <= 1

    def test_get_task_performance_result(self, evaluation_config):
        """Test getting task performance result."""
        evaluator = RQ1Evaluator(evaluation_config)

        # Add a dummy result
        result = TaskPerformanceResults(
            model_name="test_model",
            seed=42,
            dataset_name="test_dataset",
            accuracy=0.85,
            auroc_macro=0.88,
            auroc_weighted=0.89,
            f1_macro=0.84,
            f1_weighted=0.85,
            mcc=0.82,
        )
        evaluator.results["task_performance"].append(result)

        # Retrieve it
        retrieved = evaluator._get_task_performance_result(
            "test_model", 42, "test_dataset"
        )

        assert retrieved is not None
        assert retrieved.model_name == "test_model"
        assert retrieved.accuracy == 0.85

    def test_get_robust_accuracies(self, evaluation_config):
        """Test getting robust accuracies."""
        evaluator = RQ1Evaluator(evaluation_config)

        # Add dummy results
        for i in range(3):
            result = RobustnessResults(
                model_name=f"baseline_seed{i}",
                seed=i,
                attack_name="pgd",
                attack_params={"epsilon": 8 / 255, "num_steps": 40},
                clean_accuracy=0.85,
                robust_accuracy=0.40 + i * 0.01,
                clean_auroc=0.88,
                robust_auroc=0.75,
                attack_success_rate=0.50,
            )
            evaluator.results["robustness"].append(result)

        # Get accuracies
        accuracies = evaluator._get_robust_accuracies("baseline", "pgd", 8 / 255)

        assert len(accuracies) == 3
        assert all(0 <= acc <= 1 for acc in accuracies)

    def test_get_cross_site_drops(self, evaluation_config):
        """Test getting cross-site drops."""
        evaluator = RQ1Evaluator(evaluation_config)

        # Add dummy results
        for i in range(3):
            result = CrossSiteResults(
                model_name=f"baseline_seed{i}",
                seed=i,
                source_dataset="source",
                target_dataset="target",
                source_auroc=0.88,
                target_auroc=0.75 - i * 0.01,
                auroc_drop=13.0 + i,
            )
            evaluator.results["cross_site"].append(result)

        # Get drops
        drops = evaluator._get_cross_site_drops("baseline")

        assert len(drops) == 3
        assert all(d > 0 for d in drops)

    def test_save_results(self, evaluation_config):
        """Test saving results to disk."""
        evaluator = RQ1Evaluator(evaluation_config)

        # Add dummy result
        result = TaskPerformanceResults(
            model_name="test",
            seed=42,
            dataset_name="test",
            accuracy=0.85,
            auroc_macro=0.88,
            auroc_weighted=0.89,
            f1_macro=0.84,
            f1_weighted=0.85,
            mcc=0.82,
        )
        evaluator.results["task_performance"].append(result)

        # Save
        evaluator._save_results()

        # Check file exists
        output_path = evaluation_config.output_dir / "task_performance.json"
        assert output_path.exists()

        # Load and verify
        with open(output_path) as f:
            loaded = json.load(f)

        assert len(loaded) == 1
        assert loaded[0]["model_name"] == "test"

    def test_generate_summary(self, evaluation_config):
        """Test generating summary statistics."""
        evaluator = RQ1Evaluator(evaluation_config)

        # Add some results
        evaluator.results["task_performance"].append(
            TaskPerformanceResults(
                model_name="test",
                seed=42,
                dataset_name="test",
                accuracy=0.85,
                auroc_macro=0.88,
                auroc_weighted=0.89,
                f1_macro=0.84,
                f1_weighted=0.85,
                mcc=0.82,
            )
        )

        summary = evaluator.generate_summary()

        assert "num_models" in summary
        assert "task_performance_evaluations" in summary
        assert summary["task_performance_evaluations"] == 1


class TestFactoryFunctions:
    """Test factory functions."""

    def test_create_rq1_evaluator(self, model_checkpoints, dummy_dataset, tmp_path):
        """Test create_rq1_evaluator factory function."""
        evaluator = create_rq1_evaluator(
            models=model_checkpoints[:2],
            datasets={"test": dummy_dataset},
            output_dir=tmp_path,
            source_dataset_name="test",
            target_dataset_names=[],  # No target datasets
        )

        assert isinstance(evaluator, RQ1Evaluator)
        assert len(evaluator.config.models) == 2


# ============================================================================
# REPORT GENERATOR TESTS
# ============================================================================


class TestRQ1ReportGenerator:
    """Test RQ1ReportGenerator class."""

    def test_initialization(self, dummy_results, tmp_path):
        """Test report generator initialization."""
        generator = RQ1ReportGenerator(dummy_results, tmp_path)

        assert generator.results == dummy_results
        assert generator.output_dir == tmp_path
        assert generator.tables_dir.exists()
        assert generator.figures_dir.exists()

    def test_generate_task_performance_table(self, dummy_results, tmp_path):
        """Test generating task performance table."""
        generator = RQ1ReportGenerator(dummy_results, tmp_path)
        generator.generate_task_performance_table()

        # Check files exist
        assert (generator.tables_dir / "table1_task_performance.csv").exists()
        assert (generator.tables_dir / "table1_task_performance.tex").exists()
        assert (generator.tables_dir / "table1_task_performance.md").exists()

    def test_generate_robustness_table(self, dummy_results, tmp_path):
        """Test generating robustness table."""
        generator = RQ1ReportGenerator(dummy_results, tmp_path)
        generator.generate_robustness_table()

        assert (generator.tables_dir / "table2_robustness.csv").exists()

    def test_generate_cross_site_table(self, dummy_results, tmp_path):
        """Test generating cross-site table."""
        generator = RQ1ReportGenerator(dummy_results, tmp_path)
        generator.generate_cross_site_table()

        assert (generator.tables_dir / "table3_cross_site.csv").exists()

    def test_generate_calibration_table(self, dummy_results, tmp_path):
        """Test generating calibration table."""
        generator = RQ1ReportGenerator(dummy_results, tmp_path)
        generator.generate_calibration_table()

        assert (generator.tables_dir / "table4_calibration.csv").exists()

    def test_generate_statistical_tests_table(self, dummy_results, tmp_path):
        """Test generating statistical tests table."""
        generator = RQ1ReportGenerator(dummy_results, tmp_path)
        generator.generate_statistical_tests_table()

        assert (generator.tables_dir / "table5_statistical_tests.csv").exists()

    def test_generate_pareto_figures(self, dummy_results, tmp_path):
        """Test generating Pareto figures."""
        generator = RQ1ReportGenerator(dummy_results, tmp_path)

        # Should complete without error
        try:
            generator.generate_pareto_figures()
        except Exception as e:
            pytest.fail(f"generate_pareto_figures raised {type(e).__name__}: {e}")

    @patch("matplotlib.pyplot.savefig")
    @patch("matplotlib.pyplot.close")
    def test_generate_calibration_figures(
        self, mock_close, mock_savefig, dummy_results, tmp_path
    ):
        """Test generating calibration figures."""
        generator = RQ1ReportGenerator(dummy_results, tmp_path)

        # Should complete without error even if no figures generated
        try:
            generator.generate_calibration_figures()
        except Exception as e:
            pytest.fail(f"generate_calibration_figures raised {type(e).__name__}: {e}")

    def test_generate_summary_report(self, dummy_results, tmp_path):
        """Test generating summary report."""
        generator = RQ1ReportGenerator(dummy_results, tmp_path)
        generator.generate_summary_report()

        report_path = tmp_path / "RQ1_EVALUATION_REPORT.md"
        assert report_path.exists()

        # Check content
        content = report_path.read_text()
        assert "# RQ1 Evaluation Report" in content
        assert "Hypothesis Testing Results" in content
        assert "H1a" in content

    def test_format_attack_name(self, dummy_results, tmp_path):
        """Test formatting attack names."""
        generator = RQ1ReportGenerator(dummy_results, tmp_path)

        # FGSM
        name = generator._format_attack_name("fgsm", {"epsilon": 8 / 255})
        assert "FGSM" in name
        assert "8/255" in name

        # PGD
        name = generator._format_attack_name(
            "pgd", {"epsilon": 8 / 255, "num_steps": 40}
        )
        assert "PGD" in name
        assert "40" in name

        # C&W
        name = generator._format_attack_name("cw", {"confidence": 10.0})
        assert "C&W" in name
        assert "10" in name

    def test_get_model_color(self, dummy_results, tmp_path):
        """Test getting model colors."""
        generator = RQ1ReportGenerator(dummy_results, tmp_path)

        assert generator._get_model_color("baseline") == "gray"
        assert generator._get_model_color("pgd-at") == "blue"
        assert generator._get_model_color("trades") == "green"
        assert generator._get_model_color("tri-objective") == "red"

    def test_get_model_marker(self, dummy_results, tmp_path):
        """Test getting model markers."""
        generator = RQ1ReportGenerator(dummy_results, tmp_path)

        assert generator._get_model_marker("baseline") == "o"
        assert generator._get_model_marker("pgd-at") == "s"
        assert generator._get_model_marker("trades") == "^"
        assert generator._get_model_marker("tri-objective") == "*"

    def test_factory_function(self, dummy_results, tmp_path):
        """Test factory function."""
        generator = create_rq1_report_generator(dummy_results, tmp_path)

        assert isinstance(generator, RQ1ReportGenerator)
        assert generator.output_dir == tmp_path


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestRQ1Integration:
    """Integration tests for RQ1 evaluation pipeline."""

    @patch("src.evaluation.rq1_evaluator.RQ1Evaluator._load_model")
    def test_full_pipeline_minimal(
        self, mock_load_model, evaluation_config, dummy_model
    ):
        """Test minimal full pipeline execution."""
        mock_load_model.return_value = dummy_model

        # Create evaluator
        evaluator = RQ1Evaluator(evaluation_config)

        # Run task performance only (fastest)
        evaluator._evaluate_task_performance()

        # Check results
        assert len(evaluator.results["task_performance"]) > 0

        # Generate reports
        generator = RQ1ReportGenerator(evaluator.results, evaluation_config.output_dir)
        generator.generate_task_performance_table()

        # Verify output
        assert (generator.tables_dir / "table1_task_performance.csv").exists()

    def test_end_to_end_with_dummy_data(self, dummy_results, tmp_path):
        """Test end-to-end with dummy data."""
        # Create report generator
        generator = RQ1ReportGenerator(dummy_results, tmp_path)

        # Generate all reports
        generator.generate_all_reports()

        # Verify all outputs exist
        assert (generator.tables_dir / "table1_task_performance.csv").exists()
        assert (generator.tables_dir / "table2_robustness.csv").exists()
        assert (generator.tables_dir / "table3_cross_site.csv").exists()
        assert (generator.tables_dir / "table4_calibration.csv").exists()
        assert (generator.tables_dir / "table5_statistical_tests.csv").exists()
        assert (tmp_path / "RQ1_EVALUATION_REPORT.md").exists()


# ============================================================================
# RUN TESTS
# ============================================================================


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
