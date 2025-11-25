"""
Tests for orthogonality analysis module.

Tests cover:
- Configuration validation
- Result loading and parsing
- Statistical computations
- Report generation
- Error handling

Author: Viraj Pankaj Jain
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.evaluation.orthogonality import (
    ModelResults,
    OrthogonalityAnalyzer,
    OrthogonalityConfig,
    StatisticalTest,
)


@pytest.fixture
def temp_results_dir(tmp_path: Path) -> Path:
    """Create temporary results directory with mock data."""
    results_dir = tmp_path / "results"
    results_dir.mkdir()

    seeds = [42, 123, 456]
    models = {
        "baseline": {
            "clean_accuracy": [0.85, 0.84, 0.86],
            "robust_accuracy": [0.10, 0.12, 0.11],
            "cross_site_auroc": [0.75, 0.74, 0.76],
        },
        "pgd_at": {
            "clean_accuracy": [0.80, 0.79, 0.81],
            "robust_accuracy": [0.65, 0.64, 0.66],
            "cross_site_auroc": [0.74, 0.73, 0.75],
        },
        "trades": {
            "clean_accuracy": [0.82, 0.81, 0.83],
            "robust_accuracy": [0.68, 0.67, 0.69],
            "cross_site_auroc": [0.73, 0.72, 0.74],
        },
    }

    for model_name, metrics in models.items():
        for i, seed in enumerate(seeds):
            model_dir = results_dir / f"{model_name}_seed{seed}"
            model_dir.mkdir()

            metrics_data = {
                "clean_accuracy": metrics["clean_accuracy"][i],
                "robust_accuracy": metrics["robust_accuracy"][i],
                "cross_site_auroc": metrics["cross_site_auroc"][i],
                "epoch": 100,
            }

            with open(model_dir / "metrics.json", "w") as f:
                json.dump(metrics_data, f)

    return results_dir


@pytest.fixture
def valid_config(temp_results_dir: Path, tmp_path: Path) -> OrthogonalityConfig:
    """Create valid configuration."""
    output_dir = tmp_path / "output"
    return OrthogonalityConfig(
        results_dir=temp_results_dir,
        output_dir=output_dir,
        seeds=[42, 123, 456],
        dataset="test_dataset",
    )


class TestOrthogonalityConfig:
    """Tests for OrthogonalityConfig dataclass."""

    def test_valid_config(self, temp_results_dir: Path, tmp_path: Path):
        """Test creating valid configuration."""
        config = OrthogonalityConfig(
            results_dir=temp_results_dir,
            output_dir=tmp_path / "output",
            seeds=[42, 123, 456],
            dataset="isic2018",
        )

        assert config.results_dir.exists()
        assert config.output_dir.exists()  # Created by __post_init__
        assert len(config.seeds) == 3
        assert config.significance_level == 0.05

    def test_invalid_results_dir(self, tmp_path: Path):
        """Test error when results directory doesn't exist."""
        with pytest.raises(FileNotFoundError):
            OrthogonalityConfig(
                results_dir=tmp_path / "nonexistent",
                output_dir=tmp_path / "output",
                seeds=[42],
            )

    def test_invalid_significance_level(self, temp_results_dir: Path, tmp_path: Path):
        """Test error with invalid significance level."""
        with pytest.raises(ValueError, match="significance_level must be in"):
            OrthogonalityConfig(
                results_dir=temp_results_dir,
                output_dir=tmp_path / "output",
                seeds=[42, 123],
                significance_level=1.5,
            )

        with pytest.raises(ValueError, match="significance_level must be in"):
            OrthogonalityConfig(
                results_dir=temp_results_dir,
                output_dir=tmp_path / "output",
                seeds=[42, 123],
                significance_level=0.0,
            )

    def test_insufficient_seeds(self, temp_results_dir: Path, tmp_path: Path):
        """Test error with insufficient seeds for statistics."""
        with pytest.raises(ValueError, match="At least 2 seeds required"):
            OrthogonalityConfig(
                results_dir=temp_results_dir,
                output_dir=tmp_path / "output",
                seeds=[42],
            )


class TestModelResults:
    """Tests for ModelResults dataclass."""

    def test_valid_model_results(self):
        """Test creating valid ModelResults."""
        results = ModelResults(
            model_name="baseline",
            clean_accuracy=[0.85, 0.84, 0.86],
            robust_accuracy=[0.10, 0.12, 0.11],
            cross_site_auroc=[0.75, 0.74, 0.76],
            seeds=[42, 123, 456],
        )

        assert results.model_name == "baseline"
        assert len(results.clean_accuracy) == 3

    def test_mismatched_lengths(self):
        """Test error when metric lists have wrong length."""
        with pytest.raises(ValueError, match="clean_accuracy length"):
            ModelResults(
                model_name="baseline",
                clean_accuracy=[0.85, 0.84],  # Wrong length
                robust_accuracy=[0.10, 0.12, 0.11],
                cross_site_auroc=[0.75, 0.74, 0.76],
                seeds=[42, 123, 456],
            )

    def test_get_mean(self):
        """Test computing mean."""
        results = ModelResults(
            model_name="baseline",
            clean_accuracy=[0.85, 0.84, 0.86],
            robust_accuracy=[0.10, 0.12, 0.11],
            cross_site_auroc=[0.75, 0.74, 0.76],
            seeds=[42, 123, 456],
        )

        assert results.get_mean("clean_accuracy") == pytest.approx(0.85, abs=1e-6)
        assert results.get_mean("robust_accuracy") == pytest.approx(0.11, abs=1e-6)

    def test_get_std(self):
        """Test computing standard deviation."""
        results = ModelResults(
            model_name="baseline",
            clean_accuracy=[0.85, 0.84, 0.86],
            robust_accuracy=[0.10, 0.12, 0.11],
            cross_site_auroc=[0.75, 0.74, 0.76],
            seeds=[42, 123, 456],
        )

        std = results.get_std("clean_accuracy")
        assert std > 0
        assert std == pytest.approx(np.std([0.85, 0.84, 0.86], ddof=1), abs=1e-6)


class TestOrthogonalityAnalyzer:
    """Tests for OrthogonalityAnalyzer."""

    def test_load_model_results(self, valid_config: OrthogonalityConfig):
        """Test loading model results from disk."""
        analyzer = OrthogonalityAnalyzer(valid_config)
        results = analyzer.load_model_results("baseline")

        assert results.model_name == "baseline"
        assert len(results.clean_accuracy) == 3
        assert len(results.robust_accuracy) == 3
        assert len(results.cross_site_auroc) == 3
        assert results.seeds == [42, 123, 456]

    def test_load_missing_model(self, valid_config: OrthogonalityConfig):
        """Test error when model results are missing."""
        analyzer = OrthogonalityAnalyzer(valid_config)

        with pytest.raises(FileNotFoundError, match="Result file not found"):
            analyzer.load_model_results("nonexistent_model")

    def test_compute_statistical_test(self, valid_config: OrthogonalityConfig):
        """Test computing statistical test."""
        analyzer = OrthogonalityAnalyzer(valid_config)

        baseline = ModelResults(
            model_name="baseline",
            clean_accuracy=[0.85, 0.84, 0.86],
            robust_accuracy=[0.10, 0.12, 0.11],
            cross_site_auroc=[0.75, 0.74, 0.76],
            seeds=[42, 123, 456],
        )

        pgd_at = ModelResults(
            model_name="pgd_at",
            clean_accuracy=[0.80, 0.79, 0.81],
            robust_accuracy=[0.65, 0.64, 0.66],
            cross_site_auroc=[0.74, 0.73, 0.75],
            seeds=[42, 123, 456],
        )

        test = analyzer.compute_statistical_test("robust_accuracy", pgd_at, baseline)

        assert test.test_name == "paired_t_test"
        assert test.metric == "robust_accuracy"
        assert test.model_a == "pgd_at"
        assert test.model_b == "baseline"
        assert isinstance(test.p_value, float)
        assert isinstance(test.effect_size, float)
        assert isinstance(test.is_significant, bool)

    def test_create_comparison_table(self, valid_config: OrthogonalityConfig):
        """Test creating comparison table."""
        analyzer = OrthogonalityAnalyzer(valid_config)

        model_results = {
            "baseline": ModelResults(
                model_name="baseline",
                clean_accuracy=[0.85, 0.84, 0.86],
                robust_accuracy=[0.10, 0.12, 0.11],
                cross_site_auroc=[0.75, 0.74, 0.76],
                seeds=[42, 123, 456],
            ),
            "pgd_at": ModelResults(
                model_name="pgd_at",
                clean_accuracy=[0.80, 0.79, 0.81],
                robust_accuracy=[0.65, 0.64, 0.66],
                cross_site_auroc=[0.74, 0.73, 0.75],
                seeds=[42, 123, 456],
            ),
        }

        table = analyzer.create_comparison_table(model_results)

        assert len(table) == 2
        assert "Model" in table.columns
        assert "Clean Acc (%)" in table.columns
        assert "Robust Acc (%)" in table.columns
        assert "Cross-Site AUROC" in table.columns

    def test_determine_orthogonality_confirmed(self, valid_config: OrthogonalityConfig):
        """Test orthogonality confirmation."""
        analyzer = OrthogonalityAnalyzer(valid_config)

        # Create tests showing robustness improvement but NOT generalization
        tests = [
            StatisticalTest(
                test_name="paired_t_test",
                metric="robust_accuracy",
                model_a="pgd_at",
                model_b="baseline",
                statistic=10.0,
                p_value=0.001,
                is_significant=True,
                effect_size=5.0,
                interpretation="PGD-AT has significantly higher robust_accuracy",
            ),
            StatisticalTest(
                test_name="paired_t_test",
                metric="cross_site_auroc",
                model_a="pgd_at",
                model_b="baseline",
                statistic=-0.5,
                p_value=0.60,
                is_significant=False,
                effect_size=-0.2,
                interpretation="No significant difference in cross_site_auroc",
            ),
        ]

        is_orthogonal, summary = analyzer.determine_orthogonality(tests)

        assert is_orthogonal is True
        assert "CONFIRMED" in summary

    def test_determine_orthogonality_rejected(self, valid_config: OrthogonalityConfig):
        """Test orthogonality rejection."""
        analyzer = OrthogonalityAnalyzer(valid_config)

        # Create tests showing NO robustness improvement
        tests = [
            StatisticalTest(
                test_name="paired_t_test",
                metric="robust_accuracy",
                model_a="pgd_at",
                model_b="baseline",
                statistic=1.0,
                p_value=0.40,
                is_significant=False,
                effect_size=0.5,
                interpretation="No significant difference in robust_accuracy",
            ),
        ]

        is_orthogonal, summary = analyzer.determine_orthogonality(tests)

        assert is_orthogonal is False
        assert "NOT CONFIRMED" in summary

    def test_run_analysis(self, valid_config: OrthogonalityConfig):
        """Test running complete analysis."""
        analyzer = OrthogonalityAnalyzer(valid_config)
        results = analyzer.run_analysis()

        assert results.config == valid_config
        assert len(results.model_results) == 3
        assert len(results.statistical_tests) > 0
        assert isinstance(results.is_orthogonal, bool)
        assert len(results.summary) > 0
        assert len(results.comparison_table) == 3

        # Check output files were created
        assert (valid_config.output_dir / "orthogonality_results.json").exists()
        assert (valid_config.output_dir / "comparison_table.csv").exists()

    def test_plot_generation(self, valid_config: OrthogonalityConfig):
        """Test that plots are generated without errors."""
        valid_config.save_figures = True
        valid_config.figure_format = "png"

        analyzer = OrthogonalityAnalyzer(valid_config)
        results = analyzer.run_analysis()

        # Check that plot files exist
        expected_plots = [
            "comparison_clean_accuracy.png",
            "comparison_robust_accuracy.png",
            "comparison_cross_site_auroc.png",
            "orthogonality_scatter.png",
        ]

        for plot_name in expected_plots:
            plot_path = valid_config.output_dir / plot_name
            assert plot_path.exists(), f"Plot not found: {plot_name}"


class TestOrthogonalityResults:
    """Tests for OrthogonalityResults."""

    def test_save_results(self, valid_config: OrthogonalityConfig, tmp_path: Path):
        """Test saving results to disk."""
        analyzer = OrthogonalityAnalyzer(valid_config)
        results = analyzer.run_analysis()

        custom_output = tmp_path / "custom_output"
        custom_output.mkdir()

        results.save(custom_output)

        assert (custom_output / "orthogonality_results.json").exists()
        assert (custom_output / "comparison_table.csv").exists()

        # Check JSON content
        with open(custom_output / "orthogonality_results.json") as f:
            data = json.load(f)
            assert "dataset" in data
            assert "is_orthogonal" in data
            assert "model_results" in data
            assert "statistical_tests" in data


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
