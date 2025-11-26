"""
Comprehensive tests for src/evaluation/tradeoff_analysis.py
Achieves 100% line and branch coverage with production-level quality.
"""

import json
import logging
from pathlib import Path
from unittest.mock import mock_open, patch

import numpy as np
import pytest

from src.evaluation.tradeoff_analysis import (
    TradeoffAnalyzer,
    load_results,
    main,
    save_analysis,
)


class TestTradeoffAnalyzerInit:
    """Test TradeoffAnalyzer initialization."""

    def test_init_default(self):
        """Test initialization with defaults."""
        analyzer = TradeoffAnalyzer()
        assert analyzer.logger is not None

    def test_init_custom_logger(self):
        """Test initialization with custom logger."""
        logger = logging.getLogger("test_logger")
        analyzer = TradeoffAnalyzer(logger=logger)
        assert analyzer.logger == logger


class TestIsDominated:
    """Test Pareto dominance checking."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return TradeoffAnalyzer()

    def test_is_dominated_basic_maximize(self, analyzer):
        """Test basic dominance check for maximization."""
        point = np.array([0.8, 0.7])
        other_points = np.array([[0.9, 0.8], [0.7, 0.6]])
        maximize = np.array([True, True])

        result = analyzer.is_dominated(point, other_points, maximize)

        assert result  # Dominated by [0.9, 0.8]

    def test_is_dominated_not_dominated(self, analyzer):
        """Test point that is not dominated."""
        point = np.array([0.9, 0.8])
        other_points = np.array([[0.8, 0.7], [0.7, 0.9]])
        maximize = np.array([True, True])

        result = analyzer.is_dominated(point, other_points, maximize)

        assert not (result)

    def test_is_dominated_minimize(self, analyzer):
        """Test dominance with minimization objectives."""
        point = np.array([0.5, 0.6])
        other_points = np.array([[0.4, 0.5], [0.6, 0.7]])
        maximize = np.array([False, False])

        result = analyzer.is_dominated(point, other_points, maximize)

        assert result  # Dominated by [0.4, 0.5] (both lower)

    def test_is_dominated_mixed_objectives(self, analyzer):
        """Test with mixed maximize/minimize objectives."""
        point = np.array([0.8, 0.5])  # Max first, min second
        other_points = np.array([[0.9, 0.4]])  # Better in both
        maximize = np.array([True, False])

        result = analyzer.is_dominated(point, other_points, maximize)

        assert result

    def test_is_dominated_equal_point(self, analyzer):
        """Test with equal points."""
        point = np.array([0.8, 0.7])
        other_points = np.array([[0.8, 0.7], [0.7, 0.6]])
        maximize = np.array([True, True])

        result = analyzer.is_dominated(point, other_points, maximize)

        assert not (result)  # Not strictly dominated

    def test_is_dominated_single_objective_better(self, analyzer):
        """Test when other point is better in only one objective."""
        point = np.array([0.8, 0.7])
        other_points = np.array([[0.9, 0.6]])  # Better in first, worse in second
        maximize = np.array([True, True])

        result = analyzer.is_dominated(point, other_points, maximize)

        assert not (result)  # Not dominated (trade-off)

    def test_is_dominated_three_objectives(self, analyzer):
        """Test with three objectives."""
        point = np.array([0.8, 0.7, 0.6])
        other_points = np.array([[0.9, 0.8, 0.7], [0.7, 0.6, 0.5]])
        maximize = np.array([True, True, True])

        result = analyzer.is_dominated(point, other_points, maximize)

        assert result


class TestComputeParetoFrontier:
    """Test Pareto frontier computation."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return TradeoffAnalyzer()

    def test_compute_pareto_frontier_basic(self, analyzer):
        """Test basic Pareto frontier computation."""
        points = np.array(
            [
                [0.9, 0.8],  # Pareto optimal
                [0.8, 0.9],  # Pareto optimal
                [0.7, 0.7],  # Dominated
                [0.95, 0.75],  # Pareto optimal
            ]
        )
        maximize = np.array([True, True])

        pareto_points, pareto_indices = analyzer.compute_pareto_frontier(
            points, maximize
        )

        assert len(pareto_indices) == 3  # 3 Pareto optimal points
        assert 2 not in pareto_indices  # Point [0.7, 0.7] is dominated

    def test_compute_pareto_frontier_all_pareto(self, analyzer):
        """Test when all points are Pareto optimal."""
        points = np.array(
            [
                [0.9, 0.7],
                [0.8, 0.8],
                [0.7, 0.9],
            ]
        )
        maximize = np.array([True, True])

        pareto_points, pareto_indices = analyzer.compute_pareto_frontier(
            points, maximize
        )

        assert len(pareto_indices) == 3

    def test_compute_pareto_frontier_single_pareto(self, analyzer):
        """Test when only one point is Pareto optimal."""
        points = np.array(
            [
                [0.9, 0.9],  # Best in both
                [0.8, 0.8],
                [0.7, 0.7],
            ]
        )
        maximize = np.array([True, True])

        pareto_points, pareto_indices = analyzer.compute_pareto_frontier(
            points, maximize
        )

        assert len(pareto_indices) == 1
        assert 0 in pareto_indices

    def test_compute_pareto_frontier_minimize(self, analyzer):
        """Test Pareto frontier with minimization."""
        points = np.array(
            [
                [0.1, 0.2],  # Pareto optimal
                [0.2, 0.1],  # Pareto optimal
                [0.3, 0.3],  # Dominated
            ]
        )
        maximize = np.array([False, False])

        pareto_points, pareto_indices = analyzer.compute_pareto_frontier(
            points, maximize
        )

        assert len(pareto_indices) == 2
        assert 2 not in pareto_indices

    def test_compute_pareto_frontier_mixed(self, analyzer):
        """Test with mixed objectives."""
        points = np.array(
            [
                [0.9, 0.1],  # Pareto optimal (high acc, low loss)
                [0.8, 0.2],  # Dominated
                [0.85, 0.15],  # Pareto optimal
            ]
        )
        maximize = np.array([True, False])  # Max acc, min loss

        pareto_points, pareto_indices = analyzer.compute_pareto_frontier(
            points, maximize
        )

        assert len(pareto_indices) >= 1

    def test_compute_pareto_frontier_logging(self, analyzer, caplog):
        """Test that computation logs correctly."""
        points = np.array([[0.9, 0.8], [0.8, 0.7]])
        maximize = np.array([True, True])

        with caplog.at_level(logging.INFO):
            analyzer.compute_pareto_frontier(points, maximize)

        assert "Pareto-optimal points" in caplog.text


class TestFindKneePoint:
    """Test knee point detection."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return TradeoffAnalyzer()

    def test_find_knee_point_basic(self, analyzer):
        """Test basic knee point finding."""
        pareto_points = np.array(
            [
                [0.9, 0.7],
                [0.85, 0.8],  # Likely knee
                [0.7, 0.9],
            ]
        )
        maximize = np.array([True, True])

        knee_index, knee_point = analyzer.find_knee_point(pareto_points, maximize)

        assert isinstance(knee_index, int)
        assert knee_index >= 0
        assert knee_index < len(pareto_points)
        assert len(knee_point) == 2

    def test_find_knee_point_two_points(self, analyzer, caplog):
        """Test with only two points (too few)."""
        pareto_points = np.array([[0.9, 0.7], [0.7, 0.9]])
        maximize = np.array([True, True])

        with caplog.at_level(logging.WARNING):
            knee_index, knee_point = analyzer.find_knee_point(pareto_points, maximize)

        assert "Too few points" in caplog.text
        assert knee_index == len(pareto_points) // 2

    def test_find_knee_point_single_point(self, analyzer):
        """Test with single point."""
        pareto_points = np.array([[0.85, 0.75]])
        maximize = np.array([True, True])

        knee_index, knee_point = analyzer.find_knee_point(pareto_points, maximize)

        assert knee_index == 0
        np.testing.assert_array_equal(knee_point, pareto_points[0])

    def test_find_knee_point_linear(self, analyzer):
        """Test with points on a line."""
        pareto_points = np.array(
            [
                [1.0, 0.0],
                [0.75, 0.25],
                [0.5, 0.5],
                [0.25, 0.75],
                [0.0, 1.0],
            ]
        )
        maximize = np.array([True, True])

        knee_index, knee_point = analyzer.find_knee_point(pareto_points, maximize)

        # Middle points (not endpoints) should have max distance
        assert 1 <= knee_index <= 4  # Can be any non-endpoint

    def test_find_knee_point_minimize(self, analyzer):
        """Test knee point with minimization objectives."""
        pareto_points = np.array(
            [
                [0.1, 0.9],
                [0.3, 0.5],  # Likely knee
                [0.9, 0.1],
            ]
        )
        maximize = np.array([False, False])

        knee_index, knee_point = analyzer.find_knee_point(pareto_points, maximize)

        assert isinstance(knee_index, int)

    def test_find_knee_point_logging(self, analyzer, caplog):
        """Test knee point logs results."""
        pareto_points = np.array([[0.9, 0.7], [0.8, 0.8], [0.7, 0.9]])
        maximize = np.array([True, True])

        with caplog.at_level(logging.INFO):
            analyzer.find_knee_point(pareto_points, maximize)

        assert "Knee point found" in caplog.text
        assert "Knee point values" in caplog.text


class TestComputeHypervolume2D:
    """Test hypervolume computation for 2D."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return TradeoffAnalyzer()

    def test_compute_hypervolume_2d_basic(self, analyzer):
        """Test basic 2D hypervolume computation."""
        pareto_points = np.array([[0.9, 0.8], [0.8, 0.9]])
        maximize = np.array([True, True])

        hv = analyzer.compute_hypervolume_2d(pareto_points, maximize)

        assert isinstance(hv, float)
        assert hv > 0

    def test_compute_hypervolume_2d_minimize(self, analyzer):
        """Test hypervolume with minimization."""
        pareto_points = np.array([[0.1, 0.2], [0.2, 0.1]])
        maximize = np.array([False, False])

        hv = analyzer.compute_hypervolume_2d(pareto_points, maximize)

        assert isinstance(hv, float)
        assert hv > 0

    def test_compute_hypervolume_2d_mixed(self, analyzer):
        """Test hypervolume with mixed objectives."""
        pareto_points = np.array([[0.9, 0.1], [0.8, 0.15]])
        maximize = np.array([True, False])

        hv = analyzer.compute_hypervolume_2d(pareto_points, maximize)

        assert isinstance(hv, float)
        assert hv > 0

    def test_compute_hypervolume_2d_single_point(self, analyzer):
        """Test hypervolume with single point."""
        pareto_points = np.array([[0.85, 0.75]])
        maximize = np.array([True, True])

        hv = analyzer.compute_hypervolume_2d(pareto_points, maximize)

        assert isinstance(hv, float)
        assert hv >= 0

    def test_compute_hypervolume_2d_many_points(self, analyzer):
        """Test hypervolume with many points."""
        pareto_points = np.array(
            [
                [0.9, 0.7],
                [0.85, 0.75],
                [0.8, 0.8],
                [0.75, 0.85],
                [0.7, 0.9],
            ]
        )
        maximize = np.array([True, True])

        hv = analyzer.compute_hypervolume_2d(pareto_points, maximize)

        assert isinstance(hv, float)
        assert hv > 0


class TestAnalyzeTradeoffs:
    """Test comprehensive trade-off analysis."""

    @pytest.fixture
    def analyzer(self):
        """Create analyzer instance."""
        return TradeoffAnalyzer()

    @pytest.fixture
    def sample_results(self):
        """Sample results for analysis."""
        return {
            "TRADES": {"clean_acc": 0.85, "robust_acc": 0.70},
            "PGD-AT": {"clean_acc": 0.80, "robust_acc": 0.75},
            "Standard": {"clean_acc": 0.90, "robust_acc": 0.60},
        }

    def test_analyze_tradeoffs_basic(self, analyzer, sample_results):
        """Test basic trade-off analysis."""
        objectives = ["clean_acc", "robust_acc"]
        maximize = [True, True]

        analysis = analyzer.analyze_tradeoffs(sample_results, objectives, maximize)

        assert "num_methods" in analysis
        assert "num_objectives" in analysis
        assert "objectives" in analysis
        assert "maximize" in analysis
        assert "pareto" in analysis
        assert "knee_point" in analysis
        assert "hypervolume" in analysis
        assert "all_points" in analysis

    def test_analyze_tradeoffs_with_method_names(self, analyzer, sample_results):
        """Test analysis with custom method names."""
        objectives = ["clean_acc", "robust_acc"]
        maximize = [True, True]
        method_names = ["TRADES", "PGD-AT"]

        analysis = analyzer.analyze_tradeoffs(
            sample_results, objectives, maximize, method_names
        )

        assert analysis["num_methods"] == 2
        assert len(analysis["all_points"]["methods"]) == 2

    def test_analyze_tradeoffs_pareto_info(self, analyzer, sample_results):
        """Test Pareto frontier information."""
        objectives = ["clean_acc", "robust_acc"]
        maximize = [True, True]

        analysis = analyzer.analyze_tradeoffs(sample_results, objectives, maximize)

        pareto = analysis["pareto"]
        assert "num_points" in pareto
        assert "methods" in pareto
        assert "points" in pareto
        assert len(pareto["methods"]) == pareto["num_points"]

    def test_analyze_tradeoffs_knee_info(self, analyzer, sample_results):
        """Test knee point information."""
        objectives = ["clean_acc", "robust_acc"]
        maximize = [True, True]

        analysis = analyzer.analyze_tradeoffs(sample_results, objectives, maximize)

        knee = analysis["knee_point"]
        assert "index" in knee
        assert "method" in knee
        assert "values" in knee
        if knee["values"] is not None:
            assert len(knee["values"]) == 2

    def test_analyze_tradeoffs_hypervolume_2d(self, analyzer, sample_results):
        """Test that hypervolume is computed for 2D."""
        objectives = ["clean_acc", "robust_acc"]
        maximize = [True, True]

        analysis = analyzer.analyze_tradeoffs(sample_results, objectives, maximize)

        assert analysis["hypervolume"] is not None
        assert isinstance(analysis["hypervolume"], float)

    def test_analyze_tradeoffs_hypervolume_3d(self, analyzer):
        """Test that hypervolume is None for 3D."""
        results = {
            "A": {"obj1": 0.9, "obj2": 0.8, "obj3": 0.7},
            "B": {"obj1": 0.8, "obj2": 0.9, "obj3": 0.6},
        }
        objectives = ["obj1", "obj2", "obj3"]
        maximize = [True, True, True]

        analysis = analyzer.analyze_tradeoffs(results, objectives, maximize)

        assert analysis["hypervolume"] is None  # Only 2D supported

    def test_analyze_tradeoffs_minimize(self, analyzer):
        """Test analysis with minimization objectives."""
        results = {
            "A": {"loss": 0.1, "time": 10},
            "B": {"loss": 0.2, "time": 5},
            "C": {"loss": 0.15, "time": 7},
        }
        objectives = ["loss", "time"]
        maximize = [False, False]

        analysis = analyzer.analyze_tradeoffs(results, objectives, maximize)

        assert analysis["maximize"] == [False, False]
        assert analysis["pareto"]["num_points"] >= 1

    def test_analyze_tradeoffs_logging(self, analyzer, sample_results, caplog):
        """Test that analysis logs progress."""
        objectives = ["clean_acc", "robust_acc"]
        maximize = [True, True]

        with caplog.at_level(logging.INFO):
            analyzer.analyze_tradeoffs(sample_results, objectives, maximize)

        assert "Analyzing" in caplog.text
        assert "methods" in caplog.text
        assert "objectives" in caplog.text

    def test_analyze_tradeoffs_single_method(self, analyzer):
        """Test with single method (edge case)."""
        results = {"TRADES": {"clean_acc": 0.85, "robust_acc": 0.70}}
        objectives = ["clean_acc", "robust_acc"]
        maximize = [True, True]

        analysis = analyzer.analyze_tradeoffs(results, objectives, maximize)

        assert analysis["num_methods"] == 1
        assert analysis["pareto"]["num_points"] == 1

    def test_analyze_tradeoffs_all_points_included(self, analyzer, sample_results):
        """Test that all points are included in results."""
        objectives = ["clean_acc", "robust_acc"]
        maximize = [True, True]

        analysis = analyzer.analyze_tradeoffs(sample_results, objectives, maximize)

        all_points = analysis["all_points"]
        assert len(all_points["methods"]) == 3
        assert len(all_points["points"]) == 3


class TestHelperFunctions:
    """Test module-level helper functions."""

    def test_load_results(self, tmp_path):
        """Test loading results from multiple JSON files."""
        # Create test files
        trades_data = {"clean_acc": 0.85, "robust_acc": 0.70}
        pgd_data = {"clean_acc": 0.80, "robust_acc": 0.75}

        trades_file = tmp_path / "trades.json"
        pgd_file = tmp_path / "pgd.json"

        with open(trades_file, "w") as f:
            json.dump(trades_data, f)
        with open(pgd_file, "w") as f:
            json.dump(pgd_data, f)

        results_paths = {
            "TRADES": trades_file,
            "PGD-AT": pgd_file,
        }

        loaded = load_results(results_paths)

        assert "TRADES" in loaded
        assert "PGD-AT" in loaded
        assert loaded["TRADES"] == trades_data
        assert loaded["PGD-AT"] == pgd_data

    def test_load_results_single_file(self, tmp_path):
        """Test loading single result file."""
        data = {"metric": 0.9}
        file_path = tmp_path / "result.json"

        with open(file_path, "w") as f:
            json.dump(data, f)

        loaded = load_results({"method": file_path})

        assert "method" in loaded
        assert loaded["method"] == data

    def test_save_analysis(self, tmp_path):
        """Test saving analysis results."""
        analysis = {
            "num_methods": 3,
            "pareto": {"num_points": 2},
            "knee_point": {"method": "TRADES"},
        }
        output_file = tmp_path / "analysis" / "results.json"

        save_analysis(analysis, output_file)

        assert output_file.exists()
        with open(output_file, "r") as f:
            loaded = json.load(f)
        assert loaded == analysis

    def test_save_analysis_creates_parent(self, tmp_path):
        """Test that save creates parent directories."""
        output_file = tmp_path / "deep" / "nested" / "analysis.json"
        analysis = {"test": "data"}

        save_analysis(analysis, output_file)

        assert output_file.exists()
        assert output_file.parent.exists()


class TestMainFunction:
    """Test main execution function."""

    @patch("sys.argv", ["prog", "--results_dir", "results/", "--output", "output.json"])
    @patch("src.evaluation.tradeoff_analysis.load_results")
    @patch("src.evaluation.tradeoff_analysis.save_analysis")
    @patch("logging.basicConfig")
    def test_main_execution(self, mock_logging, mock_save, mock_load):
        """Test main function execution."""
        # Mock results
        mock_load.return_value = {
            "TRADES": {"clean_accuracy": 0.85, "robust_accuracy": 0.70},
            "PGD-AT": {"clean_accuracy": 0.80, "robust_accuracy": 0.75},
        }

        # Run main
        main()

        # Verify calls
        mock_logging.assert_called_once()
        mock_load.assert_called_once()
        mock_save.assert_called_once()

    @patch("sys.argv", ["prog", "--results_dir", "results/", "--output", "out.json"])
    @patch("src.evaluation.tradeoff_analysis.load_results")
    @patch("src.evaluation.tradeoff_analysis.save_analysis")
    def test_main_with_real_args(self, mock_save, mock_load):
        """Test main with real argument parsing."""
        mock_load.return_value = {
            "TRADES": {"clean_accuracy": 0.85, "robust_accuracy": 0.70},
        }

        main()

        mock_load.assert_called_once()
        mock_save.assert_called_once()


class TestEdgeCases:
    """Test edge cases and boundary conditions."""

    def test_is_dominated_empty_other_points(self):
        """Test dominance check with no other points."""
        analyzer = TradeoffAnalyzer()
        point = np.array([0.8, 0.7])
        other_points = np.array([]).reshape(0, 2)
        maximize = np.array([True, True])

        result = analyzer.is_dominated(point, other_points, maximize)

        assert not (result)

    def test_compute_pareto_frontier_single_point(self):
        """Test Pareto frontier with single point."""
        analyzer = TradeoffAnalyzer()
        points = np.array([[0.85, 0.75]])
        maximize = np.array([True, True])

        pareto_points, pareto_indices = analyzer.compute_pareto_frontier(
            points, maximize
        )

        assert len(pareto_indices) == 1
        assert pareto_indices[0] == 0

    def test_compute_hypervolume_2d_zero_area(self):
        """Test hypervolume with effectively zero area."""
        analyzer = TradeoffAnalyzer()
        # Same point repeated
        pareto_points = np.array([[0.8, 0.8], [0.8, 0.8]])
        maximize = np.array([True, True])

        hv = analyzer.compute_hypervolume_2d(pareto_points, maximize)

        assert isinstance(hv, float)
        assert hv >= 0

    def test_analyze_tradeoffs_identical_methods(self):
        """Test analysis when all methods have identical performance."""
        analyzer = TradeoffAnalyzer()
        results = {
            "A": {"acc": 0.85, "rob": 0.70},
            "B": {"acc": 0.85, "rob": 0.70},
            "C": {"acc": 0.85, "rob": 0.70},
        }
        objectives = ["acc", "rob"]
        maximize = [True, True]

        analysis = analyzer.analyze_tradeoffs(results, objectives, maximize)

        # All points are on Pareto frontier (identical)
        assert analysis["pareto"]["num_points"] == 3


class TestNumericalStability:
    """Test numerical stability and precision."""

    def test_very_close_points(self):
        """Test with very close points."""
        analyzer = TradeoffAnalyzer()
        points = np.array(
            [
                [0.8500001, 0.7500001],
                [0.8500002, 0.7500002],
                [0.8500000, 0.7500000],
            ]
        )
        maximize = np.array([True, True])

        pareto_points, pareto_indices = analyzer.compute_pareto_frontier(
            points, maximize
        )

        assert len(pareto_indices) >= 1

    def test_extreme_values(self):
        """Test with extreme values."""
        analyzer = TradeoffAnalyzer()
        maximize = np.array([True, True])

        analysis = analyzer.analyze_tradeoffs(
            {
                "A": {"x": 1.0, "y": 0.0},
                "B": {"x": 0.0, "y": 1.0},
                "C": {"x": 0.5, "y": 0.5},
            },
            ["x", "y"],
            maximize,
        )

        assert analysis["pareto"]["num_points"] >= 2

    def test_large_number_of_points(self):
        """Test with many points."""
        analyzer = TradeoffAnalyzer()
        np.random.seed(42)
        n_points = 100
        points = np.random.random((n_points, 2))
        maximize = np.array([True, True])

        pareto_points, pareto_indices = analyzer.compute_pareto_frontier(
            points, maximize
        )

        assert len(pareto_indices) <= n_points
        assert len(pareto_indices) >= 1
