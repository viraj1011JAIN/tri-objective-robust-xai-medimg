"""
Tests for Pareto Analysis Module.

Phase 9.1: Comprehensive Evaluation Infrastructure
Author: Viraj Jain
MSc Dissertation - University of Glasgow
Date: November 2024

Tests cover:
- ParetoSolution and ParetoFrontier dataclasses
- Dominance checking
- Pareto frontier computation
- Non-dominated sorting
- Knee point detection (multiple methods)
- Hypervolume computation
- Visualization functions
- Trade-off analysis
- Save/load functionality
"""

import json
import tempfile
from pathlib import Path

# Set non-interactive backend before importing pyplot
import matplotlib
import numpy as np
import pytest

matplotlib.use("Agg")

from src.evaluation.pareto_analysis import (
    ParetoFrontier,
    ParetoSolution,
    analyze_tradeoffs,
    compute_hypervolume,
    compute_hypervolume_2d,
    compute_pareto_frontier,
    find_knee_point_angle,
    find_knee_point_curvature,
    find_knee_point_distance,
    find_knee_points,
    get_dominated_solutions,
    is_dominated,
    load_frontier,
    non_dominated_sort,
    plot_parallel_coordinates,
    plot_pareto_2d,
    plot_pareto_3d,
    save_frontier,
    select_best_solution,
)

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def simple_2d_objectives():
    """Simple 2D objectives (minimize both)."""
    return np.array(
        [
            [1, 5],  # Pareto optimal
            [2, 3],  # Pareto optimal
            [3, 2],  # Pareto optimal
            [5, 1],  # Pareto optimal
            [3, 4],  # Dominated by [2, 3]
            [4, 3],  # Dominated by [2, 3] and [3, 2]
        ]
    )


@pytest.fixture
def maximize_2d_objectives():
    """2D objectives where we maximize both."""
    return np.array(
        [
            [0.95, 0.70],  # Good accuracy, moderate robustness
            [0.90, 0.80],  # Balanced
            [0.85, 0.90],  # Good robustness
            [0.80, 0.75],  # Dominated
            [0.92, 0.82],  # Pareto optimal
        ]
    )


@pytest.fixture
def convex_frontier():
    """Generate convex Pareto frontier for knee detection."""
    # Convex curve: y = 1/x for x in [0.1, 1]
    x = np.linspace(0.1, 1.0, 20)
    y = 1.0 / x
    return np.column_stack([x, y])


@pytest.fixture
def three_objective_data():
    """3D objective data."""
    np.random.seed(42)
    n = 30
    objectives = np.random.rand(n, 3)

    # Scale to make interesting Pareto frontier
    objectives[:, 0] *= 0.9
    objectives[:, 1] *= 0.85
    objectives[:, 2] *= 0.95

    return objectives


# ============================================================================
# PARETO SOLUTION TESTS
# ============================================================================


class TestParetoSolution:
    """Tests for ParetoSolution dataclass."""

    def test_creation(self):
        """Test basic creation."""
        sol = ParetoSolution(
            objectives=np.array([0.9, 0.8]),
            index=0,
            metadata={"name": "model_a"},
        )
        assert np.allclose(sol.objectives, [0.9, 0.8])
        assert sol.index == 0
        assert sol.metadata["name"] == "model_a"
        assert sol.is_dominated is False
        assert sol.is_knee is False

    def test_dominates_minimization(self):
        """Test dominance checking for minimization."""
        sol1 = ParetoSolution(objectives=np.array([1, 2]))
        sol2 = ParetoSolution(objectives=np.array([2, 3]))

        # sol1 dominates sol2 (minimize)
        assert sol1.dominates(sol2, minimize=[True, True]) == True
        assert sol2.dominates(sol1, minimize=[True, True]) == False

    def test_dominates_maximization(self):
        """Test dominance checking for maximization."""
        sol1 = ParetoSolution(objectives=np.array([0.9, 0.8]))
        sol2 = ParetoSolution(objectives=np.array([0.8, 0.7]))

        # sol1 dominates sol2 (maximize)
        assert sol1.dominates(sol2, minimize=[False, False]) == True
        assert sol2.dominates(sol1, minimize=[False, False]) == False

    def test_dominates_mixed(self):
        """Test dominance with mixed objectives."""
        sol1 = ParetoSolution(objectives=np.array([0.9, 0.1]))  # High acc, low error
        sol2 = ParetoSolution(
            objectives=np.array([0.8, 0.2])
        )  # Lower acc, higher error

        # Maximize first, minimize second
        assert sol1.dominates(sol2, minimize=[False, True]) == True

    def test_no_dominance(self):
        """Test when neither dominates."""
        sol1 = ParetoSolution(objectives=np.array([0.9, 0.7]))
        sol2 = ParetoSolution(objectives=np.array([0.8, 0.8]))

        # Neither dominates (maximize both)
        assert sol1.dominates(sol2, minimize=[False, False]) == False
        assert sol2.dominates(sol1, minimize=[False, False]) == False

    def test_to_dict(self):
        """Test dictionary conversion."""
        sol = ParetoSolution(
            objectives=np.array([0.9, 0.8]),
            index=1,
            metadata={"name": "test"},
            is_knee=True,
            crowding_distance=0.5,
        )
        d = sol.to_dict()

        assert d["objectives"] == [0.9, 0.8]
        assert d["index"] == 1
        assert d["is_knee"] is True
        assert d["crowding_distance"] == 0.5


class TestParetoFrontier:
    """Tests for ParetoFrontier dataclass."""

    def test_creation(self):
        """Test basic creation."""
        solutions = [
            ParetoSolution(objectives=np.array([0.9, 0.7])),
            ParetoSolution(objectives=np.array([0.8, 0.8])),
        ]
        frontier = ParetoFrontier(
            solutions=solutions,
            objective_names=["Accuracy", "Robustness"],
            minimize=[False, False],
        )

        assert len(frontier) == 2
        assert frontier.objective_names == ["Accuracy", "Robustness"]

    def test_get_objectives_matrix(self):
        """Test objectives matrix extraction."""
        solutions = [
            ParetoSolution(objectives=np.array([0.9, 0.7])),
            ParetoSolution(objectives=np.array([0.8, 0.8])),
        ]
        frontier = ParetoFrontier(solutions=solutions)

        matrix = frontier.get_objectives_matrix()

        assert matrix.shape == (2, 2)
        assert np.allclose(matrix[0], [0.9, 0.7])
        assert np.allclose(matrix[1], [0.8, 0.8])

    def test_to_dict(self):
        """Test dictionary conversion."""
        solutions = [
            ParetoSolution(objectives=np.array([0.9, 0.7]), index=0),
        ]
        frontier = ParetoFrontier(
            solutions=solutions,
            objective_names=["Acc", "Rob"],
            hypervolume=0.5,
        )

        d = frontier.to_dict()

        assert d["n_solutions"] == 1
        assert len(d["solutions"]) == 1
        assert d["hypervolume"] == 0.5

    def test_summary(self):
        """Test summary generation."""
        solutions = [
            ParetoSolution(objectives=np.array([0.9, 0.7])),
            ParetoSolution(objectives=np.array([0.8, 0.8])),
        ]
        frontier = ParetoFrontier(
            solutions=solutions,
            objective_names=["Accuracy", "Robustness"],
        )

        summary = frontier.summary()

        assert "PARETO FRONTIER" in summary
        assert "2" in summary  # Number of solutions


# ============================================================================
# DOMINANCE TESTS
# ============================================================================


class TestDominance:
    """Tests for dominance functions."""

    def test_is_dominated_simple(self, simple_2d_objectives):
        """Test is_dominated with simple data."""
        obj = simple_2d_objectives

        # Point [3, 4] should be dominated
        assert is_dominated(obj[4], obj[[0, 1, 2, 3]], minimize=[True, True]) is True

        # Point [1, 5] should not be dominated
        assert (
            is_dominated(obj[0], obj[[1, 2, 3, 4, 5]], minimize=[True, True]) is False
        )

    def test_is_dominated_maximize(self, maximize_2d_objectives):
        """Test is_dominated with maximization."""
        obj = maximize_2d_objectives

        # Point [0.80, 0.75] should be dominated when maximizing
        assert is_dominated(obj[3], obj[[0, 1, 2, 4]], minimize=[False, False]) is True

    def test_get_dominated_solutions(self, simple_2d_objectives):
        """Test getting dominated solutions."""
        dominated = get_dominated_solutions(simple_2d_objectives, minimize=[True, True])

        # Points 4 and 5 are dominated
        assert 4 in dominated
        assert 5 in dominated
        assert len(dominated) == 2


# ============================================================================
# PARETO FRONTIER COMPUTATION
# ============================================================================


class TestParetoFrontierComputation:
    """Tests for Pareto frontier computation."""

    def test_compute_pareto_frontier_2d(self, simple_2d_objectives):
        """Test 2D Pareto frontier computation."""
        frontier = compute_pareto_frontier(
            simple_2d_objectives,
            minimize=[True, True],
            objective_names=["Obj1", "Obj2"],
        )

        # 4 points are Pareto optimal
        assert len(frontier) == 4

        # Check indices
        pareto_indices = {s.index for s in frontier.solutions}
        assert pareto_indices == {0, 1, 2, 3}

    def test_compute_pareto_frontier_maximize(self, maximize_2d_objectives):
        """Test Pareto frontier with maximization."""
        frontier = compute_pareto_frontier(
            maximize_2d_objectives,
            minimize=[False, False],
            objective_names=["Accuracy", "Robustness"],
        )

        # Check that dominated point is excluded
        pareto_indices = {s.index for s in frontier.solutions}
        assert 3 not in pareto_indices  # [0.80, 0.75] is dominated

    def test_crowding_distance_computed(self, simple_2d_objectives):
        """Test that crowding distances are computed."""
        frontier = compute_pareto_frontier(
            simple_2d_objectives,
            minimize=[True, True],
        )

        # Boundary points should have infinite crowding distance
        crowding = [s.crowding_distance for s in frontier.solutions]
        assert any(np.isinf(c) for c in crowding)

    def test_metadata_preserved(self, simple_2d_objectives):
        """Test that metadata is preserved."""
        metadata = [{"name": f"model_{i}"} for i in range(len(simple_2d_objectives))]

        frontier = compute_pareto_frontier(
            simple_2d_objectives,
            minimize=[True, True],
            metadata_list=metadata,
        )

        for sol in frontier.solutions:
            assert "name" in sol.metadata


class TestNonDominatedSort:
    """Tests for non-dominated sorting."""

    def test_single_front(self, convex_frontier):
        """Test when all points are on one front."""
        fronts = non_dominated_sort(convex_frontier, minimize=[True, True])

        # All points should be in front 0
        assert len(fronts[0]) == len(convex_frontier)

    def test_multiple_fronts(self, simple_2d_objectives):
        """Test with multiple fronts."""
        fronts = non_dominated_sort(simple_2d_objectives, minimize=[True, True])

        # Front 0 should have 4 points
        assert len(fronts[0]) == 4

        # Front 1 should have 2 dominated points
        assert len(fronts[1]) == 2


# ============================================================================
# KNEE POINT DETECTION
# ============================================================================


class TestKneePointDetection:
    """Tests for knee point detection methods."""

    def test_knee_point_angle(self, convex_frontier):
        """Test angle-based knee detection."""
        frontier = compute_pareto_frontier(
            convex_frontier,
            minimize=[True, True],
        )

        knee_idx = find_knee_point_angle(frontier)

        assert 0 <= knee_idx < len(frontier)

    def test_knee_point_distance(self, convex_frontier):
        """Test distance-based knee detection."""
        frontier = compute_pareto_frontier(
            convex_frontier,
            minimize=[True, True],
        )

        knee_idx = find_knee_point_distance(frontier)

        assert 0 <= knee_idx < len(frontier)

    def test_knee_point_curvature(self, convex_frontier):
        """Test curvature-based knee detection."""
        frontier = compute_pareto_frontier(
            convex_frontier,
            minimize=[True, True],
        )

        knee_idx = find_knee_point_curvature(frontier)

        assert 0 <= knee_idx < len(frontier)

    def test_find_knee_points(self, convex_frontier):
        """Test find_knee_points with multiple methods."""
        frontier = compute_pareto_frontier(
            convex_frontier,
            minimize=[True, True],
        )

        knees = find_knee_points(frontier, method="distance", n_knees=1)

        assert len(knees) == 1
        assert frontier.knee_indices == knees

        # Check knee solution is marked
        assert frontier.solutions[knees[0]].is_knee is True

    def test_find_multiple_knees(self, convex_frontier):
        """Test finding multiple knee points."""
        frontier = compute_pareto_frontier(
            convex_frontier,
            minimize=[True, True],
        )

        knees = find_knee_points(frontier, method="angle", n_knees=3)

        assert len(knees) == 3
        assert len(set(knees)) == 3  # All unique

    def test_knee_with_small_frontier(self):
        """Test knee detection with small frontier."""
        objectives = np.array([[1, 2], [2, 1]])
        frontier = compute_pareto_frontier(objectives, minimize=[True, True])

        knee_idx = find_knee_point_angle(frontier)

        # Should return 0 for small frontiers
        assert knee_idx in [0, 1]


# ============================================================================
# HYPERVOLUME
# ============================================================================


class TestHypervolume:
    """Tests for hypervolume computation."""

    def test_hypervolume_2d_simple(self):
        """Test 2D hypervolume with simple case."""
        objectives = np.array(
            [
                [1, 3],
                [2, 2],
                [3, 1],
            ]
        )
        frontier = compute_pareto_frontier(objectives, minimize=[True, True])

        hv = compute_hypervolume_2d(frontier, reference_point=np.array([4, 4]))

        assert hv > 0

    def test_hypervolume_2d_stored(self, simple_2d_objectives):
        """Test that hypervolume is stored in frontier."""
        frontier = compute_pareto_frontier(
            simple_2d_objectives,
            minimize=[True, True],
        )

        hv = compute_hypervolume_2d(frontier)

        assert frontier.hypervolume == hv

    def test_hypervolume_general(self, simple_2d_objectives):
        """Test general hypervolume function."""
        hv = compute_hypervolume(
            simple_2d_objectives[[0, 1, 2, 3]],  # Pareto optimal only
            minimize=[True, True],
        )

        assert hv > 0

    def test_hypervolume_3d(self, three_objective_data):
        """Test hypervolume for 3D (Monte Carlo)."""
        frontier = compute_pareto_frontier(
            three_objective_data,
            minimize=[False, False, False],  # Maximize all
        )

        pareto_obj = frontier.get_objectives_matrix()

        hv = compute_hypervolume(
            pareto_obj,
            minimize=[False, False, False],
        )

        assert hv >= 0


# ============================================================================
# VISUALIZATION
# ============================================================================


class TestVisualization:
    """Tests for visualization functions."""

    def test_plot_pareto_2d(self, simple_2d_objectives, tmp_path):
        """Test 2D Pareto plot."""
        frontier = compute_pareto_frontier(
            simple_2d_objectives,
            minimize=[True, True],
            objective_names=["Obj1", "Obj2"],
        )
        find_knee_points(frontier)

        save_path = tmp_path / "pareto_2d.png"
        fig = plot_pareto_2d(
            frontier=frontier,
            all_solutions=simple_2d_objectives,
            title="Test Pareto",
            save_path=str(save_path),
        )

        assert fig is not None
        assert save_path.exists()

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_pareto_3d(self, three_objective_data, tmp_path):
        """Test 3D Pareto plot."""
        frontier = compute_pareto_frontier(
            three_objective_data,
            minimize=[False, False, False],
            objective_names=["Acc", "Rob", "Interp"],
        )

        save_path = tmp_path / "pareto_3d.png"
        fig = plot_pareto_3d(
            frontier=frontier,
            all_solutions=three_objective_data,
            save_path=str(save_path),
        )

        assert fig is not None
        assert save_path.exists()

        import matplotlib.pyplot as plt

        plt.close(fig)

    def test_plot_parallel_coordinates(self, three_objective_data, tmp_path):
        """Test parallel coordinates plot."""
        frontier = compute_pareto_frontier(
            three_objective_data,
            minimize=[False, False, False],
            objective_names=["Accuracy", "Robustness", "Interpretability"],
        )
        find_knee_points(frontier)

        save_path = tmp_path / "parallel.png"
        fig = plot_parallel_coordinates(
            frontier=frontier,
            save_path=str(save_path),
        )

        assert fig is not None
        assert save_path.exists()

        import matplotlib.pyplot as plt

        plt.close(fig)


# ============================================================================
# TRADE-OFF ANALYSIS
# ============================================================================


class TestTradeoffAnalysis:
    """Tests for trade-off analysis."""

    def test_analyze_tradeoffs(self, simple_2d_objectives):
        """Test trade-off analysis."""
        frontier = compute_pareto_frontier(
            simple_2d_objectives,
            minimize=[True, True],
            objective_names=["Obj1", "Obj2"],
        )

        analysis = analyze_tradeoffs(frontier)

        assert "n_pareto_solutions" in analysis
        assert "objective_ranges" in analysis
        assert "marginal_rates_of_substitution" in analysis

    def test_tradeoffs_include_correlations(self, simple_2d_objectives):
        """Test that correlations are included."""
        frontier = compute_pareto_frontier(
            simple_2d_objectives,
            minimize=[True, True],
            objective_names=["Obj1", "Obj2"],
        )

        analysis = analyze_tradeoffs(frontier)

        if "objective_correlations" in analysis:
            # Pareto objectives should be negatively correlated
            for key, corr in analysis["objective_correlations"].items():
                assert -1 <= corr <= 1


class TestSolutionSelection:
    """Tests for solution selection methods."""

    def test_select_weighted_sum(self, simple_2d_objectives):
        """Test weighted sum selection."""
        frontier = compute_pareto_frontier(
            simple_2d_objectives,
            minimize=[True, True],
        )

        # Equal weights
        idx = select_best_solution(frontier, weights=[0.5, 0.5], method="weighted_sum")

        assert 0 <= idx < len(frontier)

    def test_select_min_max(self, simple_2d_objectives):
        """Test min-max selection."""
        frontier = compute_pareto_frontier(
            simple_2d_objectives,
            minimize=[True, True],
        )

        idx = select_best_solution(frontier, method="min_max")

        assert 0 <= idx < len(frontier)

    def test_select_knee(self, convex_frontier):
        """Test knee-based selection."""
        frontier = compute_pareto_frontier(
            convex_frontier,
            minimize=[True, True],
        )
        find_knee_points(frontier)

        idx = select_best_solution(frontier, method="knee")

        assert idx == frontier.knee_indices[0]


# ============================================================================
# SAVE/LOAD
# ============================================================================


class TestSaveLoad:
    """Tests for save and load functionality."""

    def test_save_frontier(self, simple_2d_objectives, tmp_path):
        """Test saving frontier to file."""
        frontier = compute_pareto_frontier(
            simple_2d_objectives,
            minimize=[True, True],
            objective_names=["Obj1", "Obj2"],
        )

        save_path = tmp_path / "frontier.json"
        save_frontier(frontier, save_path)

        assert save_path.exists()

        # Verify JSON is valid
        with open(save_path) as f:
            data = json.load(f)

        assert "solutions" in data
        assert "objective_names" in data

    def test_load_frontier(self, simple_2d_objectives, tmp_path):
        """Test loading frontier from file."""
        frontier = compute_pareto_frontier(
            simple_2d_objectives,
            minimize=[True, True],
            objective_names=["Obj1", "Obj2"],
        )
        compute_hypervolume_2d(frontier)
        find_knee_points(frontier)

        save_path = tmp_path / "frontier.json"
        save_frontier(frontier, save_path)

        # Load it back
        loaded = load_frontier(save_path)

        assert len(loaded) == len(frontier)
        assert loaded.objective_names == frontier.objective_names
        assert loaded.hypervolume == frontier.hypervolume
        assert loaded.knee_indices == frontier.knee_indices

    def test_roundtrip_preserves_data(self, simple_2d_objectives, tmp_path):
        """Test that save/load roundtrip preserves all data."""
        metadata = [{"model": f"m{i}"} for i in range(len(simple_2d_objectives))]

        frontier = compute_pareto_frontier(
            simple_2d_objectives,
            minimize=[True, True],
            objective_names=["X", "Y"],
            metadata_list=metadata,
        )

        save_path = tmp_path / "frontier.json"
        save_frontier(frontier, save_path)
        loaded = load_frontier(save_path)

        # Compare objectives
        orig_obj = frontier.get_objectives_matrix()
        loaded_obj = loaded.get_objectives_matrix()

        assert np.allclose(orig_obj, loaded_obj)


# ============================================================================
# EDGE CASES
# ============================================================================


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_single_solution(self):
        """Test with single solution."""
        objectives = np.array([[0.9, 0.8]])

        frontier = compute_pareto_frontier(objectives, minimize=[False, False])

        assert len(frontier) == 1

    def test_identical_solutions(self):
        """Test with identical solutions."""
        objectives = np.array(
            [
                [0.5, 0.5],
                [0.5, 0.5],
                [0.5, 0.5],
            ]
        )

        frontier = compute_pareto_frontier(objectives, minimize=[True, True])

        # All are Pareto optimal (none dominates another)
        assert len(frontier) == 3

    def test_all_dominated(self):
        """Test when all but one are dominated."""
        objectives = np.array(
            [
                [1, 1],  # Dominates all others (minimize)
                [2, 3],
                [3, 2],
                [4, 4],
            ]
        )

        frontier = compute_pareto_frontier(objectives, minimize=[True, True])

        assert len(frontier) == 1
        assert frontier.solutions[0].index == 0

    def test_empty_frontier_selection(self):
        """Test selection from empty frontier."""
        frontier = ParetoFrontier(solutions=[])

        with pytest.raises(ValueError):
            select_best_solution(frontier)

    def test_1d_objectives(self):
        """Test with single objective."""
        objectives = np.array([0.9, 0.8, 0.85, 0.95])

        frontier = compute_pareto_frontier(objectives, minimize=[False])

        # Only the maximum is Pareto optimal
        assert len(frontier) == 1


# ============================================================================
# INTEGRATION TESTS
# ============================================================================


class TestIntegration:
    """Integration tests for full Pareto analysis workflow."""

    def test_full_workflow(self, maximize_2d_objectives, tmp_path):
        """Test complete Pareto analysis workflow."""
        # 1. Compute frontier
        frontier = compute_pareto_frontier(
            maximize_2d_objectives,
            minimize=[False, False],
            objective_names=["Accuracy", "Robustness"],
            metadata_list=[
                {"model": f"m{i}"} for i in range(len(maximize_2d_objectives))
            ],
        )

        # 2. Compute hypervolume
        hv = compute_hypervolume_2d(frontier)
        assert hv > 0

        # 3. Find knee points
        knees = find_knee_points(frontier, method="distance", n_knees=1)
        assert len(knees) >= 1

        # 4. Analyze trade-offs
        analysis = analyze_tradeoffs(frontier)
        assert analysis["n_pareto_solutions"] >= 1

        # 5. Select best solution
        best_idx = select_best_solution(frontier, method="weighted_sum")
        assert 0 <= best_idx < len(frontier)

        # 6. Generate plots
        fig = plot_pareto_2d(
            frontier=frontier,
            all_solutions=maximize_2d_objectives,
            save_path=str(tmp_path / "pareto.png"),
        )
        import matplotlib.pyplot as plt

        plt.close(fig)

        # 7. Save and load
        save_frontier(frontier, tmp_path / "frontier.json")
        loaded = load_frontier(tmp_path / "frontier.json")

        assert len(loaded) == len(frontier)

        # 8. Generate summary
        summary = frontier.summary()
        assert "PARETO FRONTIER" in summary

    def test_tri_objective_workflow(self, three_objective_data, tmp_path):
        """Test with three objectives (tri-objective problem)."""
        frontier = compute_pareto_frontier(
            three_objective_data,
            minimize=[False, False, False],
            objective_names=["Accuracy", "Robustness", "Interpretability"],
        )

        assert len(frontier) > 0

        # 3D visualization
        fig = plot_pareto_3d(
            frontier=frontier,
            all_solutions=three_objective_data,
            save_path=str(tmp_path / "pareto_3d.png"),
        )
        import matplotlib.pyplot as plt

        plt.close(fig)

        # Parallel coordinates
        fig = plot_parallel_coordinates(
            frontier=frontier,
            save_path=str(tmp_path / "parallel.png"),
        )
        plt.close(fig)


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-x"])
