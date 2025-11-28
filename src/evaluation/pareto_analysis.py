"""
Pareto Analysis Module for Multi-Objective Optimization Evaluation.

This module provides production-grade Pareto analysis capabilities:
- Pareto frontier computation
- Dominated solution identification
- Knee point selection (multiple methods)
- Hypervolume computation
- 2D and 3D Pareto plots
- Trade-off analysis

Phase 9.1: Comprehensive Evaluation Infrastructure
Author: Viraj Jain
MSc Dissertation - University of Glasgow
Date: November 2024

References
----------
- Deb et al. (2002). "A fast and elitist multiobjective genetic algorithm: NSGA-II"
- Zitzler et al. (2003). "Performance assessment of multiobjective optimizers"
- Branke et al. (2004). "Finding knees in multi-objective optimization"
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure

logger = logging.getLogger(__name__)

# ============================================================================
# DATA CLASSES
# ============================================================================


@dataclass
class ParetoSolution:
    """
    Represents a solution in objective space.

    Attributes
    ----------
    objectives : np.ndarray
        Objective values for this solution.
    index : int
        Original index in the solution set.
    metadata : Dict[str, Any]
        Additional metadata (e.g., hyperparameters, model name).
    is_dominated : bool
        Whether this solution is dominated.
    is_knee : bool
        Whether this solution is a knee point.
    crowding_distance : float
        Crowding distance for diversity preservation.
    """

    objectives: np.ndarray
    index: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)
    is_dominated: bool = False
    is_knee: bool = False
    crowding_distance: float = 0.0

    def __post_init__(self):
        self.objectives = np.asarray(self.objectives)

    def dominates(self, other: "ParetoSolution", minimize: List[bool] = None) -> bool:
        """
        Check if this solution dominates another.

        Parameters
        ----------
        other : ParetoSolution
            Solution to compare against.
        minimize : List[bool], optional
            For each objective, True if minimizing.
            Default: all True (minimize all).

        Returns
        -------
        bool
            True if this solution dominates other.
        """
        if minimize is None:
            minimize = [True] * len(self.objectives)

        # Convert to minimization problem
        self_obj = np.array(
            [obj if mini else -obj for obj, mini in zip(self.objectives, minimize)]
        )
        other_obj = np.array(
            [obj if mini else -obj for obj, mini in zip(other.objectives, minimize)]
        )

        # Dominates if: at least as good in all, strictly better in at least one
        at_least_as_good = np.all(self_obj <= other_obj)
        strictly_better = np.any(self_obj < other_obj)

        return at_least_as_good and strictly_better

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "objectives": self.objectives.tolist(),
            "index": int(self.index),
            "metadata": self.metadata,
            "is_dominated": bool(self.is_dominated),
            "is_knee": bool(self.is_knee),
            "crowding_distance": float(self.crowding_distance),
        }


@dataclass
class ParetoFrontier:
    """
    Represents a Pareto frontier.

    Attributes
    ----------
    solutions : List[ParetoSolution]
        Non-dominated solutions on the frontier.
    objective_names : List[str]
        Names of objectives.
    minimize : List[bool]
        Whether to minimize each objective.
    hypervolume : Optional[float]
        Hypervolume indicator (if computed).
    knee_indices : List[int]
        Indices of knee points.
    """

    solutions: List[ParetoSolution] = field(default_factory=list)
    objective_names: List[str] = field(default_factory=list)
    minimize: List[bool] = field(default_factory=list)
    hypervolume: Optional[float] = None
    knee_indices: List[int] = field(default_factory=list)

    def __len__(self) -> int:
        return len(self.solutions)

    def get_objectives_matrix(self) -> np.ndarray:
        """Get matrix of objective values (n_solutions, n_objectives)."""
        if not self.solutions:
            return np.array([])
        return np.vstack([s.objectives for s in self.solutions])

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary."""
        return {
            "n_solutions": len(self.solutions),
            "solutions": [s.to_dict() for s in self.solutions],
            "objective_names": self.objective_names,
            "minimize": [bool(m) for m in self.minimize],
            "hypervolume": (
                float(self.hypervolume) if self.hypervolume is not None else None
            ),
            "knee_indices": [int(i) for i in self.knee_indices],
        }

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            "=" * 60,
            "PARETO FRONTIER ANALYSIS",
            "=" * 60,
            f"Number of Pareto-optimal solutions: {len(self.solutions)}",
            f"Objectives: {', '.join(self.objective_names) if self.objective_names else 'N/A'}",
        ]

        if self.solutions:
            obj_matrix = self.get_objectives_matrix()
            lines.append("\nObjective Ranges:")
            for i, name in enumerate(
                self.objective_names or [f"Obj_{i}" for i in range(obj_matrix.shape[1])]
            ):
                lines.append(
                    f"  {name}: [{obj_matrix[:, i].min():.4f}, {obj_matrix[:, i].max():.4f}]"
                )

        if self.hypervolume is not None:
            lines.append(f"\nHypervolume: {self.hypervolume:.6f}")

        if self.knee_indices:
            lines.append(f"\nKnee point indices: {self.knee_indices}")

        lines.append("=" * 60)
        return "\n".join(lines)


# ============================================================================
# PARETO DOMINANCE AND FRONTIER
# ============================================================================


def is_dominated(
    solution: np.ndarray, other_solutions: np.ndarray, minimize: List[bool] = None
) -> bool:
    """
    Check if a solution is dominated by any other solution.

    Parameters
    ----------
    solution : np.ndarray
        Solution to check, shape (n_objectives,).
    other_solutions : np.ndarray
        Other solutions, shape (n_solutions, n_objectives).
    minimize : List[bool], optional
        For each objective, True if minimizing.

    Returns
    -------
    bool
        True if solution is dominated.
    """
    solution = np.asarray(solution)
    other_solutions = np.asarray(other_solutions)

    if other_solutions.ndim == 1:
        other_solutions = other_solutions.reshape(1, -1)

    n_objectives = len(solution)

    if minimize is None:
        minimize = [True] * n_objectives

    # Convert to minimization
    solution_min = np.array(
        [obj if mini else -obj for obj, mini in zip(solution, minimize)]
    )

    others_min = np.array(
        [
            [obj if mini else -obj for obj, mini in zip(sol, minimize)]
            for sol in other_solutions
        ]
    )

    # Check dominance by any other solution
    for other in others_min:
        if np.all(other <= solution_min) and np.any(other < solution_min):
            return True

    return False


def compute_pareto_frontier(
    objectives: np.ndarray,
    minimize: List[bool] = None,
    objective_names: List[str] = None,
    metadata_list: List[Dict[str, Any]] = None,
) -> ParetoFrontier:
    """
    Compute the Pareto frontier from a set of solutions.

    Parameters
    ----------
    objectives : np.ndarray
        Objective values, shape (n_solutions, n_objectives).
    minimize : List[bool], optional
        For each objective, True if minimizing. Default: all True.
    objective_names : List[str], optional
        Names of objectives.
    metadata_list : List[Dict], optional
        Metadata for each solution.

    Returns
    -------
    ParetoFrontier
        The computed Pareto frontier.
    """
    objectives = np.asarray(objectives)

    if objectives.ndim == 1:
        objectives = objectives.reshape(-1, 1)

    n_solutions, n_objectives = objectives.shape

    if minimize is None:
        minimize = [True] * n_objectives

    if objective_names is None:
        objective_names = [f"Objective_{i+1}" for i in range(n_objectives)]

    if metadata_list is None:
        metadata_list = [{} for _ in range(n_solutions)]

    # Create solution objects
    solutions = [
        ParetoSolution(
            objectives=objectives[i],
            index=i,
            metadata=metadata_list[i] if i < len(metadata_list) else {},
        )
        for i in range(n_solutions)
    ]

    # Find non-dominated solutions
    pareto_solutions = []

    for i, sol in enumerate(solutions):
        # Check if dominated by any other solution
        other_indices = [j for j in range(n_solutions) if j != i]

        if len(other_indices) == 0:
            # Single solution is always Pareto-optimal
            pareto_solutions.append(sol)
        else:
            other_objs = np.vstack([objectives[j] for j in other_indices])
            if not is_dominated(sol.objectives, other_objs, minimize):
                pareto_solutions.append(sol)
            else:
                sol.is_dominated = True

    # Compute crowding distances for pareto solutions
    if len(pareto_solutions) > 2:
        _compute_crowding_distances(pareto_solutions, minimize)

    return ParetoFrontier(
        solutions=pareto_solutions,
        objective_names=objective_names,
        minimize=minimize,
    )


def _compute_crowding_distances(
    solutions: List[ParetoSolution], minimize: List[bool]
) -> None:
    """
    Compute crowding distances for solutions (in-place).

    Crowding distance measures how isolated a solution is in objective space.
    Used for diversity preservation in multi-objective optimization.
    """
    n_solutions = len(solutions)
    n_objectives = len(solutions[0].objectives)

    if n_solutions <= 2:
        for sol in solutions:
            sol.crowding_distance = float("inf")
        return

    # Initialize distances
    for sol in solutions:
        sol.crowding_distance = 0.0

    # For each objective
    for m in range(n_objectives):
        # Sort by this objective
        sorted_sols = sorted(
            solutions, key=lambda s: s.objectives[m], reverse=not minimize[m]
        )

        # Boundary solutions get infinite distance
        sorted_sols[0].crowding_distance = float("inf")
        sorted_sols[-1].crowding_distance = float("inf")

        # Range for normalization
        obj_range = sorted_sols[-1].objectives[m] - sorted_sols[0].objectives[m]

        if obj_range > 0:
            # Interior solutions
            for i in range(1, n_solutions - 1):
                if np.isfinite(sorted_sols[i].crowding_distance):
                    distance = (
                        sorted_sols[i + 1].objectives[m]
                        - sorted_sols[i - 1].objectives[m]
                    ) / obj_range
                    sorted_sols[i].crowding_distance += distance


def get_dominated_solutions(
    objectives: np.ndarray, minimize: List[bool] = None
) -> List[int]:
    """
    Get indices of dominated solutions.

    Parameters
    ----------
    objectives : np.ndarray
        Objective values, shape (n_solutions, n_objectives).
    minimize : List[bool], optional
        For each objective, True if minimizing.

    Returns
    -------
    List[int]
        Indices of dominated solutions.
    """
    objectives = np.asarray(objectives)
    n_solutions = len(objectives)

    dominated_indices = []

    for i in range(n_solutions):
        other_objs = np.vstack([objectives[j] for j in range(n_solutions) if j != i])
        if is_dominated(objectives[i], other_objs, minimize):
            dominated_indices.append(i)

    return dominated_indices


def non_dominated_sort(
    objectives: np.ndarray, minimize: List[bool] = None
) -> List[List[int]]:
    """
    Perform non-dominated sorting (NSGA-II style).

    Returns solutions grouped by Pareto rank (front 0, front 1, etc.).

    Parameters
    ----------
    objectives : np.ndarray
        Objective values, shape (n_solutions, n_objectives).
    minimize : List[bool], optional
        For each objective, True if minimizing.

    Returns
    -------
    List[List[int]]
        List of fronts, each containing solution indices.
    """
    objectives = np.asarray(objectives)
    n_solutions = len(objectives)
    n_objectives = objectives.shape[1] if objectives.ndim > 1 else 1

    if minimize is None:
        minimize = [True] * n_objectives

    # Convert to minimization
    obj_min = np.array(
        [
            [obj if mini else -obj for obj, mini in zip(sol, minimize)]
            for sol in objectives
        ]
    )

    # For each solution, find who it dominates and who dominates it
    domination_count = np.zeros(n_solutions, dtype=int)
    dominated_sets = [[] for _ in range(n_solutions)]

    for i in range(n_solutions):
        for j in range(n_solutions):
            if i == j:
                continue

            i_dominates_j = np.all(obj_min[i] <= obj_min[j]) and np.any(
                obj_min[i] < obj_min[j]
            )
            j_dominates_i = np.all(obj_min[j] <= obj_min[i]) and np.any(
                obj_min[j] < obj_min[i]
            )

            if i_dominates_j:
                dominated_sets[i].append(j)
            elif j_dominates_i:
                domination_count[i] += 1

    # Build fronts
    fronts = []
    remaining = set(range(n_solutions))

    while remaining:
        # Current front: solutions with domination_count = 0
        current_front = [i for i in remaining if domination_count[i] == 0]

        if not current_front:
            # Shouldn't happen if algorithm is correct
            current_front = list(remaining)

        fronts.append(current_front)

        # Update domination counts for next iteration
        for i in current_front:
            remaining.discard(i)
            for j in dominated_sets[i]:
                if j in remaining:
                    domination_count[j] -= 1

    return fronts


# ============================================================================
# KNEE POINT SELECTION
# ============================================================================


def find_knee_point_angle(frontier: ParetoFrontier) -> int:
    """
    Find knee point using the angle-based method.

    The knee point is the solution that has the maximum angle
    between its neighbors.

    Parameters
    ----------
    frontier : ParetoFrontier
        Pareto frontier.

    Returns
    -------
    int
        Index of knee point in frontier.solutions.
    """
    if len(frontier) <= 2:
        return 0

    objectives = frontier.get_objectives_matrix()
    n_solutions, n_objectives = objectives.shape

    # Normalize objectives
    obj_min = objectives.min(axis=0)
    obj_max = objectives.max(axis=0)
    obj_range = obj_max - obj_min
    obj_range[obj_range == 0] = 1.0

    normalized = (objectives - obj_min) / obj_range

    # Sort by first objective
    sort_idx = np.argsort(normalized[:, 0])
    sorted_norm = normalized[sort_idx]

    # Compute angles
    max_angle = -float("inf")
    knee_idx = 0

    for i in range(1, n_solutions - 1):
        # Vectors to neighbors
        v1 = sorted_norm[i - 1] - sorted_norm[i]
        v2 = sorted_norm[i + 1] - sorted_norm[i]

        # Angle between vectors
        cos_angle = np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-10)
        angle = np.arccos(np.clip(cos_angle, -1, 1))

        if angle > max_angle:
            max_angle = angle
            knee_idx = sort_idx[i]

    return knee_idx


def find_knee_point_distance(frontier: ParetoFrontier) -> int:
    """
    Find knee point using perpendicular distance method.

    The knee point is the solution with maximum perpendicular
    distance from the line connecting extreme points.

    Parameters
    ----------
    frontier : ParetoFrontier
        Pareto frontier.

    Returns
    -------
    int
        Index of knee point in frontier.solutions.
    """
    if len(frontier) <= 2:
        return 0

    objectives = frontier.get_objectives_matrix()
    n_solutions, n_objectives = objectives.shape

    # Normalize
    obj_min = objectives.min(axis=0)
    obj_max = objectives.max(axis=0)
    obj_range = obj_max - obj_min
    obj_range[obj_range == 0] = 1.0

    normalized = (objectives - obj_min) / obj_range

    # Find extreme points (best in each objective)
    extreme_indices = []
    for m in range(n_objectives):
        if frontier.minimize[m] if frontier.minimize else True:
            extreme_indices.append(np.argmin(objectives[:, m]))
        else:
            extreme_indices.append(np.argmax(objectives[:, m]))

    # For 2D: simple perpendicular distance to line
    if n_objectives == 2:
        p1 = normalized[extreme_indices[0]]
        p2 = normalized[extreme_indices[1]]

        max_dist = -float("inf")
        knee_idx = 0

        for i in range(n_solutions):
            if i in extreme_indices:
                continue

            p = normalized[i]

            # Distance from point to line
            # d = |cross(p2-p1, p1-p)| / |p2-p1|
            v = p2 - p1
            w = p1 - p

            # 2D cross product magnitude
            cross = abs(v[0] * w[1] - v[1] * w[0])
            line_len = np.linalg.norm(v) + 1e-10

            dist = cross / line_len

            if dist > max_dist:
                max_dist = dist
                knee_idx = i

        return knee_idx
    else:
        # For higher dimensions, use distance to hyperplane
        # Simplified: use max crowding distance
        max_crowd = -float("inf")
        knee_idx = 0

        for i, sol in enumerate(frontier.solutions):
            if np.isfinite(sol.crowding_distance) and sol.crowding_distance > max_crowd:
                max_crowd = sol.crowding_distance
                knee_idx = i

        return knee_idx


def find_knee_point_curvature(frontier: ParetoFrontier) -> int:
    """
    Find knee point using curvature estimation.

    The knee point has maximum local curvature.

    Parameters
    ----------
    frontier : ParetoFrontier
        Pareto frontier.

    Returns
    -------
    int
        Index of knee point.
    """
    if len(frontier) <= 2:
        return 0

    objectives = frontier.get_objectives_matrix()
    n_solutions = len(objectives)

    # Normalize
    obj_min = objectives.min(axis=0)
    obj_max = objectives.max(axis=0)
    obj_range = obj_max - obj_min
    obj_range[obj_range == 0] = 1.0

    normalized = (objectives - obj_min) / obj_range

    # Sort by first objective
    sort_idx = np.argsort(normalized[:, 0])
    sorted_norm = normalized[sort_idx]

    # Estimate curvature using finite differences
    max_curvature = -float("inf")
    knee_idx = 0

    for i in range(1, n_solutions - 1):
        # First derivatives (finite difference)
        dx = sorted_norm[i + 1, 0] - sorted_norm[i - 1, 0]
        dy = (
            sorted_norm[i + 1, 1] - sorted_norm[i - 1, 1]
            if sorted_norm.shape[1] > 1
            else 0
        )

        # Second derivatives
        ddx = sorted_norm[i + 1, 0] - 2 * sorted_norm[i, 0] + sorted_norm[i - 1, 0]
        ddy = (
            sorted_norm[i + 1, 1] - 2 * sorted_norm[i, 1] + sorted_norm[i - 1, 1]
            if sorted_norm.shape[1] > 1
            else 0
        )

        # Curvature formula: |x'y'' - y'x''| / (x'^2 + y'^2)^(3/2)
        numerator = abs(dx * ddy - dy * ddx)
        denominator = (dx**2 + dy**2) ** 1.5 + 1e-10

        curvature = numerator / denominator

        if curvature > max_curvature:
            max_curvature = curvature
            knee_idx = sort_idx[i]

    return knee_idx


def find_knee_points(
    frontier: ParetoFrontier, method: str = "angle", n_knees: int = 1
) -> List[int]:
    """
    Find knee points in the Pareto frontier.

    Parameters
    ----------
    frontier : ParetoFrontier
        Pareto frontier.
    method : str, optional (default='angle')
        Method: 'angle', 'distance', or 'curvature'.
    n_knees : int, optional (default=1)
        Number of knee points to find.

    Returns
    -------
    List[int]
        Indices of knee points in frontier.solutions.
    """
    if len(frontier) == 0:
        return []

    if len(frontier) <= n_knees:
        return list(range(len(frontier)))

    if method == "angle":
        knee_fn = find_knee_point_angle
    elif method == "distance":
        knee_fn = find_knee_point_distance
    elif method == "curvature":
        knee_fn = find_knee_point_curvature
    else:
        raise ValueError(f"Unknown method: {method}")

    # Find primary knee
    knee_idx = knee_fn(frontier)
    knees = [knee_idx]

    # For multiple knees, iteratively find more
    # (simplified: add solutions with highest crowding distance)
    if n_knees > 1 and len(frontier) > 1:
        crowding = [
            (i, sol.crowding_distance) for i, sol in enumerate(frontier.solutions)
        ]
        crowding = sorted(
            crowding, key=lambda x: -x[1] if np.isfinite(x[1]) else float("-inf")
        )

        for idx, _ in crowding:
            if idx not in knees:
                knees.append(idx)
            if len(knees) >= n_knees:
                break

    # Mark knee solutions
    for i, sol in enumerate(frontier.solutions):
        sol.is_knee = i in knees

    frontier.knee_indices = knees

    return knees


# ============================================================================
# HYPERVOLUME
# ============================================================================


def compute_hypervolume_2d(
    frontier: ParetoFrontier, reference_point: np.ndarray = None
) -> float:
    """
    Compute hypervolume indicator for 2D Pareto frontiers.

    Parameters
    ----------
    frontier : ParetoFrontier
        2D Pareto frontier.
    reference_point : np.ndarray, optional
        Reference point (nadir point). If None, uses worst + 10%.

    Returns
    -------
    float
        Hypervolume indicator.
    """
    if len(frontier) == 0:
        return 0.0

    objectives = frontier.get_objectives_matrix()

    if objectives.shape[1] != 2:
        logger.warning("compute_hypervolume_2d only supports 2 objectives")
        return 0.0

    # Convert to minimization
    if frontier.minimize:
        for i, mini in enumerate(frontier.minimize):
            if not mini:
                objectives[:, i] = -objectives[:, i]

    # Set reference point
    if reference_point is None:
        worst = objectives.max(axis=0)
        reference_point = worst + 0.1 * np.abs(worst)

    # Sort by first objective
    sorted_idx = np.argsort(objectives[:, 0])
    sorted_obj = objectives[sorted_idx]

    # Compute hypervolume
    hypervolume = 0.0
    prev_y = reference_point[1]

    for i in range(len(sorted_obj)):
        x = sorted_obj[i, 0]
        y = sorted_obj[i, 1]

        if y < prev_y:
            width = reference_point[0] - x
            height = prev_y - y
            hypervolume += width * height
            prev_y = y

    frontier.hypervolume = hypervolume

    return hypervolume


def compute_hypervolume(
    objectives: np.ndarray,
    reference_point: np.ndarray = None,
    minimize: List[bool] = None,
) -> float:
    """
    Compute hypervolume indicator (general case).

    For 2D uses efficient algorithm, for higher dimensions uses
    Monte Carlo estimation.

    Parameters
    ----------
    objectives : np.ndarray
        Objective values of Pareto front, shape (n_solutions, n_objectives).
    reference_point : np.ndarray, optional
        Reference point.
    minimize : List[bool], optional
        Whether to minimize each objective.

    Returns
    -------
    float
        Hypervolume indicator.
    """
    objectives = np.asarray(objectives)

    if objectives.ndim == 1:
        objectives = objectives.reshape(-1, 1)

    n_objectives = objectives.shape[1]

    if minimize is None:
        minimize = [True] * n_objectives

    # Create frontier for 2D case
    if n_objectives == 2:
        frontier = compute_pareto_frontier(objectives, minimize=minimize)
        return compute_hypervolume_2d(frontier, reference_point)

    # For higher dimensions, use Monte Carlo estimation
    return _monte_carlo_hypervolume(objectives, reference_point, minimize)


def _monte_carlo_hypervolume(
    objectives: np.ndarray,
    reference_point: np.ndarray = None,
    minimize: List[bool] = None,
    n_samples: int = 100000,
) -> float:
    """Monte Carlo hypervolume estimation for n>2 objectives."""
    n_solutions, n_objectives = objectives.shape

    # Convert to minimization
    obj_min = objectives.copy()
    if minimize:
        for i, mini in enumerate(minimize):
            if not mini:
                obj_min[:, i] = -obj_min[:, i]

    # Set reference point
    if reference_point is None:
        worst = obj_min.max(axis=0)
        reference_point = worst + 0.1 * np.abs(worst)

    best = obj_min.min(axis=0)

    # Volume of bounding box
    box_volume = np.prod(reference_point - best)

    # Monte Carlo sampling
    rng = np.random.RandomState(42)
    samples = rng.uniform(best, reference_point, size=(n_samples, n_objectives))

    # Count dominated samples
    dominated_count = 0
    for sample in samples:
        # Check if sample is dominated by any solution
        for sol in obj_min:
            if np.all(sol <= sample):
                dominated_count += 1
                break

    # Hypervolume estimate
    hypervolume = box_volume * dominated_count / n_samples

    return hypervolume


# ============================================================================
# VISUALIZATION
# ============================================================================


def plot_pareto_2d(
    frontier: ParetoFrontier,
    all_solutions: np.ndarray = None,
    title: str = "Pareto Frontier",
    xlabel: str = None,
    ylabel: str = None,
    show_knee: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8),
) -> Figure:
    """
    Plot 2D Pareto frontier.

    Parameters
    ----------
    frontier : ParetoFrontier
        Pareto frontier to plot.
    all_solutions : np.ndarray, optional
        All solutions (including dominated) for reference.
    title : str, optional
        Plot title.
    xlabel : str, optional
        X-axis label.
    ylabel : str, optional
        Y-axis label.
    show_knee : bool, optional
        Whether to highlight knee points.
    save_path : str, optional
        Path to save figure.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    # Plot all solutions if provided
    if all_solutions is not None:
        all_solutions = np.asarray(all_solutions)
        ax.scatter(
            all_solutions[:, 0],
            all_solutions[:, 1],
            c="lightgray",
            alpha=0.5,
            s=30,
            label="Dominated",
        )

    # Plot Pareto frontier
    if len(frontier) > 0:
        pareto_obj = frontier.get_objectives_matrix()

        # Sort for line plot
        sort_idx = np.argsort(pareto_obj[:, 0])
        sorted_obj = pareto_obj[sort_idx]

        ax.plot(sorted_obj[:, 0], sorted_obj[:, 1], "b-", linewidth=2, alpha=0.7)
        ax.scatter(
            pareto_obj[:, 0],
            pareto_obj[:, 1],
            c="blue",
            s=100,
            label="Pareto Optimal",
            zorder=5,
        )

        # Highlight knee points
        if show_knee and frontier.knee_indices:
            for idx in frontier.knee_indices:
                knee = frontier.solutions[idx].objectives
                ax.scatter(
                    knee[0],
                    knee[1],
                    c="red",
                    s=200,
                    marker="*",
                    label="Knee Point" if idx == frontier.knee_indices[0] else "",
                    zorder=10,
                )

    # Labels
    if xlabel:
        ax.set_xlabel(xlabel, fontsize=12)
    elif frontier.objective_names:
        ax.set_xlabel(frontier.objective_names[0], fontsize=12)

    if ylabel:
        ax.set_ylabel(ylabel, fontsize=12)
    elif len(frontier.objective_names) > 1:
        ax.set_ylabel(frontier.objective_names[1], fontsize=12)

    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=10)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Pareto plot saved to {save_path}")

    return fig


def plot_pareto_3d(
    frontier: ParetoFrontier,
    all_solutions: np.ndarray = None,
    title: str = "3D Pareto Frontier",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 10),
) -> Figure:
    """
    Plot 3D Pareto frontier.

    Parameters
    ----------
    frontier : ParetoFrontier
        Pareto frontier with 3 objectives.
    all_solutions : np.ndarray, optional
        All solutions for reference.
    title : str, optional
        Plot title.
    save_path : str, optional
        Path to save figure.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401

    fig = plt.figure(figsize=figsize)
    ax = fig.add_subplot(111, projection="3d")

    # Plot all solutions
    if all_solutions is not None:
        ax.scatter(
            all_solutions[:, 0],
            all_solutions[:, 1],
            all_solutions[:, 2],
            c="lightgray",
            alpha=0.3,
            s=20,
            label="Dominated",
        )

    # Plot Pareto frontier
    if len(frontier) > 0:
        pareto_obj = frontier.get_objectives_matrix()

        ax.scatter(
            pareto_obj[:, 0],
            pareto_obj[:, 1],
            pareto_obj[:, 2],
            c="blue",
            s=80,
            label="Pareto Optimal",
        )

        # Highlight knee points
        if frontier.knee_indices:
            for idx in frontier.knee_indices:
                knee = frontier.solutions[idx].objectives
                ax.scatter(
                    knee[0], knee[1], knee[2], c="red", s=200, marker="*", label="Knee"
                )

    # Labels
    if frontier.objective_names and len(frontier.objective_names) >= 3:
        ax.set_xlabel(frontier.objective_names[0], fontsize=10)
        ax.set_ylabel(frontier.objective_names[1], fontsize=10)
        ax.set_zlabel(frontier.objective_names[2], fontsize=10)

    ax.set_title(title, fontsize=14)
    ax.legend(loc="best", fontsize=10)

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"3D Pareto plot saved to {save_path}")

    return fig


def plot_parallel_coordinates(
    frontier: ParetoFrontier,
    highlight_knee: bool = True,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (12, 6),
) -> Figure:
    """
    Plot Pareto solutions using parallel coordinates.

    Useful for visualizing high-dimensional Pareto fronts.

    Parameters
    ----------
    frontier : ParetoFrontier
        Pareto frontier.
    highlight_knee : bool, optional
        Whether to highlight knee points.
    save_path : str, optional
        Path to save figure.
    figsize : tuple, optional
        Figure size.

    Returns
    -------
    Figure
        Matplotlib figure.
    """
    if len(frontier) == 0:
        fig, ax = plt.subplots(figsize=figsize)
        ax.text(0.5, 0.5, "No solutions", ha="center", va="center")
        return fig

    objectives = frontier.get_objectives_matrix()
    n_solutions, n_objectives = objectives.shape

    # Normalize objectives
    obj_min = objectives.min(axis=0)
    obj_max = objectives.max(axis=0)
    obj_range = obj_max - obj_min
    obj_range[obj_range == 0] = 1.0

    normalized = (objectives - obj_min) / obj_range

    fig, ax = plt.subplots(figsize=figsize)

    x = np.arange(n_objectives)

    # Plot each solution
    for i in range(n_solutions):
        is_knee = i in frontier.knee_indices if frontier.knee_indices else False

        if is_knee and highlight_knee:
            ax.plot(x, normalized[i], "r-", linewidth=3, alpha=0.8, zorder=10)
        else:
            ax.plot(x, normalized[i], "b-", linewidth=1, alpha=0.4)

    # Axis labels
    ax.set_xticks(x)
    if frontier.objective_names:
        ax.set_xticklabels(frontier.objective_names, fontsize=10)

    ax.set_ylabel("Normalized Value", fontsize=12)
    ax.set_title("Parallel Coordinates: Pareto Solutions", fontsize=14)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=300, bbox_inches="tight")
        logger.info(f"Parallel coordinates plot saved to {save_path}")

    return fig


# ============================================================================
# TRADE-OFF ANALYSIS
# ============================================================================


def analyze_tradeoffs(frontier: ParetoFrontier) -> Dict[str, Any]:
    """
    Analyze trade-offs in the Pareto frontier.

    Parameters
    ----------
    frontier : ParetoFrontier
        Pareto frontier.

    Returns
    -------
    dict
        Trade-off analysis results.
    """
    if len(frontier) < 2:
        return {"error": "Need at least 2 solutions for trade-off analysis"}

    objectives = frontier.get_objectives_matrix()
    n_solutions, n_objectives = objectives.shape

    # Objective ranges
    obj_min = objectives.min(axis=0)
    obj_max = objectives.max(axis=0)
    obj_range = obj_max - obj_min

    # Compute marginal rates of substitution (MRS) between objectives
    # MRS = change in obj_j / change in obj_i
    mrs = {}

    # Sort by first objective
    sort_idx = np.argsort(objectives[:, 0])
    sorted_obj = objectives[sort_idx]

    for i in range(n_objectives):
        for j in range(i + 1, n_objectives):
            key = (
                f"MRS_{frontier.objective_names[i]}_{frontier.objective_names[j]}"
                if frontier.objective_names
                else f"MRS_{i}_{j}"
            )

            deltas_i = np.diff(sorted_obj[:, i])
            deltas_j = np.diff(sorted_obj[:, j])

            # Avoid division by zero
            valid = np.abs(deltas_i) > 1e-10

            if np.any(valid):
                mrs_values = deltas_j[valid] / deltas_i[valid]
                mrs[key] = {
                    "mean": float(np.mean(mrs_values)),
                    "std": float(np.std(mrs_values)),
                    "min": float(np.min(mrs_values)),
                    "max": float(np.max(mrs_values)),
                }

    # Correlation between objectives on Pareto front
    if n_solutions >= 3 and n_objectives >= 2:
        correlations = {}
        for i in range(n_objectives):
            for j in range(i + 1, n_objectives):
                key = (
                    f"corr_{frontier.objective_names[i]}_{frontier.objective_names[j]}"
                    if frontier.objective_names
                    else f"corr_{i}_{j}"
                )
                corr = np.corrcoef(objectives[:, i], objectives[:, j])[0, 1]
                correlations[key] = float(corr)
    else:
        correlations = {}

    return {
        "n_pareto_solutions": n_solutions,
        "objective_ranges": {
            name: {
                "min": float(obj_min[i]),
                "max": float(obj_max[i]),
                "range": float(obj_range[i]),
            }
            for i, name in enumerate(
                frontier.objective_names or [f"Obj_{i}" for i in range(n_objectives)]
            )
        },
        "marginal_rates_of_substitution": mrs,
        "objective_correlations": correlations,
        "hypervolume": frontier.hypervolume,
        "knee_indices": frontier.knee_indices,
    }


def select_best_solution(
    frontier: ParetoFrontier, weights: List[float] = None, method: str = "weighted_sum"
) -> int:
    """
    Select best solution from Pareto frontier.

    Parameters
    ----------
    frontier : ParetoFrontier
        Pareto frontier.
    weights : List[float], optional
        Weights for each objective. Default: equal weights.
    method : str, optional (default='weighted_sum')
        Selection method: 'weighted_sum', 'min_max', or 'knee'.

    Returns
    -------
    int
        Index of best solution in frontier.solutions.
    """
    if len(frontier) == 0:
        raise ValueError("Empty frontier")

    if len(frontier) == 1:
        return 0

    if method == "knee":
        if frontier.knee_indices:
            return frontier.knee_indices[0]
        else:
            return find_knee_point_distance(frontier)

    objectives = frontier.get_objectives_matrix()
    n_objectives = objectives.shape[1]

    if weights is None:
        weights = [1.0 / n_objectives] * n_objectives

    weights = np.array(weights)

    # Normalize objectives
    obj_min = objectives.min(axis=0)
    obj_max = objectives.max(axis=0)
    obj_range = obj_max - obj_min
    obj_range[obj_range == 0] = 1.0

    normalized = (objectives - obj_min) / obj_range

    # Convert to minimization
    if frontier.minimize:
        for i, mini in enumerate(frontier.minimize):
            if not mini:
                normalized[:, i] = 1 - normalized[:, i]

    if method == "weighted_sum":
        # Weighted sum of normalized objectives
        scores = np.sum(normalized * weights, axis=1)
        return int(np.argmin(scores))

    elif method == "min_max":
        # Minimize the maximum weighted normalized objective
        weighted = normalized * weights
        max_obj = np.max(weighted, axis=1)
        return int(np.argmin(max_obj))

    else:
        raise ValueError(f"Unknown method: {method}")


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def save_frontier(frontier: ParetoFrontier, filepath: Union[str, Path]) -> None:
    """Save Pareto frontier to JSON file."""
    filepath = Path(filepath)
    filepath.parent.mkdir(parents=True, exist_ok=True)

    with open(filepath, "w") as f:
        f.write(frontier.to_json())

    logger.info(f"Pareto frontier saved to {filepath}")


def load_frontier(filepath: Union[str, Path]) -> ParetoFrontier:
    """Load Pareto frontier from JSON file."""
    with open(filepath, "r") as f:
        data = json.load(f)

    solutions = [
        ParetoSolution(
            objectives=np.array(s["objectives"]),
            index=s["index"],
            metadata=s.get("metadata", {}),
            is_dominated=s.get("is_dominated", False),
            is_knee=s.get("is_knee", False),
            crowding_distance=s.get("crowding_distance", 0.0),
        )
        for s in data["solutions"]
    ]

    return ParetoFrontier(
        solutions=solutions,
        objective_names=data.get("objective_names", []),
        minimize=data.get("minimize", []),
        hypervolume=data.get("hypervolume"),
        knee_indices=data.get("knee_indices", []),
    )


# ============================================================================
# MODULE EXPORTS
# ============================================================================


__all__ = [
    # Data classes
    "ParetoSolution",
    "ParetoFrontier",
    # Dominance
    "is_dominated",
    "compute_pareto_frontier",
    "get_dominated_solutions",
    "non_dominated_sort",
    # Knee points
    "find_knee_point_angle",
    "find_knee_point_distance",
    "find_knee_point_curvature",
    "find_knee_points",
    # Hypervolume
    "compute_hypervolume_2d",
    "compute_hypervolume",
    # Visualization
    "plot_pareto_2d",
    "plot_pareto_3d",
    "plot_parallel_coordinates",
    # Trade-off analysis
    "analyze_tradeoffs",
    "select_best_solution",
    # I/O
    "save_frontier",
    "load_frontier",
]
