"""
================================================================================
Trade-off Analysis Module - Phase 5.3
================================================================================
Analyze clean-robust accuracy trade-offs in adversarial training.

Features:
    - Multi-objective trade-off analysis
    - Pareto frontier computation
    - Knee point detection
    - Dominated solution filtering
    - Beta sensitivity analysis

Author: Viraj Pankaj Jain
Institution: University of Glasgow
Date: November 2025
================================================================================
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from scipy.spatial.distance import cdist


class TradeoffAnalyzer:
    """Analyze trade-offs between multiple objectives."""

    def __init__(self, logger: Optional[logging.Logger] = None):
        """
        Initialize trade-off analyzer.

        Args:
            logger: Logger instance
        """
        self.logger = logger or logging.getLogger(__name__)

    def is_dominated(
        self, point: np.ndarray, other_points: np.ndarray, maximize: np.ndarray
    ) -> bool:
        """
        Check if a point is Pareto-dominated.

        A point is dominated if there exists another point that is
        at least as good in all objectives and strictly better in at least one.

        Args:
            point: Point to check [num_objectives]
            other_points: Other points [num_points, num_objectives]
            maximize: Boolean array indicating maximize (True) or minimize (False)

        Returns:
            True if point is dominated, False otherwise
        """
        # Adjust signs for minimization objectives
        adjusted_point = point * (2 * maximize - 1)
        adjusted_others = other_points * (2 * maximize - 1)

        # Check dominance
        # A point is dominated if any other point is >= in all objectives
        # and strictly > in at least one objective
        at_least_as_good = np.all(adjusted_others >= adjusted_point, axis=1)
        strictly_better = np.any(adjusted_others > adjusted_point, axis=1)

        return np.any(at_least_as_good & strictly_better)

    def compute_pareto_frontier(
        self, points: np.ndarray, maximize: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute Pareto frontier from set of points.

        Args:
            points: Points to analyze [num_points, num_objectives]
            maximize: Boolean array for each objective

        Returns:
            pareto_points: Points on Pareto frontier
            pareto_indices: Indices of Pareto points
        """
        num_points = points.shape[0]
        is_pareto = np.ones(num_points, dtype=bool)

        for i in range(num_points):
            if is_pareto[i]:
                # Check if point i is dominated by any other Pareto point
                other_pareto_points = points[is_pareto]
                other_pareto_points = other_pareto_points[
                    other_pareto_points != points[i]
                ].reshape(-1, points.shape[1])

                if len(other_pareto_points) > 0:
                    if self.is_dominated(points[i], other_pareto_points, maximize):
                        is_pareto[i] = False

        pareto_indices = np.where(is_pareto)[0]
        pareto_points = points[pareto_indices]

        self.logger.info(
            f"Found {len(pareto_indices)} Pareto-optimal points "
            f"out of {num_points} total points"
        )

        return pareto_points, pareto_indices

    def find_knee_point(
        self, pareto_points: np.ndarray, maximize: np.ndarray
    ) -> Tuple[int, np.ndarray]:
        """
        Find knee point on Pareto frontier (best trade-off).

        Uses point-to-line distance method:
        Knee point is the point with maximum perpendicular distance
        to the line connecting extreme points.

        Args:
            pareto_points: Points on Pareto frontier [num_points, num_objectives]
            maximize: Boolean array for each objective

        Returns:
            knee_index: Index of knee point in pareto_points
            knee_point: Coordinates of knee point
        """
        if len(pareto_points) < 3:
            # If too few points, return middle point
            self.logger.warning(
                "Too few points for knee detection, returning middle point"
            )
            knee_index = len(pareto_points) // 2
            return knee_index, pareto_points[knee_index]

        # Normalize to [0, 1] range for each objective
        normalized = (pareto_points - pareto_points.min(axis=0)) / (
            pareto_points.max(axis=0) - pareto_points.min(axis=0) + 1e-10
        )

        # Adjust for minimization objectives
        adjusted = normalized * (2 * maximize - 1)

        # Sort points by first objective
        sorted_indices = np.argsort(adjusted[:, 0])
        sorted_points = adjusted[sorted_indices]

        # Line from first to last point
        p1 = sorted_points[0]
        p2 = sorted_points[-1]

        # Compute perpendicular distance for each point
        line_vec = p2 - p1
        line_length = np.linalg.norm(line_vec)

        distances = []
        for point in sorted_points:
            point_vec = point - p1
            # Perpendicular distance = |cross product| / line_length
            cross_product = np.abs(np.cross(line_vec, point_vec))
            distance = cross_product / (line_length + 1e-10)
            distances.append(distance)

        distances = np.array(distances)

        # Knee point has maximum distance
        knee_index_sorted = np.argmax(distances)
        knee_index = sorted_indices[knee_index_sorted]
        knee_point = pareto_points[knee_index]

        self.logger.info(f"Knee point found at index {knee_index}")
        self.logger.info(f"Knee point values: {knee_point}")

        return int(knee_index), knee_point

    def analyze_tradeoffs(
        self,
        results: Dict[str, Dict[str, float]],
        objectives: List[str],
        maximize: List[bool],
        method_names: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Comprehensive trade-off analysis.

        Args:
            results: Dictionary mapping method -> objective values
            objectives: List of objective names
            maximize: List of booleans (True = maximize, False = minimize)
            method_names: Optional list of method names (uses results keys if None)

        Returns:
            Dictionary with analysis results
        """
        if method_names is None:
            method_names = list(results.keys())

        # Extract objective values
        points = []
        for method in method_names:
            point = [results[method][obj] for obj in objectives]
            points.append(point)

        points = np.array(points)
        maximize = np.array(maximize)

        self.logger.info(
            f"Analyzing {len(points)} methods across {len(objectives)} objectives"
        )

        # Compute Pareto frontier
        pareto_points, pareto_indices = self.compute_pareto_frontier(points, maximize)
        pareto_methods = [method_names[i] for i in pareto_indices]

        # Find knee point
        if len(pareto_points) >= 3:
            knee_index, knee_point = self.find_knee_point(pareto_points, maximize)
            knee_method = pareto_methods[knee_index]
        else:
            knee_index = 0
            knee_point = pareto_points[0] if len(pareto_points) > 0 else None
            knee_method = pareto_methods[0] if len(pareto_methods) > 0 else None

        # Compute hypervolume (2D only)
        hypervolume = None
        if points.shape[1] == 2:
            hypervolume = self.compute_hypervolume_2d(pareto_points, maximize)

        # Aggregate results
        analysis = {
            "num_methods": len(method_names),
            "num_objectives": len(objectives),
            "objectives": objectives,
            "maximize": maximize.tolist(),
            "pareto": {
                "num_points": len(pareto_points),
                "methods": pareto_methods,
                "points": pareto_points.tolist(),
            },
            "knee_point": {
                "index": int(knee_index),
                "method": knee_method,
                "values": knee_point.tolist() if knee_point is not None else None,
            },
            "hypervolume": float(hypervolume) if hypervolume is not None else None,
            "all_points": {"methods": method_names, "points": points.tolist()},
        }

        return analysis

    def compute_hypervolume_2d(
        self, pareto_points: np.ndarray, maximize: np.ndarray
    ) -> float:
        """
        Compute hypervolume indicator for 2D Pareto frontier.

        Args:
            pareto_points: Pareto points [num_points, 2]
            maximize: Boolean array [2]

        Returns:
            Hypervolume value
        """
        # Reference point (nadir point)
        if maximize[0]:
            ref_x = pareto_points[:, 0].min() - 0.1
        else:
            ref_x = pareto_points[:, 0].max() + 0.1

        if maximize[1]:
            ref_y = pareto_points[:, 1].min() - 0.1
        else:
            ref_y = pareto_points[:, 1].max() + 0.1

        # Sort points by first objective
        if maximize[0]:
            sorted_indices = np.argsort(-pareto_points[:, 0])
        else:
            sorted_indices = np.argsort(pareto_points[:, 0])

        sorted_points = pareto_points[sorted_indices]

        # Compute hypervolume
        hv = 0.0
        prev_x = ref_x

        for point in sorted_points:
            width = abs(point[0] - prev_x)
            height = abs(point[1] - ref_y)
            hv += width * height
            prev_x = point[0]

        return hv


def load_results(results_paths: Dict[str, Path]) -> Dict[str, Dict[str, float]]:
    """
    Load results from multiple JSON files.

    Args:
        results_paths: Dictionary mapping method_name -> results_path

    Returns:
        Dictionary mapping method_name -> metrics
    """
    results = {}

    for method_name, path in results_paths.items():
        with open(path, "r") as f:
            results[method_name] = json.load(f)

    return results


def save_analysis(analysis: Dict[str, Any], output_path: Path) -> None:
    """Save trade-off analysis results."""
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w") as f:
        json.dump(analysis, f, indent=2)


def main():
    """Example usage."""
    import argparse

    parser = argparse.ArgumentParser(description="Trade-off Analysis")
    parser.add_argument("--results_dir", type=str, required=True)
    parser.add_argument("--output", type=str, required=True)
    args = parser.parse_args()

    # Setup logger
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)

    # Load results
    results_dir = Path(args.results_dir)
    results_paths = {
        "TRADES": results_dir / "trades" / "evaluation_results.json",
        "PGD-AT": results_dir / "pgd_at" / "evaluation_results.json",
    }

    results = load_results(results_paths)

    # Initialize analyzer
    analyzer = TradeoffAnalyzer(logger=logger)

    # Analyze trade-offs
    objectives = ["clean_accuracy", "robust_accuracy"]
    maximize = [True, True]

    analysis = analyzer.analyze_tradeoffs(results, objectives, maximize)

    # Save results
    save_analysis(analysis, Path(args.output))
    logger.info(f"Saved analysis to {args.output}")


if __name__ == "__main__":
    main()
