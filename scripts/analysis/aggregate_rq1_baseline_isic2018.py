from __future__ import annotations

"""
Aggregate RQ1 baseline results on ISIC 2018 (ResNet-50, seeds 42/123/456).

This script:
- Queries MLflow for the three baseline runs
- Computes mean ± std across seeds for key metrics
- Saves a CSV summary table
- Plots mean training/validation curves with a std band

Usage
-----
    python scripts/analysis/aggregate_rq1_baseline_isic2018.py
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from mlflow.tracking import MlflowClient

EXPERIMENT_NAME = "rq1_baseline_isic2018_resnet50"
SEEDS: List[int] = [42, 123, 456]

# Final metrics we care about (whatever your training code logs)
FINAL_METRICS = [
    "train_loss",
    "val_loss",
    "train_accuracy",
    "val_accuracy",
]

# Per-epoch metrics for curves
CURVE_METRICS = [
    "train_loss",
    "val_loss",
    "train_accuracy",
    "val_accuracy",
]


@dataclass
class RunSummary:
    run_id: str
    seed: int
    metrics: Dict[str, float]


def _get_latest_run_for_seed(seed: int) -> RunSummary:
    """
    Fetch the most recent MLflow run for a given seed in the experiment.
    """
    runs = mlflow.search_runs(
        experiment_names=[EXPERIMENT_NAME],
        filter_string=f"params.seed = '{seed}'",
        order_by=["attributes.start_time DESC"],
        max_results=1,
    )

    if runs.empty:
        raise RuntimeError(
            f"No MLflow run found for experiment={EXPERIMENT_NAME!r}, seed={seed}"
        )

    row = runs.iloc[0]
    run_id = row["run_id"]

    metrics: Dict[str, float] = {}
    for name in FINAL_METRICS:
        col = f"metrics.{name}"
        value = float(row[col]) if col in row and pd.notna(row[col]) else float("nan")
        metrics[name] = value

    return RunSummary(run_id=run_id, seed=seed, metrics=metrics)


def collect_run_summaries(seeds: Sequence[int]) -> List[RunSummary]:
    """
    Collect latest run summary for each seed.
    """
    summaries: List[RunSummary] = []
    for seed in seeds:
        summaries.append(_get_latest_run_for_seed(seed))
    return summaries


def build_summary_table(run_summaries: Sequence[RunSummary]) -> pd.DataFrame:
    """
    Build a mean ± std table across seeds for FINAL_METRICS.
    """
    rows: List[Dict[str, float]] = []

    for metric_name in FINAL_METRICS:
        values = [
            rs.metrics[metric_name]
            for rs in run_summaries
            if not np.isnan(rs.metrics[metric_name])
        ]
        if not values:
            mean = float("nan")
            std = float("nan")
        else:
            arr = np.asarray(values, dtype=float)
            mean = float(arr.mean())
            std = float(arr.std(ddof=0))

        rows.append(
            {
                "metric": metric_name,
                "mean": mean,
                "std": std,
                "n_seeds": len(values),
            }
        )

    df = pd.DataFrame(rows).set_index("metric")
    return df


def _ensure_output_dirs() -> Dict[str, Path]:
    """
    Prepare output directories for CSVs and figures.
    """
    base = Path("results") / "analysis" / "rq1_baseline_isic2018_resnet50"
    figs = base / "figures"

    base.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    return {"base": base, "figs": figs}


def plot_curves_with_bands(
    run_summaries: Sequence[RunSummary],
    metric_name: str,
    client: MlflowClient,
    out_path: Path,
) -> None:
    """
    Plot mean ± std curves across seeds for a single metric.

    Assumes:
    - Metric is logged once per epoch with 'step' equal to epoch index.
    """
    all_steps: List[np.ndarray] = []
    all_values: List[np.ndarray] = []

    for rs in run_summaries:
        history = client.get_metric_history(rs.run_id, metric_name)
        if not history:
            continue

        steps = np.array([pt.step for pt in history], dtype=int)
        values = np.array([pt.value for pt in history], dtype=float)

        # Sort by step to be safe
        order = np.argsort(steps)
        steps = steps[order]
        values = values[order]

        all_steps.append(steps)
        all_values.append(values)

    if not all_steps:
        print(f"[WARN] No history found for metric {metric_name!r}; skipping curve.")
        return

    # Align sequences by the shortest length (conservative, avoids NaNs)
    min_len = min(len(s) for s in all_steps)
    epochs = all_steps[0][:min_len]

    stacked = np.stack(
        [v[:min_len] for v in all_values], axis=0
    )  # shape: (num_seeds, L)
    mean_curve = stacked.mean(axis=0)
    std_curve = stacked.std(axis=0, ddof=0)

    plt.figure()
    plt.plot(epochs, mean_curve, label=f"{metric_name} (mean)")
    plt.fill_between(
        epochs,
        mean_curve - std_curve,
        mean_curve + std_curve,
        alpha=0.3,
        label=f"{metric_name} ± std",
    )
    plt.xlabel("Epoch")
    plt.ylabel(metric_name.replace("_", " ").title())
    plt.title(f"{EXPERIMENT_NAME} – {metric_name} (mean ± std, seeds={SEEDS})")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_path)
    plt.close()
    print(f"[INFO] Saved curve for {metric_name!r} to {out_path}")


def main() -> None:
    mlflow.set_experiment(EXPERIMENT_NAME)
    client = MlflowClient()

    out_dirs = _ensure_output_dirs()

    # 1) Collect runs
    run_summaries = collect_run_summaries(SEEDS)
    print("[INFO] Found runs:")
    for rs in run_summaries:
        print(f"  seed={rs.seed} -> run_id={rs.run_id}")

    # 2) Build and save summary table
    summary_df = build_summary_table(run_summaries)
    csv_path = out_dirs["base"] / "baseline_isic2018_resnet50_summary.csv"
    summary_df.to_csv(csv_path)
    print(f"[INFO] Saved summary table to {csv_path}")
    print(summary_df)

    # 3) Plot curves with mean ± std band
    for metric in CURVE_METRICS:
        fig_path = out_dirs["figs"] / f"{metric}_mean_std.png"
        plot_curves_with_bands(run_summaries, metric, client, fig_path)


if __name__ == "__main__":
    main()
