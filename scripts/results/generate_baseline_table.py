# scripts/results/generate_baseline_table.py

import json
import os

import numpy as np
import pandas as pd

RESULTS_DIR = "results/metrics/baseline_isic2018_resnet50"
OUTPUT_FILE = "results/metrics/rq1_robustness/baseline_summary.csv"


def flatten_run(run_data: dict) -> dict:
    """
    Flatten one run's JSON into a 1-row dict.
    Assumes structure like:
      {
        "seed": 42,
        "model": "resnet50",
        "dataset": "isic2018",
        "best_epoch": 2,
        "best_val_loss": 1.94,
        "history": {
            "train_loss": [...],
            "val_loss":   [...],
            "train_acc":  [...],
            "val_acc":    [...]
        },
        ...
      }
    Any top-level scalar numeric fields are kept.
    From `history`, we derive final_* and best_* metrics where possible.
    """
    flat = {}

    # 1) Copy top-level simple fields
    for k, v in run_data.items():
        if k == "history":
            continue
        # keep scalars (numbers, strings)
        if isinstance(v, (int, float, str)):
            flat[k] = v

    # 2) Derive metrics from history if present
    history = run_data.get("history", {})
    if isinstance(history, dict):
        for name, series in history.items():
            if not isinstance(series, (list, tuple)) or len(series) == 0:
                continue

            # final value
            flat[f"final_{name}"] = series[-1]

            # best value depending on what it is
            if "loss" in name.lower():
                flat[f"best_{name}"] = float(min(series))
                flat[f"best_{name}_epoch"] = int(np.argmin(series))
            elif "acc" in name.lower():
                flat[f"best_{name}"] = float(max(series))
                flat[f"best_{name}_epoch"] = int(np.argmax(series))

    return flat


def generate_table():
    if not os.path.exists(RESULTS_DIR):
        print(f"[ERROR] Directory not found: {RESULTS_DIR}")
        return

    json_files = [f for f in os.listdir(RESULTS_DIR) if f.endswith(".json")]
    if not json_files:
        print(f"[ERROR] No JSON files found in {RESULTS_DIR}")
        return

    print(f"[INFO] Found {len(json_files)} JSON files in {RESULTS_DIR}")

    rows = []
    for jf in json_files:
        path = os.path.join(RESULTS_DIR, jf)
        with open(path, "r") as f:
            try:
                run_data = json.load(f)
            except json.JSONDecodeError:
                print(f"[WARN] Error reading {jf}, skipping.")
                continue

        flat = flatten_run(run_data)
        # fall back if no seed present
        flat.setdefault("seed", run_data.get("seed", "unknown"))
        rows.append(flat)

    if not rows:
        print("[ERROR] No valid runs to aggregate.")
        return

    df = pd.DataFrame(rows)
    print("[INFO] Raw aggregated data:")
    print(df)

    # numeric columns except 'seed'
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    if "seed" in numeric_cols:
        numeric_cols.remove("seed")

    if not numeric_cols:
        print("[WARN] No numeric metrics found to aggregate.")
        return

    summary = df[numeric_cols].agg(["mean", "std"]).T
    summary["formatted"] = summary.apply(
        lambda x: f"{x['mean']:.3f} ± {x['std']:.3f}", axis=1
    )

    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    summary.to_csv(OUTPUT_FILE)

    print("\n[INFO] Summary (mean ± std):")
    print(summary["formatted"])
    print(f"\n[SUCCESS] Table saved to {OUTPUT_FILE}")


if __name__ == "__main__":
    generate_table()
