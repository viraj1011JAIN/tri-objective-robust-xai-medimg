# scripts/results/validate_baseline_isic2018.py

import math
import os

import pandas as pd

SUMMARY_CSV = "results/metrics/rq1_robustness/baseline_summary.csv"


def main():
    if not os.path.exists(SUMMARY_CSV):
        print(f"[ERROR] Summary file not found: {SUMMARY_CSV}")
        return

    summary = pd.read_csv(SUMMARY_CSV, index_col=0)
    print("[INFO] Loaded summary:\n", summary, "\n")

    # Helper to read mean values (row name like 'best_val_loss')
    def get_mean(metric_name: str, default=None):
        if metric_name not in summary.index:
            return default
        return float(summary.loc[metric_name, "mean"])

    best_val_loss = get_mean("best_val_loss")
    final_val_acc = get_mean("final_val_acc")
    final_train_acc = get_mean("final_train_acc")
    # best_val_acc = summary.loc["best_val_acc", "mean"]
    # _final_val_loss = summary.loc["final_val_loss", "mean"]
    # _final_train_loss = summary.loc["final_train_loss", "mean"]
    # _best_val_acc = summary.loc["best_val_acc", "mean"]

    # 1) Random-baseline check (7 classes for ISIC 2018)
    num_classes = 7
    random_ce = math.log(num_classes)  # ≈ 1.946

    print(f"[INFO] Expected random CE loss for {num_classes} classes: {random_ce:.3f}")

    if best_val_loss is not None:
        print(f"[INFO] best_val_loss (mean): {best_val_loss:.3f}")
        if abs(best_val_loss - random_ce) < 0.15:
            print("[WARNING] Validation loss is at random-guessing level.")
        elif best_val_loss > random_ce + 0.2:
            print("[WARNING] Validation loss is worse than random guessing.")
        else:
            print("[OK] Validation loss is clearly better than random baseline.")

    # 2) Overfitting / underfitting diagnostics
    if final_train_acc is not None and final_val_acc is not None:
        print(f"[INFO] final_train_acc (mean): {final_train_acc:.3f}")
        print(f"[INFO] final_val_acc   (mean): {final_val_acc:.3f}")

        if final_train_acc > 0.95 and final_val_acc < 0.30:
            print(
                "[CRITICAL] Severe overfitting or data issue "
                "(train≈1.0, val near random)."
            )
        elif final_train_acc < 0.60 and final_val_acc < 0.30:
            print("[WARNING] Model may be underfitting; both accuracies low.")
        elif final_train_acc > 0.80 and final_val_acc > 0.60:
            print("[OK] Both train and val accuracy look reasonable.")
        else:
            print(
                "[INFO] Accuracies are in an intermediate regime; "
                "inspect learning curves for more detail."
            )

    else:
        print(
            "[WARN] Accuracy metrics not found in summary. "
            "Make sure train_acc and val_acc are being logged in history."
        )


if __name__ == "__main__":
    main()
