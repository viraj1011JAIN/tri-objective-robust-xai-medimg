"""
Helper instructions for generating baseline learning curves via MLflow UI.

We use MLflow's built-in comparison plots instead of re-implementing
aggregation logic here. This script just documents the exact steps.
"""


def mock_plot() -> None:
    print("Manual Curve Generation Instructions:")
    print("1. In a terminal, run: mlflow ui")
    print("2. Open: http://127.0.0.1:5000 in your browser")
    print(
        "3. In the UI, filter to the 3 baseline runs on ISIC 2018 (seeds 42, 123, 456)."
    )
    print("4. Click 'Compare' -> 'Charts'.")
    print("5. Select metrics such as `val_accuracy` and `val_loss`.")
    print("6. Screenshot the curves (PNG).")
    print("7. Save the screenshot to:")
    print("   results/plots/training_curves/baseline_learning_curve.png")
    print("8. Optionally commit this PNG as an artifact for RQ1.")


if __name__ == "__main__":
    mock_plot()
