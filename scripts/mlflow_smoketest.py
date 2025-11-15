from pathlib import Path

import mlflow


def main() -> None:
    # 1) Backend store: SQLite DB in repo root
    tracking_uri = "sqlite:///mlruns.db"
    mlflow.set_tracking_uri(tracking_uri)

    # 2) Experiment naming convention: triobj/<dataset>/<model>
    experiment_name = "triobj/cifar10/debug-cnn"
    mlflow.set_experiment(experiment_name)

    # 3) Start run
    with mlflow.start_run(run_name="smoke-test"):
        # Example "automatic" params – pretend these came from a config
        params = {
            "dataset": "CIFAR10",
            "model": "TinyCNN",
            "epochs": 2,
            "batch_size": 32,
            "optimizer": "SGD",
            "lr": 0.01,
            "seed": 42,
        }
        for k, v in params.items():
            mlflow.log_param(k, v)

        # Log a few fake metrics
        mlflow.log_metric("acc_tiny_debug", 0.2773, step=2)
        mlflow.log_metric("loss_tiny_debug", 2.00, step=2)

        # Log a tiny artifact
        tmp = Path("tmp_artifacts")
        tmp.mkdir(exist_ok=True)
        f = tmp / "hello_mlflow.txt"
        f.write_text("hello from Phase 1.2!")
        mlflow.log_artifact(str(f), artifact_path="debug")

        print("Tracking URI:", mlflow.get_tracking_uri())
        print("Experiment:", experiment_name)
        print("Run created successfully.")


if __name__ == "__main__":
    main()
