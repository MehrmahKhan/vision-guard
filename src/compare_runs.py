"""Task 3.5 - pull every run out of MLflow into one comparison table.

Run:
    python -m src.compare_runs
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import pandas as pd

from src import config


def main():
    config.ensure_dirs()
    if config.MLFLOW_TRACKING_URI != "":
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
    if experiment is None:
        raise SystemExit("No experiment named " + config.EXPERIMENT_NAME)

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if len(runs) == 0:
        raise SystemExit("No runs found yet.")

    wanted = [
        "tags.mlflow.runName",
        "params.model",
        "params.backbone",
        "params.img_size",
        "params.learning_rate",
        "params.dropout",
        "params.unfrozen_layers",
        "params.augmentation",
        "params.decision_threshold",
        "metrics.best_epoch",
        "metrics.test_accuracy",
        "metrics.test_precision",
        "metrics.test_recall",
        "metrics.test_f1",
        "metrics.test_roc_auc",
        "run_id",
    ]

    columns = []
    for name in wanted:
        if name in runs.columns:
            columns.append(name)

    table = runs[columns].copy()
    renamed = {}
    for name in columns:
        renamed[name] = name.replace("tags.mlflow.runName", "run").replace(
            "params.", ""
        ).replace("metrics.", "")
    table = table.rename(columns=renamed)

    if "test_roc_auc" in table.columns:
        table = table.sort_values("test_roc_auc", ascending=False)

    output = os.path.join(config.ARTIFACT_DIR, "run_comparison.csv")
    table.to_csv(output, index=False)
    print(table.to_string(index=False))
    print("")
    print("wrote " + output)

    if "test_roc_auc" in table.columns and "run" in table.columns:
        plot_frame = table.dropna(subset=["test_roc_auc"])
        if len(plot_frame) > 0:
            figure, axis = plt.subplots(figsize=(9, 4.5))
            labels = plot_frame["run"].astype(str).tolist()
            values = plot_frame["test_roc_auc"].tolist()
            positions = []
            index = 0
            while index < len(labels):
                positions.append(index)
                index = index + 1
            axis.bar(positions, values)
            axis.set_xticks(positions)
            axis.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
            axis.set_ylabel("test ROC-AUC")
            axis.set_title("Run comparison, VisionGuard")
            axis.set_ylim(0.0, 1.0)
            figure.tight_layout()
            chart = os.path.join(config.ARTIFACT_DIR, "run_comparison.png")
            figure.savefig(chart, dpi=130)
            plt.close(figure)
            print("wrote " + chart)


if __name__ == "__main__":
    main()
