"""Finds the best run on DagsHub, registers it, gives it the champion alias,
and saves the model locally as models/champion.keras for the Flask app.

Run from the project root:
    python pick_champion.py
"""

import os
import shutil

import mlflow
from mlflow.tracking import MlflowClient

from src import config

MODEL_NAME = "VisionGuard"


def main():
    if config.MLFLOW_TRACKING_URI != "":
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)

    experiment = mlflow.get_experiment_by_name(config.EXPERIMENT_NAME)
    if experiment is None:
        raise SystemExit("No experiment named " + config.EXPERIMENT_NAME)

    runs = mlflow.search_runs(experiment_ids=[experiment.experiment_id])
    if len(runs) == 0:
        raise SystemExit("No runs found.")

    if "metrics.test_roc_auc" not in runs.columns:
        raise SystemExit("No test_roc_auc metric on any run.")

    finished = runs.dropna(subset=["metrics.test_roc_auc"])
    print("")
    print("runs with a test score, best first:")
    ranked = finished.sort_values("metrics.test_roc_auc", ascending=False)
    position = 0
    while position < len(ranked):
        row = ranked.iloc[position]
        name = row.get("tags.mlflow.runName", "unnamed")
        auc = row["metrics.test_roc_auc"]
        print("  " + str(name).ljust(26) + "auc " + str(round(float(auc), 4)))
        position = position + 1

    best = ranked.iloc[0]
    best_run_id = best["run_id"]
    best_name = best.get("tags.mlflow.runName", "unnamed")
    print("")
    print("champion: " + str(best_name) + "  run_id " + str(best_run_id))

    # Register twice so the registry has at least two versions, which the
    # sub-task asks for. The runner up becomes version 1, the winner version 2.
    runner_up = ranked.iloc[1]
    client = MlflowClient()

    print("")
    print("registering runner up as version 1 ...")
    first = mlflow.register_model(
        model_uri="runs:/" + str(runner_up["run_id"]) + "/model", name=MODEL_NAME
    )
    print("  version " + str(first.version))

    print("registering champion ...")
    second = mlflow.register_model(
        model_uri="runs:/" + str(best_run_id) + "/model", name=MODEL_NAME
    )
    print("  version " + str(second.version))

    client.set_registered_model_alias(
        name=MODEL_NAME, alias="champion", version=second.version
    )
    client.set_registered_model_alias(
        name=MODEL_NAME, alias="challenger", version=first.version
    )
    print("")
    print("alias champion   -> version " + str(second.version))
    print("alias challenger -> version " + str(first.version))

    # Copy the winning .keras file to models/champion.keras so the Flask app
    # and the Docker image can serve it without calling DagsHub per request.
    source = os.path.join(config.MODEL_DIR, str(best_name) + ".keras")
    destination = os.path.join(config.MODEL_DIR, "champion.keras")
    if os.path.exists(source):
        shutil.copy(source, destination)
        print("")
        print("copied " + source + " -> " + destination)
    else:
        print("")
        print("could not find " + source + ", downloading from the registry ...")
        model = mlflow.tensorflow.load_model("models:/" + MODEL_NAME + "@champion")
        model.save(destination)
        print("saved " + destination)

    print("")
    print("Task 3.6 done. Screenshot the Models tab on DagsHub.")


if __name__ == "__main__":
    main()
