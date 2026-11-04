"""Task 2.7 (apply part) - Keras Tuner on VisionGuard.

Searches learning rate, dropout and the number of unfrozen backbone layers.
Default is 10 trials, which clears the "at least 8" requirement.

Run:
    python -m src.tune --trials 10
"""

import argparse
import json
import os

import keras_tuner as kt
import mlflow
import tensorflow as tf

from src import config, models
from src.data_pipeline import make_dataset


def build_hypermodel(hp):
    learning_rate = hp.Choice("learning_rate", [1e-3, 5e-4, 1e-4])
    dropout = hp.Float("dropout", min_value=0.1, max_value=0.5, step=0.1)
    unfreeze = hp.Choice("unfrozen_layers", [0, 10, 20, 40])

    model, _preprocess, backbone = models.build_transfer_model(
        backbone_name="MobileNetV2",
        img_size=config.IMG_SIZE,
        dropout=dropout,
        learning_rate=learning_rate,
    )

    if unfreeze > 0:
        models.unfreeze_last_layers(
            model, backbone, how_many=unfreeze, learning_rate=learning_rate / 10.0
        )

    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=10)
    parser.add_argument("--epochs", type=int, default=5)
    args = parser.parse_args()

    config.ensure_dirs()
    config.set_seeds()

    _model, preprocess, _backbone = models.build_transfer_model("MobileNetV2")
    train_dataset = make_dataset(
        "train", augment=True, shuffle=True, preprocess_fn=preprocess
    )
    validation_dataset = make_dataset("val", preprocess_fn=preprocess)

    tuner = kt.RandomSearch(
        build_hypermodel,
        objective=kt.Objective("val_auc", direction="max"),
        max_trials=args.trials,
        seed=config.SEED,
        directory=os.path.join(config.ARTIFACT_DIR, "tuner"),
        project_name="visionguard",
        overwrite=True,
    )

    tuner.search_space_summary()

    stop_early = tf.keras.callbacks.EarlyStopping(monitor="val_loss", patience=2)
    tuner.search(
        train_dataset,
        validation_data=validation_dataset,
        epochs=args.epochs,
        callbacks=[stop_early],
        verbose=2,
    )

    print("")
    tuner.results_summary(num_trials=args.trials)

    best_hyperparameters = tuner.get_best_hyperparameters(num_trials=1)[0]
    best_values = {}
    for key in best_hyperparameters.values:
        best_values[key] = best_hyperparameters.values[key]

    print("")
    print("=== best hyperparameters ===")
    for key in best_values:
        print("  " + str(key) + " = " + str(best_values[key]))

    output = os.path.join(config.ARTIFACT_DIR, "tuner_best.json")
    handle = open(output, "w", encoding="utf-8")
    json.dump(best_values, handle, indent=2)
    handle.close()
    print("wrote " + output)

    # Record the search itself as one MLflow run so it sits next to the others.
    if config.MLFLOW_TRACKING_URI != "":
        mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(config.EXPERIMENT_NAME)
    with mlflow.start_run(run_name="keras-tuner-search"):
        mlflow.log_param("tuner", "RandomSearch")
        mlflow.log_param("max_trials", args.trials)
        mlflow.log_param("epochs_per_trial", args.epochs)
        for key in best_values:
            mlflow.log_param("best_" + str(key), best_values[key])
        mlflow.log_artifact(output, artifact_path="tuner")


if __name__ == "__main__":
    main()
