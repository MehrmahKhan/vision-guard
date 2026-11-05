"""Task 3.2 - a small, fast MLflow experiment used to learn the tracking API
before wiring it into the real VisionGuard training loop.

Dataset: Fashion-MNIST (built into Keras, downloads once, ~30 MB).
Model:   a tiny dense network. Nothing here is meant to be accurate; the point
         is to show parameters, metrics, tags and artifacts landing in MLflow.

Three runs sweep the learning rate with manual logging, then a fourth run uses
mlflow.tensorflow.autolog() so the two styles can be compared side by side in
the DagsHub UI.

Run:  python -m src.toy_mlflow
"""

import json
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import numpy as np
import tensorflow as tf
from sklearn.metrics import classification_report, confusion_matrix

from src import config

TOY_EXPERIMENT = "toy-fashion-mnist"
OUT_DIR = os.path.join(config.ARTIFACT_DIR, "toy")
SUBSET_TRAIN = 12000
SUBSET_TEST = 3000
EPOCHS = 4
LEARNING_RATES = [0.01, 0.001, 0.0001]

CLASSES = [
    "tshirt", "trouser", "pullover", "dress", "coat",
    "sandal", "shirt", "sneaker", "bag", "boot",
]


def load_subset():
    """Fashion-MNIST, cut down so each run finishes in well under a minute."""
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train[:SUBSET_TRAIN].astype("float32") / 255.0
    y_train = y_train[:SUBSET_TRAIN]
    x_test = x_test[:SUBSET_TEST].astype("float32") / 255.0
    y_test = y_test[:SUBSET_TEST]
    return (x_train, y_train), (x_test, y_test)


def build_model(learning_rate, hidden_units=128):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(hidden_units, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def save_confusion_matrix(y_true, y_pred, path):
    matrix = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(10), CLASSES, rotation=45, ha="right")
    ax.set_yticks(range(10), CLASSES)
    ax.set_xlabel("predicted")
    ax.set_ylabel("true")
    ax.set_title("Toy run confusion matrix")
    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def manual_run(learning_rate, data):
    """One run with every value logged by hand, which is what Task 3 asks for."""
    (x_train, y_train), (x_test, y_test) = data
    config.set_seeds()

    run_name = "manual-lr-" + str(learning_rate)
    with mlflow.start_run(run_name=run_name) as run:
        mlflow.set_tags({
            "task": "3.2-toy-tracking",
            "logging_style": "manual",
            "dataset": "fashion-mnist-subset",
            "student_roll": "23L-0623",
        })
        mlflow.log_params({
            "learning_rate": learning_rate,
            "hidden_units": 128,
            "dropout": 0.2,
            "optimizer": "adam",
            "epochs": EPOCHS,
            "batch_size": 64,
            "train_images": SUBSET_TRAIN,
            "test_images": SUBSET_TEST,
            "seed": config.SEED,
        })

        model = build_model(learning_rate)
        history = model.fit(
            x_train, y_train,
            validation_split=0.15,
            epochs=EPOCHS,
            batch_size=64,
            verbose=2,
        )

        # per-epoch curves, so the MLflow metric plots have a shape to show
        for epoch in range(EPOCHS):
            mlflow.log_metrics({
                "train_loss": float(history.history["loss"][epoch]),
                "train_accuracy": float(history.history["accuracy"][epoch]),
                "val_loss": float(history.history["val_loss"][epoch]),
                "val_accuracy": float(history.history["val_accuracy"][epoch]),
            }, step=epoch)

        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        probabilities = model.predict(x_test, verbose=0)
        predictions = probabilities.argmax(axis=1)

        mlflow.log_metrics({
            "test_loss": float(test_loss),
            "test_accuracy": float(test_accuracy),
        })

        os.makedirs(OUT_DIR, exist_ok=True)
        cm_path = os.path.join(OUT_DIR, "confusion_lr_" + str(learning_rate) + ".png")
        save_confusion_matrix(y_test, predictions, cm_path)
        mlflow.log_artifact(cm_path, artifact_path="plots")

        report = classification_report(
            y_test, predictions, target_names=CLASSES, output_dict=True, zero_division=0
        )
        report_path = os.path.join(OUT_DIR, "report_lr_" + str(learning_rate) + ".json")
        with open(report_path, "w", encoding="utf-8") as handle:
            json.dump(report, handle, indent=2)
        mlflow.log_artifact(report_path, artifact_path="reports")

        print("  manual run", run.info.run_id, "test_accuracy", round(float(test_accuracy), 4))
        return float(test_accuracy)


def autolog_run(data):
    """Same model, but MLflow fills the run in for us. Kept separate so the
    report can show what autolog captures that manual logging misses."""
    (x_train, y_train), (x_test, y_test) = data
    config.set_seeds()

    mlflow.tensorflow.autolog(log_models=False)
    with mlflow.start_run(run_name="autolog-lr-0.001") as run:
        mlflow.set_tags({
            "task": "3.2-toy-tracking",
            "logging_style": "autolog",
            "dataset": "fashion-mnist-subset",
            "student_roll": "23L-0623",
        })
        model = build_model(0.001)
        model.fit(
            x_train, y_train,
            validation_split=0.15,
            epochs=EPOCHS,
            batch_size=64,
            verbose=2,
        )
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
        print("  autolog run", run.info.run_id, "test_accuracy", round(float(test_accuracy), 4))
    mlflow.tensorflow.autolog(disable=True)
    return float(test_accuracy)


def main():
    if not config.MLFLOW_TRACKING_URI:
        raise SystemExit(
            "MLFLOW_TRACKING_URI is empty. Check your .env file before running this."
        )

    mlflow.set_tracking_uri(config.MLFLOW_TRACKING_URI)
    mlflow.set_experiment(TOY_EXPERIMENT)
    print("tracking to", config.MLFLOW_TRACKING_URI)
    print("experiment", TOY_EXPERIMENT)

    config.ensure_dirs()
    os.makedirs(OUT_DIR, exist_ok=True)

    print("loading fashion-mnist subset")
    data = load_subset()

    results = {}
    for learning_rate in LEARNING_RATES:
        print("manual run, learning rate", learning_rate)
        results["manual_lr_" + str(learning_rate)] = manual_run(learning_rate, data)

    print("autolog run")
    results["autolog_lr_0.001"] = autolog_run(data)

    print()
    print("summary")
    for name, accuracy in sorted(results.items(), key=lambda item: -item[1]):
        print("  " + name.ljust(22), round(accuracy, 4))
    best = max(results, key=results.get)
    print()
    print("best toy run:", best, round(results[best], 4))
    print("open the", TOY_EXPERIMENT, "experiment on DagsHub to screenshot it")


if __name__ == "__main__":
    main()
