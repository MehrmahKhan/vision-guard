"""Tasks 3.1 and 3.2 (toy labs) - a tiny Keras model on Fashion-MNIST with full
MLflow logging.

Run it twice with different learning rates so there are two runs to compare:

  Local (3.1):
      python toy/toy_mlflow.py --lr 0.001
      python toy/toy_mlflow.py --lr 0.0001
      mlflow ui --backend-store-uri ./mlruns

  Against DagsHub (3.2) - the tracking URI comes from .env, so just:
      python toy/toy_mlflow.py --lr 0.001 --remote
"""

import argparse
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mlflow
import tensorflow as tf
from dotenv import load_dotenv

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
load_dotenv()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--units", type=int, default=64)
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument(
        "--remote",
        action="store_true",
        help="log to the DagsHub MLflow server from MLFLOW_TRACKING_URI",
    )
    args = parser.parse_args()

    if args.remote:
        uri = os.environ.get("MLFLOW_TRACKING_URI", "")
        if uri == "":
            raise SystemExit("MLFLOW_TRACKING_URI is not set in .env")
        mlflow.set_tracking_uri(uri)
        print("tracking to " + uri)
        mlflow.set_experiment("toy-fashion-mnist")
    else:
        mlflow.set_tracking_uri("file:./mlruns")
        print("tracking locally to ./mlruns")
        mlflow.set_experiment("toy-fashion-mnist-local")

    tf.keras.utils.set_random_seed(42)

    (train_images, train_labels), (test_images, test_labels) = (
        tf.keras.datasets.fashion_mnist.load_data()
    )
    train_images = train_images.astype("float32") / 255.0
    test_images = test_images.astype("float32") / 255.0

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(args.units, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=args.lr),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )

    run_name = "lr" + str(args.lr) + "-units" + str(args.units)
    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("learning_rate", args.lr)
        mlflow.log_param("units", args.units)
        mlflow.log_param("epochs", args.epochs)
        mlflow.log_param("optimizer", "adam")
        mlflow.set_tag("stage", "toy")
        mlflow.set_tag("dataset", "fashion_mnist")

        history = model.fit(
            train_images,
            train_labels,
            validation_split=0.1,
            epochs=args.epochs,
            batch_size=128,
            verbose=2,
        )

        epoch = 0
        while epoch < args.epochs:
            for key in history.history:
                mlflow.log_metric(key, float(history.history[key][epoch]), step=epoch)
            epoch = epoch + 1

        test_loss, test_accuracy = model.evaluate(test_images, test_labels, verbose=0)
        mlflow.log_metric("test_loss", float(test_loss))
        mlflow.log_metric("test_accuracy", float(test_accuracy))
        print("test accuracy: " + str(round(float(test_accuracy), 4)))

        os.makedirs("artifacts", exist_ok=True)
        figure, axis = plt.subplots(figsize=(6, 4))
        axis.plot(history.history["loss"], label="train loss")
        axis.plot(history.history["val_loss"], label="val loss")
        axis.set_xlabel("epoch")
        axis.set_ylabel("loss")
        axis.set_title("Toy run, lr=" + str(args.lr))
        axis.legend()
        figure.tight_layout()
        plot_path = os.path.join("artifacts", "toy_" + run_name + "_loss.png")
        figure.savefig(plot_path, dpi=130)
        plt.close(figure)
        mlflow.log_artifact(plot_path, artifact_path="figures")

        model_path = os.path.join("artifacts", "toy_" + run_name + ".keras")
        model.save(model_path)
        mlflow.log_artifact(model_path, artifact_path="keras_model")

        print("logged run " + run_name)


if __name__ == "__main__":
    main()
