"""Task 2.7 (toy part) - Keras Tuner on Fashion-MNIST.

Searches the learning rate and the number of units with random search.

Run:
    python toy/toy_tuner.py --trials 6
"""

import argparse

import keras_tuner as kt
import tensorflow as tf


def build_model(hp):
    units = hp.Int("units", min_value=32, max_value=256, step=32)
    learning_rate = hp.Choice("learning_rate", [1e-2, 1e-3, 1e-4])

    model = tf.keras.Sequential(
        [
            tf.keras.layers.Flatten(input_shape=(28, 28)),
            tf.keras.layers.Dense(units, activation="relu"),
            tf.keras.layers.Dense(10, activation="softmax"),
        ]
    )
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--trials", type=int, default=6)
    parser.add_argument("--epochs", type=int, default=3)
    args = parser.parse_args()

    tf.keras.utils.set_random_seed(42)

    (train_images, train_labels), _test = tf.keras.datasets.fashion_mnist.load_data()
    train_images = train_images.astype("float32") / 255.0

    tuner = kt.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=args.trials,
        seed=42,
        directory="artifacts/tuner_toy",
        project_name="fashion_mnist",
        overwrite=True,
    )

    tuner.search_space_summary()
    tuner.search(
        train_images,
        train_labels,
        validation_split=0.1,
        epochs=args.epochs,
        batch_size=128,
        verbose=2,
    )
    print("")
    tuner.results_summary(num_trials=args.trials)

    best = tuner.get_best_hyperparameters(num_trials=1)[0]
    print("")
    print("=== best toy settings ===")
    for key in best.values:
        print("  " + str(key) + " = " + str(best.values[key]))


if __name__ == "__main__":
    main()
