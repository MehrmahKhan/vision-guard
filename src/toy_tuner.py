"""Sub-task 2.7, the toy half. A small Keras Tuner random search on
Fashion-MNIST, run before tuning the real model so that the API and the
output format were familiar first.

Search space: learning rate and the number of units in the hidden layer,
which are the two the sub-task asks for.

Run:  python -m src.toy_tuner
"""

import json
import os
import shutil

import keras_tuner as kt
import tensorflow as tf

from src import config

OUT_DIR = os.path.join(config.ARTIFACT_DIR, "toy_tuner")
TUNER_DIR = os.path.join(OUT_DIR, "search")
PROJECT = "fashion_mnist_toy"
SUBSET_TRAIN = 10000
SUBSET_TEST = 2500
MAX_TRIALS = 8
EPOCHS = 3


def load_subset():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.fashion_mnist.load_data()
    x_train = x_train[:SUBSET_TRAIN].astype("float32") / 255.0
    y_train = y_train[:SUBSET_TRAIN]
    x_test = x_test[:SUBSET_TEST].astype("float32") / 255.0
    y_test = y_test[:SUBSET_TEST]
    return (x_train, y_train), (x_test, y_test)


def build_model(hp):
    """Keras Tuner calls this once per trial with a different hp sample."""
    units = hp.Int("units", min_value=32, max_value=256, step=32)
    learning_rate = hp.Choice("learning_rate", [0.01, 0.003, 0.001, 0.0003, 0.0001])

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(28, 28)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units, activation="relu"),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(10, activation="softmax"),
    ])
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def main():
    config.set_seeds()
    config.ensure_dirs()
    os.makedirs(OUT_DIR, exist_ok=True)

    # start from a clean search so the trial numbering is not resumed
    if os.path.isdir(TUNER_DIR):
        shutil.rmtree(TUNER_DIR)

    (x_train, y_train), (x_test, y_test) = load_subset()

    tuner = kt.RandomSearch(
        build_model,
        objective="val_accuracy",
        max_trials=MAX_TRIALS,
        executions_per_trial=1,
        directory=TUNER_DIR,
        project_name=PROJECT,
        seed=config.SEED,
    )

    print("search space:")
    tuner.search_space_summary()
    print()
    print("running " + str(MAX_TRIALS) + " trials of " + str(EPOCHS) + " epochs each")
    print()

    tuner.search(
        x_train, y_train,
        validation_split=0.15,
        epochs=EPOCHS,
        batch_size=64,
        verbose=2,
    )

    print()
    print("=" * 68)
    print("  toy tuner results, sorted by validation accuracy")
    print("=" * 68)
    print()
    header = "trial".ljust(7) + "units".ljust(8) + "learning_rate".ljust(16) + "val_accuracy"
    print(header)
    print("-" * len(header))

    rows = []
    trials = tuner.oracle.get_best_trials(num_trials=MAX_TRIALS)
    rank = 0
    for trial in trials:
        rank = rank + 1
        units = trial.hyperparameters.get("units")
        learning_rate = trial.hyperparameters.get("learning_rate")
        score = trial.score
        if score is None:
            score_text = "failed"
            score_value = None
        else:
            score_text = str(round(float(score), 4))
            score_value = round(float(score), 4)
        line = str(rank).ljust(7) + str(units).ljust(8) + str(learning_rate).ljust(16) + score_text
        print(line)
        rows.append({
            "rank": rank,
            "trial_id": trial.trial_id,
            "units": units,
            "learning_rate": learning_rate,
            "val_accuracy": score_value,
        })

    best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
    print()
    print("best settings found:")
    print("  units          : " + str(best_hp.get("units")))
    print("  learning_rate  : " + str(best_hp.get("learning_rate")))

    best_model = tuner.get_best_models(num_models=1)[0]
    test_loss, test_accuracy = best_model.evaluate(x_test, y_test, verbose=0)
    print()
    print("best model on the held out test set:")
    print("  test_accuracy  : " + str(round(float(test_accuracy), 4)))
    print("  test_loss      : " + str(round(float(test_loss), 4)))

    # which hyperparameter moved the score more
    by_lr = {}
    by_units = {}
    for row in rows:
        if row["val_accuracy"] is None:
            continue
        by_lr.setdefault(row["learning_rate"], []).append(row["val_accuracy"])
        bucket = "small" if row["units"] <= 128 else "large"
        by_units.setdefault(bucket, []).append(row["val_accuracy"])

    def spread(groups):
        means = []
        for key in groups:
            values = groups[key]
            total = 0.0
            for v in values:
                total = total + v
            means.append(total / len(values))
        if len(means) < 2:
            return 0.0
        return max(means) - min(means)

    lr_spread = spread(by_lr)
    units_spread = spread(by_units)

    print()
    print("which setting mattered more:")
    print("  spread across learning rates : " + str(round(lr_spread, 4)))
    print("  spread across unit counts    : " + str(round(units_spread, 4)))
    if lr_spread > units_spread:
        print("  -> learning rate dominated")
    else:
        print("  -> unit count dominated")

    summary = {
        "trials": rows,
        "best": {
            "units": best_hp.get("units"),
            "learning_rate": best_hp.get("learning_rate"),
            "test_accuracy": round(float(test_accuracy), 4),
            "test_loss": round(float(test_loss), 4),
        },
        "spread_learning_rate": round(lr_spread, 4),
        "spread_units": round(units_spread, 4),
    }
    out_path = os.path.join(OUT_DIR, "toy_tuner_results.json")
    with open(out_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print()
    print("results written to " + out_path)


if __name__ == "__main__":
    main()
