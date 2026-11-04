"""Tasks 2.4 - 2.6: curves, threshold choice, metrics, confusion matrix, ROC,
and the grid of misclassified images.

Imported by src/train.py, but also runnable on a saved model:
    python -m src.evaluate --model models/champion.keras --run-name champion
"""

import argparse
import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from src import config


def collect_predictions(model, dataset):
    """Runs the model over a dataset and returns (probabilities, labels)."""
    probability_batches = []
    label_batches = []
    for images, labels in dataset:
        predictions = model.predict(images, verbose=0)
        probability_batches.append(np.array(predictions).reshape(-1))
        label_batches.append(np.array(labels).reshape(-1))

    probabilities = np.concatenate(probability_batches)
    labels = np.concatenate(label_batches)
    return probabilities, labels


def choose_threshold(probabilities, labels, beta=0.5):
    """Sweeps thresholds on the VALIDATION set and picks the best F-beta score.

    beta = 0.5 weights precision higher than recall. For VisionGuard, the
    positive class is "AI-generated", so a false positive means calling a real
    photograph fake. That is the worse error: it accuses a genuine image. Being
    stricter about claiming "AI" therefore favours precision, which is what
    beta < 1 does.

    The sweep is done on validation data, never on test, because choosing a
    threshold is a fitting step. Doing it on test would leak the test set.
    """
    rows = []
    best_threshold = 0.5
    best_score = -1.0

    candidate = 0.05
    while candidate <= 0.951:
        predictions = (probabilities >= candidate).astype(int)
        precision = precision_score(labels, predictions, zero_division=0)
        recall = recall_score(labels, predictions, zero_division=0)
        if precision + recall == 0:
            f_beta = 0.0
        else:
            numerator = (1 + beta * beta) * precision * recall
            denominator = (beta * beta * precision) + recall
            f_beta = numerator / denominator

        row = {}
        row["threshold"] = round(candidate, 3)
        row["precision"] = precision
        row["recall"] = recall
        row["f1"] = f1_score(labels, predictions, zero_division=0)
        row["f_beta_0.5"] = f_beta
        row["accuracy"] = accuracy_score(labels, predictions)
        rows.append(row)

        if f_beta > best_score:
            best_score = f_beta
            best_threshold = round(candidate, 3)

        candidate = candidate + 0.05

    return best_threshold, pd.DataFrame(rows)


def full_evaluation(probabilities, labels, threshold, run_name):
    predictions = (probabilities >= threshold).astype(int)

    metrics = {}
    metrics["accuracy"] = accuracy_score(labels, predictions)
    metrics["precision"] = precision_score(labels, predictions, zero_division=0)
    metrics["recall"] = recall_score(labels, predictions, zero_division=0)
    metrics["f1"] = f1_score(labels, predictions, zero_division=0)
    try:
        metrics["roc_auc"] = roc_auc_score(labels, probabilities)
    except ValueError:
        metrics["roc_auc"] = 0.0

    matrix = confusion_matrix(labels, predictions)
    metrics["true_negative"] = int(matrix[0][0])
    metrics["false_positive"] = int(matrix[0][1])
    metrics["false_negative"] = int(matrix[1][0])
    metrics["true_positive"] = int(matrix[1][1])

    figure_paths = []
    figure_paths.append(plot_confusion(matrix, run_name))
    figure_paths.append(plot_roc(labels, probabilities, metrics["roc_auc"], run_name))

    metrics_path = os.path.join(config.ARTIFACT_DIR, run_name + "_test_metrics.csv")
    pd.DataFrame([metrics]).to_csv(metrics_path, index=False)

    return metrics, figure_paths


def plot_confusion(matrix, run_name):
    figure, axis = plt.subplots(figsize=(5, 4.5))
    image = axis.imshow(matrix, cmap="Blues")
    axis.set_xticks([0, 1])
    axis.set_yticks([0, 1])
    axis.set_xticklabels(config.CLASS_NAMES)
    axis.set_yticklabels(config.CLASS_NAMES)
    axis.set_xlabel("predicted")
    axis.set_ylabel("actual")
    axis.set_title("Confusion matrix, " + run_name)

    row = 0
    while row < 2:
        column = 0
        while column < 2:
            axis.text(
                column,
                row,
                str(matrix[row][column]),
                ha="center",
                va="center",
                color="black",
                fontsize=14,
            )
            column = column + 1
        row = row + 1

    figure.colorbar(image, ax=axis)
    figure.tight_layout()
    output = os.path.join(config.ARTIFACT_DIR, run_name + "_confusion_matrix.png")
    figure.savefig(output, dpi=130)
    plt.close(figure)
    return output


def plot_roc(labels, probabilities, auc_value, run_name):
    false_positive_rate, true_positive_rate, _thresholds = roc_curve(labels, probabilities)
    figure, axis = plt.subplots(figsize=(5.5, 5))
    axis.plot(
        false_positive_rate,
        true_positive_rate,
        label="AUC = " + str(round(float(auc_value), 4)),
    )
    axis.plot([0, 1], [0, 1], linestyle="--", color="grey", label="random")
    axis.set_xlabel("false positive rate")
    axis.set_ylabel("true positive rate")
    axis.set_title("ROC curve, " + run_name)
    axis.legend()
    figure.tight_layout()
    output = os.path.join(config.ARTIFACT_DIR, run_name + "_roc_curve.png")
    figure.savefig(output, dpi=130)
    plt.close(figure)
    return output


def plot_history(history, best_epoch, run_name):
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.5))

    axes[0].plot(history["loss"], label="train loss")
    if "val_loss" in history:
        axes[0].plot(history["val_loss"], label="val loss")
    axes[0].axvline(
        best_epoch - 1, linestyle="--", color="red", label="best epoch " + str(best_epoch)
    )
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("binary cross-entropy")
    axes[0].set_title("Loss")
    axes[0].legend()

    if "accuracy" in history:
        axes[1].plot(history["accuracy"], label="train accuracy")
    if "val_accuracy" in history:
        axes[1].plot(history["val_accuracy"], label="val accuracy")
    axes[1].axvline(
        best_epoch - 1, linestyle="--", color="red", label="best epoch " + str(best_epoch)
    )
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()

    figure.suptitle("Training curves, " + run_name)
    figure.tight_layout()
    output = os.path.join(config.ARTIFACT_DIR, run_name + "_curves.png")
    figure.savefig(output, dpi=130)
    plt.close(figure)
    return output


def error_grid(probabilities, labels, threshold, run_name, img_size, how_many=10):
    """Task 2.6 - ten misclassified test images with their probabilities."""
    test_csv = os.path.join(config.SPLIT_DIR, "test.csv")
    if not os.path.exists(test_csv):
        return None

    frame = pd.read_csv(test_csv)
    if len(frame) != len(probabilities):
        print("warning: test.csv length does not match prediction count, skipping grid")
        return None

    predictions = (probabilities >= threshold).astype(int)
    wrong_indices = []
    index = 0
    while index < len(predictions):
        if int(predictions[index]) != int(labels[index]):
            wrong_indices.append(index)
        index = index + 1

    if len(wrong_indices) == 0:
        print("no misclassified test images, nothing to plot")
        return None

    # Show the most confident mistakes first, they are the most informative.
    def confidence_of(position):
        value = float(probabilities[position])
        if int(labels[position]) == 1:
            return 1.0 - value
        return value

    wrong_indices.sort(key=confidence_of, reverse=True)
    selected = wrong_indices[:how_many]

    columns = 5
    rows = 2
    figure, axes = plt.subplots(rows, columns, figsize=(columns * 2.6, rows * 3.0))

    slot = 0
    row = 0
    while row < rows:
        column = 0
        while column < columns:
            axis = axes[row][column]
            axis.set_xticks([])
            axis.set_yticks([])
            if slot < len(selected):
                position = selected[slot]
                path = frame.iloc[position]["path"]
                try:
                    image = Image.open(path).convert("RGB").resize((img_size, img_size))
                    axis.imshow(image)
                    image.close()
                except Exception:
                    axis.text(0.5, 0.5, "unreadable", ha="center")
                actual = config.CLASS_NAMES[int(labels[position])]
                predicted = config.CLASS_NAMES[int(predictions[position])]
                axis.set_title(
                    "true " + actual + "\npred " + predicted
                    + "  p=" + str(round(float(probabilities[position]), 3)),
                    fontsize=8,
                )
            slot = slot + 1
            column = column + 1
        row = row + 1

    figure.suptitle("Most confident misclassifications, " + run_name)
    figure.tight_layout()
    output = os.path.join(config.ARTIFACT_DIR, run_name + "_errors.png")
    figure.savefig(output, dpi=130)
    plt.close(figure)

    error_rows = []
    for position in selected:
        row_data = {}
        row_data["path"] = frame.iloc[position]["path"]
        row_data["true_label"] = config.CLASS_NAMES[int(labels[position])]
        row_data["predicted"] = config.CLASS_NAMES[int(predictions[position])]
        row_data["probability_ai"] = float(probabilities[position])
        error_rows.append(row_data)
    pd.DataFrame(error_rows).to_csv(
        os.path.join(config.ARTIFACT_DIR, run_name + "_errors.csv"), index=False
    )

    print("wrote " + output)
    return output


def main():
    import tensorflow as tf

    from src.data_pipeline import make_dataset

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--run-name", default="standalone")
    parser.add_argument("--threshold", type=float, default=None)
    parser.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    args = parser.parse_args()

    config.ensure_dirs()
    model = tf.keras.models.load_model(args.model)

    validation_dataset = make_dataset("val", img_size=args.img_size)
    test_dataset = make_dataset("test", img_size=args.img_size)

    if args.threshold is None:
        validation_probabilities, validation_labels = collect_predictions(
            model, validation_dataset
        )
        threshold, table = choose_threshold(validation_probabilities, validation_labels)
        table.to_csv(
            os.path.join(config.ARTIFACT_DIR, args.run_name + "_threshold_sweep.csv"),
            index=False,
        )
    else:
        threshold = args.threshold

    print("threshold: " + str(threshold))
    probabilities, labels = collect_predictions(model, test_dataset)
    metrics, _figures = full_evaluation(probabilities, labels, threshold, args.run_name)
    for key in metrics:
        print("  " + key.ljust(16) + str(metrics[key]))
    error_grid(probabilities, labels, threshold, args.run_name, args.img_size)


if __name__ == "__main__":
    main()
