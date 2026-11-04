"""Task 1.3 - visual EDA.

Reads artifacts/dataset_inventory.csv (produced by src/audit_dataset.py) and
writes three figures into artifacts/.

Run:
    python -m src.eda
"""

import os

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image

from src import config


def load_inventory():
    path = os.path.join(config.ARTIFACT_DIR, "dataset_inventory.csv")
    if not os.path.exists(path):
        raise SystemExit("Run `python -m src.audit_dataset` first, " + path + " is missing.")
    frame = pd.read_csv(path)
    return frame[frame["corrupt"] == False]  # noqa: E712


def sample_grid(frame, per_class=6):
    classes = sorted(frame["class"].unique())
    figure, axes = plt.subplots(
        len(classes), per_class, figsize=(per_class * 2.2, len(classes) * 2.4)
    )

    row_index = 0
    for class_name in classes:
        subset = frame[frame["class"] == class_name].sample(
            n=per_class, random_state=config.SEED
        )
        paths = subset["path"].tolist()
        column_index = 0
        for path in paths:
            axis = axes[row_index][column_index]
            try:
                image = Image.open(path).convert("RGB")
                axis.imshow(image)
                image.close()
            except Exception:
                axis.text(0.5, 0.5, "unreadable", ha="center")
            axis.set_xticks([])
            axis.set_yticks([])
            if column_index == 0:
                axis.set_ylabel(str(class_name), fontsize=11)
            column_index = column_index + 1
        row_index = row_index + 1

    figure.suptitle("Sample images per class", fontsize=13)
    figure.tight_layout()
    output = os.path.join(config.ARTIFACT_DIR, "eda_samples.png")
    figure.savefig(output, dpi=130)
    plt.close(figure)
    print("wrote " + output)


def histograms(frame):
    columns = ["width", "height", "file_size_bytes"]
    titles = ["Image width (px)", "Image height (px)", "File size (bytes)"]
    classes = sorted(frame["class"].unique())

    figure, axes = plt.subplots(1, 3, figsize=(15, 4))
    index = 0
    while index < len(columns):
        axis = axes[index]
        for class_name in classes:
            values = frame[frame["class"] == class_name][columns[index]]
            axis.hist(values, bins=40, alpha=0.55, label=str(class_name))
        axis.set_title(titles[index])
        axis.set_xlabel(columns[index])
        axis.set_ylabel("count")
        axis.legend()
        index = index + 1

    figure.tight_layout()
    output = os.path.join(config.ARTIFACT_DIR, "eda_histograms.png")
    figure.savefig(output, dpi=130)
    plt.close(figure)
    print("wrote " + output)


def aspect_ratio_plot(frame):
    figure, axis = plt.subplots(figsize=(6, 5))
    classes = sorted(frame["class"].unique())
    for class_name in classes:
        subset = frame[frame["class"] == class_name]
        axis.scatter(subset["width"], subset["height"], s=6, alpha=0.4, label=str(class_name))
    axis.set_xlabel("width (px)")
    axis.set_ylabel("height (px)")
    axis.set_title("Resolution scatter, width against height")
    axis.legend()
    figure.tight_layout()
    output = os.path.join(config.ARTIFACT_DIR, "eda_resolution_scatter.png")
    figure.savefig(output, dpi=130)
    plt.close(figure)
    print("wrote " + output)


def main():
    config.ensure_dirs()
    frame = load_inventory()
    sample_grid(frame)
    histograms(frame)
    aspect_ratio_plot(frame)


if __name__ == "__main__":
    main()
