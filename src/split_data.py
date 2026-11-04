"""Builds train / validation / test lists as CSV files.

Splitting happens on the file list, not on the images, and duplicates are
dropped first so the same picture can never sit in two splits (that would leak
test information into training).

Run:
    python -m src.split_data
"""

import os

import pandas as pd
from sklearn.model_selection import train_test_split

from src import config


def main():
    config.ensure_dirs()
    inventory_path = os.path.join(config.ARTIFACT_DIR, "dataset_inventory.csv")
    if not os.path.exists(inventory_path):
        raise SystemExit("Run `python -m src.audit_dataset` first.")

    frame = pd.read_csv(inventory_path)
    before = len(frame)

    frame = frame[frame["corrupt"] == False]  # noqa: E712
    frame = frame.drop_duplicates(subset=["sha256"], keep="first")
    after = len(frame)
    print("rows before cleaning: " + str(before) + ", after: " + str(after))

    # Map the folder name to a 0/1 label. Anything whose folder name contains
    # "ai" or "fake" becomes the positive class (1).
    labels = []
    for class_name in frame["class"].tolist():
        lowered = str(class_name).lower()
        if "ai" in lowered or "fake" in lowered:
            labels.append(1)
        else:
            labels.append(0)
    frame = frame.assign(label=labels)

    print("")
    print("label counts (0 = REAL, 1 = AI):")
    print(frame["label"].value_counts().to_string())

    train_frame, temp_frame = train_test_split(
        frame,
        test_size=0.30,
        random_state=config.SEED,
        stratify=frame["label"],
    )
    validation_frame, test_frame = train_test_split(
        temp_frame,
        test_size=0.50,
        random_state=config.SEED,
        stratify=temp_frame["label"],
    )

    columns = ["path", "label", "class"]
    names = ["train", "val", "test"]
    frames = [train_frame, validation_frame, test_frame]

    index = 0
    while index < len(names):
        output = os.path.join(config.SPLIT_DIR, names[index] + ".csv")
        frames[index][columns].to_csv(output, index=False)
        print("wrote " + output + "  rows=" + str(len(frames[index])))
        index = index + 1


if __name__ == "__main__":
    main()
