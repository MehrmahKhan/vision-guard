"""Task 1.2 - dataset audit.

Walks the dataset folder, records one row per image, flags unreadable files and
exact duplicates (same SHA-256 of the raw bytes), and prints the summary
statistics the report asks for.

Run:
    python -m src.audit_dataset
"""

import hashlib
import os
import sys

import pandas as pd
from PIL import Image

from src import config

VALID_EXTENSIONS = [".jpg", ".jpeg", ".png", ".bmp", ".webp", ".gif", ".tif", ".tiff"]


def sha256_of_file(path):
    """Hash the raw bytes in 1 MB chunks so a big file never loads fully."""
    hasher = hashlib.sha256()
    handle = open(path, "rb")
    while True:
        chunk = handle.read(1024 * 1024)
        if not chunk:
            break
        hasher.update(chunk)
    handle.close()
    return hasher.hexdigest()


def list_class_folders(root):
    """Every immediate sub-folder of root is one class."""
    folders = []
    for entry in sorted(os.listdir(root)):
        full = os.path.join(root, entry)
        if os.path.isdir(full):
            folders.append(entry)
    return folders


def audit(root):
    rows = []
    class_folders = list_class_folders(root)
    if len(class_folders) == 0:
        print("No class sub-folders found under " + root)
        sys.exit(1)

    for class_name in class_folders:
        class_path = os.path.join(root, class_name)
        for current_dir, _subdirs, files in os.walk(class_path):
            for file_name in files:
                extension = os.path.splitext(file_name)[1].lower()
                if extension not in VALID_EXTENSIONS:
                    continue

                full_path = os.path.join(current_dir, file_name)
                row = {}
                row["path"] = full_path.replace("\\", "/")
                row["class"] = class_name
                row["file_name"] = file_name
                row["extension"] = extension

                try:
                    row["file_size_bytes"] = os.path.getsize(full_path)
                except OSError:
                    row["file_size_bytes"] = 0

                row["width"] = 0
                row["height"] = 0
                row["mode"] = ""
                row["format"] = ""
                row["corrupt"] = False

                try:
                    image = Image.open(full_path)
                    image.verify()
                    # verify() leaves the file unusable, so reopen to read metadata
                    image = Image.open(full_path)
                    row["width"] = image.width
                    row["height"] = image.height
                    row["mode"] = image.mode
                    if image.format is None:
                        row["format"] = ""
                    else:
                        row["format"] = image.format
                    image.close()
                except Exception as error:
                    row["corrupt"] = True
                    row["error"] = str(error)

                try:
                    row["sha256"] = sha256_of_file(full_path)
                except OSError:
                    row["sha256"] = ""

                rows.append(row)

    return pd.DataFrame(rows)


def summarise(frame):
    print("")
    print("=== counts per class ===")
    print(frame["class"].value_counts().to_string())

    readable = frame[frame["corrupt"] == False]  # noqa: E712

    print("")
    print("=== width / height / file size per class (min, median, max) ===")
    columns = ["width", "height", "file_size_bytes"]
    for class_name in sorted(readable["class"].unique()):
        subset = readable[readable["class"] == class_name]
        print("")
        print("class: " + str(class_name) + "  (n=" + str(len(subset)) + ")")
        for column in columns:
            minimum = subset[column].min()
            median = subset[column].median()
            maximum = subset[column].max()
            print(
                "  "
                + column.ljust(18)
                + " min="
                + str(minimum)
                + "  median="
                + str(median)
                + "  max="
                + str(maximum)
            )

    print("")
    print("=== colour modes ===")
    print(readable["mode"].value_counts().to_string())

    print("")
    print("=== formats ===")
    print(readable["format"].value_counts().to_string())

    corrupt_count = int(frame["corrupt"].sum())
    print("")
    print("=== integrity ===")
    print("corrupt / unreadable files: " + str(corrupt_count))

    duplicate_mask = frame.duplicated(subset=["sha256"], keep=False)
    duplicate_frame = frame[duplicate_mask & (frame["sha256"] != "")]
    unique_hashes = duplicate_frame["sha256"].nunique()
    print("duplicate files (exact byte match): " + str(len(duplicate_frame)))
    print("distinct hashes involved in duplication: " + str(unique_hashes))

    cross_class = 0
    for _hash_value, group in duplicate_frame.groupby("sha256"):
        if group["class"].nunique() > 1:
            cross_class = cross_class + 1
    print("duplicate groups that span BOTH classes (label conflicts): " + str(cross_class))

    return duplicate_frame


def main():
    config.ensure_dirs()
    root = config.DATA_DIR
    print("auditing: " + os.path.abspath(root))

    frame = audit(root)
    print("total image files found: " + str(len(frame)))

    duplicate_frame = summarise(frame)

    inventory_path = os.path.join(config.ARTIFACT_DIR, "dataset_inventory.csv")
    duplicates_path = os.path.join(config.ARTIFACT_DIR, "duplicates.csv")
    corrupt_path = os.path.join(config.ARTIFACT_DIR, "corrupt_files.csv")

    frame.to_csv(inventory_path, index=False)
    duplicate_frame.to_csv(duplicates_path, index=False)
    frame[frame["corrupt"] == True].to_csv(corrupt_path, index=False)  # noqa: E712

    print("")
    print("wrote " + inventory_path)
    print("wrote " + duplicates_path)
    print("wrote " + corrupt_path)


if __name__ == "__main__":
    main()
