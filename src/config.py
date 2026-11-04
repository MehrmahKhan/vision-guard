"""Central settings. Everything comes from environment variables so that the
same code runs on your laptop, inside Docker and on the cloud VM without edits.
"""

import os
import random

import numpy as np
from dotenv import load_dotenv

load_dotenv()


def _get_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _get_float(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


SEED = _get_int("SEED", 42)
IMG_SIZE = _get_int("IMG_SIZE", 160)
BATCH_SIZE = _get_int("BATCH_SIZE", 32)

DATA_DIR = os.environ.get("DATA_DIR", "data/raw")
SPLIT_DIR = "data/splits"
ARTIFACT_DIR = "artifacts"
MODEL_DIR = "models"

MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", "")
EXPERIMENT_NAME = os.environ.get("EXPERIMENT_NAME", "VisionGuard")

DECISION_THRESHOLD = _get_float("DECISION_THRESHOLD", 0.5)

# class_index -> name. Label 1 is the positive class for the sigmoid output.
CLASS_NAMES = ["REAL", "AI"]


def set_seeds(seed=None):
    """Fix every random source we touch so a rerun gives the same numbers."""
    if seed is None:
        seed = SEED
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    import tensorflow as tf

    tf.random.set_seed(seed)
    tf.keras.utils.set_random_seed(seed)
    return seed


def ensure_dirs():
    for folder in [SPLIT_DIR, ARTIFACT_DIR, MODEL_DIR]:
        os.makedirs(folder, exist_ok=True)
