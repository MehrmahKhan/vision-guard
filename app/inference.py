"""Task 4.1 - the inference module.

The model is loaded ONCE, at import time, into a module-level singleton.

Why loading per request is unacceptable: reading a Keras file rebuilds the graph
and copies tens of megabytes of weights into memory. That is roughly 2 to 10
seconds on a CPU box. Doing it per request means every user waits for it, the
process allocates and frees that memory over and over, and a handful of
concurrent uploads will exhaust RAM and kill the container. Loading once costs
that time at start-up, where nobody is waiting.

Why training and serving preprocessing must match: the model learned a mapping
from a very specific input distribution, 160x160 pixels scaled to -1..1 by
MobileNetV2's preprocess_input. If serving instead sends 0..255, or resizes with
a different interpolation, or keeps the aspect ratio with padding, then every
feature the convolutions learned is being fed something it never saw. The model
will still return a confident-looking number, and that number will be noise.
This class of bug is silent, which is exactly what makes it dangerous.
"""

import io
import os

import numpy as np
from PIL import Image

CLASS_NAMES = ["REAL", "AI"]

_model = None
_model_path = None
_img_size = None


def _env_int(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def get_threshold():
    return _env_float("DECISION_THRESHOLD", 0.5)


def model_version_label():
    """What goes in the database as 'which model decided this'."""
    label = os.environ.get("MODEL_VERSION", "")
    if label != "":
        return label
    if _model_path is None:
        return "unknown"
    return os.path.basename(_model_path)


def load_model(path=None):
    """Loads the champion model once and caches it."""
    global _model, _model_path, _img_size

    if _model is not None:
        return _model

    if path is None:
        path = os.environ.get("MODEL_PATH", "models/champion.keras")

    if not os.path.exists(path):
        raise FileNotFoundError(
            "Model file not found at " + path
            + ". Fetch the champion with `python -m src.load_champion "
            + "--image <any.jpg> --download-to " + path + "`."
        )

    # Imported here, not at module top, so that the Flask app can start in
    # API-only mode (Task 8) on a machine without TensorFlow installed.
    import tensorflow as tf

    print("loading model from " + path + " ...")
    _model = tf.keras.models.load_model(path)
    _model_path = path

    shape = _model.input_shape
    if shape is not None and len(shape) == 4 and shape[1] is not None:
        _img_size = int(shape[1])
    else:
        _img_size = _env_int("IMG_SIZE", 160)

    print("model loaded, expects " + str(_img_size) + "x" + str(_img_size) + " input")
    return _model


def preprocess_bytes(raw_bytes, img_size=None):
    """Same steps as training: decode, convert to RGB, resize to a square,
    scale to -1..1. Raises on anything that is not a decodable image.
    """
    if img_size is None:
        if _img_size is None:
            img_size = _env_int("IMG_SIZE", 160)
        else:
            img_size = _img_size

    stream = io.BytesIO(raw_bytes)
    image = Image.open(stream)
    image.verify()

    stream = io.BytesIO(raw_bytes)
    image = Image.open(stream).convert("RGB")
    image = image.resize((img_size, img_size), Image.BILINEAR)

    array = np.array(image).astype("float32")
    image.close()
    array = (array / 127.5) - 1.0
    return np.expand_dims(array, axis=0)


def predict_bytes(raw_bytes):
    """Returns a dict: verdict, confidence, probability_ai, threshold, model."""
    model = load_model()
    batch = preprocess_bytes(raw_bytes)
    probability = float(model.predict(batch, verbose=0)[0][0])
    return build_result(probability)


def build_result(probability):
    threshold = get_threshold()
    if probability >= threshold:
        verdict = "AI"
        confidence = probability
    else:
        verdict = "REAL"
        confidence = 1.0 - probability

    result = {}
    result["verdict"] = verdict
    result["probability_ai"] = round(probability, 6)
    result["confidence"] = round(confidence, 6)
    result["confidence_percent"] = round(confidence * 100.0, 2)
    result["threshold"] = threshold
    result["model"] = model_version_label()
    result["caveat"] = (
        "This is a statistical guess from a model trained on one dataset of "
        "scraped and generated pictures. It has not seen every generator, and "
        "heavy editing, compression or screenshots of real photos can push it "
        "either way. Treat the verdict as one weak signal, never as proof."
    )
    return result


if __name__ == "__main__":
    # Task 4.1 evidence: console output for two known images.
    import sys

    if len(sys.argv) < 2:
        print("usage: python -m app.inference image1.jpg [image2.jpg ...]")
        raise SystemExit(1)

    load_model()
    index = 1
    while index < len(sys.argv):
        path = sys.argv[index]
        handle = open(path, "rb")
        data = handle.read()
        handle.close()
        outcome = predict_bytes(data)
        print("")
        print("file      : " + path)
        print("verdict   : " + outcome["verdict"])
        print("p(AI)     : " + str(outcome["probability_ai"]))
        print("confidence: " + str(outcome["confidence_percent"]) + "%")
        print("model     : " + outcome["model"])
        index = index + 1
