"""Task 7 - the model API that goes on the cloud.

This is a deliberately small, separate Flask service. It holds the model and
nothing else: no HTML, no database, no sessions. Flask (the UI from Task 4) runs
on your laptop and calls this over the public internet, which is exactly the
split the assignment asks for.

It is protected by a shared secret in the X-API-Key header. Without that, anyone
who finds the URL gets free use of your inference, and a public model endpoint
is also a free way for someone to probe your model's behaviour.

Endpoints:
    GET  /health      no key needed, so the platform's own health check works
    POST /predict     multipart form field "file", or a raw image body

Run locally:
    MODEL_PATH=models/champion.keras MODEL_API_KEY=secret python model_api/main.py
"""

import io
import os
import time

import numpy as np
from flask import Flask, jsonify, request
from PIL import Image

app = Flask(__name__)

MAX_BYTES = int(float(os.environ.get("MAX_UPLOAD_MB", "5")) * 1024 * 1024)
app.config["MAX_CONTENT_LENGTH"] = MAX_BYTES

_model = None
_img_size = None


def threshold():
    value = os.environ.get("DECISION_THRESHOLD", "0.5")
    return float(value)


def model_label():
    label = os.environ.get("MODEL_VERSION", "")
    if label != "":
        return label
    return os.path.basename(os.environ.get("MODEL_PATH", "champion.keras"))


def load_model():
    global _model, _img_size
    if _model is not None:
        return _model

    path = os.environ.get("MODEL_PATH", "models/champion.keras")
    if not os.path.exists(path):
        raise FileNotFoundError("model file not found at " + path)

    import tensorflow as tf

    print("loading model: " + path)
    _model = tf.keras.models.load_model(path)

    shape = _model.input_shape
    if shape is not None and len(shape) == 4 and shape[1] is not None:
        _img_size = int(shape[1])
    else:
        _img_size = int(os.environ.get("IMG_SIZE", "160"))
    print("ready, input " + str(_img_size) + "x" + str(_img_size))
    return _model


def key_ok():
    expected = os.environ.get("MODEL_API_KEY", "").strip()
    if expected == "":
        # No key configured means the service is open. Fine for a local test,
        # never for the deployed one.
        return True
    supplied = request.headers.get("X-API-Key", "")
    return supplied == expected


def preprocess(raw_bytes):
    stream = io.BytesIO(raw_bytes)
    image = Image.open(stream)
    image.verify()

    stream = io.BytesIO(raw_bytes)
    image = Image.open(stream).convert("RGB")
    image = image.resize((_img_size, _img_size), Image.BILINEAR)
    array = np.array(image).astype("float32")
    image.close()
    array = (array / 127.5) - 1.0
    return np.expand_dims(array, axis=0)


@app.route("/health", methods=["GET"])
def health():
    payload = {}
    payload["status"] = "ok"
    payload["service"] = "visionguard-model-api"
    payload["model"] = model_label()
    payload["threshold"] = threshold()
    try:
        load_model()
        payload["model_loaded"] = True
        payload["input_size"] = _img_size
    except Exception as error:
        payload["status"] = "degraded"
        payload["model_loaded"] = False
        payload["error"] = str(error)
        return jsonify(payload), 503
    return jsonify(payload), 200


@app.route("/predict", methods=["POST"])
def predict():
    if not key_ok():
        return jsonify({"error": "invalid or missing X-API-Key"}), 401

    started = time.time()

    raw_bytes = None
    file_name = "unnamed"
    if "file" in request.files:
        uploaded = request.files["file"]
        if uploaded.filename is not None and uploaded.filename != "":
            file_name = uploaded.filename
        raw_bytes = uploaded.read()
    else:
        raw_bytes = request.get_data()

    if raw_bytes is None or len(raw_bytes) == 0:
        return jsonify({"error": "no image data received"}), 400

    if len(raw_bytes) > MAX_BYTES:
        return jsonify({"error": "image larger than the limit"}), 413

    try:
        load_model()
    except Exception as error:
        return jsonify({"error": "model unavailable: " + str(error)}), 503

    try:
        batch = preprocess(raw_bytes)
    except Exception:
        return jsonify({"error": "the data could not be decoded as an image"}), 422

    probability = float(_model.predict(batch, verbose=0)[0][0])
    limit = threshold()
    if probability >= limit:
        verdict = "AI"
        confidence = probability
    else:
        verdict = "REAL"
        confidence = 1.0 - probability

    payload = {}
    payload["verdict"] = verdict
    payload["probability_ai"] = round(probability, 6)
    payload["confidence"] = round(confidence, 6)
    payload["confidence_percent"] = round(confidence * 100.0, 2)
    payload["threshold"] = limit
    payload["model"] = model_label()
    payload["file_name"] = file_name
    payload["latency_ms"] = int((time.time() - started) * 1000)
    payload["caveat"] = (
        "Statistical estimate from a model trained on one dataset. Not proof."
    )
    return jsonify(payload), 200


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "image larger than the limit"}), 413


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "8080"))
    app.run(host="0.0.0.0", port=port, debug=False)
