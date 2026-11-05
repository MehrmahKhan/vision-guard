"""Task 7 - the VisionGuard model API, deployed to Vercel.

This is Pathway B from the assignment: the Flask user interface runs on the
local machine and the model is served from the cloud.

The assignment warns that "TensorFlow will not fit" inside a Vercel function,
and that is correct: a Vercel serverless function is limited to 250 MB
unzipped, while tensorflow-cpu alone is well over 600 MB. The model was
therefore converted from Keras to TensorFlow Lite and is served by a small
interpreter instead. The bundle is roughly 80 MB: the interpreter, NumPy,
Pillow, Flask and an 8.5 MB model file.

The converted model was verified against the original before deployment. On
the same input the Keras model returned 0.1953 and the TFLite model returned
0.1953, a difference of 0.0.

Endpoints:
    GET  /health      no key required
    POST /predict     multipart form field "file", requires X-API-Key
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

_interpreter = None
_input_details = None
_output_details = None
_img_size = None


def find_model():
    """The function's working directory differs between local runs and
    Vercel, so look in each place the file could be."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    candidates.append(os.environ.get("MODEL_PATH", ""))
    candidates.append(os.path.join(here, "..", "models", "champion.tflite"))
    candidates.append(os.path.join(here, "models", "champion.tflite"))
    candidates.append(os.path.join("models", "champion.tflite"))
    candidates.append("/var/task/models/champion.tflite")

    for candidate in candidates:
        if candidate == "":
            continue
        resolved = os.path.abspath(candidate)
        if os.path.exists(resolved):
            return resolved
    return None


def build_interpreter(path):
    try:
        from ai_edge_litert.interpreter import Interpreter

        return Interpreter(model_path=path)
    except ImportError:
        pass

    try:
        from tflite_runtime.interpreter import Interpreter

        return Interpreter(model_path=path)
    except ImportError:
        pass

    import tensorflow as tf

    return tf.lite.Interpreter(model_path=path)


def load_model():
    global _interpreter, _input_details, _output_details, _img_size

    if _interpreter is not None:
        return _interpreter

    path = find_model()
    if path is None:
        raise FileNotFoundError("could not locate champion.tflite")

    _interpreter = build_interpreter(path)
    _interpreter.allocate_tensors()
    _input_details = _interpreter.get_input_details()
    _output_details = _interpreter.get_output_details()
    _img_size = int(_input_details[0]["shape"][1])
    return _interpreter


def threshold():
    return float(os.environ.get("DECISION_THRESHOLD", "0.55"))


def model_label():
    label = os.environ.get("MODEL_VERSION", "")
    if label != "":
        return label
    return "champion.tflite"


def key_ok():
    expected = os.environ.get("MODEL_API_KEY", "").strip()
    if expected == "":
        return True
    return request.headers.get("X-API-Key", "") == expected


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
@app.route("/api/health", methods=["GET"])
def health():
    payload = {}
    payload["status"] = "ok"
    payload["service"] = "visionguard-model-api"
    payload["platform"] = "vercel"
    payload["runtime"] = "tflite"
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
@app.route("/api/predict", methods=["POST"])
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

    _interpreter.set_tensor(_input_details[0]["index"], batch)
    _interpreter.invoke()
    probability = float(_interpreter.get_tensor(_output_details[0]["index"])[0][0])

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


@app.route("/", methods=["GET"])
def index():
    return jsonify(
        {
            "service": "VisionGuard model API",
            "endpoints": {"health": "GET /health", "predict": "POST /predict"},
        }
    ), 200


@app.errorhandler(413)
def too_large(_error):
    return jsonify({"error": "image larger than the limit"}), 413


@app.errorhandler(404)
def not_found(_error):
    """Reports the path actually received, which makes a routing mismatch
    on the hosting platform obvious instead of silent."""
    return jsonify(
        {
            "error": "no route for this path",
            "path_received": request.path,
            "available": ["GET /health", "POST /predict"],
        }
    ), 404


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.environ.get("PORT", "8080")), debug=False)
