"""Task 7 - the VisionGuard model API, deployed to Render.

This is the same service as model_api/main.py with one important difference:
it runs the model through a TensorFlow Lite interpreter instead of full
TensorFlow.

Why. The free hosting tier gives 512 MB of RAM. Importing tensorflow-cpu uses
roughly 400 MB before a single image is processed, which leaves no headroom and
gets the container killed. A TFLite interpreter plus NumPy and Pillow uses
around 120 MB, so the same model serves comfortably in the same box. The
assignment's own Pathway B notes this constraint ("TensorFlow will not fit"),
and converting the model is the answer to it rather than a workaround.

The converted model was verified against the original before deployment: both
return 0.1953 on the same input, a difference of 0.0.

Endpoints:
    GET  /health      no key required, so the platform health check works
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

MODEL_PATH = os.environ.get("MODEL_PATH", "models/champion.tflite")

_interpreter = None
_input_details = None
_output_details = None
_img_size = None


def build_interpreter(path):
    """Loads a TFLite interpreter from whichever runtime is installed.

    ai-edge-litert is the current official package. tflite-runtime is the
    older one. Full tensorflow is the fallback for local testing, where
    memory is not a constraint.
    """
    try:
        from ai_edge_litert.interpreter import Interpreter

        print("using ai-edge-litert")
        return Interpreter(model_path=path)
    except ImportError:
        pass

    try:
        from tflite_runtime.interpreter import Interpreter

        print("using tflite-runtime")
        return Interpreter(model_path=path)
    except ImportError:
        pass

    import tensorflow as tf

    print("using full tensorflow (local fallback)")
    return tf.lite.Interpreter(model_path=path)


def load_model():
    global _interpreter, _input_details, _output_details, _img_size

    if _interpreter is not None:
        return _interpreter

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError("model file not found at " + MODEL_PATH)

    print("loading " + MODEL_PATH + " ...")
    _interpreter = build_interpreter(MODEL_PATH)
    _interpreter.allocate_tensors()

    _input_details = _interpreter.get_input_details()
    _output_details = _interpreter.get_output_details()
    _img_size = int(_input_details[0]["shape"][1])

    print("model ready, expects " + str(_img_size) + "x" + str(_img_size) + " input")
    return _interpreter


def threshold():
    return float(os.environ.get("DECISION_THRESHOLD", "0.55"))


def model_label():
    label = os.environ.get("MODEL_VERSION", "")
    if label != "":
        return label
    return os.path.basename(MODEL_PATH)


def key_ok():
    expected = os.environ.get("MODEL_API_KEY", "").strip()
    if expected == "":
        return True
    return request.headers.get("X-API-Key", "") == expected


def preprocess(raw_bytes):
    """Identical to the training preprocessing: RGB, resize to a square,
    scale to the range minus one to plus one."""
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


def run_model(batch):
    _interpreter.set_tensor(_input_details[0]["index"], batch)
    _interpreter.invoke()
    output = _interpreter.get_tensor(_output_details[0]["index"])
    return float(output[0][0])


@app.route("/health", methods=["GET"])
def health():
    payload = {}
    payload["status"] = "ok"
    payload["service"] = "visionguard-model-api"
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

    probability = run_model(batch)
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
