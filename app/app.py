"""Task 4 - the Flask web application.

Routes:
    GET  /            upload page
    POST /predict     accepts the file, validates it, classifies it, logs it
    GET  /history     the database trail (Task 5)
    GET  /health      health check, used by Docker and by the report
    GET  /api/predict-info   small JSON helper that says which inference path is live

Inference path:
    If MODEL_API_URL is set, the image is forwarded to the hosted model API
    (Task 8). Otherwise the local champion .keras file is used (Task 4).
"""

import os
import time

from dotenv import load_dotenv
from flask import (
    Flask,
    jsonify,
    redirect,
    render_template,
    request,
    url_for,
)
from werkzeug.utils import secure_filename

from app import api_client, db

load_dotenv()

ALLOWED_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".webp", ".gif"}


def max_upload_bytes():
    value = os.environ.get("MAX_UPLOAD_MB", "5")
    return int(float(value) * 1024 * 1024)


def create_app():
    application = Flask(__name__)
    application.config["MAX_CONTENT_LENGTH"] = max_upload_bytes()
    application.config["SECRET_KEY"] = os.environ.get(
        "FLASK_SECRET_KEY", "dev-only-not-for-production"
    )

    try:
        db.init_db()
    except Exception as error:
        print("WARNING: could not initialise the database: " + str(error))

    # Preload the local model only when there is no remote API configured.
    if not api_client.is_configured():
        try:
            from app import inference

            inference.load_model()
        except Exception as error:
            print("WARNING: local model not loaded: " + str(error))

    register_routes(application)
    register_error_handlers(application)
    return application


def inference_mode():
    if api_client.is_configured():
        return "remote"
    return "local"


def extension_allowed(file_name):
    extension = os.path.splitext(file_name)[1].lower()
    return extension in ALLOWED_EXTENSIONS


def register_routes(application):
    @application.route("/", methods=["GET"])
    def index():
        return render_template("upload.html", mode=inference_mode())

    @application.route("/predict", methods=["POST"])
    def predict():
        started = time.time()

        if "file" not in request.files:
            return (
                render_template(
                    "error.html",
                    code="No file",
                    message="No file part was sent with the form.",
                    hint="Pick an image and submit again.",
                ),
                400,
            )

        uploaded = request.files["file"]
        if uploaded.filename is None or uploaded.filename.strip() == "":
            return (
                render_template(
                    "error.html",
                    code="Empty selection",
                    message="You submitted the form without choosing a file.",
                    hint="Click Choose image first.",
                ),
                400,
            )

        # Failure case 1: wrong file type
        safe_name = secure_filename(uploaded.filename)
        if safe_name == "":
            safe_name = "upload"
        if not extension_allowed(safe_name):
            return (
                render_template(
                    "error.html",
                    code="Unsupported file type",
                    message="'" + safe_name + "' is not an image type this service accepts.",
                    hint="Allowed types: PNG, JPG, JPEG, BMP, WEBP, GIF.",
                ),
                415,
            )

        # The file never touches disk. It is read into memory, used, discarded.
        raw_bytes = uploaded.read()
        size = len(raw_bytes)

        # Failure case 2: too large. MAX_CONTENT_LENGTH catches most of these
        # before we get here (that path lands in the 413 handler below), this
        # check covers the rest.
        if size > max_upload_bytes():
            return (
                render_template(
                    "error.html",
                    code="File too large",
                    message="That file is "
                    + str(round(size / (1024 * 1024), 2))
                    + " MB. The limit is "
                    + os.environ.get("MAX_UPLOAD_MB", "5")
                    + " MB.",
                    hint="Resize or compress the image and try again.",
                ),
                413,
            )

        if size == 0:
            return (
                render_template(
                    "error.html",
                    code="Empty file",
                    message="The uploaded file contained no data.",
                    hint="Check the file and try again.",
                ),
                400,
            )

        # Failure case 3: looks like an image but cannot be decoded.
        mode = inference_mode()
        if mode == "remote":
            result, error = api_client.predict_remote(raw_bytes, safe_name)
            if error is not None:
                return (
                    render_template(
                        "error.html",
                        code="Model API unavailable",
                        message=error,
                        hint="Check MODEL_API_URL and MODEL_API_KEY in your .env, "
                        "then confirm the cloud service is running.",
                    ),
                    502,
                )
        else:
            from app import inference

            try:
                result = inference.predict_bytes(raw_bytes)
            except FileNotFoundError as error:
                return (
                    render_template(
                        "error.html",
                        code="Model missing",
                        message=str(error),
                        hint="Fetch the champion model, then restart the app.",
                    ),
                    503,
                )
            except Exception:
                return (
                    render_template(
                        "error.html",
                        code="Undecodable image",
                        message="The file has an image extension but its contents "
                        "could not be decoded as an image.",
                        hint="It may be renamed, truncated or corrupt. Open it "
                        "locally to confirm, then upload a valid image.",
                    ),
                    422,
                )

        latency_ms = int((time.time() - started) * 1000)

        try:
            db.log_prediction(
                file_name=safe_name,
                file_size_bytes=size,
                result=result,
                source=mode,
                latency_ms=latency_ms,
                client_ip=request.remote_addr,
            )
        except Exception as error:
            print("WARNING: failed to write the history row: " + str(error))

        # raw_bytes goes out of scope here. Nothing was written to disk.
        return render_template(
            "result.html",
            result=result,
            file_name=safe_name,
            file_size_kb=round(size / 1024.0, 1),
            latency_ms=latency_ms,
            mode=mode,
        )

    @application.route("/history", methods=["GET"])
    def history():
        try:
            rows = db.fetch_history(limit=200)
            totals = db.summary_counts()
            return render_template("history.html", rows=rows, totals=totals)
        except Exception as error:
            return (
                render_template(
                    "error.html",
                    code="History unavailable",
                    message="The database could not be read: " + str(error),
                    hint="Check DATABASE_URL and that the database service is healthy.",
                ),
                503,
            )

    @application.route("/health", methods=["GET"])
    def health():
        payload = {}
        payload["status"] = "ok"
        payload["inference_mode"] = inference_mode()
        payload["threshold"] = os.environ.get("DECISION_THRESHOLD", "0.5")
        payload["max_upload_mb"] = os.environ.get("MAX_UPLOAD_MB", "5")

        if inference_mode() == "local":
            payload["model_path"] = os.environ.get("MODEL_PATH", "models/champion.keras")
            try:
                from app import inference

                payload["model_loaded"] = inference.load_model() is not None
                payload["model_version"] = inference.model_version_label()
            except Exception as error:
                payload["status"] = "degraded"
                payload["model_loaded"] = False
                payload["model_error"] = str(error)
        else:
            payload["model_api_url"] = api_client.api_url()
            reachable, detail = api_client.health_remote()
            payload["model_api_reachable"] = reachable
            payload["model_api_detail"] = detail
            if not reachable:
                payload["status"] = "degraded"

        try:
            db.fetch_history(limit=1)
            payload["database"] = "reachable"
        except Exception as error:
            payload["status"] = "degraded"
            payload["database"] = "unreachable: " + str(error)

        if payload["status"] == "ok":
            return jsonify(payload), 200
        return jsonify(payload), 503

    @application.route("/api/predict-info", methods=["GET"])
    def predict_info():
        return jsonify({"mode": inference_mode(), "api_url": api_client.api_url()})

    @application.route("/favicon.ico")
    def favicon():
        return redirect(url_for("static", filename="style.css"))


def register_error_handlers(application):
    @application.errorhandler(413)
    def too_large(_error):
        return (
            render_template(
                "error.html",
                code="File too large",
                message="The upload exceeded the "
                + os.environ.get("MAX_UPLOAD_MB", "5")
                + " MB limit, so the server refused it before reading it.",
                hint="Resize or compress the image and try again.",
            ),
            413,
        )

    @application.errorhandler(404)
    def not_found(_error):
        return (
            render_template(
                "error.html",
                code="Page not found",
                message="There is nothing at that address.",
                hint="Go back to the upload page.",
            ),
            404,
        )

    @application.errorhandler(500)
    def server_error(_error):
        return (
            render_template(
                "error.html",
                code="Server error",
                message="Something broke on our side.",
                hint="Check the container logs with `docker compose logs web`.",
            ),
            500,
        )


app = create_app()


if __name__ == "__main__":
    # Local development only. Debug is read from the environment and defaults to
    # off, because debug mode exposes the Werkzeug console: anyone who can
    # trigger an exception gets an interactive Python prompt inside the
    # container, with the database password in os.environ.
    debug_flag = os.environ.get("FLASK_DEBUG", "0") == "1"
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=debug_flag)
