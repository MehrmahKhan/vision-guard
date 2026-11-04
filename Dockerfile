# Task 6.2 - image for the VisionGuard web app.
#
# Why the instructions are in this order:
#   Docker caches one layer per instruction and reuses it as long as that
#   instruction and everything above it is unchanged. Source code changes on
#   every commit, dependencies change maybe once a week. So requirements.txt is
#   copied and installed FIRST, in its own layer. Editing app/app.py then
#   invalidates only the layers below the pip install, and the rebuild takes
#   seconds instead of re-downloading TensorFlow every time.
#
# Why tensorflow-cpu:
#   The full `tensorflow` wheel pulls in the CUDA/cuDNN runtime, which is well
#   over 2 GB of libraries that can never be used here. Inference on one image
#   at a time is not GPU work, there is no GPU on the deployment VM, and the
#   smaller image pushes and pulls far faster. `tensorflow-cpu` is pinned
#   exactly so the serving environment matches the training environment.
#
# Secrets:
#   Nothing secret is copied or baked in. .dockerignore excludes .env, and every
#   setting arrives as an environment variable at run time.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=5000 \
    FLASK_DEBUG=0 \
    MODEL_PATH=/app/models/champion.keras \
    IMG_SIZE=160 \
    DECISION_THRESHOLD=0.5 \
    MAX_UPLOAD_MB=5

WORKDIR /app

# libgomp1 is TensorFlow's OpenMP runtime; curl is only for the health check.
RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# --- dependency layer (changes rarely) ---
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

# --- source layer (changes every commit) ---
COPY app/ ./app/

RUN useradd --create-home --uid 10001 appuser \
    && mkdir -p /app/models /app/data \
    && chown -R appuser:appuser /app
USER appuser

EXPOSE 5000

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

# Gunicorn, not the Flask development server: the dev server is single-threaded,
# single-process and explicitly documented as unfit for production. One worker
# with four threads keeps only one copy of the model in memory.
CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 180 --access-logfile - app.app:app"]
