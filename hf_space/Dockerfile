# VisionGuard model API, deployed to Hugging Face Spaces (Task 7).
#
# Spaces serves whatever listens on port 7860, so PORT is set to that here
# rather than the 8080 used locally. Everything else matches the local image.

FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=7860 \
    MODEL_PATH=/app/models/champion.keras \
    IMG_SIZE=224 \
    DECISION_THRESHOLD=0.55 \
    MODEL_VERSION=champion-v2 \
    MAX_UPLOAD_MB=5

WORKDIR /app

RUN apt-get update \
    && apt-get install --no-install-recommends -y libgomp1 curl \
    && rm -rf /var/lib/apt/lists/*

# Dependencies first so editing main.py does not reinstall TensorFlow.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
# The model is baked into the image: Spaces gives no persistent disk to
# mount it from, which is the opposite of the Compose choice in Task 6.
COPY models/ /app/models/

# Spaces expects the container to run as uid 1000.
RUN useradd --create-home --uid 1000 apiuser && chown -R apiuser:apiuser /app
USER apiuser

EXPOSE 7860

HEALTHCHECK --interval=30s --timeout=10s --start-period=120s --retries=3 \
  CMD curl -fsS "http://127.0.0.1:${PORT}/health" || exit 1

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 180 --access-logfile - main:app"]
