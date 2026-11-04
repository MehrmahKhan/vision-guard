# Image for the cloud-hosted model API (Task 7).
FROM python:3.11-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PORT=8080 \
    MODEL_PATH=/app/models/champion.keras \
    IMG_SIZE=160 \
    DECISION_THRESHOLD=0.5

WORKDIR /app

# Dependencies first, so editing main.py does not reinstall TensorFlow.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY main.py .
# The model is baked in here on purpose: a serverless or managed container
# platform gives you no persistent disk to mount it from.
COPY models/ /app/models/

RUN useradd --create-home --uid 10001 apiuser && chown -R apiuser:apiuser /app
USER apiuser

EXPOSE 8080

HEALTHCHECK --interval=30s --timeout=10s --start-period=90s --retries=3 \
  CMD python -c "import urllib.request,os,sys; \
u='http://127.0.0.1:'+os.environ.get('PORT','8080')+'/health'; \
sys.exit(0 if urllib.request.urlopen(u, timeout=8).status==200 else 1)"

CMD ["sh", "-c", "gunicorn --bind 0.0.0.0:${PORT} --workers 1 --threads 4 --timeout 120 main:app"]
