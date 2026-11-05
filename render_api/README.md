# VisionGuard model API

Deployed model endpoint for MLOps Assignment 2 (23L-0623_A2_Ops).

Runs a fine-tuned MobileNetV2, converted to TensorFlow Lite so it fits the
free tier's 512 MB memory limit. Called over the public internet by a Flask
application running locally.

## Endpoints

| Method | Path | Auth | Purpose |
|---|---|---|---|
| GET | `/health` | none | status, model version, input size |
| POST | `/predict` | `X-API-Key` header | classify a multipart `file` field |

## Deploy settings on Render

- Root directory: `render_api`
- Runtime: Python 3
- Build command: `pip install -r requirements.txt`
- Start command: `gunicorn --bind 0.0.0.0:$PORT --workers 1 --threads 4 --timeout 120 main:app`
- Environment variables: `MODEL_API_KEY`, `DECISION_THRESHOLD=0.55`, `MODEL_VERSION=champion-v2`
