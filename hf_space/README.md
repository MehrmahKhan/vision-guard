---
title: VisionGuard API
emoji: 🛡️
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# VisionGuard model API

Model serving endpoint for MLOps Assignment 2 (23L-0623_A2_Ops).

Classifies an uploaded image as AI-generated or real using a fine-tuned
MobileNetV2, and is called by a Flask application running locally.

## Endpoints

- `GET /health` returns service status and the loaded model version
- `POST /predict` accepts multipart form field `file`, requires header `X-API-Key`

## Example

```
curl -X POST -H "X-API-Key: <key>" -F "file=@image.jpg" https://<space>.hf.space/predict
```
