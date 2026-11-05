# VisionGuard model API (Vercel)

Task 7, Pathway B: the Flask interface runs locally, the model is served here.

TensorFlow does not fit inside Vercel's 250 MB function limit, so the Keras
champion was converted to TensorFlow Lite and is served by a small
interpreter. The whole bundle is roughly 80 MB.

| Method | Path | Auth |
|---|---|---|
| GET | `/health` | none |
| POST | `/predict` | `X-API-Key` header, multipart `file` field |
