"""Tests that CI can run without a model file or a database.

They cover the validation rules from Task 4.2, which are the parts most likely
to break silently.
"""

import io
import os

import pytest

os.environ.setdefault("DATABASE_URL", "sqlite:///test_visionguard.db")
os.environ.setdefault("MAX_UPLOAD_MB", "1")
os.environ.setdefault("DECISION_THRESHOLD", "0.5")
os.environ.setdefault("MODEL_PATH", "models/does-not-exist.keras")

from app.app import app as flask_app  # noqa: E402


@pytest.fixture
def client():
    flask_app.config["TESTING"] = True
    test_client = flask_app.test_client()
    return test_client


def test_upload_page_loads(client):
    response = client.get("/")
    assert response.status_code == 200
    assert b"VisionGuard" in response.data


def test_history_page_loads(client):
    response = client.get("/history")
    assert response.status_code in (200, 503)


def test_health_returns_json(client):
    response = client.get("/health")
    assert response.status_code in (200, 503)
    payload = response.get_json()
    assert "status" in payload
    assert "inference_mode" in payload


def test_missing_file_part_is_rejected(client):
    response = client.post("/predict", data={})
    assert response.status_code == 400


def test_wrong_extension_is_rejected(client):
    data = {}
    data["file"] = (io.BytesIO(b"not an image at all"), "payload.txt")
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 415


def test_empty_filename_is_rejected(client):
    data = {}
    data["file"] = (io.BytesIO(b""), "")
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 400


def test_oversize_file_is_rejected(client):
    big = b"x" * (2 * 1024 * 1024)
    data = {}
    data["file"] = (io.BytesIO(big), "big.jpg")
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    assert response.status_code == 413


def test_undecodable_image_is_rejected(client):
    """An extension that says .jpg over bytes that are not an image."""
    data = {}
    data["file"] = (io.BytesIO(b"\x00\x01\x02 definitely not a jpeg"), "fake.jpg")
    response = client.post("/predict", data=data, content_type="multipart/form-data")
    # 422 when a model is present, 503 when the model file is missing in CI.
    assert response.status_code in (422, 503)


def test_404_page(client):
    response = client.get("/no-such-page")
    assert response.status_code == 404
