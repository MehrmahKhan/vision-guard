"""Task 8 - the bridge between the Flask view and the model API in the cloud.

The assignment asks for Flask running on localhost while the model sits behind a
public URL. This module posts the raw image bytes to that URL and returns the
same dict shape that app/inference.py returns, so the view code does not care
which path was taken.
"""

import os

import requests


def api_url():
    return os.environ.get("MODEL_API_URL", "").strip()


def api_key():
    return os.environ.get("MODEL_API_KEY", "").strip()


def is_configured():
    return api_url() != ""


def predict_remote(raw_bytes, file_name, timeout=60):
    """POSTs the image to the hosted model API.

    Returns (result_dict, None) on success or (None, error_string) on failure.
    """
    url = api_url()
    if url == "":
        return None, "MODEL_API_URL is not set"

    headers = {}
    key = api_key()
    if key != "":
        headers["X-API-Key"] = key

    files = {}
    files["file"] = (file_name, raw_bytes, "application/octet-stream")

    try:
        response = requests.post(url, files=files, headers=headers, timeout=timeout)
    except requests.exceptions.Timeout:
        return None, "the model API did not answer within " + str(timeout) + " seconds"
    except requests.exceptions.RequestException as error:
        return None, "could not reach the model API: " + str(error)

    if response.status_code == 401 or response.status_code == 403:
        return None, "the model API rejected the API key (HTTP " + str(response.status_code) + ")"

    if response.status_code != 200:
        return None, "model API returned HTTP " + str(response.status_code) + ": " + response.text[:300]

    try:
        payload = response.json()
    except ValueError:
        return None, "the model API did not return JSON"

    if "error" in payload:
        return None, str(payload["error"])

    # Fill in anything the API left out so the template never breaks.
    if "model" not in payload:
        payload["model"] = "remote-api"
    if "confidence_percent" not in payload and "confidence" in payload:
        payload["confidence_percent"] = round(float(payload["confidence"]) * 100.0, 2)
    if "caveat" not in payload:
        payload["caveat"] = (
            "This verdict comes from a hosted model and is a statistical guess, "
            "not proof."
        )

    return payload, None


def health_remote(timeout=15):
    url = api_url()
    if url == "":
        return False, "MODEL_API_URL is not set"
    health = url.replace("/predict", "/health")
    try:
        response = requests.get(health, timeout=timeout)
    except requests.exceptions.RequestException as error:
        return False, str(error)
    if response.status_code == 200:
        return True, response.text[:200]
    return False, "HTTP " + str(response.status_code)
