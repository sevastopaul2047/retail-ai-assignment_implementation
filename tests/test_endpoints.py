"""
Module C - automated tests for the FastAPI endpoints.
Run with: pytest tests/ -v
"""

import io
import os
import sys

import numpy as np
import cv2
import pytest
from fastapi.testclient import TestClient

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
os.environ.setdefault("API_KEY", "dev-demo-key-change-me")

from app.main import app  # noqa: E402

client = TestClient(app)
HEADERS = {"X-API-Key": os.environ["API_KEY"]}


def _dummy_image_bytes() -> bytes:
    """A small synthetic in-memory JPEG for upload tests."""
    img = np.full((100, 100, 3), 120, dtype=np.uint8)
    ok, buf = cv2.imencode(".jpg", img)
    assert ok
    return buf.tobytes()


def test_root():
    r = client.get("/")
    assert r.status_code == 200
    assert "endpoints" in r.json()


def test_health():
    r = client.get("/health")
    assert r.status_code == 200
    assert r.json()["status"] == "ok"


def test_missing_api_key_rejected():
    r = client.post("/analyze-sentiment", json={"text": "great product"})
    assert r.status_code == 401


def test_invalid_api_key_rejected():
    r = client.post("/analyze-sentiment", json={"text": "great product"}, headers={"X-API-Key": "wrong"})
    assert r.status_code == 401


def test_analyze_sentiment_positive():
    r = client.post("/analyze-sentiment", json={"text": "Absolutely love this product, amazing quality!"}, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["sentiment"] in {"positive", "negative", "neutral"}
    assert "cleaned_text" in body


def test_analyze_sentiment_negative():
    r = client.post("/analyze-sentiment", json={"text": "Terrible product, broke after two days."}, headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["sentiment"] in {"positive", "negative", "neutral"}


def test_analyze_sentiment_rejects_empty_text():
    r = client.post("/analyze-sentiment", json={"text": ""}, headers=HEADERS)
    assert r.status_code == 422


def test_chatbot_rule_based_return_policy():
    r = client.post("/chatbot", json={"message": "What is your return policy?"}, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["intent"] == "return_policy"
    assert body["method"] == "rule_based"


def test_chatbot_greeting():
    r = client.post("/chatbot", json={"message": "hello"}, headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["intent"] == "greeting"


def test_chatbot_fallback_handles_gibberish():
    r = client.post("/chatbot", json={"message": "asdkjfh qweoiuwer"}, headers=HEADERS)
    assert r.status_code == 200
    assert "reply" in r.json()


def test_classify_product():
    img_bytes = _dummy_image_bytes()
    files = {"image": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    r = client.post("/classify-product", files=files, headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert body["category"] in {"shoes", "bags", "electronics", "clothing", "groceries"}
    assert 0.0 <= body["confidence"] <= 1.0


def test_classify_product_rejects_bad_content_type():
    files = {"image": ("test.txt", io.BytesIO(b"not an image"), "text/plain")}
    r = client.post("/classify-product", files=files, headers=HEADERS)
    assert r.status_code == 415


def test_recognize_face_no_face_in_blank_image():
    img_bytes = _dummy_image_bytes()  # flat gray image, no face
    files = {"image": ("test.jpg", io.BytesIO(img_bytes), "image/jpeg")}
    r = client.post("/recognize-face", files=files, headers=HEADERS)
    assert r.status_code == 200
    assert r.json()["status"] == "no_face_detected"


def test_dashboard_stats():
    r = client.get("/dashboard/stats", headers=HEADERS)
    assert r.status_code == 200
    body = r.json()
    assert "visits" in body
    assert "sentiment" in body
