"""
Computer Vision service layer.
Loads the product classifier (Module A2) and face DB (Module A3) once at
import time and exposes simple predict functions used by app/routers/vision.py.
"""

import json
import os
import pickle
import tempfile
from datetime import datetime, timezone

import cv2
import numpy as np
import tensorflow as tf

from app.services import cv_utils

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

# --- Product classifier (Module A2) -------------------------------------------------
_classifier = tf.keras.models.load_model(os.path.join(MODELS_DIR, "product_classifier.h5"))
with open(os.path.join(MODELS_DIR, "class_names.json")) as f:
    _CLASS_NAMES = json.load(f)

# --- Face recognition DB (Module A3) -------------------------------------------------
with open(os.path.join(MODELS_DIR, "face_db.pkl"), "rb") as f:
    _face_bundle = pickle.load(f)

_face_recognizer = cv2.face.LBPHFaceRecognizer_create()
with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as tmp:
    tmp.write(_face_bundle["recognizer_yml_bytes"])
    _tmp_path = tmp.name
_face_recognizer.read(_tmp_path)
os.remove(_tmp_path)
_FACE_LABELS = _face_bundle["labels"]  # {"0": "Aarav_Sharma", ...}

# In-memory visit log (swap for a real DB - e.g. Postgres/SQLite - in production)
_VISIT_LOG = []

# LBPH confidence: LOWER is a better match. Above this threshold -> "unknown".
LBPH_CONFIDENCE_THRESHOLD = 70.0


def classify_product(image_bytes: bytes) -> dict:
    """Module A2 - Image Classification"""
    img = cv_utils.decode_image_bytes(image_bytes)
    processed = cv_utils.preprocess_for_classifier(img, size=(96, 96))
    batch = np.expand_dims(processed, axis=0)
    preds = _classifier.predict(batch, verbose=0)[0]
    top_idx = int(np.argmax(preds))
    return {
        "category": _CLASS_NAMES[top_idx],
        "confidence": float(preds[top_idx]),
        "all_scores": {name: float(score) for name, score in zip(_CLASS_NAMES, preds)},
    }


def recognize_face(image_bytes: bytes) -> dict:
    """Module A3 - Face Recognition & customer visit logging"""
    img = cv_utils.decode_image_bytes(image_bytes)
    boxes = cv_utils.detect_faces(img)

    if len(boxes) == 0:
        return {"status": "no_face_detected", "customer": None, "confidence": None}

    # Use the largest detected face
    box = max(boxes, key=lambda b: b[2] * b[3])
    face_crop = cv_utils.crop_face(img, box)
    face_processed = cv_utils.preprocess_face_for_recognition(face_crop)

    label_id, confidence = _face_recognizer.predict(face_processed)

    if confidence > LBPH_CONFIDENCE_THRESHOLD:
        status = "unknown_customer"
        customer_name = None
    else:
        status = "returning_customer"
        customer_name = _FACE_LABELS.get(str(label_id), "unknown")
        _VISIT_LOG.append({
            "customer": customer_name,
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "confidence": float(confidence),
        })

    return {
        "status": status,
        "customer": customer_name,
        "confidence": float(confidence),
        "face_box": [int(v) for v in box],
    }


def get_visit_stats() -> dict:
    """Aggregate stats for GET /dashboard/stats"""
    counts = {}
    for visit in _VISIT_LOG:
        counts[visit["customer"]] = counts.get(visit["customer"], 0) + 1
    return {
        "total_visits_logged": len(_VISIT_LOG),
        "unique_customers_seen": len(counts),
        "visits_by_customer": counts,
        "recent_visits": _VISIT_LOG[-10:],
    }
