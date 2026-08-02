"""
Module A3 - Face Recognition Fundamentals
Builds a demo customer face database using OpenCV's LBPH (Local Binary
Patterns Histograms) recognizer, which needs no dlib/face_recognition
compile step and is lightweight enough to run/deploy anywhere.

Pipeline: detect face (Haar Cascade) -> crop + preprocess -> train LBPH
recognizer on labeled faces -> save recognizer + label map.

This script generates a small SYNTHETIC demo dataset of "faces" (distinct
grayscale patterns per identity) so the pipeline is fully runnable without
needing real photos or an internet download (e.g. LFW). For a real deployment:
  1. Put real (consenting) customer photos in data/faces/<customer_name>/*.jpg
  2. Replace `generate_synthetic_faces()` with a loader that reads those
     folders and runs `cv_utils.detect_faces` + `crop_face` on each image.

Output:
    app/models/face_db.pkl       (pickle: LBPH weights bytes + id->name label map)
"""

import json
import os
import pickle
import sys
import tempfile

import cv2
import numpy as np

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
from app.services import cv_utils  # noqa: E402,F401 (imported for use in real-data version)

FACE_SIZE = (100, 100)
MODEL_OUT = os.path.join(os.path.dirname(__file__), "..", "app", "models", "face_db.pkl")

DEMO_CUSTOMERS = ["Aarav_Sharma", "Priya_Nair", "Rohan_Gupta", "Ishita_Roy"]


def generate_synthetic_faces(samples_per_person=15, seed=7):
    """
    Generates distinct grayscale gradient/blob patterns per identity to stand
    in for real face crops, purely so the LBPH training pipeline is testable
    end-to-end without external face datasets.
    """
    rng = np.random.default_rng(seed)
    faces, labels = [], []
    for label_id, _ in enumerate(DEMO_CUSTOMERS):
        cx, cy = rng.integers(20, 80), rng.integers(20, 80)
        for _ in range(samples_per_person):
            img = np.zeros(FACE_SIZE, dtype=np.uint8)
            yy, xx = np.mgrid[0:FACE_SIZE[0], 0:FACE_SIZE[1]]
            blob = np.exp(-(((xx - cx) ** 2 + (yy - cy) ** 2) / (2 * 15 ** 2)))
            noise = rng.normal(0, 8, FACE_SIZE)
            img = np.clip(blob * 200 + noise, 0, 255).astype(np.uint8)
            faces.append(img)
            labels.append(label_id)
    return faces, labels


def main():
    print("Generating demo face dataset...")
    faces, labels = generate_synthetic_faces()

    print("Training LBPH recognizer...")
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train(faces, np.array(labels))

    # LBPH recognizers only support .write()/.read() to a yml/xml file, so we
    # serialize via a temp file and bundle the bytes + label map into one
    # face_db.pkl, matching the deliverable name from the project spec.
    with tempfile.NamedTemporaryFile(suffix=".yml", delete=False) as tmp:
        tmp_path = tmp.name
    recognizer.write(tmp_path)
    with open(tmp_path, "rb") as f:
        recognizer_bytes = f.read()
    os.remove(tmp_path)

    label_map = {str(i): name for i, name in enumerate(DEMO_CUSTOMERS)}

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    with open(MODEL_OUT, "wb") as f:
        pickle.dump({"recognizer_yml_bytes": recognizer_bytes, "labels": label_map}, f)

    print(f"Saved face DB (recognizer + labels) to {MODEL_OUT}")
    print("Demo customers:", DEMO_CUSTOMERS)


if __name__ == "__main__":
    main()
