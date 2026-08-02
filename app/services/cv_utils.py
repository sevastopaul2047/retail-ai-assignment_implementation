"""
cv_utils.py
Module A1 - OpenCV Basics
Reusable preprocessing functions used across the CV pipeline
(product classifier + face recognition).
"""

import cv2
import numpy as np

# Haar Cascade for face detection (ships with opencv-contrib-python)
_FACE_CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
face_cascade = cv2.CascadeClassifier(_FACE_CASCADE_PATH)


def read_image(path: str) -> np.ndarray:
    """Read an image from disk as a BGR numpy array."""
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(f"Could not read image at {path}")
    return img


def decode_image_bytes(image_bytes: bytes) -> np.ndarray:
    """Decode raw bytes (e.g. from an uploaded file) into a BGR numpy array."""
    arr = np.frombuffer(image_bytes, dtype=np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image bytes - unsupported or corrupt image")
    return img


def to_grayscale(img: np.ndarray) -> np.ndarray:
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


def resize_image(img: np.ndarray, size=(224, 224)) -> np.ndarray:
    return cv2.resize(img, size, interpolation=cv2.INTER_AREA)


def apply_blur(img: np.ndarray, ksize=(5, 5)) -> np.ndarray:
    return cv2.GaussianBlur(img, ksize, 0)


def canny_edges(img: np.ndarray, low=100, high=200) -> np.ndarray:
    gray = to_grayscale(img) if img.ndim == 3 else img
    return cv2.Canny(gray, low, high)


def detect_faces(img: np.ndarray, scale_factor=1.1, min_neighbors=5):
    """
    Detect faces using Haar Cascades.
    Returns a list of bounding boxes (x, y, w, h).
    """
    gray = to_grayscale(img) if img.ndim == 3 else img
    faces = face_cascade.detectMultiScale(
        gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=(40, 40)
    )
    return faces


def crop_face(img: np.ndarray, box) -> np.ndarray:
    x, y, w, h = box
    return img[y:y + h, x:x + w]


def draw_boxes(img: np.ndarray, boxes, color=(0, 255, 0), thickness=2) -> np.ndarray:
    out = img.copy()
    for (x, y, w, h) in boxes:
        cv2.rectangle(out, (x, y), (x + w, y + h), color, thickness)
    return out


def preprocess_for_classifier(img: np.ndarray, size=(96, 96)) -> np.ndarray:
    """
    Standard preprocessing pipeline before feeding an image
    into the product classifier: resize + normalize to [0, 1].
    """
    resized = resize_image(img, size)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    normalized = rgb.astype("float32") / 255.0
    return normalized


def preprocess_face_for_recognition(img: np.ndarray, size=(100, 100)) -> np.ndarray:
    """Grayscale + resize a cropped face for the LBPH recognizer."""
    gray = to_grayscale(img) if img.ndim == 3 else img
    return cv2.resize(gray, size, interpolation=cv2.INTER_AREA)


def capture_from_webcam(camera_index: int = 0):
    """
    Open a webcam feed and yield frames.
    NOTE: only works in an environment with a physical/virtual camera device.
    Use as: `for frame in capture_from_webcam(): ...` then `break` to release.
    """
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Is a camera connected?")
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            yield frame
    finally:
        cap.release()
