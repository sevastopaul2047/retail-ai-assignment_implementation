"""
NLP service layer.
Loads the sentiment model + vectorizer (Module B2) once at import time.
"""

import os

import joblib

from notebooks.train_sentiment_model import clean_text  # reuse the exact training-time cleaner

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")

_sentiment_model = joblib.load(os.path.join(MODELS_DIR, "sentiment_model.pkl"))
_vectorizer = joblib.load(os.path.join(MODELS_DIR, "vectorizer.pkl"))

# In-memory feedback log for the dashboard (swap for a real DB in production)
_FEEDBACK_LOG = []


def analyze_sentiment(text: str) -> dict:
    """Module B1 + B2 - Text Preprocessing & Sentiment Analysis"""
    cleaned = clean_text(text)
    vec = _vectorizer.transform([cleaned])
    prediction = _sentiment_model.predict(vec)[0]

    # LinearSVC/LogisticRegression: use decision_function or predict_proba if available
    confidence = None
    if hasattr(_sentiment_model, "predict_proba"):
        proba = _sentiment_model.predict_proba(vec)[0]
        confidence = float(max(proba))

    _FEEDBACK_LOG.append({"text": text, "sentiment": prediction})

    return {
        "sentiment": prediction,
        "confidence": confidence,
        "cleaned_text": cleaned,
    }


def get_sentiment_stats() -> dict:
    counts = {}
    for entry in _FEEDBACK_LOG:
        counts[entry["sentiment"]] = counts.get(entry["sentiment"], 0) + 1
    return {
        "total_feedback_analyzed": len(_FEEDBACK_LOG),
        "sentiment_breakdown": counts,
    }
