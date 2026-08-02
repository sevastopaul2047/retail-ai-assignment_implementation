"""
Chatbot service layer (Module B3).
Hybrid approach:
  1. Rule-based: exact/substring match of the incoming message against
     each intent's known `patterns` (fast, precise for common FAQs).
  2. ML fallback: TF-IDF + LinearSVC intent classifier trained in
     notebooks/train_chatbot_model.py, used when no rule matches directly.
"""

import json
import os
import random

import joblib

MODELS_DIR = os.path.join(os.path.dirname(__file__), "..", "models")
DATA_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "data")

_bundle = joblib.load(os.path.join(MODELS_DIR, "chatbot_model.pkl"))
_vectorizer = _bundle["vectorizer"]
_classifier = _bundle["classifier"]
_RESPONSES_BY_TAG = _bundle["responses_by_tag"]

with open(os.path.join(DATA_DIR, "intents.json")) as f:
    _INTENTS = json.load(f)["intents"]

FALLBACK_RESPONSE = (
    "I'm not fully sure I understood that. Could you rephrase, or ask about "
    "order status, returns, store hours, payments, or say 'human agent' to talk to support?"
)


def _rule_based_match(message: str):
    """Try exact/substring matching against known patterns first."""
    msg = message.lower().strip()
    for intent in _INTENTS:
        for pattern in intent["patterns"]:
            if pattern.lower() in msg or msg in pattern.lower():
                return intent["tag"]
    return None


def get_bot_reply(message: str) -> dict:
    tag = _rule_based_match(message)
    method = "rule_based"

    if tag is None:
        method = "ml_fallback"
        vec = _vectorizer.transform([message.lower()])
        tag = _classifier.predict(vec)[0]
        # LinearSVC decision_function gives a rough confidence proxy
        margin = float(max(_classifier.decision_function(vec)[0]))
        if margin < 0.15:  # too uncertain -> generic fallback
            return {"reply": FALLBACK_RESPONSE, "intent": "fallback", "method": method}

    responses = _RESPONSES_BY_TAG.get(tag, [FALLBACK_RESPONSE])
    return {"reply": random.choice(responses), "intent": tag, "method": method}
