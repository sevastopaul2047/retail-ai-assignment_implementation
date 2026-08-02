"""
Module B3 - Chatbot Basics
Trains an ML intent-classification fallback (TF-IDF + LinearSVC) on top of
data/intents.json. The chatbot service (app/services/chatbot_service.py)
first tries exact/substring rule-based matching against `patterns`, and
falls back to this ML classifier for anything that doesn't match directly.

Output:
    app/models/chatbot_model.pkl   (dict: {"vectorizer", "classifier", "intents"})
"""

import json
import os

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "intents.json")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "..", "app", "models", "chatbot_model.pkl")


def main():
    print("Loading intents...")
    with open(DATA_PATH) as f:
        intents_data = json.load(f)["intents"]

    X, y = [], []
    for intent in intents_data:
        for pattern in intent["patterns"]:
            X.append(pattern.lower())
            y.append(intent["tag"])

    print(f"Training on {len(X)} patterns across {len(intents_data)} intents...")
    vectorizer = TfidfVectorizer(ngram_range=(1, 2))
    X_vec = vectorizer.fit_transform(X)

    clf = LinearSVC()
    clf.fit(X_vec, y)

    # Responses keyed by tag, for the service layer to look up after prediction
    responses_by_tag = {intent["tag"]: intent["responses"] for intent in intents_data}

    bundle = {
        "vectorizer": vectorizer,
        "classifier": clf,
        "responses_by_tag": responses_by_tag,
    }

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(bundle, MODEL_OUT)
    print(f"Saved chatbot model bundle to {MODEL_OUT}")


if __name__ == "__main__":
    main()
