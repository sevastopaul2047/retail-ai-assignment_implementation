"""
Module B1 + B2 - Text Preprocessing & Sentiment Analysis
Cleans data/reviews.csv and trains a TF-IDF + Logistic Regression baseline
sentiment classifier (Positive / Negative / Neutral).

Upgrade path (see README "Stretch Goals"): swap this for a fine-tuned
DistilBERT model via HuggingFace `transformers` for higher accuracy.

Output:
    app/models/sentiment_model.pkl
    app/models/vectorizer.pkl
"""

import os
import re
import string

import joblib
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

DATA_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "reviews.csv")
MODEL_OUT = os.path.join(os.path.dirname(__file__), "..", "app", "models", "sentiment_model.pkl")
VECT_OUT = os.path.join(os.path.dirname(__file__), "..", "app", "models", "vectorizer.pkl")

STOPWORDS = {
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being", "this", "that",
    "these", "those", "i", "it", "its", "and", "or", "but", "of", "to", "in", "on", "for",
    "with", "as", "at", "by", "from", "than", "then", "so", "very", "my", "me", "we", "you",
}


def clean_text(text: str) -> str:
    """
    Module B1 - Text Preprocessing
    Lowercase -> remove punctuation/digits -> tokenize -> remove stopwords.
    (spaCy/NLTK lemmatization can be swapped in for stemming/lemmatizing tokens;
    kept dependency-light here so it runs anywhere with zero downloads.)
    """
    text = text.lower()
    text = re.sub(f"[{re.escape(string.punctuation)}]", " ", text)
    text = re.sub(r"\d+", " ", text)
    tokens = text.split()
    tokens = [t for t in tokens if t not in STOPWORDS and len(t) > 1]
    return " ".join(tokens)


def main():
    print("Loading dataset...")
    df = pd.read_csv(DATA_PATH)
    df["clean_text"] = df["review_text"].apply(clean_text)

    X_train, X_test, y_train, y_test = train_test_split(
        df["clean_text"], df["sentiment"], test_size=0.2, random_state=42, stratify=df["sentiment"]
    )

    print("Vectorizing (TF-IDF)...")
    vectorizer = TfidfVectorizer(max_features=3000, ngram_range=(1, 2))
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    print("Training Logistic Regression classifier...")
    clf = LogisticRegression(max_iter=1000, class_weight="balanced")
    clf.fit(X_train_vec, y_train)

    preds = clf.predict(X_test_vec)
    print(classification_report(y_test, preds))

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    joblib.dump(clf, MODEL_OUT)
    joblib.dump(vectorizer, VECT_OUT)
    print(f"Saved model to {MODEL_OUT}")
    print(f"Saved vectorizer to {VECT_OUT}")


if __name__ == "__main__":
    main()
