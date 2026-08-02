"""
Module A2 - Image Classification
Trains a MobileNetV2 transfer-learning model on a 5-class product-category
dataset (shoes, bags, electronics, clothing, groceries).

This script ships with a small SYNTHETIC placeholder dataset generator so the
full pipeline (train -> save -> serve via API) runs out of the box without
any external downloads. For real results, replace `generate_synthetic_dataset()`
with a loader for a real dataset such as:
  - Kaggle "Retail Product Checkout Dataset"
  - Fashion-MNIST (reshaped to 3-channel + resized to 96x96)

Usage:
    python notebooks/train_image_classifier.py
Output:
    app/models/product_classifier.h5
    app/models/class_names.json
"""

import json
import os

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.applications import MobileNetV2

CLASS_NAMES = ["shoes", "bags", "electronics", "clothing", "groceries"]
IMG_SIZE = (96, 96)
MODEL_OUT = os.path.join(os.path.dirname(__file__), "..", "app", "models", "product_classifier.h5")
CLASSES_OUT = os.path.join(os.path.dirname(__file__), "..", "app", "models", "class_names.json")


def generate_synthetic_dataset(samples_per_class=60, seed=42):
    """
    Generates simple synthetic images per class (distinct color/shape motifs)
    so the training pipeline is fully self-contained and reproducible.
    Swap this out for a real image loader (e.g. tf.keras.utils.image_dataset_from_directory)
    once you have a real dataset in data/product_images/<class_name>/*.jpg
    """
    rng = np.random.default_rng(seed)
    X, y = [], []
    base_colors = [
        (200, 50, 50),    # shoes -> reddish
        (50, 200, 50),    # bags -> greenish
        (50, 50, 200),    # electronics -> blueish
        (200, 200, 50),   # clothing -> yellowish
        (200, 50, 200),   # groceries -> purplish
    ]
    for label, color in enumerate(base_colors):
        for _ in range(samples_per_class):
            img = np.ones((*IMG_SIZE, 3), dtype=np.float32)
            noise = rng.normal(0, 0.08, (*IMG_SIZE, 3))
            for c in range(3):
                img[:, :, c] = (color[c] / 255.0) + noise[:, :, c]
            img = np.clip(img, 0, 1)
            X.append(img)
            y.append(label)
    X = np.array(X, dtype="float32")
    y = np.array(y, dtype="int32")
    idx = rng.permutation(len(X))
    return X[idx], y[idx]


def build_model(num_classes: int):
    base = MobileNetV2(
        input_shape=(*IMG_SIZE, 3),
        include_top=False,
        weights=None,  # set weights="imagenet" if internet access is available at train time
        pooling="avg",
    )
    # NOTE: with weights="imagenet" (recommended, requires internet at train time)
    # keep base.trainable = False and only fine-tune the head below.
    # Since this script trains from random init (no internet dependency), the
    # backbone must be trainable or it will never learn anything useful.
    base.trainable = True

    inputs = layers.Input(shape=(*IMG_SIZE, 3))
    x = base(inputs, training=False)
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)
    outputs = layers.Dense(num_classes, activation="softmax")(x)
    model = models.Model(inputs, outputs)
    model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"])
    return model


def main():
    print("Generating dataset...")
    X, y = generate_synthetic_dataset()
    split = int(0.85 * len(X))
    X_train, X_val = X[:split], X[split:]
    y_train, y_val = y[:split], y[split:]

    print("Building MobileNetV2 model...")
    model = build_model(num_classes=len(CLASS_NAMES))
    model.summary()

    print("Training...")
    model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=5,
        batch_size=16,
        verbose=2,
    )

    val_loss, val_acc = model.evaluate(X_val, y_val, verbose=0)
    print(f"Validation accuracy: {val_acc:.3f}")

    os.makedirs(os.path.dirname(MODEL_OUT), exist_ok=True)
    model.save(MODEL_OUT)
    with open(CLASSES_OUT, "w") as f:
        json.dump(CLASS_NAMES, f)

    print(f"Saved model to {MODEL_OUT}")
    print(f"Saved class names to {CLASSES_OUT}")


if __name__ == "__main__":
    main()
