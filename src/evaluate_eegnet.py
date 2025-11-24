# evaluate_eegnet_fast.py
import os
import numpy as np
from collections import Counter
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras

from dataloader_eegnet import load_epochs, filter_lr, split_train_test

SEED = 42
TEST_SIZE = 0.2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")

# 🔴 CHANGE THIS to match the model you just trained
MODEL_NAME = "eegnet_fast_global_20251124_112120.h5"
MODEL_PATH = os.path.join(MODELS_DIR, MODEL_NAME)

def load_and_preprocess():
    X, y_raw, subjects = load_epochs()
    X, y_bin, subjects = filter_lr(X, y_raw, subjects)

    # same normalization as in train_eegnet_fast.py
    X = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-6)
    return X.astype(np.float32), y_bin.astype(np.int32), subjects

def main():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

    print(f"📦 Loading trained EEGNet from: {MODEL_PATH}")
    model = keras.models.load_model(MODEL_PATH)

    print("🔍 Loading and preprocessing dataset...")
    X, y, subjects = load_and_preprocess()
    print(f"Data shape: {X.shape}, label distribution: {Counter(y)}")

    # Recreate SAME 80/20 split as in training
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=True
    )
    print(f"Using test set of size: {X_test.shape[0]}")

    X_test_in = X_test[..., np.newaxis]

    print("✅ Evaluating model on test set...")
    y_prob = model.predict(X_test_in, batch_size=128).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Test accuracy (LEFT vs RIGHT): {acc:.4f}\n")

    print("📈 Classification report:")
    print(classification_report(
        y_test, y_pred,
        target_names=["LEFT(0)", "RIGHT(1)"]
    ))

    cm = confusion_matrix(y_test, y_pred)
    print("📉 Confusion matrix (rows=true, cols=pred):")
    print(cm)

if __name__ == "__main__":
    main()
