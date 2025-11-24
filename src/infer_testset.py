# infer_testset_fixed.py
import numpy as np
import os
from tensorflow import keras
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "eegnet_fast_global.h5")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

def preprocess(X):
    return (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-6)

print("🔍 Loading data...")
X = np.load(os.path.join(DATA_DIR, "eegnet_epochs.npy")).astype(np.float32)
y = np.load(os.path.join(DATA_DIR, "eegnet_labels.npy")).astype(int)

# IMPORTANT: convert labels 2→0, 3→1 exactly like training
y = np.where(y == 2, 0, 1)

print("🔁 Reproducing SAME 80/20 split as training...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# preprocess
X_train = preprocess(X_train)
X_test = preprocess(X_test)

X_test_in = X_test[..., np.newaxis]

print("🧠 Loading model:", MODEL_PATH)
model = keras.models.load_model(MODEL_PATH)

print("⚡ Running inference...")
probs = model.predict(X_test_in, batch_size=256).ravel()
preds = (probs >= 0.5).astype(int)

print("\n🎯 Test Accuracy:", accuracy_score(y_test, preds))
print("\n📈 Classification Report:")
print(classification_report(y_test, preds, digits=4))

print("\n📊 Confusion Matrix:")
print(confusion_matrix(y_test, preds))
