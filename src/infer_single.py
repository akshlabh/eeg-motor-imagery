# infer_single.py
import numpy as np
import os
from tensorflow import keras

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "..", "models", "eegnet_fast_global.h5")
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

def load_model():
    model = keras.models.load_model(MODEL_PATH)
    return model

def load_random_trial():
    X = np.load(os.path.join(DATA_DIR, "eegnet_epochs.npy")).astype(np.float32)
    y = np.load(os.path.join(DATA_DIR, "eegnet_labels.npy")).astype(int)
    # pick a random example
    i = np.random.randint(0, len(X))
    return X[i], y[i], i

if __name__ == "__main__":
    model = load_model()
    x, y_true, idx = load_random_trial()
    # preprocess same as training
    x = (x - x.mean(axis=1, keepdims=True)) / (x.std(axis=1, keepdims=True) + 1e-6)
    x_in = x[np.newaxis, ..., np.newaxis]  # shape (1, ch, t, 1)
    prob = model.predict(x_in)[0,0]
    pred = int(prob >= 0.5)
    print(f"Index: {idx}  True label: {y_true}  Pred: {pred}  Prob: {prob:.4f}")
