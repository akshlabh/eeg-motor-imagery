# generate_eegnet_report.py
import os
import json
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from tensorflow import keras

from dataloader_eegnet import load_epochs, filter_lr, split_train_test

# -----------------------------
# CONFIGURATION
# -----------------------------
SEED = 42
TEST_SIZE = 0.2

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")

# 🔴 UPDATE THIS TIMESTAMP to your run
TIMESTAMP = "20251125_193714"

MODEL_PATH = os.path.join(MODELS_DIR, f"eegnet_fast_global_{TIMESTAMP}.h5")
HISTORY_PATH = os.path.join(ARTIFACTS_DIR, f"history_eegnet_fast_{TIMESTAMP}.json")
CM_PATH = os.path.join(ARTIFACTS_DIR, f"eegnet_fast_{TIMESTAMP}_confusion.npy")
Y_TRUE_PATH = os.path.join(ARTIFACTS_DIR, f"eegnet_fast_{TIMESTAMP}_y_true.npy")
Y_PRED_PATH = os.path.join(ARTIFACTS_DIR, f"eegnet_fast_{TIMESTAMP}_y_pred.npy")

OUTPUT_DIR = os.path.join(SCRIPT_DIR, "report_outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# -----------------------------
# DATA LOADING
# -----------------------------
def load_and_preprocess():
    X, y_raw, subjects = load_epochs()
    X, y_bin, subjects = filter_lr(X, y_raw, subjects)
    X = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-6)
    return X.astype(np.float32), y_bin.astype(np.int32), subjects


# -----------------------------
# PLOT 1: Training curves
# -----------------------------
def plot_training_curves(history, out_prefix):
    acc = history["accuracy"]
    val_acc = history["val_accuracy"]
    loss = history["loss"]
    val_loss = history["val_loss"]
    epochs = range(1, len(acc) + 1)

    plt.figure(figsize=(12, 4))

    # Accuracy plot
    plt.subplot(1, 2, 1)
    plt.plot(epochs, acc, label="Train Accuracy")
    plt.plot(epochs, val_acc, label="Val Accuracy")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.title("EEGNet Training Accuracy")
    plt.grid(True)
    plt.legend()

    # Loss plot
    plt.subplot(1, 2, 2)
    plt.plot(epochs, loss, label="Train Loss")
    plt.plot(epochs, val_loss, label="Val Loss")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("EEGNet Training Loss")
    plt.grid(True)
    plt.legend()

    path = f"{out_prefix}_training_curves.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    print(f"📊 Saved training curves → {path}")


# -----------------------------
# PLOT 2: Confusion Matrix
# -----------------------------
def plot_confusion_matrix(cm, out_prefix):
    classes = ["LEFT", "RIGHT"]
    fig, ax = plt.subplots(figsize=(5, 4))

    im = ax.imshow(cm, interpolation="nearest", cmap="Blues")
    plt.title("EEGNet Confusion Matrix")
    plt.colorbar(im, ax=ax)

    tick_marks = np.arange(len(classes))
    ax.set_xticks(tick_marks)
    ax.set_yticks(tick_marks)
    ax.set_xticklabels(classes)
    ax.set_yticklabels(classes)
    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("True Label")

    # Write values in cells
    thresh = cm.max() / 2.0
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(
                j, i, format(cm[i, j], "d"),
                ha="center", va="center",
                color="white" if cm[i, j] > thresh else "black"
            )

    path = f"{out_prefix}_confusion_matrix.png"
    plt.tight_layout()
    plt.savefig(path, dpi=200)
    print(f"📉 Saved confusion matrix plot → {path}")


# -----------------------------
# MAIN REPORT FUNCTION
# -----------------------------
def main():
    print("\n📦 Loading model:", MODEL_PATH)
    model = keras.models.load_model(MODEL_PATH)

    print("🔍 Loading and preprocessing dataset...")
    X, y, subjects = load_and_preprocess()

    # Recreate SAME split used during training
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=TEST_SIZE, random_state=SEED, stratify=True
    )
    X_test_in = X_test[..., np.newaxis]

    print("\n🔎 Evaluating model on test set...")
    y_prob = model.predict(X_test_in, batch_size=128).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 FINAL TEST ACCURACY: {acc:.4f}\n")

    print("📈 Classification report:")
    print(classification_report(y_test, y_pred, target_names=["LEFT", "RIGHT"]))

    cm = confusion_matrix(y_test, y_pred)
    print("📉 Confusion matrix:\n", cm)

    # Save numpy arrays
    np.save(f"{OUTPUT_DIR}/y_true.npy", y_test)
    np.save(f"{OUTPUT_DIR}/y_pred.npy", y_pred)
    np.save(f"{OUTPUT_DIR}/y_prob.npy", y_prob)
    np.save(f"{OUTPUT_DIR}/confusion.npy", cm)

    # -----------------------------
    # LOAD HISTORY
    # -----------------------------
    if os.path.exists(HISTORY_PATH):
        print("\n📚 Loading training history...")
        with open(HISTORY_PATH, "r") as f:
            history = json.load(f)

        plot_training_curves(history, os.path.join(OUTPUT_DIR, "EEGNet"))
    else:
        print("⚠️ Training history not found.")

    # -----------------------------
    # CONFUSION MATRIX PLOT
    # -----------------------------
    plot_confusion_matrix(cm, os.path.join(OUTPUT_DIR, "EEGNet"))

    print("\n✅ FULL REPORT GENERATED!")
    print(f"All outputs saved to: {OUTPUT_DIR}\n")


if __name__ == "__main__":
    main()
