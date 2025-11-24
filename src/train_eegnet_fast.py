# train_eegnet_fast.py
import os
import json
import time
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from collections import Counter

from dataloader_eegnet import load_epochs, filter_lr, split_train_test, batch_generator

# ----------------- Reproducibility -----------------
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
ARTIFACTS_DIR = os.path.join(SCRIPT_DIR, "artifacts")
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(ARTIFACTS_DIR, exist_ok=True)

# ----------------- EEGNet model -----------------
def EEGNet_simple(nb_channels, nb_samples, drop_prob=0.5, kern_len=64):
    """
    Compact EEGNet-style CNN for motor imagery (binary).
    Input shape: (channels, samples, 1)
    """
    inputs = keras.Input(shape=(nb_channels, nb_samples, 1))

    # Temporal conv
    x = layers.Conv2D(16, (1, kern_len), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)

    # Depthwise (spatial) conv
    x = layers.DepthwiseConv2D((nb_channels, 1), depth_multiplier=2,
                               use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(drop_prob)(x)

    # Separable conv
    x = layers.SeparableConv2D(16, (1, 16), padding='same', use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 8))(x)
    x = layers.Dropout(drop_prob)(x)

    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='elu')(x)
    x = layers.BatchNormalization()(x)   # small stability boost vs old script
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(1e-3),
        loss='binary_crossentropy',
        metrics=['accuracy'],
    )
    return model

# ----------------- Data loading + preprocessing -----------------
def load_and_preprocess():
    """
    Loads eegnet_epochs.npy / eegnet_labels.npy / eegnet_subjects.npy
    Keeps only LEFT (2) and RIGHT (3), maps to 0/1.
    Applies per-channel z-score normalization across time.
    """
    X, y_raw, subjects = load_epochs()

    # Keep only LEFT(2) and RIGHT(3) and map → 0 / 1
    X, y_bin, subjects = filter_lr(X, y_raw, subjects)

    # Per-channel z-score normalization across time
    X = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-6)

    return X.astype(np.float32), y_bin.astype(np.int32), subjects

# ----------------- Main training function -----------------
def main(test_size=0.2, batch_size=64, max_epochs=80, save_history=True):
    print("🔍 Loading EEGNet-ready dataset...")
    X, y, subjects = load_and_preprocess()
    n_samples, n_ch, n_t = X.shape
    label_counts = Counter(y)
    print(f"Data shape: {X.shape}, Labels distribution: {label_counts} (0=LEFT,1=RIGHT)")

    # 80/20 split, stratified on label (trial-level split, FAST)
    X_train, X_test, y_train, y_test = split_train_test(
        X, y, test_size=test_size, random_state=SEED, stratify=True
    )
    print(f"Train: {X_train.shape[0]}  |  Test: {X_test.shape[0]}")

    # Build model
    model = EEGNet_simple(nb_channels=n_ch, nb_samples=n_t,
                          drop_prob=0.5, kern_len=min(64, n_t // 2))

    # ----- File names with timestamp (for multiple runs) -----
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    ckpt_path = os.path.join(MODELS_DIR, f"eegnet_fast_global_{timestamp}.h5")
    history_path = os.path.join(ARTIFACTS_DIR, f"history_eegnet_fast_{timestamp}.json")
    preds_prefix = os.path.join(ARTIFACTS_DIR, f"eegnet_fast_{timestamp}")

    # Callbacks (early stopping, LR scheduling, checkpoint)
    callbacks = [
        keras.callbacks.ModelCheckpoint(
            ckpt_path, monitor='val_loss', save_best_only=True, mode='min'
        ),
        keras.callbacks.EarlyStopping(
            monitor='val_loss', patience=12, restore_best_weights=True
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss', factor=0.5, patience=6, min_lr=1e-5
        ),
    ]

    # Training generator with augmentation (same as your old working script)
    train_gen = batch_generator(
        X_train, y_train, batch_size=batch_size, shuffle=True, augment=True
    )
    steps_per_epoch = max(1, X_train.shape[0] // batch_size)

    # Validation data (no augmentation)
    X_val = X_test[..., np.newaxis]
    y_val = y_test

    print("🚀 Training EEGNet (fast 80/20)...")
    history = model.fit(
        train_gen,
        steps_per_epoch=steps_per_epoch,
        epochs=max_epochs,
        validation_data=(X_val, y_val),
        callbacks=callbacks,
        verbose=1,
        # ⚠️ NO class_weight here – Keras doesn't support it with Python generators
    )

    # Load best weights (based on val_loss)
    model.load_weights(ckpt_path)

    # ---------- Save training history for plots ----------
    if save_history:
        try:
            with open(history_path, "w") as f:
                json.dump(history.history, f)
            print(f"💾 Saved training history to: {history_path}")
        except Exception as e:
            print("⚠ Could not save history:", e)

    # Final evaluation
    print("\n✅ Evaluating on held-out test set...")
    y_prob = model.predict(X_val, batch_size=128).ravel()
    y_pred = (y_prob >= 0.5).astype(int)

    acc = accuracy_score(y_val, y_pred)
    print(f"\n🎯 Test accuracy (LEFT vs RIGHT): {acc:.4f}\n")

    print("📈 Classification report:")
    print(classification_report(
        y_val, y_pred,
        target_names=["LEFT(0)", "RIGHT(1)"]
    ))

    cm = confusion_matrix(y_val, y_pred)
    print("📉 Confusion matrix (rows=true, cols=pred):")
    print(cm)

    # ---------- Save evaluation artifacts for later plotting/report ----------
    try:
        np.save(preds_prefix + "_confusion.npy", cm)
        np.save(preds_prefix + "_y_true.npy", y_val)
        np.save(preds_prefix + "_y_pred.npy", y_pred)
        np.save(preds_prefix + "_y_prob.npy", y_prob)
        print(f"💾 Saved predictions/confusion to prefix: {preds_prefix}_*.npy")
    except Exception as e:
        print("⚠ Could not save prediction artifacts:", e)

    print(f"\n💾 Saved best EEGNet model to: {ckpt_path}")

if __name__ == "__main__":
    main()
