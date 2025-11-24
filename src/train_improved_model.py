"""
train_improved_model.py
Robust stacked ensemble: XGBoost (tree) + Keras MLP (neural) -> LogisticRegression meta.
Uses already-extracted features in src/data:
 - global_vectors.npy
 - global_labels.npy
 - global_subjects.npy
 - additional_psd_aug.npy
 - additional_cov_ts_aug.npy
 - additional_labels_aug.npy
 - additional_subjects_aug.npy

Outputs:
 - models/xgb_base.joblib
 - models/mlp_base.h5
 - models/stack_meta.joblib
 - models/stack_pipeline.joblib  (useful wrapper)
"""

import os
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
import joblib
import warnings
warnings.filterwarnings("ignore")

# XGBoost
from xgboost import XGBClassifier

# Keras for MLP
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# reproducibility
SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def load_and_merge():
    X_csp = np.load(os.path.join(DATA_DIR, "global_vectors.npy"))
    y_csp = np.load(os.path.join(DATA_DIR, "global_labels.npy"))
    subj_csp = np.load(os.path.join(DATA_DIR, "global_subjects.npy"))

    X_psd = np.load(os.path.join(DATA_DIR, "additional_psd_aug.npy"))
    X_ts = np.load(os.path.join(DATA_DIR, "additional_cov_ts_aug.npy"))
    y_add = np.load(os.path.join(DATA_DIR, "additional_labels_aug.npy"))
    subj_add = np.load(os.path.join(DATA_DIR, "additional_subjects_aug.npy"))

    assert X_csp.shape[0] == X_psd.shape[0] == X_ts.shape[0]
    assert np.all(y_csp == y_add)
    assert np.all(subj_csp == subj_add)

    X = np.concatenate([X_csp, X_psd, X_ts], axis=1)
    return X, y_csp, subj_csp


def build_mlp(input_dim, dropout_rate=0.4):
    inputs = keras.Input(shape=(input_dim,), name="input")
    x = layers.BatchNormalization()(inputs)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    x = layers.Dense(128, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(dropout_rate)(x)
    outputs = layers.Dense(1, activation="sigmoid", name="out")(x)

    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )
    return model


def get_oof_predictions(X, y_binary, n_splits=5):
    """
    Train base models with CV and return out-of-fold predictions for stacking.
    Returns:
      oof_xgb (n_samples,), oof_mlp (n_samples,), test_xgb_models list, test_mlp_models list
    """
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)

    oof_xgb = np.zeros(len(y_binary), dtype=float)
    oof_mlp = np.zeros(len(y_binary), dtype=float)

    # to save fold models if needed
    xgb_models = []
    mlp_models = []

    fold_idx = 0
    for train_index, valid_index in skf.split(X, y_binary):
        fold_idx += 1
        print(f"\n--- OOF fold {fold_idx}/{n_splits} ---")
        X_tr, X_val = X[train_index], X[valid_index]
        y_tr, y_val = y_binary[train_index], y_binary[valid_index]

        # Standardize per fold
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        # XGBoost base (compatible for older versions)
        xgb = XGBClassifier(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            use_label_encoder=False,
            eval_metric="logloss",
            random_state=SEED,
            n_jobs=-1,
        )

        # ✔️ add early stopping via set_params (old + new XGB compatible)
        xgb.set_params(early_stopping_rounds=25)

        xgb.fit(
            X_tr_s,
            y_tr,
            eval_set=[(X_val_s, y_val)],
            verbose=False
        )
        pred_xgb = xgb.predict_proba(X_val_s)[:, 1]
        oof_xgb[valid_index] = pred_xgb
        xgb_models.append((xgb, scaler))

        pred_xgb = xgb.predict_proba(X_val_s)[:, 1]
        oof_xgb[valid_index] = pred_xgb
        xgb_models.append((xgb, scaler))

        # MLP base
        input_dim = X.shape[1]
        mlp = build_mlp(input_dim)
        # Use class weights to balance
        # compute class weights
        from sklearn.utils import class_weight
        cw = class_weight.compute_class_weight("balanced", classes=np.unique(y_tr), y=y_tr)
        cw_dict = {0: float(cw[0]), 1: float(cw[1])}

        # train
        # scale for MLP (same scaler)
        X_tr_mlp = scaler.transform(X_tr)
        X_val_mlp = scaler.transform(X_val)

        early = keras.callbacks.EarlyStopping(monitor="val_loss", patience=12, restore_best_weights=True, verbose=0)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=6, verbose=0)
        history = mlp.fit(
            X_tr_mlp,
            y_tr,
            validation_data=(X_val_mlp, y_val),
            epochs=200,
            batch_size=128,
            class_weight=cw_dict,
            callbacks=[early, reduce_lr],
            verbose=0,
        )
        pred_mlp = mlp.predict(X_val_mlp).ravel()
        oof_mlp[valid_index] = pred_mlp
        mlp_models.append((mlp, scaler))

        # quick fold evaluation
        val_pred_bin = (pred_mlp >= 0.5).astype(int)
        acc_val = accuracy_score(y_val, val_pred_bin)
        print(f" Fold {fold_idx} MLP val accuracy: {acc_val:.4f} | XGB val AUC-like (approx): {np.mean(pred_xgb>=0.5):.3f}")

    return oof_xgb, oof_mlp, xgb_models, mlp_models


def train_stack_and_evaluate(X_train, y_train_bin, X_test, y_test_bin, oof_xgb, oof_mlp, xgb_models, mlp_models):
    # Build stacking features for training meta-model
    # Stacking training matrix = [oof_xgb, oof_mlp]
    stack_train = np.vstack([oof_xgb, oof_mlp]).T

    # Meta model: LogisticRegression with L2 (probabilistic)
    meta = LogisticRegression(C=1.0, class_weight="balanced", random_state=SEED, max_iter=2000)
    meta.fit(stack_train, y_train_bin)

    # Build test features (average predictions from base models)
    # For X_test, we need to average prediction across all saved fold scalers/models
    # Use each fold model to predict X_test and average probabilities
    def avg_preds(models_list, X):
        preds = []
        for model, scaler in models_list:
            Xs = scaler.transform(X)
            if isinstance(model, XGBClassifier):
                p = model.predict_proba(Xs)[:, 1]
            else:
                p = model.predict(Xs).ravel()
            preds.append(p)
        preds = np.stack(preds, axis=0)
        return preds.mean(axis=0)

    test_pred_xgb = avg_preds(xgb_models, X_test)
    test_pred_mlp = avg_preds(mlp_models, X_test)

    stack_test = np.vstack([test_pred_xgb, test_pred_mlp]).T
    final_proba = meta.predict_proba(stack_test)[:, 1]
    final_pred = (final_proba >= 0.5).astype(int)

    test_acc = accuracy_score(y_test_bin, final_pred)
    print("\n=== Final Ensemble Results ===")
    print("Test accuracy (binary):", test_acc)
    # Convert to original label space for reporting: 0->2, 1->3
    y_test_orig = np.where(y_test_bin == 0, 2, 3)
    final_pred_orig = np.where(final_pred == 0, 2, 3)
    print(classification_report(y_test_orig, final_pred_orig, labels=[2, 3], target_names=["LEFT (2)", "RIGHT (3)"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test_orig, final_pred_orig, labels=[2, 3]))

    # Save artifacts
    joblib.dump(meta, os.path.join(MODELS_DIR, "stack_meta.joblib"))
    joblib.dump(xgb_models, os.path.join(MODELS_DIR, "xgb_models_folds.joblib"))
    joblib.dump(mlp_models, os.path.join(MODELS_DIR, "mlp_models_folds.joblib"))

    # For convenience, save a wrapper pipeline (SelectKBest + StandardScaler on top of meta)
    wrapper = {
        "meta": meta,
        "xgb_models": xgb_models,
        "mlp_models": mlp_models,
    }
    joblib.dump(wrapper, os.path.join(MODELS_DIR, "stack_pipeline.joblib"))
    print("Saved stacked ensemble artifacts to models/")

    return test_acc


def main():
    X, y, subj = load_and_merge()
    # filter left (2) vs right (3)
    mask = (y == 2) | (y == 3)
    X = X[mask]
    y = y[mask]
    subj = subj[mask]
    print("Filtered shape:", X.shape, "labels:", np.unique(y, return_counts=True))

    # map labels to binary 0/1: left=0, right=1
    y_bin = (y == 3).astype(int)

    # split into train/test (stratified)
    X_train, X_test, y_train_bin, y_test_bin = train_test_split(
        X, y_bin, test_size=0.2, random_state=SEED, stratify=y_bin
    )

    # Feature selection BEFORE stacking? We'll let base models use full features but
    # Optionally apply SelectKBest to reduce dimension for MLP/XGB individually.
    # For speed you might want to reduce dims. We'll train on full and let models handle it.

    # Create OOF predictions for stacking
    oof_xgb, oof_mlp, xgb_models, mlp_models = get_oof_predictions(X_train, y_train_bin, n_splits=5)

    # Train meta-model and evaluate
    test_acc = train_stack_and_evaluate(X_train, y_train_bin, X_test, y_test_bin, oof_xgb, oof_mlp, xgb_models, mlp_models)

    print("\nDone. Ensemble Test accuracy:", test_acc)


if __name__ == "__main__":
    main()
