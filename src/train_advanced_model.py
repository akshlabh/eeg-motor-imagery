import os
import numpy as np

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

from xgboost import XGBClassifier
import joblib

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")


def load_and_merge_features():
    """
    Load CSP, PSD and Covariance/Tangent features and merge into a single matrix.

    Returns
    -------
    X : np.ndarray, shape (n_trials, n_features)
    y : np.ndarray, shape (n_trials,)
        Original labels (1 = rest, 2 = left, 3 = right)
    subjects : np.ndarray, shape (n_trials,)
        Subject ID per trial (string codes like 'S001')
    """
    # CSP
    X_csp = np.load(os.path.join(DATA_DIR, "global_vectors.npy"))
    y_csp = np.load(os.path.join(DATA_DIR, "global_labels.npy"))
    subj_csp = np.load(os.path.join(DATA_DIR, "global_subjects.npy"))

    # Additional features (Left/Right only, already aligned)
    X_psd = np.load(os.path.join(DATA_DIR, "additional_psd_aug.npy"))
    X_cov = np.load(os.path.join(DATA_DIR, "additional_cov_ts_aug.npy"))
    y_add = np.load(os.path.join(DATA_DIR, "additional_labels_aug.npy"))
    subj_add = np.load(os.path.join(DATA_DIR, "additional_subjects_aug.npy"))

    # Basic sanity checks
    assert X_csp.shape[0] == X_psd.shape[0] == X_cov.shape[0], \
        "Number of trials mismatch between CSP and additional features."
    assert np.all(y_csp == y_add), "Label arrays for CSP and additional features differ."
    assert np.all(subj_csp == subj_add), "Subject arrays for CSP and additional features differ."

    # Merge features along feature dimension
    X_full = np.concatenate([X_csp, X_psd, X_cov], axis=1)
    return X_full, y_csp, subj_csp


def main(test_size=0.2, random_state=42):
    # 1. Load & merge all features
    X, y, subjects = load_and_merge_features()
    print("📊 Full feature matrix:", X.shape)

    # 2. Keep only LEFT (2) and RIGHT (3) trials
    mask = (y == 2) | (y == 3)
    X = X[mask]
    y = y[mask]
    subjects = subjects[mask]

    print("🎯 Using LEFT (2) vs RIGHT (3) only")
    print("   Filtered feature matrix:", X.shape)
    print("   Label distribution:", {lbl: int((y == lbl).sum()) for lbl in np.unique(y)})

    # 3. Map labels {2, 3} -> {0, 1} for XGBoost
    #    0 -> LEFT (2), 1 -> RIGHT (3)
    y_bin = (y == 3).astype(int)

    # 4. Train/validation split
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y_bin,
        test_size=test_size,
        random_state=random_state,
        stratify=y_bin,
    )
    print(f"\n🧪 Train size: {X_train.shape[0]} | Test size: {X_test.shape[0]}")

    # 5. Build pipeline: scaler -> feature selection -> XGBoost
    pipe = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("select", SelectKBest(score_func=f_classif, k=min(200, X.shape[1]))),
            (
                "clf",
                XGBClassifier(
                    use_label_encoder=False,
                    eval_metric="logloss",
                    objective="binary:logistic",
                    n_jobs=-1,
                    random_state=random_state,
                ),
            ),
        ]
    )

    # Hyperparameter search space (kept moderate to avoid huge runtime)
    param_dist = {
        "select__k": [20, 50, 100, min(200, X.shape[1])],
        "clf__n_estimators": [100, 200],
        "clf__max_depth": [3, 5, 8],
        "clf__learning_rate": [0.01, 0.05, 0.1],
        "clf__subsample": [0.8, 1.0],
        "clf__colsample_bytree": [0.8, 1.0],
    }

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=random_state)

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=24,                # 24 random combos
        cv=cv,
        scoring="accuracy",
        n_jobs=-1,
        verbose=2,
        random_state=random_state,
        error_score="raise",      # show real error if something goes wrong
    )

    # 6. Hyperparameter search
    print("\n🔍 Running RandomizedSearchCV for XGBoost (this may take a bit)...")
    search.fit(X_train, y_train)

    print("\n🏆 Best CV accuracy:", search.best_score_)
    print("🏅 Best parameters:", search.best_params_)

    best_model = search.best_estimator_

    # 7. Evaluate on held-out test set
    y_pred_bin = best_model.predict(X_test)

    # Convert back to original labels for reporting:
    # 0 -> 2 (LEFT), 1 -> 3 (RIGHT)
    y_test_orig = np.where(y_test == 0, 2, 3)
    y_pred_orig = np.where(y_pred_bin == 0, 2, 3)

    test_acc = accuracy_score(y_test_orig, y_pred_orig)

    print(f"\n✅ Test Accuracy (LEFT vs RIGHT): {test_acc:.4f}\n")

    print("📈 Classification report (labels in original space 2/3):")
    print(
        classification_report(
            y_test_orig,
            y_pred_orig,
            labels=[2, 3],
            target_names=["LEFT (2)", "RIGHT (3)"],
        )
    )

    print("📉 Confusion matrix (rows=true, cols=pred, labels [2,3]):")
    print(confusion_matrix(y_test_orig, y_pred_orig, labels=[2, 3]))

    # 8. Save final model (pipeline + XGB) for deployment
    os.makedirs(MODELS_DIR, exist_ok=True)
    model_path = os.path.join(MODELS_DIR, "xgb_global_augmented.joblib")
    joblib.dump(
        {
            "pipeline": best_model,
            "label_mapping": {0: 2, 1: 3},  # for converting predictions back
        },
        model_path,
    )
    print(f"\n💾 Saved optimized XGBoost pipeline to: {model_path}")


if __name__ == "__main__":
    main()
