# src/vectorize.py
import numpy as np
from sklearn.preprocessing import StandardScaler
from mne.decoding import CSP
import joblib
import os

def extract_embeddings(epochs, n_components=6, save_models=True, models_folder="models", prefix="csp"):
    print("🧠 Extracting CSP embeddings...")
    X = epochs.get_data()   # (n_trials, n_ch, n_times)
    if getattr(epochs, "events", None) is not None and epochs.events.shape[1] >= 3:
        y = epochs.events[:, 2].astype(int)
    elif getattr(epochs, "metadata", None) is not None and "label" in epochs.metadata:
        y = epochs.metadata["label"].astype(int).values
    else:
        raise RuntimeError("Cannot extract labels from epochs")

    csp = CSP(n_components=n_components, log=True, reg="ledoit_wolf")
    X_csp = csp.fit_transform(X, y)          # (n_trials, n_components)
    scaler = StandardScaler().fit(X_csp)
    X_scaled = scaler.transform(X_csp).astype(np.float32)

    if save_models:
        os.makedirs(models_folder, exist_ok=True)
        joblib.dump(csp, os.path.join(models_folder, f"{prefix}_csp.joblib"))
        joblib.dump(scaler, os.path.join(models_folder, f"{prefix}_scaler.joblib"))
        print(f"  → Saved models to {models_folder}/")

    return X_scaled, y.astype(int), {"csp_components": n_components}


def transform_epochs(epochs, csp_joblib, scaler_joblib):
    import joblib
    csp = joblib.load(csp_joblib)
    scaler = joblib.load(scaler_joblib)
    X = epochs.get_data()
    X_csp = csp.transform(X)
    X_scaled = scaler.transform(X_csp).astype(np.float32)
    return X_scaled
