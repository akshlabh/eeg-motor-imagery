import os
import numpy as np
from tqdm import tqdm
from scipy.signal import welch
from pyriemann.estimation import Covariances
from pyriemann.tangentspace import TangentSpace

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

OUT_PSD = os.path.join(DATA_DIR, "additional_psd.npy")
OUT_TS = os.path.join(DATA_DIR, "additional_tangent.npy")


def compute_psd_features(epochs_data, sfreq=160):
    """
    Compute bandpower PSD over mu (8-13 Hz) & beta (14-30 Hz).
    Output per trial: [BP_mu, BP_beta] per channel.
    """
    bands = {"mu": (8, 13), "beta": (14, 30)}

    psd_features = []

    for trial in epochs_data:
        ch_features = []
        for ch in trial:
            f, Pxx = welch(ch, fs=sfreq, nperseg=256)

            features_ch = []
            for band, (l, h) in bands.items():
                idx = (f >= l) & (f <= h)
                bp = np.trapz(Pxx[idx], f[idx])
                features_ch.append(bp)
            ch_features.extend(features_ch)

        psd_features.append(ch_features)

    return np.array(psd_features)


def compute_tangent_space_features(epochs_data):
    """
    Compute Riemannian Tangent Space features.
    BEST for MI classification.
    """
    covs = Covariances().fit_transform(epochs_data)
    ts = TangentSpace().fit_transform(covs)
    return ts


def main():
    print("🔍 Loading CSP dataset...")
    X_csp = np.load(os.path.join(DATA_DIR, "global_vectors.npy"))  # (19674, 4)
    y = np.load(os.path.join(DATA_DIR, "global_labels.npy"))
    subjects = np.load(os.path.join(DATA_DIR, "global_subjects.npy"))

    print("🔍 Loading raw epoch data for PSD+TS extraction...")
    # You already have epoch data in CSP pipeline — reuse the cached epochs
    # They must be saved during build_global_db; if not, regenerate minimal raw epoch dataset
    X_epochs = np.load(os.path.join(DATA_DIR, "global_epochs_raw.npy"))  # shape: (19674, 64, N)

    print("📡 Computing PSD features...")
    X_psd = compute_psd_features(X_epochs)
    np.save(OUT_PSD, X_psd)
    print("✅ PSD saved:", OUT_PSD, X_psd.shape)

    print("📡 Computing Tangent Space features...")
    X_ts = compute_tangent_space_features(X_epochs)
    np.save(OUT_TS, X_ts)
    print("✅ TS saved:", OUT_TS, X_ts.shape)

    print("🎉 All additional features extracted successfully!")


if __name__ == "__main__":
    main()
