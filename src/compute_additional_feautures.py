import os
import numpy as np
from scipy.signal import welch
from tqdm import tqdm

from preprocess import load_eeg_data, clean_data, create_epochs

try:
    from pyriemann.tangentspace import TangentSpace
    from pyriemann.estimation import Covariances
    HAVE_PYRIEMANN = True
except:
    HAVE_PYRIEMANN = False
    print("⚠ pyriemann missing — tangent-space -> fallback")

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUT_DIR   = DATA_DIR

TARGET_LABELS = [2, 3]   # LEFT, RIGHT ONLY


def subject_folders(base):
    return sorted([
        os.path.join(base, d)
        for d in os.listdir(base)
        if d.lower().startswith("s") and os.path.isdir(os.path.join(base, d))
    ])


def bandpower(epoch, sf=160, bands=[(8,13),(13,30)], nperseg=128):
    from numpy import trapezoid
    bp = []
    for ch in epoch:
        f, Pxx = welch(ch, fs=sf, nperseg=nperseg)
        for lo, hi in bands:
            idx = (f >= lo) & (f <= hi)
            bp.append(trapezoid(Pxx[idx], f[idx]))
    return np.array(bp, dtype=np.float32)


def cov_matrix(epoch):
    X = np.asarray(epoch, dtype=np.float64)
    C = X @ X.T
    tr = np.trace(C)
    if tr > 0:
        C /= tr
    C = 0.5 * (C + C.T)
    C += 1e-6 * np.eye(C.shape[0])
    return C


def main():
    psd_list = []
    cov_list = []
    label_list = []
    subj_list = []

    print("\n📌 Extracting LEFT–RIGHT PSD + Cov features...\n")

    for subj_dir in tqdm(subject_folders(DATA_DIR)):
        subj = os.path.basename(subj_dir)
        try:
            print(f"\n📌 Subject {subj}")

            raw = load_eeg_data(subj_dir)
            raw_clean, ica = clean_data(raw, apply_ica=False)  # light mode

            epochs = create_epochs(raw_clean, tmin=0.5, tmax=2.5)
            X = epochs.get_data()
            y = epochs.events[:, 2].astype(int)

            # FILTER LEFT + RIGHT
            mask = np.isin(y, TARGET_LABELS)
            X = X[mask]
            y = y[mask]

            print(f"   → Using {X.shape[0]} LR epochs")

            for i in range(X.shape[0]):
                ep = X[i]

                # PSD
                psd_list.append(bandpower(ep))

                # Covariance
                cov_list.append(cov_matrix(ep))

                label_list.append(int(y[i]))
                subj_list.append(subj)

        except Exception as e:
            print("   ❌ Skip", subj, "error:", e)


    psd_arr = np.vstack(psd_list).astype(np.float32)
    labels  = np.array(label_list, dtype=np.int32)
    subjects= np.array(subj_list)

    print("\n➡ PSD shape:", psd_arr.shape)

    # ---- Riemannian Tangent Space (robust) ----
    print("\n📌 Computing covariance-based features...")

    try:
        if HAVE_PYRIEMANN:
            print("   → Using pyriemann OAS + TangentSpace")
            covs = np.stack(cov_list)
            Cs = Covariances(estimator="oas").fit_transform(covs)
            ts = TangentSpace().fit_transform(Cs)
            cov_ts = ts.astype(np.float32)
        else:
            raise Exception("pyriemann unavailable")

    except Exception as e:
        print("⚠ Tangent-space failed:", e)
        print("   → Falling back to log-diagonal + upper-tri features")
        fallback_feats = []
        for C in cov_list:
            ld = np.log(np.diag(C) + 1e-12)
            ut = C[np.triu_indices(C.shape[0])]
            fallback_feats.append(np.concatenate([ld, ut[:50]], dtype=np.float32))
        cov_ts = np.vstack(fallback_feats)
    
    print("➡ Cov/TS shape:", cov_ts.shape)

    # ---- Save ----
    np.save(os.path.join(OUT_DIR, "additional_psd_aug.npy"), psd_arr)
    np.save(os.path.join(OUT_DIR, "additional_cov_ts_aug.npy"), cov_ts)
    np.save(os.path.join(OUT_DIR, "additional_labels_aug.npy"), labels)
    np.save(os.path.join(OUT_DIR, "additional_subjects_aug.npy"), subjects)

    print("\n🎉 DONE — Saved aligned LR-only PSD + Cov features.")
    print("   →", os.path.join(OUT_DIR, "additional_psd_aug.npy"))
    print("   →", os.path.join(OUT_DIR, "additional_cov_ts_aug.npy"))

if __name__ == "__main__":
    main()
