# prepare_eegnet_data.py
import os
import numpy as np
from preprocess import load_eeg_data, clean_data, create_epochs

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

TARGET_LEN = 321  # number of time samples to keep per epoch (safe across subjects)

def subject_folders(base):
    return sorted(
        [
            os.path.join(base, d)
            for d in os.listdir(base)
            if d.lower().startswith("s") and os.path.isdir(os.path.join(base, d))
        ]
    )

def main():
    X_list, y_list, subj_list = [], [], []

    print("📌 Preparing EEGNet dataset from EDF files...")
    for subj_dir in subject_folders(DATA_DIR):
        subj = os.path.basename(subj_dir)
        print(f"\n📌 Subject {subj}")
        try:
            raw = load_eeg_data(subj_dir)
            # We won't apply ICA here to save time; preprocessing already does bandpass + ref
            raw_clean, ica = clean_data(raw, apply_ica=False)
            epochs = create_epochs(raw_clean, tmin=0.0, tmax=2.5)  # same window as before

            X = epochs.get_data()          # (n_trials, n_ch, n_t)
            y_all = epochs.events[:, 2]    # 1=rest, 2=left, 3=right

            # Keep only LEFT(2) and RIGHT(3)
            mask = (y_all == 2) | (y_all == 3)
            X = X[mask]
            y = y_all[mask]

            if X.shape[0] == 0:
                print("  → No left/right epochs, skipping.")
                continue

            # Crop to fixed time length for all subjects
            n_t = X.shape[2]
            n_keep = min(TARGET_LEN, n_t)
            if n_t != n_keep:
                print(f"  → Cropping from {n_t} to {n_keep} samples")
            X = X[:, :, :n_keep]

            X_list.append(X.astype(np.float32))
            y_list.append(y.astype(np.int32))
            subj_list.append(np.array([subj] * X.shape[0]))

            print(f"  → Kept {X.shape[0]} LR trials with shape (channels={X.shape[1]}, time={X.shape[2]})")

        except Exception as e:
            print(f"  ⚠ SKIP {subj_dir} due to error: {e}")

    if not X_list:
        raise RuntimeError("No data collected for EEGNet!")

    X_all = np.concatenate(X_list, axis=0)
    y_all = np.concatenate(y_list, axis=0)
    subj_all = np.concatenate(subj_list, axis=0)

    os.makedirs(DATA_DIR, exist_ok=True)
    np.save(os.path.join(DATA_DIR, "eegnet_epochs.npy"), X_all)
    np.save(os.path.join(DATA_DIR, "eegnet_labels.npy"), y_all)
    np.save(os.path.join(DATA_DIR, "eegnet_subjects.npy"), subj_all)

    unique, counts = np.unique(y_all, return_counts=True)
    print("\n✅ Saved EEGNet data:")
    print("  eegnet_epochs.npy  →", X_all.shape)
    print("  eegnet_labels.npy  →", y_all.shape, dict(zip(unique, counts)))
    print("  eegnet_subjects.npy →", subj_all.shape)

if __name__ == "__main__":
    main()
