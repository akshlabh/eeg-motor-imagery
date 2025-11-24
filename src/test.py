# src/test.py
import os
import numpy as np

DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

def show(name):
    path = os.path.join(DATA_DIR, name)
    print(name, "→", "FOUND" if os.path.exists(path) else "MISSING")
    if os.path.exists(path):
        print("   shape:", np.load(path).shape)

files = [
    "global_vectors.npy",
    "global_labels.npy",
    "global_subjects.npy",
    "additional_psd_aug.npy",
    "additional_cov_ts_aug.npy",
    "additional_labels_aug.npy",
    "additional_subjects_aug.npy",
]

for f in files:
    show(f)
