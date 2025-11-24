# dataloader_eegnet.py
import os
import numpy as np
from sklearn.model_selection import train_test_split

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")

def load_epochs():
    """
    Loads EEGNet-ready epochs:

      eegnet_epochs.npy   -> shape (N, n_channels, n_times)
      eegnet_labels.npy   -> shape (N,) with labels {2,3}
      eegnet_subjects.npy -> shape (N,) subject IDs

    """
    X = np.load(os.path.join(DATA_DIR, "eegnet_epochs.npy"))
    y = np.load(os.path.join(DATA_DIR, "eegnet_labels.npy"))
    subjects = np.load(os.path.join(DATA_DIR, "eegnet_subjects.npy"))
    return X.astype(np.float32), y.astype(int), subjects

def filter_lr(X, y, subjects):
    # if labels are already 2/3, this just passes them through
    mask = (y == 2) | (y == 3)
    X = X[mask]
    y = y[mask]
    subjects = subjects[mask]
    # map 2->0, 3->1
    y_bin = (y == 3).astype(int)
    return X, y_bin, subjects

# (keep the rest of this file as I gave earlier: split_train_test, augment_epoch, batch_generator)


def split_train_test(X, y, test_size=0.2, random_state=42, stratify=True):
    strat = y if stratify else None
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=strat)
    return X_tr, X_te, y_tr, y_te

# Simple augmentations: random gaussian noise, time-shift
def augment_epoch(epoch, noise_scale=0.01, shift_max=10):
    e = epoch.copy()
    # gaussian noise
    e = e + np.random.normal(scale=noise_scale, size=e.shape).astype(np.float32)
    # random circular time shift
    t = np.random.randint(-shift_max, shift_max)
    if t != 0:
        e = np.roll(e, t, axis=1)
    return e

def batch_generator(X, y, batch_size=64, shuffle=True, augment=False):
    n = X.shape[0]
    idx = np.arange(n)
    while True:
        if shuffle:
            np.random.shuffle(idx)
        for i in range(0, n, batch_size):
            batch_idx = idx[i:i+batch_size]
            batch_X = X[batch_idx].copy()
            if augment:
                for j in range(batch_X.shape[0]):
                    if np.random.rand() < 0.5:
                        batch_X[j] = augment_epoch(batch_X[j])
            batch_y = y[batch_idx]
            # EEGNet expects shape (n_samples, n_channels, n_times, 1)
            yield batch_X[..., np.newaxis], batch_y
