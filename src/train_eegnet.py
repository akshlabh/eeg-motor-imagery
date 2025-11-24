# train_eegnet.py
import os
import numpy as np
import random
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from dataloader_eegnet import load_epochs, filter_lr, split_train_test, batch_generator
from sklearn.model_selection import LeaveOneGroupOut
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

SEED = 42
np.random.seed(SEED)
random.seed(SEED)
tf.random.set_seed(SEED)

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(SCRIPT_DIR, "..", "models")
os.makedirs(MODELS_DIR, exist_ok=True)


def EEGNet_SSVEP(nb_channels, nb_samples, drop_prob=0.5, kern_len=64):
    """
    Simple EEGNet-like architecture (adapted for 1D epochs).
    Input shape = (channels, samples, 1)
    """
    inputs = keras.Input(shape=(nb_channels, nb_samples, 1))
    # temporal conv
    x = layers.Conv2D(16, (1, kern_len), padding='same', use_bias=False)(inputs)
    x = layers.BatchNormalization()(x)
    # depthwise (spatial) conv
    x = layers.DepthwiseConv2D((nb_channels, 1), depth_multiplier=2, use_bias=False, depthwise_constraint=None)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 4))(x)
    x = layers.Dropout(drop_prob)(x)
    # separable conv
    x = layers.SeparableConv2D(16, (1, 16), use_bias=False, padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation('elu')(x)
    x = layers.AveragePooling2D((1, 8))(x)
    x = layers.Dropout(drop_prob)(x)
    x = layers.Flatten()(x)
    x = layers.Dense(64, activation='elu')(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(1, activation='sigmoid')(x)
    model = keras.Model(inputs=inputs, outputs=outputs)
    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='binary_crossentropy', metrics=['accuracy'])
    return model


def loso_train_eval(X, y, groups, epochs=80, batch_size=64):
    logo = LeaveOneGroupOut()
    accs = []
    fold = 0
    for train_idx, test_idx in logo.split(X, y, groups):
        fold += 1
        subj = groups[test_idx][0]
        print(f"\n=== LOSO fold {fold}: holdout subject {subj} ===")
        X_tr, X_te = X[train_idx], X[test_idx]
        y_tr, y_te = y[train_idx], y[test_idx]
        # create model
        n_ch, n_t = X.shape[1], X.shape[2]
        model = EEGNet_SSVEP(n_ch, n_t, drop_prob=0.5, kern_len=min(64, n_t//2))
        # callbacks
        ckpt = keras.callbacks.ModelCheckpoint(os.path.join(MODELS_DIR, f"eegnet_fold_{subj}.h5"),
                                               save_best_only=True, monitor='val_loss', mode='min')
        es = keras.callbacks.EarlyStopping(monitor='val_loss', patience=12, restore_best_weights=True)
        reduce_lr = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=6, factor=0.5)
        # generator training
        train_gen = batch_generator(X_tr, y_tr, batch_size=batch_size, shuffle=True, augment=True)
        val_gen = batch_generator(X_te, y_te, batch_size=batch_size, shuffle=False, augment=False)
        steps_per_epoch = max(1, X_tr.shape[0] // batch_size)
        val_steps = max(1, X_te.shape[0] // batch_size)
        history = model.fit(
            train_gen,
            steps_per_epoch=steps_per_epoch,
            validation_data=val_gen,
            validation_steps=val_steps,
            epochs=epochs,
            callbacks=[ckpt, es, reduce_lr],
            verbose=1
        )
        # load best
        model.load_weights(os.path.join(MODELS_DIR, f"eegnet_fold_{subj}.h5"))
        # predict on held out
        X_te_in = X_te[..., np.newaxis]
        y_pred = (model.predict(X_te_in, batch_size=128).ravel() >= 0.5).astype(int)
        acc = accuracy_score(y_te, y_pred)
        print("Fold acc:", acc)
        accs.append(acc)
        # optional: print per fold classification report
        print(classification_report(y_te, y_pred, target_names=['LEFT','RIGHT']))
    print("\n=== LOSO summary ===")
    print("Mean acc:", np.mean(accs), "std:", np.std(accs))
    return accs


def main():
    X, y, subjects = load_epochs_and_preprocess()
    print("Data shapes:", X.shape, y.shape)
    accs = loso_train_eval(X, y, subjects, epochs=80, batch_size=64)
    print("Finished LOSO training. Per-subject acc mean:", np.mean(accs))

def load_epochs_and_preprocess():
    X, y_original, subjects = load_epochs()
    # filter left/right
    X, y_bin, subjects = filter_lr(X, y_original, subjects)
    # optional: normalize each epoch by channel-wise zscore
    X = (X - X.mean(axis=2, keepdims=True)) / (X.std(axis=2, keepdims=True) + 1e-6)
    return X, y_bin, subjects

if __name__ == "__main__":
    main()
