
# 🧠 EEG Motor Imagery Classification (BCI)  
**Left vs Right Motor Imagery using CSP, XGBoost/SVM, and EEGNet**

This project implements a complete EEG-based Motor Imagery classification pipeline using **BCI Competition IV Dataset 2a (PhysioNet)**.  
We train and compare multiple machine-learning and deep-learning models for **binary left vs right hand imagery**:

- ✔ CSP + SVM  
- ✔ CSP + XGBoost  
- ✔ PSD + Riemannian Tangent Space features  
- ✔ A deep-learning EEGNet model  
- ✔ Fully automated evaluation + plots + report generation  

Final best accuracy: **≈ 71% (EEGNet-Fast, stratified 80/20 split)**

---

## 📂 Project Structure (Cleaned)

```

project/
│
├── src/
│   ├── preprocess.py                  # Load EDF files, filter, epoch
│   ├── dataloader_eegnet.py           # Loader for EEGNet-ready epochs
│   ├── build_global_csp_incremental.py# CSP feature extraction
│   ├── compute_additional_features.py # PSD, covariance, tangent-space
│   ├── train_advanced_model.py        # XGBoost + CSP training
│   ├── train_improved_model.py        # Stacked ML ensemble
│   ├── train_eegnet_fast.py           # Fast EEGNet (recommended)
│   ├── evaluate_eegnet.py             # Evaluate trained EEGNet model
│   ├── infer_single.py                # Predict for single trial
│   ├── generate_eegnet_report.py      # Generate all plots + report
│   ├── report_outputs/                # Training curves, confusion plots
│   └── artifacts/                     # (ignored) predictions, history
│
├── models/            # (ignored) Saved .h5 / .joblib models
├── data/              # (ignored) Raw EDF files (S001–S109)
├── .gitignore
└── README.md

```

---

# 📥 Dataset: Where to Download

This project uses:

### **BCI Competition IV - Dataset 2a**  
Link (PhysioNet):  
https://physionet.org/content/eegmmidb/1.0.0/

Each subject folder contains .EDF files:

```

S001R03.edf → Baseline/Rest
S001R04.edf → Task Left/Right Imagery
...

```

### Tasks You Need
For Motor Imagery (left/right):

- **R03–R14** for each subject  
- Ignore R01, R02, R15 if present (not motor imagery)

### Place files here:

```

project/src/data/S001/*.edf
project/src/data/S002/*.edf
...

````

---

# ⚙️ Installation

```bash
git clone https://github.com/akshlabh/eeg-motor-imagery.git
cd eeg-motor-imagery

python -m venv .venv
.\.venv\Scripts\activate   # Windows

pip install -r src/requirements.txt
````

---

# 🧹 Step 1: Preprocess EDF → Epochs (fast)

This script loads raw EDF signals, filters, removes artifacts, and saves epochs.

```bash
python src/preprocess.py
```

This generates:

```
eegnet_epochs.npy      # shape (N, 64, T)
eegnet_labels.npy      # labels (1=left, 2=right, etc.)
eegnet_subjects.npy
```

---

# 🧠 Step 2: Extract Features for ML Models (optional)

### CSP Feature Vectors

```bash
python src/build_global_csp_incremental.py
```

Generates:

```
global_vectors.npy
global_labels.npy
```

### PSD + Riemannian Tangent Space Features

```bash
python src/compute_additional_features.py
```

Generates:

```
additional_psd_aug.npy
additional_cov_ts_aug.npy
```

---

# 🤖 Step 3: Train Machine Learning Models

## 3.1 Train XGBoost + Feature Selection (80/20)

```bash
python src/train_advanced_model.py
```

Outputs:

* `xgb_global_augmented.joblib`
* Accuracy ~62%

## 3.2 Train Improved Stacked Model

```bash
python src/train_improved_model.py
```

Outputs a stacked MLP + XGB model (~65% accuracy)

---

# 🧬 Step 4: Train Deep Learning Model (Best Choice)

### **EEGNet-Fast (Recommended)**

This is optimized for speed and accuracy.

```bash
python src/train_eegnet_fast.py
```

Outputs:

* `models/eegnet_fast_global_*.h5`
* `artifacts/history_*.json`
* `report_outputs/EEGNet_training_curves.png`

**Final Accuracy:** **~71%**

---

# 📊 Step 5: Evaluate EEGNet

```bash
python src/evaluate_eegnet.py
```

Outputs:

* Test classification report
* Confusion matrix
* `EEGNet_confusion_matrix.png`

---

# 📝 Step 6: Auto-Generate Full Report (Graphs + Metrics)

```bash
python src/generate_eegnet_report.py
```

Outputs inside `report_outputs/`:

* Training curves PNG
* Confusion matrix PNG
* Test metrics JSON
* A complete **Markdown/HTML-ready summary**

---

# 🧾 Results Summary

| Model                         | Accuracy | Notes                       |
| ----------------------------- | -------- | --------------------------- |
| SVM (Linear)                  | ~52%     | Simple CSP baseline         |
| SVM (RBF + tuning)            | ~53%     | Slight improvement          |
| XGBoost + feature selection   | ~62%     | Strong classical ML         |
| Stacked MLP + XGB             | ~65%     | Ensemble improved stability |
| **EEGNet-Fast (recommended)** | **~71%** | Best speed/accuracy         |

---

# 📌 How to Run Inference on a Single EEG Trial

```bash
python src/infer_single.py --file sample.npy
```

---

# 📘 Citations

If you use this repo in research, cite:

**EEG Motor Movement/Imagery Dataset**
Goldberger AL, et al. *PhysioBank, PhysioToolkit, and PhysioNet.*

**EEGNet: A Compact CNN for EEG-based BCIs**
Vernon Lawhern et al., 2018.

---

# 🙌 Acknowledgements

* PhysioNet & BCI Competition IV for dataset
* PyRiemann, MNE-Python, TensorFlow/Keras
* Original EEGNet authors

---

# 📬 Contact

For questions or improvements:
**Akshay** – *akshlabh (GitHub)*


