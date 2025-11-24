# src/build_global_csp_incremental.py (REPLACED CONTENT - RENAME TO 'build_subject_csp_features.py' for clarity)
import os
import numpy as np
from tqdm import tqdm
from preprocess import load_eeg_data, clean_data, create_epochs
from vectorize import extract_embeddings # <-- This is the key change!
from build_vector_db import build_faiss_db
import json

# --- (Directory/Path definitions remain the same) ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(SCRIPT_DIR, "data")
OUT_DIR = DATA_DIR
OUT_VEC = os.path.join(OUT_DIR, "global_vectors.npy")
OUT_LABELS = os.path.join(OUT_DIR, "global_labels.npy")
OUT_SUBJECTS = os.path.join(OUT_DIR, "global_subjects.npy")
OUT_META = os.path.join(OUT_DIR, "eeg_vector_db_global.jsonl")
OUT_INDEX = os.path.join(OUT_DIR, "eeg_index_global.faiss")
# ----------------------------------------------------

def subject_folders(base):
    return sorted([os.path.join(base, d) for d in os.listdir(base) if d.lower().startswith("s") and os.path.isdir(os.path.join(base, d))])

def main(n_components=8): # Use 8 components (4 for Left, 4 for Right)
    vectors = []
    labels = []
    subjects = []
    meta_entries = []
    target_labels = [2, 3] # Left and Right MI

    print("PASS 1: Computing PER-SUBJECT regularized CSP features...")

    for subj_dir in tqdm(subject_folders(DATA_DIR), desc="Processing Subjects"):
        subj = os.path.basename(subj_dir)
        try:
            raw = load_eeg_data(subj_dir)
            
            # CRITICAL: APPLY ICA for artifact cleaning before CSP
            # NOTE: Assuming EOG channel is 'eog' or you've configured it in preprocess.py
            raw_clean, ica = clean_data(raw, apply_ica=True) 
            epochs = create_epochs(raw_clean, tmin=0.0, tmax=2.5)

            # Use MNE's regularized CSP (fit/transform) and StandardScaler per subject
            V, y_all_labels, feat_info = extract_embeddings(
                epochs,
                n_components=n_components,
                save_models=True, # Saves the per-subject CSP/Scaler for future use
                models_folder=os.path.join(os.path.dirname(DATA_DIR), "models", subj) 
            )

            # Filter for Left (2) and Right (3) epochs only
            mask = np.isin(y_all_labels, target_labels)
            V_subj = V[mask]
            y_subj = y_all_labels[mask]

            if V_subj.shape[0] == 0:
                 continue # Skip subject if no target epochs found

            vectors.append(V_subj)
            labels.extend(y_subj.tolist())
            subjects.extend([subj] * V_subj.shape[0])

            # Update meta entries
            for i, lbl in enumerate(y_subj):
                meta_entries.append({"subject": subj, "label": int(lbl), "feature_pipeline": feat_info})

        except Exception as e:
            print(f"\nSKIP subject {subj_dir} error: {e}")

    if not vectors:
        raise RuntimeError("No features extracted.")

    V_all = np.vstack(vectors).astype(np.float32)
    y_all = np.array(labels, dtype=np.int32)
    subj_array = np.array(subjects, dtype='<U10')

    # 3. Save the new feature set
    np.save(OUT_VEC, V_all)
    np.save(OUT_LABELS, y_all)
    np.save(OUT_SUBJECTS, subj_array)
    print(f"\n✅ Saved new robust features: {OUT_VEC} shape {V_all.shape}")

    # Rebuild FAISS DB (optional, but necessary since vectors changed)
    # build_faiss_db(V_all, meta_entries, index_path=OUT_INDEX, meta_path=OUT_META, metric="l2")
    # print("Built new FAISS index.")

if __name__ == "__main__":
    main(n_components=4)