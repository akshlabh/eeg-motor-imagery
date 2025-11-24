# main.py
import os
import numpy as np

from preprocess import load_eeg_data, clean_data, create_epochs
from vectorize import extract_embeddings
from build_vector_db import build_faiss_db
from query_faiss_db import query_faiss_db
from visualise import visualize_embeddings, visualize_query_result

# --------- User configuration ----------
# If you have a folder with EDFs, set SUBJECT_FOLDER to that path.
# Otherwise the script will run a demo with synthetic data.
SUBJECT_FOLDER = '../data'  # e.g. "/path/to/subject01"
# ---------------------------------------

def demo_flow():
    print("🔬 Running demo with synthetic data...")
    n_trials, n_channels, n_samples = 50, 22, 256
    dummy_data = np.random.randn(n_trials, n_channels, n_samples).astype(np.float32)
    events = np.column_stack([np.arange(n_trials), np.zeros(n_trials, dtype=int), np.random.randint(0, 2, n_trials)])
    info = None
    try:
        import mne
        info = mne.create_info(ch_names=[f"EEG{i}" for i in range(n_channels)], sfreq=256, ch_types='eeg')
        epochs = mne.EpochsArray(dummy_data, info, events)
    except Exception:
        # fallback: create a minimal object-like wrapper
        class EpochsDummy:
            def __init__(self, X, events):
                self._X = X
                self._events = events
            def get_data(self):
                return self._X
            @property
            def events(self):
                return self._events
        epochs = EpochsDummy(dummy_data, events)

    embeddings, labels, feat_info = extract_embeddings(epochs, n_components=6, save_models=False)

    meta = [
    {
        "subject": "demo",
        "trial": int(i),
        "label": int(labels[i]),
        "note": "synthetic",
        "pipeline": feat_info
    }
    for i in range(len(labels))
    ]   

    build_faiss_db(
    embeddings,
    meta,   # metadata list of dicts
    index_path="data/eeg_index.faiss",
    meta_path="data/eeg_vector_db.jsonl",
    metric="l2"   # or "cosine" if you normalized vectors
    )

    visualize_embeddings(embeddings, labels)

    query_vector = embeddings[0]
    results = query_faiss_db(query_vector, top_k=3)
    print("\n🔍 FAISS Search Results:")
    for r in results:
        print(r)

    # highlight neighbor indices
    top_indices = [r["index"] for r in results]
    visualize_query_result(query_vector, embeddings, labels, top_indices=top_indices)


def real_flow(subject_folder):
    # 1. Load raw
    raw = load_eeg_data(subject_folder)

    # 2. Clean (ICA)
    raw_clean, ica = clean_data(raw)

    # 3. Create epochs
    epochs = create_epochs(raw_clean, tmin=0.0, tmax=2.5)

    # 4. Extract embeddings
    embeddings, labels, feat_info = extract_embeddings(epochs, n_components=6, save_models=False)

    # 5. Meta
    meta = [
        {
            "subject": os.path.basename(subject_folder),
            "trial": int(i),
            "label": int(labels[i]),
            "source_file": None,   # you can fill this with actual EDF path/time if you add it
            "pipeline": feat_info
        }
        for i in range(len(labels))
    ]

    # 6. Build DB
    build_faiss_db(
    embeddings,
    meta,   # metadata list of dicts
    index_path="data/eeg_index.faiss",
    meta_path="data/eeg_vector_db.jsonl",
    metric="l2"   # or "cosine" if you normalized vectors
    )

    # 7. Visualize
    visualize_embeddings(embeddings, labels)

    # 8. Query and visualize a sample
    q = embeddings[0]
    results = query_faiss_db(q, top_k=5)
    print("\n🔍 FAISS Search Results:")
    for r in results:
        print(r)
    visualize_query_result(q, embeddings, labels, top_indices=[r["index"] for r in results])


if __name__ == "__main__":
    if SUBJECT_FOLDER and os.path.isdir(SUBJECT_FOLDER):
        real_flow(SUBJECT_FOLDER)
    else:
        demo_flow()
