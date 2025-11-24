# src/build_vector_db.py
import faiss
import numpy as np
import os
import time
import json

def _ensure_dir(path):
    d = os.path.dirname(path) or "."
    os.makedirs(d, exist_ok=True)

def build_faiss_db(embeddings, metadata, index_path="data/eeg_index.faiss", meta_path="data/eeg_vector_db.jsonl",
                   metric="l2", use_gpu=False, pipeline_version="v1"):
    print("📦 Building FAISS DB...", f"metric={metric}")
    embeddings = np.asarray(embeddings, dtype=np.float32)
    assert embeddings.ndim == 2
    n, dim = embeddings.shape
    assert len(metadata) == n, "metadata length mismatch"

    if metric == "cosine":
        norms = np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-12
        embeddings = embeddings / norms
        index = faiss.IndexFlatIP(dim)
    else:
        index = faiss.IndexFlatL2(dim)

    if use_gpu:
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            print("  → FAISS: moved index to GPU")
        except Exception as e:
            print("  → FAISS GPU init failed:", e)

    index.add(embeddings)

    try:
        if use_gpu and hasattr(faiss, "index_gpu_to_cpu"):
            index_to_save = faiss.index_gpu_to_cpu(index)
        else:
            index_to_save = index
    except Exception:
        index_to_save = index

    _ensure_dir(index_path)
    _ensure_dir(meta_path)
    faiss.write_index(index_to_save, index_path)
    print(f"✅ Saved FAISS index -> {index_path}")

    with open(meta_path, "w") as f:
        header = {"created": time.strftime("%Y-%m-%d %H:%M:%S"), "pipeline_version": pipeline_version, "n": n, "dim": dim, "metric": metric}
        f.write(json.dumps({"_meta": header}) + "\n")
        for i, meta in enumerate(metadata):
            entry = dict(meta)
            entry["_id"] = i
            f.write(json.dumps(entry) + "\n")
    print(f"✅ Saved metadata jsonl -> {meta_path}")
