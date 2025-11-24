# src/query_faiss_db.py
import faiss
import json
import numpy as np
import os

def query_faiss_db(query_embedding, top_k=5, index_path="data/eeg_index.faiss", meta_path="data/eeg_vector_db.jsonl", metric="l2"):
    print("🔍 Searching FAISS DB...")
    if not os.path.exists(index_path):
        raise FileNotFoundError(f"FAISS index not found at '{index_path}'")
    if not os.path.exists(meta_path):
        raise FileNotFoundError(f"Metadata file not found at '{meta_path}'")

    index = faiss.read_index(index_path)
    with open(meta_path, "r") as f:
        lines = f.readlines()
    header = json.loads(lines[0])["_meta"]
    metas = [json.loads(l) for l in lines[1:]]

    q = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    if metric == "cosine":
        q = q / (np.linalg.norm(q, axis=1, keepdims=True) + 1e-12)
    distances, indices = index.search(q, top_k)

    results = []
    for rank, idx in enumerate(indices[0], start=1):
        if idx < 0 or idx >= len(metas):
            continue
        results.append({
            "rank": rank,
            "index": int(idx),
            "distance": float(distances[0][rank-1]),
            "meta": metas[idx]
        })
    return results
