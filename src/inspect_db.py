# src/inspect_db.py
import faiss, json, collections
def inspect(index_path="data/eeg_index.faiss", meta_path="data/eeg_vector_db.jsonl", n=5):
    idx = faiss.read_index(index_path)
    with open(meta_path) as f:
        lines = f.readlines()
    header = json.loads(lines[0])["_meta"]
    metas = [json.loads(l) for l in lines[1:]]
    print("HEADER:", header)
    print("TOTAL vectors:", len(metas))
    labels = [m.get("label") for m in metas]
    print("Label counts:", dict(collections.Counter(labels)))
    print("SAMPLE meta:", metas[:n])
if __name__=="__main__":
    inspect()
