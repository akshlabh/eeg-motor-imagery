# src/__init__.py
# package exports for easier imports
from .preprocess import load_eeg_data, clean_data, create_epochs
from .vectorize import extract_embeddings, transform_epochs
from .build_vector_db import build_faiss_db
from .query_faiss_db import query_faiss_db
