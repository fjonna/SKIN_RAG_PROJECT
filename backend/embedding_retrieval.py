import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

INDEX_PATH = "index_store/faiss.index"
METADATA_PATH = "index_store/metadata.json"

model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    data = json.load(f)


def search_diseases(query, top_k=3):
    query_embedding = model.encode([query], normalize_embeddings=True)
    query_embedding = np.array(query_embedding, dtype="float32")

    scores, indices = index.search(query_embedding, top_k)

    results = []
    for score, idx in zip(scores[0], indices[0]):
        item = data[idx]
        results.append({
            "name": item["name"],
            "description": item["description"],
            "risk_level": item["risk_level"],
            "next_steps": item["next_steps"],
            "confidence": round(float(score), 3)
        })

    return results