import json

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
INDEX_PATH = "index_store/faiss.index"
METADATA_PATH = "index_store/metadata.json"

model = SentenceTransformer(MODEL_NAME)
index = faiss.read_index(INDEX_PATH)

with open(METADATA_PATH, "r", encoding="utf-8") as file:
    documents = json.load(file)

_documents_by_key = {
    "".join(document["disease"].lower().split()): document for document in documents
}


def get_disease_info(name):
    """Return the full knowledge-base document for a disease name, if one exists."""
    return _documents_by_key.get("".join(name.lower().split()))


def search_diseases(query, top_k=3):
    """Return the most relevant disease knowledge-base documents for a query."""
    query_embedding = model.encode([query], normalize_embeddings=True)
    scores, indices = index.search(np.array(query_embedding, dtype="float32"), top_k)

    results = []
    for score, index_position in zip(scores[0], indices[0]):
        if index_position < 0:
            continue
        document = documents[index_position]
        results.append(
            {
                "disease": document["disease"],
                "description": document["full_text"],
                "risk_level": document.get("risk_level", "Unknown"),
                "next_steps": document.get(
                    "next_steps",
                    "Consult a dermatologist or healthcare professional for proper evaluation.",
                ),
                "confidence": round(float(score), 3),
            }
        )
    return results
