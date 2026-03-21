import json
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

DATA_PATH = "data/diseases.json"
INDEX_PATH = "index_store/faiss.index"
METADATA_PATH = "index_store/metadata.json"


def load_data():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_texts(data):
    texts = []
    for item in data:
        text = f"""
        Disease: {item['name']}
        Symptoms: {item['symptoms']}
        Description: {item['description']}
        Risk level: {item['risk_level']}
        Next steps: {item['next_steps']}
        """
        texts.append(text.strip())
    return texts


def main():
    print("Loading data...")
    data = load_data()

    print("Building texts...")
    texts = build_texts(data)

    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Creating embeddings...")
    embeddings = model.encode(texts, normalize_embeddings=True)
    embeddings = np.array(embeddings, dtype="float32")

    print("Building FAISS index...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatIP(dimension)
    index.add(embeddings)

    print("Saving index...")
    faiss.write_index(index, INDEX_PATH)

    print("Saving metadata...")
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print("DONE")
    

if __name__ == "__main__":
    main()
