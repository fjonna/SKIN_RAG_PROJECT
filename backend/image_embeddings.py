from pathlib import Path
from PIL import Image
import torch
import clip
import numpy as np
import faiss

DATASET_PATH = Path("../data/raw/images/train")
INDEX_PATH = "index_store/image_index.faiss"
LABELS_PATH = "index_store/image_labels.npy"

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_embeddings = []
labels = []
paths = []


def process_images():
    for folder in DATASET_PATH.iterdir():
        if folder.is_dir():
            for img_path in folder.glob("*"):
                try:
                    image = preprocess(Image.open(img_path).convert("RGB")).unsqueeze(0).to(device)

                    with torch.no_grad():
                        embedding = model.encode_image(image)

                    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
                    embedding = embedding.cpu().numpy().astype("float32")

                    image_embeddings.append(embedding[0])
                    labels.append(folder.name)
                    paths.append(str(img_path))

                except Exception as e:
                    print(f"Skipped {img_path}: {e}")


def build_index():
    embeddings_np = np.array(image_embeddings, dtype="float32")

    index = faiss.IndexFlatIP(embeddings_np.shape[1])
    index.add(embeddings_np)

    faiss.write_index(index, INDEX_PATH)
    np.save(LABELS_PATH, np.array(list(zip(labels, paths)), dtype=object))


if __name__ == "__main__":
    print("Processing images...")
    process_images()

    print(f"Total processed: {len(image_embeddings)}")

    print("Building FAISS index...")
    build_index()

    print("DONE")