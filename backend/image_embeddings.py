from pathlib import Path
from PIL import Image
import torch
import clip
import numpy as np
import faiss

DATASET_PATH = Path("../data/raw/images/train_balanced")
INDEX_PATH = "index_store/image_index.faiss"
LABELS_PATH = "index_store/image_labels.npy"
BATCH_SIZE = 32

device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

image_embeddings = []
labels = []
paths = []


def process_images():
    image_files = [
        (img_path, folder.name)
        for folder in DATASET_PATH.iterdir()
        if folder.is_dir()
        for img_path in folder.rglob("*")
        if img_path.is_file() and img_path.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"}
    ]

    for batch_start in range(0, len(image_files), BATCH_SIZE):
        batch_tensors = []
        batch_metadata = []

        for img_path, label in image_files[batch_start:batch_start + BATCH_SIZE]:
            try:
                with Image.open(img_path) as image:
                    batch_tensors.append(preprocess(image.convert("RGB")))
                batch_metadata.append((img_path, label))
            except Exception as e:
                print(f"Skipped {img_path}: {e}")

        if not batch_tensors:
            continue

        batch = torch.stack(batch_tensors).to(device)
        with torch.no_grad():
            embeddings = model.encode_image(batch)

        embeddings = embeddings / embeddings.norm(dim=-1, keepdim=True)
        embeddings = embeddings.cpu().numpy().astype("float32")

        image_embeddings.extend(embeddings)
        labels.extend(label for _, label in batch_metadata)
        paths.extend(str(img_path) for img_path, _ in batch_metadata)
        print(f"processed {len(image_embeddings)} images")


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
