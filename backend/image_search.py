import numpy as np
import faiss
import torch
import clip
from PIL import Image
from collections import defaultdict

# paths
INDEX_PATH = "index_store/image_index.faiss"
LABELS_PATH = "index_store/image_labels.npy"

# load model
device = "cuda" if torch.cuda.is_available() else "cpu"
model, preprocess = clip.load("ViT-B/32", device=device)

# load index + labels
index = faiss.read_index(INDEX_PATH)
label_data = np.load(LABELS_PATH, allow_pickle=True)


# 🔹 clean labels (fix ugly names)
def clean_label(label):
    return label.replace(" Photos", "").replace("_", " ")


# 🔹 main search function
def search_image(image_path, top_k=3, initial_k=10):
    image = preprocess(Image.open(image_path).convert("RGB")).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image)

    embedding = embedding / embedding.norm(dim=-1, keepdim=True)
    embedding = embedding.cpu().numpy().astype("float32")

    scores, indices = index.search(embedding, initial_k)

    class_scores = defaultdict(list)
    class_examples = {}

    # 🔥 group results by class
    for score, idx in zip(scores[0], indices[0]):
        label, path = label_data[idx]

        label = clean_label(str(label))  # FIX label

        class_scores[label].append(float(score))

        if label not in class_examples:
            class_examples[label] = str(path)

    # 🔹 aggregate results
    aggregated = []
    for label, score_list in class_scores.items():
        avg_score = sum(score_list) / len(score_list)

        aggregated.append({
            "label": label,
            "confidence": round(avg_score, 3),
            "matches": len(score_list),
            "example_path": class_examples[label]
        })

    # 🔥 sort by best matches + confidence
    aggregated.sort(key=lambda x: (x["matches"], x["confidence"]), reverse=True)

    return aggregated[:top_k]