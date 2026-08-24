import json
import re
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
KNOWLEDGE_BASE_PATH = Path("knowledge_base")
INDEX_PATH = "index_store/faiss.index"
METADATA_PATH = "index_store/metadata.json"
DEFAULT_NEXT_STEPS = "Consult a dermatologist or healthcare professional for proper evaluation."

SECTION_HEADING_PATTERN = re.compile(r"^##\s+(.+?)\s*$", re.MULTILINE)


def parse_sections(full_text):
    """Split a knowledge-base document's body into its '## Heading' sections."""
    headings = list(SECTION_HEADING_PATTERN.finditer(full_text))
    sections = {}
    for position, heading in enumerate(headings):
        start = heading.end()
        end = headings[position + 1].start() if position + 1 < len(headings) else len(full_text)
        sections[heading.group(1).strip().lower()] = full_text[start:end].strip()
    return sections


def load_documents():
    documents = []
    for filepath in sorted(KNOWLEDGE_BASE_PATH.glob("*.md")):
        full_text = filepath.read_text(encoding="utf-8").strip()
        if not full_text:
            continue

        first_line = full_text.splitlines()[0]
        disease = first_line.removeprefix("# ").strip()
        sections = parse_sections(full_text)

        documents.append(
            {
                "disease": disease,
                "filename": filepath.name,
                "full_text": full_text,
                "risk_level": sections.get("risk level", "Unknown"),
                "next_steps": sections.get("recommended next steps", DEFAULT_NEXT_STEPS),
            }
        )
    return documents


def main():
    print("Loading knowledge-base documents...")
    documents = load_documents()
    if not documents:
        raise RuntimeError(f"No Markdown documents found directly in {KNOWLEDGE_BASE_PATH}.")

    print(f"Loaded {len(documents)} documents.")
    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)

    print("Creating embeddings...")
    embeddings = model.encode(
        [document["full_text"] for document in documents], normalize_embeddings=True
    )
    embeddings = np.array(embeddings, dtype="float32")

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)

    print("Saving index and document metadata...")
    faiss.write_index(index, INDEX_PATH)
    with open(METADATA_PATH, "w", encoding="utf-8") as file:
        json.dump(documents, file, indent=2)

    print("DONE")


if __name__ == "__main__":
    main()
