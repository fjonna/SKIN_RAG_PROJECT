# Skin RAG Project

A multimodal retrieval-augmented (RAG) assistant for skin condition information. It combines **text retrieval** (symptom descriptions against a curated dermatology knowledge base) with **image retrieval** (CLIP-based visual similarity search over labeled skin images) and uses an LLM to turn the retrieved evidence into a natural-language explanation.

> **Disclaimer:** This project is for research and educational purposes only. It does not provide medical diagnoses and is not a substitute for professional medical advice.

## How it works

1. **Text retrieval** — Symptom descriptions are embedded with `sentence-transformers/all-MiniLM-L6-v2` and matched against a FAISS index built from markdown documents in `backend/knowledge_base/` (one file per condition, e.g. `melanoma.md`, `eczema.md`), each with sections for overview, symptoms, causes, risk level, and next steps.
2. **Image retrieval** — Uploaded images are embedded with OpenAI's CLIP (`ViT-B/32`) and matched against a FAISS index of labeled reference images.
3. **Fusion** — When both a symptom description and an image are provided, text and image matches are merged into a single ranked list of candidate conditions.
4. **Generation** — The top candidates are passed to the Anthropic API (Claude) to produce a grounded natural-language summary. If no API key is configured, or the call fails, a deterministic fallback summary is used instead.
5. **Frontend** — A Streamlit app lets a user upload an image and/or describe symptoms, and displays the diagnosis, supporting details, and visually similar reference cases.

## Project structure

```
.
├── backend/
│   ├── server.py                # FastAPI app (diagnose / diagnose-image / diagnose-multimodal)
│   ├── embedding_retrieval.py   # Text embedding search over the knowledge base
│   ├── build_index.py           # Builds the FAISS text index from knowledge_base/*.md
│   ├── image_embeddings.py      # Builds the FAISS image index from the training images
│   ├── image_search.py          # CLIP-based image similarity search
│   ├── generation.py            # LLM-based (Claude) explanation generation, with fallback
│   └── knowledge_base/          # Markdown reference documents, one per skin condition
├── frontend/
│   └── client.py                # Streamlit UI
├── prepare_image_dataset.py     # Normalizes raw Kaggle image folders into per-class folders
├── create_balanced_subset.py    # Caps each class to a max number of images
├── data/                        # Raw and processed image datasets (not tracked in git)
└── requirements.txt
```

## Setup

### 1. Install dependencies

```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

### 2. Configure environment variables

Create a `.env` file in `backend/` (used for the generation step):

```
ANTHROPIC_API_KEY=your_key_here
```

If this is not set, the app still works — it falls back to a deterministic summary instead of an LLM-generated one.

### 3. Build the text knowledge base index

```bash
cd backend
python build_index.py
```

This reads `backend/knowledge_base/*.md` and writes `backend/index_store/faiss.index` and `metadata.json`.

### 4. (Optional) Build the image search index

Image search requires a labeled image dataset under `data/raw/images/train_balanced/<Class Name>/...`.

```bash
python prepare_image_dataset.py     # from repo root: normalizes raw Kaggle data into data/raw/images/train
python create_balanced_subset.py    # caps each class to a balanced subset in train_balanced
cd backend
python image_embeddings.py          # builds index_store/image_index.faiss and image_labels.npy
```

If the image index is missing, `/diagnose-image` and the image portion of `/diagnose-multimodal` will return an explanatory error instead of failing.

## Running the app

Start the backend API:

```bash
cd backend
uvicorn server:app --reload
```

Start the frontend (in a separate terminal):

```bash
cd frontend
streamlit run client.py
```

The Streamlit app expects the API at `http://127.0.0.1:8000`.

## API

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Health check |
| `/diagnose` | POST | Text-only diagnosis from `symptoms` |
| `/diagnose-image` | POST | Image-only diagnosis from an uploaded file |
| `/diagnose-multimodal` | POST | Combined text + image diagnosis with LLM-generated explanation |

## Tech stack

FastAPI · Streamlit · Sentence-Transformers · FAISS · OpenAI CLIP · PyTorch · Anthropic API
