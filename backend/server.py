import os
import shutil
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from embedding_retrieval import search_diseases

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "temp_uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)


class DiagnosisRequest(BaseModel):
    symptoms: str
    top_k: int = 3


@app.get("/")
def home():
    return {"message": "Skin disease diagnosis API is running."}


@app.post("/diagnose")
def diagnose(request: DiagnosisRequest):
    candidates = search_diseases(request.symptoms, request.top_k)

    return {
        "mode": "text",
        "candidates": candidates,
        "disclaimer": "This system is for informational purposes only and does not replace professional medical advice."
    }


@app.post("/diagnose-image")
def diagnose_image(file: UploadFile = File(...), top_k: int = Form(3)):
    temp_path = os.path.join(UPLOAD_DIR, file.filename)

    with open(temp_path, "wb") as buffer:
        shutil.copyfileobj(file.file, buffer)

    if not os.path.exists(os.path.join("index_store", "image_index.faiss")):
        return {
            "error": (
                "Image search index is missing. Add images to "
                "data/raw/images/train, then run `python image_embeddings.py`."
            )
        }

    try:
        # Import lazily so the API can start before the image index is built.
        from image_search import search_image

        candidates = search_image(temp_path, top_k=top_k)
    except FileNotFoundError:
        return {
            "error": (
                "Image search index is missing. Add images to "
                "data/raw/images/train, then run `python image_embeddings.py`."
            )
        }
    except Exception as exc:
        return {"error": f"Image search is unavailable: {exc}"}

    return {
        "mode": "image",
        "candidates": candidates,
        "explanation": "Results are based on visual similarity to known cases.",
        "disclaimer": "This system is for informational purposes only and does not replace professional medical advice."
    }
