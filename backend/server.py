import os
import shutil
from uuid import uuid4
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

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
    from embedding_retrieval import search_diseases

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


@app.post("/diagnose-multimodal")
def diagnose_multimodal(
    symptoms: str | None = Form(None),
    file: UploadFile | None = File(None),
    top_k: int = Form(3),
):
    has_symptoms = bool(symptoms and symptoms.strip())
    if not has_symptoms and file is None:
        return {"error": "Please provide symptoms, an image, or both."}

    text_results = []
    if has_symptoms:
        from embedding_retrieval import search_diseases

        text_results = search_diseases(symptoms.strip(), top_k=5)

    image_results = []
    if file is not None:
        index_path = os.path.join("index_store", "image_index.faiss")
        labels_path = os.path.join("index_store", "image_labels.npy")
        if not (os.path.exists(index_path) and os.path.exists(labels_path)):
            return {
                "error": (
                    "Image search index is missing. Add images to "
                    "data/raw/images/train_balanced, then run "
                    "`python image_embeddings.py` from the backend folder."
                )
            }

        filename = os.path.basename(file.filename or "uploaded_image")
        temp_path = os.path.join(UPLOAD_DIR, f"{uuid4().hex}_{filename}")
        try:
            with open(temp_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            from image_search import search_image

            image_results = search_image(temp_path, top_k=5)
        except FileNotFoundError:
            return {
                "error": (
                    "Image search index is missing. Add images to "
                    "data/raw/images/train_balanced, then run "
                    "`python image_embeddings.py` from the backend folder."
                )
            }
        except Exception as exc:
            return {"error": f"Image search is unavailable: {exc}"}
        finally:
            if os.path.exists(temp_path):
                os.remove(temp_path)

    def normalized_name(name: str) -> str:
        return "".join(name.lower().split())

    combined = {}
    for result in text_results:
        key = normalized_name(result["name"])
        combined[key] = {
            "name": result["name"],
            "final_score": 0.45 * float(result["confidence"]),
            "text_score": float(result["confidence"]),
            "image_score": 0.0,
            "source": "text",
            "description": result["description"],
            "risk_level": result["risk_level"],
            "next_steps": result["next_steps"],
        }

    for result in image_results:
        key = normalized_name(result["label"])
        image_score = float(result["confidence"])
        if key in combined:
            candidate = combined[key]
            candidate["image_score"] = image_score
            candidate["final_score"] += 0.55 * image_score
            candidate["source"] = "both"
        else:
            combined[key] = {
                "name": result["label"],
                "final_score": 0.55 * image_score,
                "text_score": 0.0,
                "image_score": image_score,
                "source": "image",
                "description": "Result based on visual similarity with indexed skin disease images.",
                "risk_level": "unknown",
                "next_steps": "Consult a dermatologist or healthcare professional for proper evaluation.",
            }

    candidates = sorted(
        combined.values(), key=lambda candidate: candidate["final_score"], reverse=True
    )[:top_k]
    for candidate in candidates:
        candidate["final_score"] = round(candidate["final_score"], 3)
        candidate["text_score"] = round(candidate["text_score"], 3)
        candidate["image_score"] = round(candidate["image_score"], 3)

    return {
        "mode": "multimodal",
        "candidates": candidates,
        "disclaimer": "This system is for informational purposes only and does not replace professional medical advice.",
    }
