import os
import shutil
from uuid import uuid4
from dotenv import load_dotenv
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

load_dotenv()

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
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
    if not os.path.exists(os.path.join("index_store", "image_index.faiss")):
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

        # Import lazily so the API can start before the image index is built.
        from image_search import search_image

        candidates = search_image(temp_path, top_k=top_k)
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

    from fusion import IMAGE_CANDIDATES, IMAGE_NEIGHBOURS, TEXT_CANDIDATES, fuse

    text_results = []
    if has_symptoms:
        from embedding_retrieval import search_diseases

        # Every knowledge-base document, not just the top few: one document per
        # class means this yields a text score for every class the image index
        # can return, which is what keeps fusion a re-ranking over the full
        # class space. See fusion.py for the measurements behind this.
        text_results = search_diseases(symptoms.strip(), top_k=TEXT_CANDIDATES)

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

            # A tighter neighbour pool than the image-only endpoint uses: with
            # 5 neighbours the per-class mean cosine stays sharp, which is what
            # fusion needs. Cross-validated in evaluation/fusion_cv.py.
            image_results = search_image(
                temp_path, top_k=IMAGE_CANDIDATES, initial_k=IMAGE_NEIGHBOURS
            )
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

    from embedding_retrieval import get_disease_info

    ranked = fuse(text_results, image_results)[:top_k]

    candidates = []
    for candidate in ranked:
        disease_info = get_disease_info(candidate["name"])
        candidates.append({
            "name": candidate["name"],
            "final_score": round(candidate["final_score"], 3),
            "text_score": round(candidate["text_score"], 3),
            "image_score": round(candidate["image_score"], 3),
            "source": candidate["source"],
            "description": (
                disease_info["full_text"]
                if disease_info
                else "Result based on visual similarity with indexed skin disease images."
            ),
            "risk_level": disease_info["risk_level"] if disease_info else "Unknown",
            "next_steps": (
                disease_info["next_steps"]
                if disease_info
                else "Consult a dermatologist or healthcare professional for proper evaluation."
            ),
            "similar_image_path": candidate["similar_image_path"],
        })

    from generation import generate_explanation

    final_explanation = generate_explanation(
        candidates, symptoms=symptoms, has_image=file is not None
    )

    return {
        "mode": "multimodal",
        "candidates": candidates,
        "final_explanation": final_explanation,
        "disclaimer": "This system is for informational purposes only and does not replace professional medical advice.",
    }
