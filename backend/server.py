from fastapi import FastAPI
from pydantic import BaseModel
from embedding_retrieval import search_diseases

app = FastAPI()


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
        "candidates": candidates,
        "disclaimer": "This system is for informational purposes only and does not replace professional medical advice."
    }