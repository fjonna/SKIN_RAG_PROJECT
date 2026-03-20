from fastapi import FastAPI
from pydantic import BaseModel
from retrieval import load_data, find_most_similar_disease

app = FastAPI()

data = load_data()


class DiagnosisRequest(BaseModel):
    symptoms: str


@app.get("/")
def home():
    return {"message": "Skin disease diagnosis API is running."}


@app.post("/diagnose")
def diagnose(request: DiagnosisRequest):
    result, score = find_most_similar_disease(request.symptoms, data)

    return {
        "most_likely_disease": result["name"],
        "description": result["description"],
        "risk_level": result["risk_level"],
        "next_steps": result["next_steps"],
        "similarity_score": round(score, 3),
        "disclaimer": "This system is for informational purposes only and does not replace professional medical advice."
    }