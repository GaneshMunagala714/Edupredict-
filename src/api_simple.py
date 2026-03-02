"""
Simple API with predictor only (no RAG)
"""

import sys
from pathlib import Path
from typing import Dict, List

sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

from fastapi import FastAPI
from pydantic import BaseModel

print("Importing predictor...")
from predictor import create_model, predict_from_dict
print("Predictor OK")

app = FastAPI()

class PredictRequest(BaseModel):
    university_type: str
    region: str
    current_cs_enrollment: int
    faculty_count: int
    budget_millions: float
    market_demand_score: float
    competition_level: str

@app.get("/")
def root():
    return {"status": "ok", "api": "simple predictor"}

@app.post("/predict")
def predict(req: PredictRequest):
    data = req.dict()
    result = predict_from_dict(data)
    return {
        "recommendation": result.recommendation,
        "confidence": result.confidence,
        "predicted_enrollment": result.predicted_enrollment,
        "break_even_years": result.break_even_years,
        "roi_score": result.roi_score
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting API on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)