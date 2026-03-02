"""
Test API with predictor import
"""

from fastapi import FastAPI
from pydantic import BaseModel

# Test predictor import
print("Testing predictor import...")
try:
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).parent.parent / "models"))
    from predictor import create_model
    print("Predictor OK")
    PREDICTOR_AVAILABLE = True
except Exception as e:
    print(f"Predictor failed: {e}")
    PREDICTOR_AVAILABLE = False

app = FastAPI()

@app.get("/")
def root():
    return {
        "status": "ok",
        "predictor_available": PREDICTOR_AVAILABLE
    }

if __name__ == "__main__":
    import uvicorn
    print("Starting test API on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)