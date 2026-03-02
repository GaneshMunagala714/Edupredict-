"""
Minimal API test
"""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI()

class TestRequest(BaseModel):
    name: str

@app.get("/")
def root():
    return {"status": "ok", "message": "Minimal API works"}

@app.post("/test")
def test(req: TestRequest):
    return {"received": req.name}

if __name__ == "__main__":
    import uvicorn
    print("Starting minimal API on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)