"""
EduPredict FastAPI Backend
API endpoints for predictions and RAG queries.
"""

import sys
import json
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add src and models to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

# Import RAG (lazy to avoid segfault on import)
RAG_AVAILABLE = False
try:
    from rag.query import answer_question, get_suggested_questions
    RAG_AVAILABLE = True
except Exception as e:
    print(f"Warning: RAG not available (Chroma issue): {e}")
    def answer_question(*args, **kwargs):
        return {"error": "RAG unavailable on this system", "answer": None}
    def get_suggested_questions():
        return ["RAG unavailable"]

from fetcher import load_metadata, get_source_status

# Import predictor (with fallback)
try:
    from predictor import create_model, predict_from_dict, UniversityProfile
    PREDICTOR_AVAILABLE = True
except ImportError as e:
    print(f"Warning: Could not import predictor: {e}")
    PREDICTOR_AVAILABLE = False

app = FastAPI(
    title="EduPredict API",
    version="1.0",
    description="Predictive tool for universities to decide: Should we add an AI program?"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Configure for production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Paths
DATA_DIR = Path(__file__).parent.parent / "data"
METADATA_FILE = DATA_DIR / "metadata.json"


# Request/Response Models

class PredictRequest(BaseModel):
    university_type: str  # public, private, for_profit
    region: str
    current_cs_enrollment: int
    faculty_count: int
    budget_millions: float
    market_demand_score: float  # 0-100
    competition_level: str  # low, medium, high


class PredictResponse(BaseModel):
    recommendation: str  # YES, MAYBE, NO
    confidence: float
    predicted_enrollment: int
    break_even_years: float
    roi_score: float
    key_factors: List[str]
    risk_factors: List[str]
    market_outlook: str


class RAGQueryRequest(BaseModel):
    question: str
    top_k: int = 5
    use_llm: bool = False  # Default to fast extraction mode


class RAGQueryResponse(BaseModel):
    question: str
    answer: str
    sources: List[str]
    confidence: float
    method: str
    documents_used: int
    retrieved_documents: List[Dict]


class DataStatusResponse(BaseModel):
    last_check: Optional[str]
    sources: Dict[str, Any]
    total_records: int


class DataSourceStatus(BaseModel):
    name: str
    last_fetch: Optional[str]
    days_ago: Optional[int]
    fresh: bool
    file: Optional[str]


class HealthResponse(BaseModel):
    status: str
    version: str
    timestamp: str
    predictor_available: bool
    rag_available: bool


class PredictionHistoryResponse(BaseModel):
    predictions: List[Dict]
    total_count: int


class PredictionStatsResponse(BaseModel):
    total_predictions: int
    recommendations: Dict[str, int]
    avg_confidence: float
    avg_predicted_enrollment: float


# Endpoints

@app.get("/", response_model=HealthResponse)
def root():
    """Health check and API info."""
    return HealthResponse(
        status="healthy",
        version="1.0",
        timestamp=datetime.now().isoformat(),
        predictor_available=PREDICTOR_AVAILABLE,
        rag_available=RAG_AVAILABLE
    )


@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):
    """
    Predict whether a university should add an AI program.
    """
    if not PREDICTOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Prediction model not available"
        )
    
    try:
        # Convert to predictor input format
        data = {
            "university_type": request.university_type,
            "region": request.region,
            "current_cs_enrollment": request.current_cs_enrollment,
            "faculty_count": request.faculty_count,
            "budget_millions": request.budget_millions,
            "market_demand_score": request.market_demand_score,
            "competition_level": request.competition_level
        }
        
        # Get prediction
        result = predict_from_dict(data)
        
        return PredictResponse(
            recommendation=result.recommendation,
            confidence=result.confidence,
            predicted_enrollment=result.predicted_enrollment,
            break_even_years=result.break_even_years,
            roi_score=result.roi_score,
            key_factors=result.key_factors,
            risk_factors=result.risk_factors,
            market_outlook=result.market_outlook
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction error: {str(e)}"
        )


@app.get("/predict/history", response_model=PredictionHistoryResponse)
def prediction_history(limit: int = 10):
    """
    Get recent prediction history.
    """
    if not PREDICTOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Prediction model not available"
        )
    
    try:
        model = create_model()
        history = model.get_prediction_history(limit=limit)
        
        # Parse JSON fields
        for h in history:
            try:
                h["key_factors"] = json.loads(h.get("key_factors", "[]"))
                h["risk_factors"] = json.loads(h.get("risk_factors", "[]"))
            except:
                h["key_factors"] = []
                h["risk_factors"] = []
        
        return PredictionHistoryResponse(
            predictions=history,
            total_count=len(history)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching history: {str(e)}"
        )


@app.get("/predict/stats", response_model=PredictionStatsResponse)
def prediction_stats():
    """
    Get prediction statistics.
    """
    if not PREDICTOR_AVAILABLE:
        raise HTTPException(
            status_code=503,
            detail="Prediction model not available"
        )
    
    try:
        model = create_model()
        stats = model.get_statistics()
        
        return PredictionStatsResponse(
            total_predictions=stats.get("total_predictions", 0),
            recommendations=stats.get("recommendations", {}),
            avg_confidence=stats.get("avg_confidence", 0.0),
            avg_predicted_enrollment=stats.get("avg_predicted_enrollment", 0.0)
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching stats: {str(e)}"
        )


@app.post("/rag/query", response_model=RAGQueryResponse)
def rag_query(request: RAGQueryRequest):
    """
    Query the RAG system for relevant documents and generate an answer.
    """
    try:
        result = answer_question(
            question=request.question,
            top_k=request.top_k,
            use_llm=request.use_llm
        )
        
        if "error" in result and not result.get("answer"):
            raise HTTPException(
                status_code=503,
                detail=result["error"]
            )
        
        return RAGQueryResponse(
            question=request.question,
            answer=result.get("answer", "No answer generated"),
            sources=result.get("sources", []),
            confidence=result.get("confidence", 0.0),
            method=result.get("method", "unknown"),
            documents_used=result.get("documents_used", 0),
            retrieved_documents=result.get("retrieved_documents", [])
        )
        
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"RAG query error: {str(e)}"
        )


@app.get("/rag/suggestions")
def rag_suggestions():
    """Get suggested questions for RAG queries."""
    return {
        "suggested_questions": get_suggested_questions()
    }


@app.get("/data/status", response_model=DataStatusResponse)
def data_status():
    """
    Get data freshness status.
    """
    try:
        metadata = load_metadata()
        sources = metadata.get("sources", {})
        
        # Calculate total records
        total = 0
        for src, info in sources.items():
            if info.get("last_fetch"):
                total += 1
        
        # Get detailed status
        detailed_status = get_source_status()
        
        return DataStatusResponse(
            last_check=metadata.get("last_check"),
            sources=detailed_status,
            total_records=total
        )
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching data status: {str(e)}"
        )


@app.post("/data/update")
def data_update(background_tasks: BackgroundTasks):
    """
    Manually trigger data update.
    """
    def run_update():
        import subprocess
        try:
            result = subprocess.run(
                [sys.executable, "-m", "scheduler"],
                cwd=str(Path(__file__).parent),
                capture_output=True,
                text=True,
                timeout=300
            )
            print(result.stdout)
            if result.stderr:
                print(result.stderr)
        except Exception as e:
            print(f"Update error: {e}")
    
    # Run in background
    background_tasks.add_task(run_update)
    
    return {
        "status": "started",
        "message": "Data update triggered in background. Check /data/status for progress.",
        "triggered_at": datetime.now().isoformat()
    }


@app.get("/data/sources")
def data_sources():
    """
    List all configured data sources.
    """
    return {
        "sources": [
            {
                "name": "arxiv",
                "description": "AI education research papers from arXiv",
                "frequency": "weekly",
                "url": "https://export.arxiv.org/api/query",
                "status": "active"
            },
            {
                "name": "ipeds",
                "description": "IPEDS education statistics",
                "frequency": "annual",
                "url": "https://nces.ed.gov/ipeds",
                "status": "manual_download_required"
            },
            {
                "name": "bls",
                "description": "BLS Occupational Employment data",
                "frequency": "annual",
                "url": "https://www.bls.gov/oes/",
                "status": "reference_data"
            },
            {
                "name": "census",
                "description": "Census education demographics",
                "frequency": "annual",
                "url": "https://www.census.gov/data.html",
                "status": "reference_data"
            },
            {
                "name": "google_trends",
                "description": "Search trends for AI education",
                "frequency": "weekly",
                "url": "https://trends.google.com",
                "status": "optional"
            }
        ]
    }


@app.get("/docs/info")
def api_info():
    """Extended API documentation."""
    return {
        "name": "EduPredict API",
        "version": "1.0",
        "description": "Predictive tool for universities considering AI programs",
        "endpoints": {
            "prediction": {
                "predict": "POST /predict - Get recommendation for AI program",
                "history": "GET /predict/history - View past predictions",
                "stats": "GET /predict/stats - Prediction statistics"
            },
            "rag": {
                "query": "POST /rag/query - Ask questions about AI education data",
                "suggestions": "GET /rag/suggestions - Get suggested questions"
            },
            "data": {
                "status": "GET /data/status - Check data freshness",
                "update": "POST /data/update - Trigger data refresh",
                "sources": "GET /data/sources - List data sources"
            }
        },
        "recommendation_levels": {
            "YES": "Strong indicators suggest adding AI program",
            "MAYBE": "Mixed indicators - further analysis recommended",
            "NO": "Indicators suggest not adding program at this time"
        }
    }


# Development server
if __name__ == "__main__":
    import uvicorn
    
    print("=" * 60)
    print("EduPredict API Server")
    print("=" * 60)
    print(f"Predictor available: {PREDICTOR_AVAILABLE}")
    print("\nAPI Documentation:")
    print(f"  Swagger UI: http://localhost:8000/docs")
    print(f"  ReDoc: http://localhost:8000/redoc")
    print(f"  Health: http://localhost:8000/")
    print("\nKey endpoints:")
    print(f"  POST http://localhost:8000/predict")
    print(f"  POST http://localhost:8000/rag/query")
    print(f"  GET  http://localhost:8000/data/status")
    print("=" * 60)
    
    uvicorn.run(app, host="0.0.0.0", port=8000)
