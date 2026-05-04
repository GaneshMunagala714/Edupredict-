"""
EduPredict FastAPI Backend
API endpoints for predictions and RAG queries.
"""

import sys
import os

# Fix OpenMP conflict that causes segfault on macOS with torch + chromadb
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException, BackgroundTasks, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, field_validator
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.util import get_remote_address
from slowapi.errors import RateLimitExceeded

load_dotenv()

# Structured logging
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
logging.basicConfig(
    level=getattr(logging, LOG_LEVEL, logging.INFO),
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%Y-%m-%dT%H:%M:%S",
)
logger = logging.getLogger("edupredict")

# Add src and models to path
sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "models"))

# RAG is loaded lazily on first request to avoid segfault on macOS
# (torch + sentence-transformers OpenMP conflict crashes at import time)
RAG_AVAILABLE = False
_rag_loaded = False

def _load_rag():
    global RAG_AVAILABLE, _rag_loaded, answer_question, get_suggested_questions
    if _rag_loaded:
        return RAG_AVAILABLE
    _rag_loaded = True
    try:
        from rag.query import answer_question as _aq, get_suggested_questions as _gsq
        answer_question = _aq
        get_suggested_questions = _gsq
        RAG_AVAILABLE = True
        logger.info("RAG loaded successfully")
    except Exception as e:
        logger.warning(f"RAG not available: {e}")
        RAG_AVAILABLE = False
    return RAG_AVAILABLE

def answer_question(*args, **kwargs):
    if _load_rag() and RAG_AVAILABLE:
        from rag.query import answer_question as _aq
        return _aq(*args, **kwargs)
    return {"error": "RAG unavailable on this system", "answer": None}

def get_suggested_questions():
    if _load_rag() and RAG_AVAILABLE:
        from rag.query import get_suggested_questions as _gsq
        return _gsq()
    return ["What AI programs are growing fastest?", "What salary can AI graduates expect?"]

from fetcher import load_metadata, get_source_status

# Import predictor (with fallback)
try:
    from predictor import create_model, predict_from_dict, UniversityProfile
    PREDICTOR_AVAILABLE = True
except ImportError as e:
    logger.warning(f"Could not import predictor: {e}")
    PREDICTOR_AVAILABLE = False

limiter = Limiter(key_func=get_remote_address)
app = FastAPI(
    title="EduPredict API",
    version="1.0",
    description="Predictive tool for universities to decide: Should we add an AI program?"
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)

# CORS — read from env, fall back to localhost for dev
_raw_origins = os.getenv("ALLOWED_ORIGINS", "http://localhost:3000,http://localhost:8000")
ALLOWED_ORIGINS = [o.strip() for o in _raw_origins.split(",") if o.strip()]

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type", "Authorization"],
)

# Paths
DATA_DIR = Path(os.getenv("EDUPREDICT_DATA_DIR", Path(__file__).parent.parent / "data"))
METADATA_FILE = DATA_DIR / "metadata.json"


# Request/Response Models

VALID_UNIVERSITY_TYPES = {"public", "private", "for_profit"}
VALID_COMPETITION_LEVELS = {"low", "medium", "high"}


class PredictRequest(BaseModel):
    university_type: str  # public, private, for_profit
    region: str
    current_cs_enrollment: int
    faculty_count: int
    budget_millions: float
    market_demand_score: float  # 0-100
    competition_level: str  # low, medium, high

    @field_validator("university_type")
    @classmethod
    def validate_university_type(cls, v):
        if v.lower() not in VALID_UNIVERSITY_TYPES:
            raise ValueError(f"university_type must be one of: {', '.join(VALID_UNIVERSITY_TYPES)}")
        return v.lower()

    @field_validator("competition_level")
    @classmethod
    def validate_competition_level(cls, v):
        if v.lower() not in VALID_COMPETITION_LEVELS:
            raise ValueError(f"competition_level must be one of: {', '.join(VALID_COMPETITION_LEVELS)}")
        return v.lower()

    @field_validator("market_demand_score")
    @classmethod
    def validate_demand_score(cls, v):
        if not (0 <= v <= 100):
            raise ValueError("market_demand_score must be between 0 and 100")
        return v

    @field_validator("current_cs_enrollment", "faculty_count")
    @classmethod
    def validate_positive_int(cls, v):
        if v < 0:
            raise ValueError("Value must be non-negative")
        return v

    @field_validator("budget_millions")
    @classmethod
    def validate_budget(cls, v):
        if v < 0:
            raise ValueError("budget_millions must be non-negative")
        return v


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
@limiter.limit("30/minute")
def predict(request: PredictRequest, req: Request):
    """
    Predict whether a university should add an AI program.
    """
    if not PREDICTOR_AVAILABLE:
        raise HTTPException(status_code=503, detail="Prediction model not available")

    try:
        data = {
            "university_type": request.university_type,
            "region": request.region,
            "current_cs_enrollment": request.current_cs_enrollment,
            "faculty_count": request.faculty_count,
            "budget_millions": request.budget_millions,
            "market_demand_score": request.market_demand_score,
            "competition_level": request.competition_level,
        }
        result = predict_from_dict(data)
        logger.info(f"Prediction: {result.recommendation} (confidence={result.confidence:.2f}, type={request.university_type})")
        return PredictResponse(
            recommendation=result.recommendation,
            confidence=result.confidence,
            predicted_enrollment=result.predicted_enrollment,
            break_even_years=result.break_even_years,
            roi_score=result.roi_score,
            key_factors=result.key_factors,
            risk_factors=result.risk_factors,
            market_outlook=result.market_outlook,
        )
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


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
@limiter.limit("20/minute")
def rag_query(request: RAGQueryRequest, req: Request):
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
        logger.error(f"RAG query error: {e}")
        raise HTTPException(status_code=500, detail=f"RAG query error: {str(e)}")


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

    host = os.getenv("HOST", "0.0.0.0")
    port = int(os.getenv("PORT", "8000"))

    logger.info("=" * 60)
    logger.info("EduPredict API Server (dev mode)")
    logger.info(f"Predictor available: {PREDICTOR_AVAILABLE}")
    logger.info(f"RAG available: {RAG_AVAILABLE}")
    logger.info(f"Allowed origins: {ALLOWED_ORIGINS}")
    logger.info(f"Swagger UI: http://localhost:{port}/docs")
    logger.info("=" * 60)

    uvicorn.run(app, host=host, port=port)
