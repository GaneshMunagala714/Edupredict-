# EduPredict

Predictive tool for universities to decide: "Should we add an AI program?"

**Live Demo:** https://GaneshMunagala714.github.io/Edupredict-

## Architecture

- **Frontend:** HTML5 dashboard with visualizations + AI Advisor RAG interface
- **Backend:** FastAPI with RAG (LangChain + Chroma + HuggingFace)
- **Prediction Model:** Multi-factor decision engine with SQLite storage
- **Database:** SQLite for predictions, Chroma for vector search
- **Auto-Data:** Weekly fetch from IPEDS, BLS, Census, arXiv

## Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Run the Data Pipeline

```bash
# Fetch data from all sources
python src/fetcher.py

# Clean and standardize data
python src/cleaner.py

# Ingest into vector database
python src/rag/ingest.py
```

### 3. Start the API Server

```bash
python src/api.py
```

Server will start at `http://localhost:8000`

- API docs: http://localhost:8000/docs
- Health check: http://localhost:8000/

### 4. Open the Frontend

Open `index-2.html` in a browser (or serve via any static file server).

## API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | Health check and API info |
| `/predict` | POST | Get AI program recommendation |
| `/predict/history` | GET | View prediction history |
| `/predict/stats` | GET | Prediction statistics |
| `/rag/query` | POST | Ask RAG questions about AI education |
| `/rag/suggestions` | GET | Get suggested RAG questions |
| `/data/status` | GET | Check data freshness |
| `/data/update` | POST | Trigger data refresh |
| `/data/sources` | GET | List configured data sources |

## Project Structure

```
edupredict/
├── data/                    # Data storage
│   ├── raw/                 # Downloaded datasets
│   ├── processed/           # Cleaned data
│   └── metadata.json        # Fetch tracking
├── models/                  # ML models and vector DB
│   ├── predictor.py         # Prediction model
│   ├── chroma/              # Chroma vector database
│   └── predictions.db       # SQLite prediction history
├── src/
│   ├── fetcher.py           # Data fetching from APIs
│   ├── cleaner.py           # Data cleaning and standardization
│   ├── scheduler.py         # Job scheduling
│   ├── api.py               # FastAPI backend
│   └── rag/                 # RAG system
│       ├── ingest.py        # Document ingestion to Chroma
│       └── query.py         # RAG query handler
├── requirements.txt         # Python dependencies
└── index-2.html            # Frontend dashboard
```

## Data Sources

| Source | Frequency | Status | Notes |
|--------|-----------|--------|-------|
| **arXiv** | Weekly | ✅ Auto | AI education research papers |
| **BLS** | Annual | 📋 Reference | Occupational employment data |
| **Census** | Annual | 📋 Reference | Education demographics |
| **IPEDS** | Annual | ⚠️ Manual | Requires manual download |
| **Google Trends** | Weekly | 🔧 Optional | Search trends (needs pytrends) |

### Manual Data Steps

**IPEDS Data:**
1. Visit https://nces.ed.gov/ipeds/use-the-data/download-access-database
2. Download the latest Access database
3. Place in `data/raw/` directory
4. Run `python src/cleaner.py`

## RAG Questions It Answers

1. What are current AI program enrollment trends?
2. Which universities recently added AI programs?
3. What are job market projections for AI roles?
4. What factors predict program success?
5. What is the average salary for AI engineers?
6. What education level is needed for AI jobs?

## Prediction Model

The prediction model evaluates universities across 6 dimensions:

| Factor | Weight | Description |
|--------|--------|-------------|
| Market demand | 25% | Regional AI job market demand (0-100) |
| Budget capacity | 20% | Available budget in millions |
| Faculty strength | 15% | Number of qualified faculty |
| Competition | 15% | Low/medium/high local competition |
| CS enrollment | 15% | Current CS program size |
| Institution type | 10% | Public/private/for-profit bonus |

### Recommendation Levels

- **YES** (Score 65+): Strong indicators suggest adding AI program
- **MAYBE** (Score 45-64): Mixed indicators - further analysis recommended
- **NO** (Score <45): Indicators suggest not adding program at this time

## Running the Pipeline

### Manual Pipeline

```bash
# Check all data sources
python src/fetcher.py

# Clean fetched data
python src/cleaner.py

# Ingest to vector DB
python src/rag/ingest.py

# Test RAG queries
python src/rag/query.py

# Start API server
python src/api.py
```

### Scheduled Updates

```bash
# Setup cron job (prints instructions)
python src/scheduler.py --setup

# Run manually
python src/scheduler.py
```

## Testing

### Test API Endpoints

```bash
# Health check
curl http://localhost:8000/

# Get prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "university_type": "public",
    "region": "Northeast",
    "current_cs_enrollment": 500,
    "faculty_count": 10,
    "budget_millions": 10,
    "market_demand_score": 75,
    "competition_level": "low"
  }'

# RAG query
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are current AI program enrollment trends?"}'

# Data status
curl http://localhost:8000/data/status
```

### Test Frontend

1. Open `index-2.html` in browser
2. Navigate to "🤖 AI Advisor" tab
3. Check data status loads from API
4. Try a RAG question
5. Fill prediction form and click "Get Recommendation"

## Development

### Adding New Data Sources

1. Add fetcher function in `src/fetcher.py`
2. Add cleaner function in `src/cleaner.py`
3. Update metadata in `data/metadata.json`
4. Add document loader in `src/rag/ingest.py` (optional)

### Extending the RAG System

To use LLM generation (slower but better answers):

```python
from rag.query import answer_question

result = answer_question(
    question="Your question here",
    top_k=5,
    use_llm=True  # Requires HuggingFace transformers
)
```

## Requirements

- Python 3.9+
- See `requirements.txt` for package list

## Troubleshooting

**Chroma database not found:**
- Run `python src/rag/ingest.py` first

**API connection errors in frontend:**
- Make sure API server is running at localhost:8000
- Check browser console for CORS errors

**Missing data sources:**
- Run `python src/fetcher.py` to populate raw data
- Some sources (IPEDS) require manual download

**Slow RAG responses:**
- Set `use_llm=false` for faster extraction-based answers
- LLM mode requires downloading HuggingFace models (first time only)

## Deployment

### Backend (API Server)

```bash
# Production with uvicorn
uvicorn src.api:app --host 0.0.0.0 --port 8000 --workers 4
```

### Frontend (GitHub Pages)

```bash
# Copy index-2.html to your GitHub Pages repo
cp index-2.html /path/to/github/pages/repo/
git add index-2.html
git commit -m "Update dashboard"
git push origin main
```

**Note:** For GitHub Pages deployment, you need a hosted API server or use CORS proxy.

## License

MIT

## Deadline

March 9, 2026 - Professor deliverable

## Implementation Status

### ✅ COMPLETE (Working)

| Component | Status | Details |
|-----------|--------|---------|
| **Data Fetcher** | ✅ Working | arXiv (151 papers), BLS (reference), Census (reference), IPEDS (reference) |
| **Data Cleaner** | ✅ Working | Parses arXiv XML, standardizes all sources to common schema |
| **RAG Ingest** | ✅ Working | 159 documents ingested to Chroma (arXiv:151, BLS:6, Census:1, IPEDS:1) |
| **RAG Query** | ✅ Working | Similarity search + answer generation with confidence scores |
| **Prediction Model** | ✅ Working | Multi-factor decision engine with SQLite storage |
| **API Endpoints** | ✅ Working | FastAPI with /predict, /rag/query, /data/status endpoints |
| **Frontend** | ✅ Working | Dashboard with AI Advisor tab calling backend API |

### ⚠️ MANUAL STEPS REQUIRED

| Component | Status | Action Required |
|-----------|--------|-----------------|
| **IPEDS Full Data** | ⚠️ Manual | Download Access database from nces.ed.gov/ipeds |
| **Google Trends** | ⚠️ Optional | `pip install pytrends` to enable |
| **BLS Full Dataset** | ⚠️ Manual | Download CSV from data.bls.gov/oes |

### 🔧 KNOWN ISSUES

1. **Chroma/Sentence-Transformers**: May encounter environment-specific segfaults on some systems when loading embedding models. The database is created successfully before the issue occurs.
   - **Workaround**: Use pre-ingested vector DB or run on compatible environment
   - **Fix**: Update to `langchain-huggingface` package (future update)

2. **LLM Generation**: Optional LLM mode requires HuggingFace models download (~500MB-2GB)
   - **Workaround**: Default extraction-based answers work without LLM

## Quick Start Verification

### 1. Verify Data Pipeline

```bash
# Using Anaconda Python (recommended)
cd ~/edupredict

# Fetch data
/opt/anaconda3/bin/python3 src/fetcher.py

# Clean data  
/opt/anaconda3/bin/python3 src/cleaner.py

# Ingest to Chroma
KMP_DUPLICATE_LIB_OK=TRUE TOKENIZERS_PARALLELISM=false \
  /opt/anaconda3/bin/python3 src/rag/ingest.py
```

### 2. Test Prediction Model

```bash
/opt/anaconda3/bin/python3 -c "
import sys; sys.path.insert(0, 'models')
from predictor import create_model, UniversityProfile

model = create_model()
profile = UniversityProfile('public', 'Northeast', 800, 15, 15.0, 85, 'low')
result = model.predict(profile)

print(f'Recommendation: {result.recommendation}')
print(f'Confidence: {result.confidence:.1%}')
print(f'Predicted Enrollment: {result.predicted_enrollment}')
print(f'Break-even: {result.break_even_years:.1f} years')
"
```

### 3. Start API Server

```bash
KMP_DUPLICATE_LIB_OK=TRUE TOKENIZERS_PARALLELISM=false \
  /opt/anaconda3/bin/python3 src/api.py
```

Server starts at `http://localhost:8000`

### 4. Test API Endpoints

```bash
# Health check
curl http://localhost:8000/

# Get prediction
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "university_type": "public",
    "region": "Northeast",
    "current_cs_enrollment": 500,
    "faculty_count": 10,
    "budget_millions": 10,
    "market_demand_score": 75,
    "competition_level": "low"
  }'

# RAG query
curl -X POST http://localhost:8000/rag/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What are current AI program enrollment trends?"}'

# Data status
curl http://localhost:8000/data/status
```

## Data Pipeline Summary

### Automated (Tier 1)

| Source | Method | Frequency | Records |
|--------|--------|-----------|---------|
| arXiv | API | Weekly | 151 papers |
| BLS | Reference data | Annual | 5 occupations |
| Census | Reference data | Annual | 5 education levels |
| IPEDS | Web check | Annual | Metadata reference |

### Manual (Tier 2)

| Source | Action | When Needed |
|--------|--------|-------------|
| IPEDS | Download Access DB | Deep enrollment analysis |
| BLS | Download full CSV | Detailed wage statistics |
| Census | API key for large queries | State-level demographics |

## Production Readiness Checklist

- [x] Data pipeline: fetcher → cleaner → database
- [x] RAG system: ingest → query → answers
- [x] Prediction API: all endpoints implemented
- [x] Frontend: API integration complete
- [x] SQLite storage: predictions.db working
- [x] Chroma vector DB: 159 documents indexed
- [x] Documentation: README updated
- [ ] Environment testing: Verify on target deployment machine
- [ ] Load testing: API performance under concurrent requests
- [ ] Error handling: Graceful degradation for missing data
