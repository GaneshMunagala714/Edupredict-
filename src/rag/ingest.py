"""
EduPredict RAG Document Ingestion
Ingests documents into Chroma vector database.
"""

import json
import os
from pathlib import Path
from typing import List, Dict, Any
from datetime import datetime

# LangChain imports
try:
    from langchain_community.vectorstores import Chroma
    from langchain_community.embeddings import HuggingFaceEmbeddings
    from langchain.schema import Document
    LANGCHAIN_AVAILABLE = True
except ImportError as e:
    print(f"Error importing LangChain: {e}")
    print("Please install: pip install langchain langchain-community chromadb")
    LANGCHAIN_AVAILABLE = False

# Paths
DATA_DIR = Path(__file__).parent.parent.parent / "data"
PROCESSED_DIR = DATA_DIR / "processed"
CHROMA_DIR = Path(__file__).parent.parent.parent / "models" / "chroma"

# Embedding model (free, local)
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"

# Suppress tokenizers parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def get_embeddings():
    """Initialize HuggingFace embeddings."""
    if not LANGCHAIN_AVAILABLE:
        raise ImportError("LangChain not available")
    
    print(f"Loading embedding model: {EMBEDDING_MODEL}")
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )


def load_arxiv_documents() -> List[Document]:
    """Load arXiv papers as LangChain Documents."""
    documents = []
    
    arxiv_file = PROCESSED_DIR / "arxiv_papers.json"
    if not arxiv_file.exists():
        print("No arXiv papers found")
        return documents
    
    try:
        with open(arxiv_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        papers = data.get("papers", [])
        print(f"Loading {len(papers)} arXiv papers...")
        
        for paper in papers:
            title = paper.get("title", "")
            summary = paper.get("summary", "")
            
            if not title or not summary:
                continue
            
            # Create content by combining title and summary
            content = f"Title: {title}\n\nAbstract: {summary}"
            
            doc = Document(
                page_content=content,
                metadata={
                    "source": "arxiv",
                    "title": title,
                    "authors": ", ".join(paper.get("authors", [])[:3]),  # First 3 authors
                    "published": paper.get("published", ""),
                    "year": paper.get("year"),
                    "categories": ", ".join(paper.get("categories", [])),
                    "url": paper.get("url", ""),
                    "id": paper.get("id", ""),
                    "doc_type": "research_paper"
                }
            )
            documents.append(doc)
        
        print(f"  Loaded {len(documents)} arXiv documents")
        
    except Exception as e:
        print(f"  Error loading arXiv papers: {e}")
    
    return documents


def load_bls_documents() -> List[Document]:
    """Load BLS employment data as documents."""
    documents = []
    
    bls_file = PROCESSED_DIR / "bls_employment.json"
    if not bls_file.exists():
        print("No BLS data found")
        return documents
    
    try:
        with open(bls_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        occupations = data.get("ai_relevant_occupations", [])
        summary = data.get("summary", {})
        
        print(f"Loading {len(occupations)} BLS occupations...")
        
        # Create a summary document
        summary_content = f"""AI/ML Job Market Summary (2024)

Total Employment in AI-Relevant Occupations: {summary.get('total_employment', 'N/A'):,}
Weighted Average Wage: ${summary.get('weighted_avg_wage', 'N/A'):,}
Average Projected Growth: {summary.get('avg_projected_growth', 'N/A')}%

Key Occupations:
"""
        
        for occ in occupations:
            summary_content += f"\n- {occ['title']} (SOC {occ['soc_code']})"
            summary_content += f"\n  Median Wage: ${occ.get('median_wage_2024', 'N/A'):,}"
            summary_content += f"\n  Employment: {occ.get('employment_2023', 'N/A'):,}"
            summary_content += f"\n  Projected Growth: {occ.get('projected_growth', 'N/A')}%"
            summary_content += f"\n  Education: {occ.get('education_required', 'N/A')}"
            summary_content += f"\n  AI Relevance: {occ.get('relevance_to_ai', 'N/A')}\n"
        
        doc = Document(
            page_content=summary_content,
            metadata={
                "source": "bls",
                "title": "AI/ML Job Market Summary 2024",
                "data_year": 2024,
                "doc_type": "employment_data",
                "occupation_count": len(occupations)
            }
        )
        documents.append(doc)
        
        # Also create individual occupation documents for better retrieval
        for occ in occupations:
            content = f"""Occupation: {occ['title']}
SOC Code: {occ['soc_code']}

Salary Information:
- Median Annual Wage (2024): ${occ.get('median_wage_2024', 'N/A'):,}

Employment Statistics:
- Total Employment (2023): {occ.get('employment_2023', 'N/A'):,}
- Projected Growth: {occ.get('projected_growth', 'N/A')}%

Education Requirements:
- Typical Entry-Level Education: {occ.get('education_required', 'N/A')}

AI Relevance:
{occ.get('relevance_to_ai', 'N/A')}
"""
            doc = Document(
                page_content=content,
                metadata={
                    "source": "bls",
                    "title": occ['title'],
                    "soc_code": occ['soc_code'],
                    "median_wage": occ.get('median_wage_2024'),
                    "employment": occ.get('employment_2023'),
                    "growth": occ.get('projected_growth'),
                    "education": occ.get('education_required'),
                    "data_year": 2024,
                    "doc_type": "occupation"
                }
            )
            documents.append(doc)
        
        print(f"  Loaded {len(documents)} BLS documents")
        
    except Exception as e:
        print(f"  Error loading BLS data: {e}")
    
    return documents


def load_census_documents() -> List[Document]:
    """Load Census education data as documents."""
    documents = []
    
    census_file = PROCESSED_DIR / "census_education.json"
    if not census_file.exists():
        print("No Census data found")
        return documents
    
    try:
        with open(census_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        education_levels = data.get("education_attainment", [])
        
        print(f"Loading Census education data...")
        
        content = """US Education Attainment - Relevant to AI/ML Programs

Education Levels and AI Field Relevance:

"""
        
        for level in education_levels:
            content += f"""{level['level']} ({level['variable_code']})
- Relevance to AI/ML: {level['relevance']}
- Estimated % in AI Fields: {level['ai_field_pct_estimate']}%

"""
        
        content += """Key Insights for University AI Programs:

1. Master's degrees show high relevance to AI/ML positions
2. Doctorate degrees essential for AI research roles
3. Bachelor's degrees sufficient for entry-level positions
4. Professional degrees increasingly include AI components

Demographic factors to consider:
- Age distribution of students pursuing AI education
- Geographic concentration in tech hubs
- International student participation in AI programs
"""
        
        doc = Document(
            page_content=content,
            metadata={
                "source": "census",
                "title": "Education Attainment for AI/ML Fields",
                "acs_year": data.get("meta", {}).get("acs_year", 2023),
                "doc_type": "education_data",
                "levels_count": len(education_levels)
            }
        )
        documents.append(doc)
        
        print(f"  Loaded {len(documents)} Census documents")
        
    except Exception as e:
        print(f"  Error loading Census data: {e}")
    
    return documents


def load_trends_documents() -> List[Document]:
    """Load Google Trends data as documents."""
    documents = []
    
    trends_file = PROCESSED_DIR / "google_trends.json"
    if not trends_file.exists():
        print("No Google Trends data found")
        return documents
    
    try:
        with open(trends_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        keywords = data.get("keywords", {})
        summary = data.get("summary", {})
        
        print(f"Loading {len(keywords)} trend keywords...")
        
        content = f"""AI Education Search Trends

Overview:
- Total Keywords Tracked: {summary.get('total_keywords', 0)}
- Keywords Trending Up: {summary.get('trending_up', 0)}
- Keywords Trending Down: {summary.get('trending_down', 0)}
- Average Interest Score: {summary.get('avg_interest', 0)}

Keyword Details:

"""
        
        for kw, info in keywords.items():
            content += f"""{kw}
- Average Interest: {info['average_interest']}
- Peak Interest: {info['peak_interest']} (on {info['peak_date']})
- Trend Direction: {info['trend_direction']}
- Recent Average: {info['recent_avg']}
- Volatility: {info['volatility']}

"""
        
        doc = Document(
            page_content=content,
            metadata={
                "source": "google_trends",
                "title": "AI Education Search Trends",
                "geo": data.get("meta", {}).get("geo", "US"),
                "timeframe": data.get("meta", {}).get("timeframe", "5 years"),
                "doc_type": "trends_data",
                "keyword_count": len(keywords)
            }
        )
        documents.append(doc)
        
        print(f"  Loaded {len(documents)} trends documents")
        
    except Exception as e:
        print(f"  Error loading trends data: {e}")
    
    return documents


def load_ipeds_documents() -> List[Document]:
    """Load IPEDS metadata as documents."""
    documents = []
    
    ipeds_file = PROCESSED_DIR / "ipeds_metadata.json"
    if not ipeds_file.exists():
        print("No IPEDS data found")
        return documents
    
    try:
        with open(ipeds_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        print("Loading IPEDS reference data...")
        
        content = f"""IPEDS (Integrated Postsecondary Education Data System) Reference

Academic Year: {data.get('meta', {}).get('academic_year', 'N/A')}
Status: {data.get('status', 'N/A')}

Available Data Tables:

"""
        
        for table in data.get("available_data", []):
            content += f"""{table['table']}: {table['description']}
Fields: {', '.join(table.get('fields', []))}

"""
        
        content += f"""
Note: {data.get('note', 'Full IPEDS data requires manual download')}

Relevant AI Program Data from IPEDS:
- Computer Science enrollment trends
- AI-related degree completions
- Institution characteristics and tuition
- Faculty and instructional staff data
"""
        
        doc = Document(
            page_content=content,
            metadata={
                "source": "ipeds",
                "title": "IPEDS Education Data Reference",
                "academic_year": data.get("meta", {}).get("academic_year"),
                "status": data.get("status"),
                "doc_type": "metadata"
            }
        )
        documents.append(doc)
        
        print(f"  Loaded {len(documents)} IPEDS documents")
        
    except Exception as e:
        print(f"  Error loading IPEDS data: {e}")
    
    return documents


def load_all_documents() -> List[Document]:
    """Load all processed documents into LangChain Document format."""
    documents = []
    
    print("Loading all document sources...")
    
    documents.extend(load_arxiv_documents())
    documents.extend(load_bls_documents())
    documents.extend(load_census_documents())
    documents.extend(load_trends_documents())
    documents.extend(load_ipeds_documents())
    
    return documents


def ingest_documents():
    """
    Ingest all documents into Chroma vector database.
    """
    if not LANGCHAIN_AVAILABLE:
        print("ERROR: Required libraries not available")
        print("Run: pip install langchain langchain-community chromadb sentence-transformers")
        return
    
    print("=" * 60)
    print("EduPredict RAG Document Ingestion")
    print("=" * 60)
    
    # Initialize embeddings
    print("\nInitializing embeddings...")
    try:
        embeddings = get_embeddings()
    except Exception as e:
        print(f"Error loading embeddings: {e}")
        return
    
    # Load documents
    print("\nLoading documents...")
    documents = load_all_documents()
    
    if not documents:
        print("\nNo documents to ingest. Run cleaner.py first.")
        return
    
    print(f"\nLoaded {len(documents)} total documents")
    
    # Show breakdown by source
    source_counts = {}
    for doc in documents:
        src = doc.metadata.get("source", "unknown")
        source_counts[src] = source_counts.get(src, 0) + 1
    
    print("\nDocument counts by source:")
    for src, count in sorted(source_counts.items()):
        print(f"  {src}: {count}")
    
    # Create/Update Chroma database
    print(f"\nIngesting into Chroma at {CHROMA_DIR}...")
    CHROMA_DIR.mkdir(parents=True, exist_ok=True)
    
    try:
        # Check if database already exists
        if (CHROMA_DIR / "chroma.sqlite3").exists():
            print("  Existing database found, updating...")
            # Load existing and add new
            vectorstore = Chroma(
                persist_directory=str(CHROMA_DIR),
                embedding_function=embeddings
            )
            # Add documents
            batch_size = 100
            for i in range(0, len(documents), batch_size):
                batch = documents[i:i+batch_size]
                vectorstore.add_documents(batch)
                print(f"  Added batch {i//batch_size + 1}/{(len(documents)-1)//batch_size + 1} ({len(batch)} docs)")
        else:
            print("  Creating new database...")
            vectorstore = Chroma.from_documents(
                documents=documents,
                embedding=embeddings,
                persist_directory=str(CHROMA_DIR)
            )
        
        # Persist to disk
        vectorstore.persist()
        
        # Verify
        collection_stats = vectorstore._collection.count()
        print(f"\n✓ Successfully ingested {collection_stats} documents")
        print(f"  Vector DB location: {CHROMA_DIR}")
        
        # Save ingestion log
        log_file = CHROMA_DIR / "ingestion_log.json"
        log_data = {
            "ingested_at": datetime.now().isoformat(),
            "total_documents": len(documents),
            "source_breakdown": source_counts,
            "collection_count": collection_stats
        }
        with open(log_file, "w") as f:
            json.dump(log_data, f, indent=2)
        
    except Exception as e:
        print(f"\n✗ Error during ingestion: {e}")
        import traceback
        traceback.print_exc()


def verify_ingestion() -> Dict[str, Any]:
    """Verify that documents are properly ingested."""
    if not LANGCHAIN_AVAILABLE:
        return {"error": "LangChain not available"}
    
    try:
        if not CHROMA_DIR.exists() or not (CHROMA_DIR / "chroma.sqlite3").exists():
            return {"status": "empty", "message": "No Chroma database found"}
        
        embeddings = get_embeddings()
        vectorstore = Chroma(
            persist_directory=str(CHROMA_DIR),
            embedding_function=embeddings
        )
        
        count = vectorstore._collection.count()
        
        # Try a test query
        test_results = vectorstore.similarity_search("AI education programs", k=3)
        
        return {
            "status": "healthy",
            "document_count": count,
            "test_query_results": len(test_results),
            "sample_sources": list(set(r.metadata.get("source", "unknown") for r in test_results))
        }
        
    except Exception as e:
        return {"status": "error", "message": str(e)}


if __name__ == "__main__":
    ingest_documents()
    
    print("\n" + "=" * 60)
    print("Verifying ingestion...")
    result = verify_ingestion()
    print(json.dumps(result, indent=2))
    print("=" * 60)
