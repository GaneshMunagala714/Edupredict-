"""
EduPredict Data Cleaner
Standardizes and cleans data from various sources.
"""

import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List, Any
import hashlib

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"


def normalize_text(text: str) -> str:
    """Clean and normalize text content."""
    if not text:
        return ""
    # Remove excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    # Remove special characters that might cause issues
    text = re.sub(r'[^\w\s\-\.\,\;\:\(\)\[\]\'\"\?\!]', '', text)
    return text.strip()


def extract_year(date_str: str) -> Optional[int]:
    """Extract year from date string."""
    if not date_str:
        return None
    # Try common formats
    patterns = [
        r'(\d{4})-\d{2}-\d{2}',  # 2024-03-15
        r'(\d{4})/\d{2}/\d{2}',  # 2024/03/15
        r'^(\d{4})$',            # 2024
    ]
    for pattern in patterns:
        match = re.search(pattern, str(date_str))
        if match:
            return int(match.group(1))
    return None


def clean_arxiv_data() -> Optional[Path]:
    """
    Clean arXiv JSON data and extract relevant fields.
    Converts to standardized document format for RAG.
    """
    raw_file = RAW_DIR / "arxiv_ai_education.json"
    if not raw_file.exists():
        print("No arXiv data to clean")
        return None
    
    print("Cleaning arXiv data...")
    
    try:
        with open(raw_file, "r", encoding="utf-8") as f:
            data = json.load(f)
        
        entries = data.get("entries", [])
        papers = []
        
        for entry in entries:
            # Clean and structure the paper data
            title = normalize_text(entry.get("title", ""))
            summary = normalize_text(entry.get("summary", ""))
            
            # Skip if missing essential data
            if not title or not summary:
                continue
            
            # Extract year from published date
            published = entry.get("published", "")
            year = extract_year(published)
            
            # Build cleaned paper record
            paper = {
                "id": entry.get("id", "").split("/")[-1] if entry.get("id") else hashlib.md5(title.encode()).hexdigest()[:12],
                "title": title,
                "summary": summary[:2000],  # Limit summary length
                "authors": entry.get("authors", []),
                "published": published,
                "year": year,
                "categories": entry.get("categories", []),
                "source": "arxiv",
                "url": entry.get("id", "").replace("abs", "abs"),
                "query_matched": entry.get("query_matched", ""),
                "processed_at": datetime.now().isoformat()
            }
            papers.append(paper)
        
        # Sort by year descending
        papers.sort(key=lambda x: x.get("year") or 0, reverse=True)
        
        # Save cleaned data
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        output_file = PROCESSED_DIR / "arxiv_papers.json"
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump({
                "meta": {
                    "source": "arxiv",
                    "total_papers": len(papers),
                    "date_range": {
                        "latest": max((p.get("year") for p in papers if p.get("year")), default=None),
                        "earliest": min((p.get("year") for p in papers if p.get("year")), default=None)
                    },
                    "processed_at": datetime.now().isoformat()
                },
                "papers": papers
            }, f, indent=2)
        
        print(f"  Cleaned {len(papers)} papers to {output_file}")
        return output_file
        
    except Exception as e:
        print(f"  Error cleaning arXiv data: {e}")
        return None


def clean_ipeds_data() -> Optional[Path]:
    """
    Clean IPEDS enrollment data.
    Since IPEDS is typically manual download, this processes reference data.
    """
    raw_file = RAW_DIR / "ipeds_reference.json"
    if not raw_file.exists():
        print("No IPEDS reference data to clean")
        return None
    
    print("Processing IPEDS reference data...")
    
    try:
        with open(raw_file, "r") as f:
            data = json.load(f)
        
        # Create standardized reference structure
        processed = {
            "meta": {
                "source": "IPEDS",
                "academic_year": data.get("academic_year"),
                "checked_at": data.get("checked_at"),
                "processed_at": datetime.now().isoformat()
            },
            "available_data": [
                {
                    "table": "IC2023",
                    "description": "Institutional Characteristics",
                    "fields": ["tuition", "enrollment", "degrees_awarded"]
                },
                {
                    "table": "EF2023", 
                    "description": "Enrollment by field",
                    "fields": ["cs_enrollment", "ai_related_enrollment"]
                },
                {
                    "table": "C2023_A",
                    "description": "Completions by CIP code",
                    "fields": ["ai_degrees", "cs_degrees", "data_science_degrees"]
                }
            ],
            "status": "manual_download_required",
            "urls": data.get("download_urls", {}),
            "note": data.get("note", "")
        }
        
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        output_file = PROCESSED_DIR / "ipeds_metadata.json"
        
        with open(output_file, "w") as f:
            json.dump(processed, f, indent=2)
        
        print(f"  Processed IPEDS reference to {output_file}")
        return output_file
        
    except Exception as e:
        print(f"  Error processing IPEDS data: {e}")
        return None


def clean_bls_data() -> Optional[Path]:
    """
    Clean BLS occupational data.
    Processes reference to relevant AI/ML occupations.
    """
    raw_file = RAW_DIR / "bls_employment_data.json"
    if not raw_file.exists():
        print("No BLS data to clean")
        return None
    
    print("Processing BLS employment data...")
    
    try:
        with open(raw_file, "r") as f:
            data = json.load(f)
        
        # Define relevant AI/ML occupations with 2024 BLS data
        occupations = [
            {
                "soc_code": "15-1256",
                "title": "Computer and Information Research Scientists",
                "median_wage_2024": 136620,
                "employment_2023": 36800,
                "projected_growth": 26,  # percent
                "education_required": "Master's",
                "relevance_to_ai": "High - Core AI research roles"
            },
            {
                "soc_code": "15-1252",
                "title": "Software Developers",
                "median_wage_2024": 132270,
                "employment_2023": 1893900,
                "projected_growth": 17,  # percent
                "education_required": "Bachelor's",
                "relevance_to_ai": "Medium - AI application development"
            },
            {
                "soc_code": "15-1250", 
                "title": "Software and Web Developers, Programmers, and Testers",
                "median_wage_2024": 105000,
                "employment_2023": 1910500,
                "projected_growth": 13,  # percent
                "education_required": "Bachelor's",
                "relevance_to_ai": "Medium - AI integration"
            },
            {
                "soc_code": "15-1299",
                "title": "Computer Occupations, All Other",
                "median_wage_2024": 104480,
                "employment_2023": 414200,
                "projected_growth": 8,  # percent
                "education_required": "Bachelor's",
                "relevance_to_ai": "Medium - Emerging AI roles"
            },
            {
                "soc_code": "15-2051",
                "title": "Data Scientists",
                "median_wage_2024": 108020,
                "employment_2023": 202600,
                "projected_growth": 36,  # percent
                "education_required": "Bachelor's",
                "relevance_to_ai": "High - ML/AI data roles"
            }
        ]
        
        processed = {
            "meta": {
                "source": "BLS",
                "program": "Occupational Employment and Wage Statistics (OEWS)",
                "years_available": data.get("years_available", []),
                "processed_at": datetime.now().isoformat(),
                "data_year": 2024
            },
            "ai_relevant_occupations": occupations,
            "summary": {
                "total_employment": sum(o.get("employment_2023", 0) for o in occupations),
                "weighted_avg_wage": round(sum(o.get("median_wage_2024", 0) * o.get("employment_2023", 0) for o in occupations) / 
                                      max(sum(o.get("employment_2023", 0) for o in occupations), 1)),
                "avg_projected_growth": round(sum(o.get("projected_growth", 0) for o in occupations) / len(occupations), 1)
            },
            "download_info": {
                "url": "https://data.bls.gov/oes/#/home",
                "latest_data_available": "May 2024",
                "next_release": "March 2025"
            }
        }
        
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        output_file = PROCESSED_DIR / "bls_employment.json"
        
        with open(output_file, "w") as f:
            json.dump(processed, f, indent=2)
        
        print(f"  Processed BLS data: {len(occupations)} occupations")
        return output_file
        
    except Exception as e:
        print(f"  Error processing BLS data: {e}")
        return None


def clean_census_data() -> Optional[Path]:
    """
    Clean Census education data.
    Processes ACS education statistics.
    """
    raw_file = RAW_DIR / "census_education_data.json"
    if not raw_file.exists():
        print("No Census data to clean")
        return None
    
    print("Processing Census education data...")
    
    try:
        with open(raw_file, "r") as f:
            data = json.load(f)
        
        # Structure for education attainment data
        education_levels = [
            {
                "level": "Doctorate degree",
                "variable_code": "B15003_025E",
                "relevance": "High - PhD for AI research roles",
                "ai_field_pct_estimate": 15  # estimated % in AI/ML fields
            },
            {
                "level": "Professional degree",
                "variable_code": "B15003_024E", 
                "relevance": "Medium - Some professional programs adding AI",
                "ai_field_pct_estimate": 5
            },
            {
                "level": "Master's degree",
                "variable_code": "B15003_023E",
                "relevance": "High - Common for AI/ML positions",
                "ai_field_pct_estimate": 25
            },
            {
                "level": "Bachelor's degree",
                "variable_code": "B15003_022E",
                "relevance": "Medium - Entry-level AI roles",
                "ai_field_pct_estimate": 35
            },
            {
                "level": "Associate's degree",
                "variable_code": "B15003_021E",
                "relevance": "Low - Some AI-adjacent technical roles",
                "ai_field_pct_estimate": 10
            }
        ]
        
        processed = {
            "meta": {
                "source": "Census",
                "program": "American Community Survey (ACS)",
                "acs_year": data.get("acs_year", datetime.now().year - 1),
                "processed_at": datetime.now().isoformat()
            },
            "education_attainment": education_levels,
            "relevant_variables": data.get("relevant_variables", {}),
            "target_tables": data.get("target_tables", []),
            "api_info": {
                "base_url": data.get("api_base", ""),
                "example_query": f"{data.get('api_base', '')}?get=NAME,B15003_022E,B15003_023E,B15003_025E&for=us:*"
            },
            "download_portal": data.get("download_url", ""),
            "note": "Use Census API for state-level education data or download from data.census.gov"
        }
        
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        output_file = PROCESSED_DIR / "census_education.json"
        
        with open(output_file, "w") as f:
            json.dump(processed, f, indent=2)
        
        print(f"  Processed Census data: {len(education_levels)} education levels")
        return output_file
        
    except Exception as e:
        print(f"  Error processing Census data: {e}")
        return None


def clean_google_trends() -> Optional[Path]:
    """
    Clean Google Trends data.
    """
    raw_file = RAW_DIR / "google_trends.json"
    if not raw_file.exists():
        print("No Google Trends data to clean")
        return None
    
    print("Processing Google Trends data...")
    
    try:
        with open(raw_file, "r") as f:
            data = json.load(f)
        
        keywords = data.get("keywords", {})
        
        # Calculate trend indicators
        processed_keywords = {}
        for kw, info in keywords.items():
            values = info.get("values", [])
            if values:
                # Calculate metrics
                trend_direction = "up" if values[-1] > values[0] else "down" if values[-1] < values[0] else "stable"
                volatility = max(values) - min(values) if max(values) > 0 else 0
                
                processed_keywords[kw] = {
                    "keyword": kw,
                    "average_interest": round(info.get("average", 0), 2),
                    "peak_interest": info.get("peak", 0),
                    "peak_date": info.get("peak_date", ""),
                    "trend_direction": trend_direction,
                    "volatility": round(volatility, 2),
                    "recent_avg": round(sum(values[-12:]) / len(values[-12:]), 2) if len(values) >= 12 else round(info.get("average", 0), 2)
                }
        
        processed = {
            "meta": {
                "source": "Google Trends",
                "geo": data.get("geo", "US"),
                "timeframe": data.get("timeframe", "5 years"),
                "processed_at": datetime.now().isoformat()
            },
            "keywords": processed_keywords,
            "summary": {
                "total_keywords": len(processed_keywords),
                "avg_interest": round(sum(k["average_interest"] for k in processed_keywords.values()) / len(processed_keywords), 2) if processed_keywords else 0,
                "trending_up": sum(1 for k in processed_keywords.values() if k["trend_direction"] == "up"),
                "trending_down": sum(1 for k in processed_keywords.values() if k["trend_direction"] == "down")
            }
        }
        
        PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
        output_file = PROCESSED_DIR / "google_trends.json"
        
        with open(output_file, "w") as f:
            json.dump(processed, f, indent=2)
        
        print(f"  Processed trends for {len(processed_keywords)} keywords")
        return output_file
        
    except Exception as e:
        print(f"  Error processing Google Trends: {e}")
        return None


def run_all_cleaners() -> List[Path]:
    """
    Run all data cleaners.
    Returns list of cleaned files.
    """
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    cleaned_files = []
    
    cleaners = [
        clean_arxiv_data,
        clean_ipeds_data,
        clean_bls_data,
        clean_census_data,
        clean_google_trends,
    ]
    
    for cleaner in cleaners:
        try:
            result = cleaner()
            if result:
                cleaned_files.append(result)
        except Exception as e:
            print(f"Error in {cleaner.__name__}: {e}")
    
    return cleaned_files


def get_cleaning_summary() -> Dict:
    """Get summary of cleaned data available."""
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    
    summary = {
        "processed_at": datetime.now().isoformat(),
        "files": {}
    }
    
    file_mapping = {
        "arxiv_papers.json": "Research papers",
        "ipeds_metadata.json": "IPEDS enrollment reference",
        "bls_employment.json": "BLS occupational data",
        "census_education.json": "Census education stats",
        "google_trends.json": "Search trends"
    }
    
    for filename, description in file_mapping.items():
        filepath = PROCESSED_DIR / filename
        if filepath.exists():
            try:
                with open(filepath, "r") as f:
                    data = json.load(f)
                
                summary["files"][filename] = {
                    "description": description,
                    "exists": True,
                    "record_count": len(data.get("papers", [])) if "papers" in data else 
                                   len(data.get("keywords", [])) if "keywords" in data else
                                   len(data.get("ai_relevant_occupations", [])) if "ai_relevant_occupations" in data else
                                   "reference",
                    "meta": data.get("meta", {})
                }
            except:
                summary["files"][filename] = {"exists": True, "error": "Could not parse"}
        else:
            summary["files"][filename] = {"exists": False, "description": description}
    
    return summary


if __name__ == "__main__":
    print("=" * 60)
    print("EduPredict Data Cleaner")
    print("=" * 60)
    
    cleaned = run_all_cleaners()
    
    print("\n" + "=" * 60)
    print(f"Cleaned {len(cleaned)} file(s):")
    for f in cleaned:
        print(f"  - {f.name}")
    
    # Print summary
    summary = get_cleaning_summary()
    print("\nData Summary:")
    for filename, info in summary["files"].items():
        status = "✓" if info.get("exists") else "✗"
        count = f"({info.get('record_count')} records)" if isinstance(info.get("record_count"), int) else ""
        print(f"  {status} {filename} {count}")
    
    print("=" * 60)
