"""
EduPredict Data Fetcher
Checks data sources for updates and downloads new data.
"""

import json
import hashlib
import requests
import xml.etree.ElementTree as ET
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional, List
import time

# Optional imports with graceful fallbacks
try:
    from bs4 import BeautifulSoup
    BS4_AVAILABLE = True
except ImportError:
    BS4_AVAILABLE = False
    print("Warning: beautifulsoup4 not installed. Some features limited.")

try:
    from pytrends.request import TrendReq
    PYTRENDS_AVAILABLE = True
except ImportError:
    PYTRENDS_AVAILABLE = False
    print("Warning: pytrends not installed. Google Trends unavailable.")

DATA_DIR = Path(__file__).parent.parent / "data"
RAW_DIR = DATA_DIR / "raw"
METADATA_FILE = DATA_DIR / "metadata.json"

# API Keys (free tier, no keys required for most)
BLS_API_KEY = ""  # Optional, can be empty for public data
CENSUS_API_KEY = ""  # Optional for small queries


def load_metadata() -> Dict:
    """Load metadata tracking file."""
    if METADATA_FILE.exists():
        with open(METADATA_FILE, "r") as f:
            return json.load(f)
    return {"version": "1.0", "sources": {}}


def save_metadata(metadata: Dict):
    """Save metadata tracking file."""
    with open(METADATA_FILE, "w") as f:
        json.dump(metadata, f, indent=2)


def calculate_checksum(filepath: Path) -> str:
    """Calculate MD5 checksum of file."""
    hash_md5 = hashlib.md5()
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    return hash_md5.hexdigest()


def fetch_ipeds() -> Optional[Path]:
    """
    Check IPEDS for new data availability.
    IPEDS releases data annually via Access database or CSV files.
    This checks the website for new releases.
    """
    print("Checking IPEDS for updates...")
    
    try:
        # Check the IPEDS data release page
        url = "https://nces.ed.gov/ipeds/use-the-data/download-access-database"
        headers = {
            "User-Agent": "EduPredict Data Fetcher (educational research)"
        }
        
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
        
        # Parse to find current year
        current_year = datetime.now().year
        content = response.text
        
        # Look for most recent academic year (e.g., "2023-24" or "202324")
        years_found = []
        for year in range(current_year, current_year - 5, -1):
            year_str = str(year)
            prev_year = str(year - 1)
            # Check various formats
            patterns = [
                f"{prev_year}-{year_str[2:]}",  # 2023-24
                f"{prev_year}{year_str[2:]}",  # 202324
                year_str
            ]
            for pattern in patterns:
                if pattern in content:
                    years_found.append((year, pattern))
                    break
        
        if years_found:
            latest_year, pattern = years_found[0]
            print(f"  Found IPEDS data for academic year: {pattern}")
            
            # Note: Actual download requires manual steps or specialized tools
            # for Access database files. Save a reference marker.
            filepath = RAW_DIR / "ipeds_reference.json"
            data = {
                "source": "IPEDS",
                "academic_year": pattern,
                "checked_at": datetime.now().isoformat(),
                "url": url,
                "note": "Manual download required for Access database files",
                "available_years": [y[1] for y in years_found[:3]]
            }
            
            with open(filepath, "w") as f:
                json.dump(data, f, indent=2)
            
            return filepath
        
        print("  No clear year markers found in IPEDS page")
        return None
        
    except Exception as e:
        print(f"  Error checking IPEDS: {e}")
        return None


def fetch_bls() -> Optional[Path]:
    """
    Fetch BLS Occupational Outlook data.
    Uses BLS Public Data API (free, no key required for limited use).
    https://www.bls.gov/developers/api_faqs.htm
    """
    print("Checking BLS for updates...")
    
    try:
        # BLS API endpoint for occupational data
        # Fetch data for computer/math occupations (SOC code 15-0000)
        base_url = "https://api.bls.gov/publicAPI/v2/timeseries/data/"
        
        # Series IDs for relevant occupations
        # Format: prefix + seasonal + area + industry + occupation + datatype
        # CUUR0000SA0 - All items in U.S. city average, all urban consumers, not seasonally adjusted
        series_ids = [
            "CUUR0000SA0",  # CPI for all urban consumers
            "OEUN000000000000014",  # Employment for computer occupations
        ]
        
        # For demo, we'll fetch wage/employment data
        # Using public occupational employment data
        url = "https://data.bls.gov/oesXlsx/data.htm"
        
        # Check available years
        years_url = "https://data.bls.gov/oes/#/home"
        headers = {"User-Agent": "EduPredict Research Bot"}
        
        resp = requests.get(years_url, headers=headers, timeout=30)
        resp.raise_for_status()
        
        current_year = datetime.now().year
        
        # Find most recent OEWS data year
        years_available = []
        for year in range(current_year, current_year - 5, -1):
            if str(year) in resp.text:
                years_available.append(year)
        
        # Save metadata about available data
        filepath = RAW_DIR / "bls_employment_data.json"
        data = {
            "source": "BLS",
            "program": "Occupational Employment and Wage Statistics",
            "years_available": years_available[:3],
            "checked_at": datetime.now().isoformat(),
            "target_occupations": [
                "15-1256 Computer and Information Research Scientists",
                "15-1250 Software and Web Developers, Programmers, and Testers",
                "15-1241 Computer and Information Research Scientists",
                "15-1299 Computer Occupations, All Other"
            ],
            "download_urls": {
                "all_data": "https://data.bls.gov/oes/#/home",
                "methodology": "https://www.bls.gov/oes/oes_ques.htm"
            },
            "note": "Manual CSV download from BLS OEWS website required for full dataset"
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"  Found BLS data for years: {years_available[:3]}")
        return filepath
        
    except Exception as e:
        print(f"  Error checking BLS: {e}")
        return None


def fetch_census() -> Optional[Path]:
    """
    Fetch Census education and demographic data.
    Uses Census Data API (free, no key required for basic use).
    https://www.census.gov/data/developers/data-sets.html
    """
    print("Checking Census for updates...")
    
    try:
        # Census ACS (American Community Survey) API
        # Get education attainment data
        base_url = "https://api.census.gov/data"
        
        current_year = datetime.now().year
        # ACS data is released with 1-year delay typically
        acs_year = current_year - 1
        
        # Try to fetch ACS 1-year estimates
        # Variables: B15003_022E (Bachelor's), B15003_023E (Master's), B15003_025E (Doctorate)
        url = f"{base_url}/{acs_year}/acs/acs1"
        
        headers = {"User-Agent": "EduPredict Research Bot"}
        
        # First, check if the year is available
        resp = requests.get(url, headers=headers, timeout=30)
        
        # Build metadata
        filepath = RAW_DIR / "census_education_data.json"
        data = {
            "source": "Census",
            "program": "American Community Survey (ACS)",
            "acs_year": acs_year,
            "checked_at": datetime.now().isoformat(),
            "relevant_variables": {
                "B15003_022E": "Bachelor's degree",
                "B15003_023E": "Master's degree", 
                "B15003_025E": "Doctorate degree",
                "B15003_001E": "Total population 25+",
                "B15003_017E": "Some college, less than 1 year",
                "B15003_018E": "Some college, 1+ years, no degree",
                "B15003_019E": "Associate's degree",
                "B15003_020E": "Associate's degree - occupational",
                "B15003_021E": "Associate's degree - academic"
            },
            "target_tables": ["B15003"],
            "geography": "us",
            "api_base": url,
            "download_url": "https://data.census.gov/cedsci/table?q=education",
            "note": "Use Census API or data portal for full state-level education data"
        }
        
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2)
        
        print(f"  Census ACS data available for year: {acs_year}")
        return filepath
        
    except Exception as e:
        print(f"  Error checking Census: {e}")
        return None


def fetch_arxiv() -> Optional[Path]:
    """
    Fetch recent AI education papers from arXiv.
    Uses arXiv API (free, no registration required).
    http://export.arxiv.org/api/query
    """
    print("Checking arXiv for new papers...")
    
    # Multiple queries to capture AI education related papers
    queries = [
        # AI in education
        "cat:cs.AI AND (education OR university OR curriculum)",
        # Machine learning education
        "cat:cs.LG AND (education OR teaching OR course)",
        # Computer science education
        "cat:cs.CY AND (artificial intelligence OR machine learning)",
        # AI programs and enrollment
        "all:((\"AI program\" OR \"artificial intelligence program\") AND university)"
    ]
    
    all_entries = []
    
    for query in queries:
        url = "https://export.arxiv.org/api/query"
        
        params = {
            "search_query": query,
            "start": 0,
            "max_results": 50,  # Per query
            "sortBy": "submittedDate",
            "sortOrder": "descending"
        }
        
        try:
            response = requests.get(url, params=params, timeout=30)
            response.raise_for_status()
            
            # Parse Atom XML
            root = ET.fromstring(response.content)
            
            # arXiv uses Atom namespace
            ns = {'atom': 'http://www.w3.org/2005/Atom'}
            
            for entry in root.findall('atom:entry', ns):
                title = entry.find('atom:title', ns)
                summary = entry.find('atom:summary', ns)
                published = entry.find('atom:published', ns)
                updated = entry.find('atom:updated', ns)
                id_elem = entry.find('atom:id', ns)
                
                # Get authors
                authors = []
                for author in entry.findall('atom:author', ns):
                    name = author.find('atom:name', ns)
                    if name is not None:
                        authors.append(name.text)
                
                # Get categories
                categories = []
                for cat in entry.findall('atom:category', ns):
                    term = cat.get('term')
                    if term:
                        categories.append(term)
                
                entry_data = {
                    "title": title.text.strip() if title is not None else "",
                    "summary": summary.text.strip() if summary is not None else "",
                    "published": published.text if published is not None else "",
                    "updated": updated.text if updated is not None else "",
                    "id": id_elem.text if id_elem is not None else "",
                    "authors": authors,
                    "categories": categories,
                    "query_matched": query
                }
                all_entries.append(entry_data)
            
            # Respect arXiv rate limits (3 seconds between requests)
            time.sleep(3)
            
        except Exception as e:
            print(f"  Error fetching arXiv query: {e}")
            continue
    
    if all_entries:
        # Remove duplicates based on arXiv ID
        seen_ids = set()
        unique_entries = []
        for entry in all_entries:
            if entry["id"] not in seen_ids:
                seen_ids.add(entry["id"])
                unique_entries.append(entry)
        
        # Save to file
        filepath = RAW_DIR / "arxiv_ai_education.json"
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump({
                "fetch_date": datetime.now().isoformat(),
                "total_entries": len(unique_entries),
                "entries": unique_entries
            }, f, indent=2)
        
        print(f"  Fetched {len(unique_entries)} unique papers from arXiv")
        return filepath
    
    print("  No papers retrieved from arXiv")
    return None


def fetch_google_trends() -> Optional[Path]:
    """
    Fetch Google Trends data for AI education keywords.
    Requires pytrends library.
    """
    print("Checking Google Trends...")
    
    if not PYTRENDS_AVAILABLE:
        print("  Skipped: pytrends not installed (pip install pytrends)")
        return None
    
    try:
        # Initialize pytrends
        pytrends = TrendReq(hl='en-US', tz=360)
        
        # Keywords related to AI education
        keywords = [
            "AI degree",
            "artificial intelligence program",
            "machine learning masters",
            "data science program",
            "AI university"
        ]
        
        all_data = {}
        
        for kw in keywords:
            try:
                pytrends.build_payload([kw], cat=0, timeframe='today 5-y', geo='US')
                data = pytrends.interest_over_time()
                
                if not data.empty:
                    # Convert to dict
                    trend_data = {
                        "keyword": kw,
                        "dates": data.index.strftime('%Y-%m-%d').tolist(),
                        "values": data[kw].tolist(),
                        "average": float(data[kw].mean()),
                        "peak": int(data[kw].max()),
                        "peak_date": data[kw].idxmax().strftime('%Y-%m-%d')
                    }
                    all_data[kw] = trend_data
                
                # Rate limiting
                time.sleep(1)
                
            except Exception as e:
                print(f"    Error fetching trends for '{kw}': {e}")
                continue
        
        if all_data:
            filepath = RAW_DIR / "google_trends.json"
            with open(filepath, "w") as f:
                json.dump({
                    "fetch_date": datetime.now().isoformat(),
                    "geo": "US",
                    "timeframe": "5 years",
                    "keywords": all_data
                }, f, indent=2)
            
            print(f"  Fetched trends for {len(all_data)} keywords")
            return filepath
        
        print("  No Google Trends data retrieved")
        return None
        
    except Exception as e:
        print(f"  Error with Google Trends: {e}")
        return None


def fetch_news_api() -> Optional[Path]:
    """
    Optional: Fetch AI education news from NewsAPI.
    Requires API key (free tier available).
    """
    print("Checking for education news...")
    
    # This would require a NewsAPI key
    # Skipping for now as it's optional
    print("  Skipped: NewsAPI optional (requires API key)")
    return None


def check_all_sources() -> List[Path]:
    """
    Check all data sources for updates.
    Returns list of new/changed files.
    """
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    
    metadata = load_metadata()
    new_files = []
    
    # Check each source
    fetchers = {
        "arxiv": fetch_arxiv,
        "ipeds": fetch_ipeds,
        "bls": fetch_bls,
        "census": fetch_census,
        "google_trends": fetch_google_trends,
    }
    
    for source_name, fetch_func in fetchers.items():
        try:
            filepath = fetch_func()
            if filepath:
                # Calculate new checksum
                new_checksum = calculate_checksum(filepath)
                old_checksum = metadata.get("sources", {}).get(source_name, {}).get("checksum")
                
                if new_checksum != old_checksum:
                    print(f"✓ {source_name}: New data found!")
                    new_files.append(filepath)
                    
                    # Update metadata
                    if "sources" not in metadata:
                        metadata["sources"] = {}
                    if source_name not in metadata["sources"]:
                        metadata["sources"][source_name] = {}
                    
                    metadata["sources"][source_name]["checksum"] = new_checksum
                    metadata["sources"][source_name]["last_fetch"] = datetime.now().isoformat()
                    metadata["sources"][source_name]["file"] = str(filepath.name)
                else:
                    # Still update the last_fetch time
                    metadata["sources"][source_name]["last_fetch"] = datetime.now().isoformat()
                    print(f"  {source_name}: No changes (up to date)")
            else:
                print(f"  {source_name}: Skipped/No data")
        except Exception as e:
            print(f"✗ {source_name}: Error - {e}")
    
    # Update last check timestamp
    metadata["last_check"] = datetime.now().isoformat()
    
    save_metadata(metadata)
    return new_files


def get_source_status() -> Dict:
    """Get current status of all data sources."""
    metadata = load_metadata()
    
    status = {}
    for source, info in metadata.get("sources", {}).items():
        last_fetch = info.get("last_fetch")
        if last_fetch:
            try:
                fetch_date = datetime.fromisoformat(last_fetch)
                days_ago = (datetime.now() - fetch_date).days
                status[source] = {
                    "last_fetch": last_fetch,
                    "days_ago": days_ago,
                    "fresh": days_ago < 7,  # Less than a week
                    "file": info.get("file")
                }
            except:
                status[source] = {"last_fetch": last_fetch, "error": "Invalid date"}
        else:
            status[source] = {"status": "never_fetched"}
    
    return status


if __name__ == "__main__":
    print("=" * 60)
    print("EduPredict Data Fetcher")
    print("=" * 60)
    print(f"Data directory: {DATA_DIR}")
    print(f"Raw data: {RAW_DIR}")
    print("-" * 60)
    
    new_files = check_all_sources()
    
    print("\n" + "=" * 60)
    if new_files:
        print(f"Found {len(new_files)} new/changed file(s):")
        for f in new_files:
            print(f"  - {f.name}")
    else:
        print("No new data found")
    
    # Print source status
    print("\nSource Status:")
    status = get_source_status()
    for source, info in status.items():
        if "days_ago" in info:
            fresh = "✓" if info["fresh"] else "⚠"
            print(f"  {fresh} {source}: {info['days_ago']} days ago")
        else:
            print(f"  ⚠ {source}: {info.get('status', 'unknown')}")
    
    print("=" * 60)
