import requests
import re
import time
from pathlib import Path

# =========================================================
# CONFIGURATION
# =========================================================
# Add your specific exact-match keywords inside this list.
# The script will search for papers containing ANY of these phrases.
KEYWORDS = [
    "uterine electromyography",
    "uterine activity monitoring",
    "uterine contraction monitoring",
    "uterine contraction detection",
    "term/preterm birth prediction",
    "pregnancy/labor classification",
    "electrohysterography",
    "preterm labor prediction",
    "EHG signal processing",
    "pregnancy monitoring",
    "uterine emg features",
    "electrohysterography features",
    "uterine activity",
    "electrohysterogram"

]

# Set how many top-cited papers you want to try downloading
MAX_RESULTS = 50
# =========================================================

def sanitize_filename(title):
    """Removes illegal characters from titles to create valid filenames."""
    if not title:
        return "Unknown_Title"
    clean_title = re.sub(r'[\\/*?:"<>|]', "", title)
    return clean_title[:100].strip()

def download_open_access_papers():
    base_dir = Path(__file__).resolve().parent.parent
    raw_dir = base_dir / "data" / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)
    
    # Dynamically build the search query from the KEYWORDS list
    # This wraps each keyword in quotes for exact matching and joins them with OR
    search_query = " OR ".join([f'"{kw}"' for kw in KEYWORDS])
    
    # 1. Stricter OpenAlex API Search
    url = "https://api.openalex.org/works"
    params = {
        "search": search_query,
        "filter": "is_oa:true",
        "per-page": MAX_RESULTS,
        "sort": "cited_by_count:desc"
    }
    
    api_headers = {
        "User-Agent": "mailto:your_email@example.com" 
    }
    
    # 2. Browser Disguise for Publishers
    browser_headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36",
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,image/apng,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Connection": "keep-alive"
    }
    
    print(f"Searching OpenAlex for strictly matched papers based on your keywords...")
    print(f"Query: {search_query}\n")
    
    try:
        response = requests.get(url, params=params, headers=api_headers)
        response.raise_for_status()
        
        data = response.json()
        results = data.get("results", [])
        
        print(f"Found top {len(results)} papers. Starting downloads to {raw_dir.relative_to(base_dir)}/ ...\n")
        print("-" * 80)
        
        for i, paper in enumerate(results, start=1):
            title = paper.get("title", f"Unknown_Title_{i}")
            oa_info = paper.get("open_access", {})
            oa_url = oa_info.get("oa_url")
            
            if not oa_url:
                print(f"[{i}] Skipped (No OA URL available): {title}")
                continue
                
            print(f"[{i}] Attempting to download: {title}")
            
            try:
                # Use the browser_headers to bypass 403 Forbidden errors
                pdf_response = requests.get(oa_url, headers=browser_headers, stream=True, timeout=20)
                pdf_response.raise_for_status()
                
                content_type = pdf_response.headers.get('Content-Type', '')
                if 'application/pdf' not in content_type:
                    print(f"    -> Skipped (URL returned a landing page, not a direct PDF. Type: {content_type})")
                    continue
                    
                safe_title = sanitize_filename(title)
                file_path = raw_dir / f"{safe_title}.pdf"
                
                with open(file_path, "wb") as f:
                    for chunk in pdf_response.iter_content(chunk_size=8192):
                        f.write(chunk)
                        
                print(f"    -> Successfully saved: {file_path.name}")
                
            except requests.exceptions.RequestException as e:
                if hasattr(e, 'response') and e.response is not None:
                    print(f"    -> Failed to download: HTTP {e.response.status_code}")
                else:
                    print(f"    -> Failed to download: {e}")
            
            time.sleep(2) # Polite sleep to avoid triggering rate limits
            
    except requests.exceptions.RequestException as e:
        print(f"Error fetching data from OpenAlex: {e}")

if __name__ == "__main__":
    download_open_access_papers()