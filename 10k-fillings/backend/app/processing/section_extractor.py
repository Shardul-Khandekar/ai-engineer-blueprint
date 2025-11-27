import os
import json
import time
from sec_api import ExtractorApi
from dotenv import load_dotenv
from typing import Dict, Any, List

load_dotenv()
# Load SEC API key from environment variables
SEC_API_KEY = os.getenv("SEC_API_KEY")

# Define output directory for extracted sections
OUTPUT_DIR = "extracted_10k_sections"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the sections to extract
TARGET_SECTIONS: Dict[str, str] = {
    "1A": "Risk Factors",
    "7": "Management's Discussion & Analysis",
    "1": "Business",
    "8": "Financial Statements and Supplementary Data"
}

# Metadata for filings to process
# This should be replaced with actual CIKs and accession numbers and should be dynamically loaded
FILINGS_TO_PROCESS: List[Dict[str, Any]] = [
    {
        "cik": "0000320193",
        "ticker": "AAPL",
        "fiscal_year": 2025,
        "filing_url": "https://www.sec.gov/ix?doc=/Archives/edgar/data/0000320193/000032019325000079/aapl-20250927.htm",
        "company_name": "Apple Inc."
    },
]

