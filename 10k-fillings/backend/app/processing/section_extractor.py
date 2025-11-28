import os
import json
import time
from dotenv import load_dotenv
from sec_api import ExtractorApi
from typing import Dict, Any, List, Optional

from app.ingestion.sec.api.url_generator import get_latest_10k_filing_data

load_dotenv()
# Load SEC API key from environment variables
SEC_API_KEY = os.getenv("SEC_API_KEY")

if not SEC_API_KEY:
    raise ValueError("SEC_API_KEY not found in environment variables")

# Define output directory for extracted sections
OUTPUT_DIR = "extracted_10k_sections"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Define the sections to extract
TARGET_SECTIONS: Dict[str, str] = {
    "1A": "Risk Factors",
    "7": "Management's Discussion & Analysis",
    "1": "Business",
}

# List of tickers to process. This can be moved to a config file or database as needed.
TICKERS_TO_PROCESS: List[Dict[str, str]] = [
    {"ticker": "AAPL", "company_name": "Apple Inc."},
]

extractor_api = ExtractorApi(api_key=SEC_API_KEY)


def extract_and_save_sections(company_data: Dict[str, str]):
    """
    Retrieves filing data, extracts sections 
    using the URL, and saves them locally with Pinecone metadata.
    """

    ticker = company_data["ticker"]
    company_name = company_data["company_name"]

    print(f"Processing ticker: {ticker}, company: {company_name}")

    filing_data = get_latest_10k_filing_data(ticker)

    if not filing_data or not filing_data.get('filings'):
        print(f"Skipping {ticker}: No 10-K filing data found.")
        return

    # Extract the latest filing
    filing = filing_data['filings'][0]

    try:
        filing_url = filing.get('linkToFilingDetails')
        cik = filing.get('cik')
        
        # Use the year from the 'filedAt' field as the fiscal year proxy
        filed_at_year = filing.get('filedAt', 'YYYY')[:4]
        fiscal_year = int(filed_at_year)

    except (TypeError, ValueError) as e:
        print(f"Skipping {ticker}: Failed to parse required metadata fields. Error: {e}")
        return

    print(f"Starting extraction for {company_name} ({ticker}, FY{fiscal_year})")

    for item_id, section_title in TARGET_SECTIONS.items():
        print(f"Extracting Item {item_id}: {section_title}")

        metadata = {
            "cik": str(cik), # Ensure CIK is a string
            "ticker": ticker,
            "fiscal_year": fiscal_year,
            "section_id": item_id,
            "section_title": section_title,
            "company_name": company_name,
            "filing_url": filing_url
        }

        try:
            section_text = extractor_api.get_section(filing_url, item_id, "text")

            file_prefix = f"{cik}_{fiscal_year}_{ticker}"
            filename = os.path.join(OUTPUT_DIR, f"{file_prefix}_ITEM{item_id}.txt")

            with open(filename, 'w', encoding='utf-8') as f:
                # Save metadata as JSON header
                f.write(json.dumps(metadata) + "\n\n")
                f.write(section_text)
            
            print(f"Successfully saved {item_id} to {filename}")
            time.sleep(1) # Respect API rate limits
        
        except Exception as e:
            print(f"ERROR extracting Item {item_id} for {ticker}. Error: {e}")

if __name__ == "__main__":
    for company in TICKERS_TO_PROCESS:
        extract_and_save_sections(company)
    
    print("\nSection extraction completed")