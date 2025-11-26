import os
from dotenv import load_dotenv
from sec_api import RenderApi

load_dotenv()

SEC_API_KEY = os.getenv("SEC_API_KEY")

if not SEC_API_KEY:
    raise ValueError("SEC_API_KEY not found in environment variables")

# Initialize the RenderApi with the SEC API key
renderApi = RenderApi(api_key=os.getenv("SEC_API_KEY"))

def download_html_filing(filing_url: str, output_path: str, ticker: str) -> bool:
    """
        Downloads the raw HTML content of the filing for display/inspection
    """

    print(f"\nAttempting to download raw HTML for {ticker}")
    # SEC requires company name, email address for downloading filings
    company_name = "IterativeStartup"
    email_address = "admin@example.com"

    try:
        filing_content = renderApi.get_filing(filing_url)

        with open(output_path, 'w', encoding='utf-8') as f:
            f.write(filing_content)
        
        print(f"Successfully downloaded HTML filing for {ticker} to {output_path}")
        return True
    except requests.exceptions.RequestException as e:
        print(f"Error during HTML download for {ticker}: {e}")
        return False

# Example usage
filing_url = "https://www.sec.gov/Archives/edgar/data/320193/000032019325000079/aapl-20250927.htm"
output_path = "./sec_filings/AAPL_10K_2025.html"
ticker = "AAPL"
download_html_filing(filing_url, output_path, ticker)