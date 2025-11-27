import os
from dotenv import load_dotenv
from sec_api import QueryApi

# Load environment variables from .env file
load_dotenv()
SEC_API_KEY = os.getenv("SEC_API_KEY")

if not SEC_API_KEY:
    raise ValueError("SEC_API_KEY not found in environment variables")

# Initialize the QueryApi with the SEC API key
query_api = QueryApi(api_key=SEC_API_KEY)

def get_latest_10k_filing_data(ticker: str) -> str | None:
    """
        Queries the SEC API to find the URL of the most recent 10-K filing
    """
    query = {
        "query": {
            "query_string": {
                "query": f"ticker:{ticker} AND formType:\"10-K\"",
                "default_operator": "AND"
            }
        },
        "from": "0",
        "size": "1", # We only need the latest one
        "sort": [{ "filedAt": { "order": "desc" } }]
    }

    try:
        response = query_api.get_filings(query)
        # print(f"SEC API response for ticker {ticker}: {response}")
        filings = response.get('filings', [])

        if response:
            return response
        else:
            print(f"No 10-K filing found for ticker: {ticker}")
            return None
    except Exception as e:
        print(f"Error querying SEC API for ticker {ticker}: {e}")
        return None

# Example usage
filing_data = get_latest_10k_filing_data("AAPL")
filing_url = filing_data['filings'][0].get('linkToFilingDetails')
print(f"Found latest 10-K URL for AAPL: {filing_url}")