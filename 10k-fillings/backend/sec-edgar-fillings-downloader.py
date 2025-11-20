import sys
from sec_edgar_downloader import Downloader


def fetch_latest_10k(ticker_symbol):

    # SEC requires company name, email address for downloading filings
    company_name = "IterativeStartup"
    email_address = "admin@example.com"

    print(f"Initializing downloader for {company_name} ({email_address})")
    dl = Downloader(company_name, email_address, "./sec_filings")

    print(f"Fetching latest 10-K for {ticker_symbol}")

    try:
        # Ticker -> CIK -> Latest 10-K -> Download
        ticker_filling = dl.get("10-K", ticker_symbol, limit=1)
        if ticker_filling:
            print(
                f"Successfully downloaded the latest 10-K for {ticker_symbol}")
        else:
            print(f"No 10-K filings found for {ticker_symbol}")

    except Exception as e:
        print(f"An error occurred: {e}")


if __name__ == "__main__":

    # Ask user for Ticker Symbol
    ticker = input(
        "Enter the Ticker Symbol of the company (e.g., AAPL for Apple Inc.): ").strip().upper()
    # Fetch the latest 10-K filing for the given ticker symbol
    fetch_latest_10k(ticker)
