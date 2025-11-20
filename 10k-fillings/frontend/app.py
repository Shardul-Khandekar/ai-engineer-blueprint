import streamlit as st
import sys
import os
import glob

# Add parent directory to sys.path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

# Import backend module
from backend import sec_edgar_fillings_downloader

# Streamlit app configuration
st.set_page_config(layout="wide", page_title="Financial Agent")
DATA_DIR = "sec_filings"

if "messages" not in st.session_state:
    st.session_state.messages = []


def get_downloaded_files():
    """
    Scans the DATA_DIR for downloaded 10-K files.
    Returns a dictionary: {Ticker: [list of file paths]}
    """

    files_map = {}
    if not os.path.exists(DATA_DIR):
        return files_map

    # The sec-edgar-downloader creates a structure like:
    # sec_filings/sec-edgar-filings/AAPL/10-K/000.../primary-document.html
    search_path = os.path.join(
        DATA_DIR, "sec-edgar-filings", "*", "10-K", "*", "*.html")
    found_files = glob.glob(search_path)

    # Also look for .txt files
    if not found_files:
        search_path_txt = os.path.join(
            DATA_DIR, "sec-edgar-filings", "*", "10-K", "*", "*.txt")
        found_files = glob.glob(search_path_txt)

    for filepath in found_files:
        # Extract ticker from the file path
        parts = filepath.split(os.sep)

        try:
            # Find index of 'sec-edgar-filings' and get the next folder (Ticker)
            idx = parts.index("sec-edgar-filings")
            ticker = parts[idx + 1]

            if ticker not in files_map:
                files_map[ticker] = []
            files_map[ticker].append(filepath)

        except ValueError:
            continue

    return files_map


def read_file_content(filepath):
    """
    Reads the content of a file given its path.
    Supports both .html and .txt files.
    """
    try:
        with open(filepath, "r", encoding="utf-8", errors='ignore') as f:
            return f.read()
    except Exception as e:
        return f"Error reading file: {e}"


# Sidebar file explorer
st.sidebar.header("🗂️ Knowledge Base")
if st.sidebar.button("Refresh Files"):
    st.rerun()

files_data = get_downloaded_files()

if not files_data:
    st.sidebar.info("No documents found yet. Ask the agent to download one!")
else:
    for ticker, paths in files_data.items():
        with st.sidebar.expander(f"📂 {ticker}", expanded=False):
            for i, path in enumerate(paths):
                # Use unique key for each button
                btn_label = f"📄 10-K Filing ({i+1})"
                if st.button(btn_label, key=f"btn_{ticker}_{i}"):
                    st.session_state['active_file'] = path
                    st.session_state['active_ticker'] = ticker

# Main chat interface
st.title("🤖 10-K Analyst Agent")

if 'active_file' in st.session_state:
    st.info(f"Viewing 10-K for **{st.session_state['active_ticker']}**")

    with st.expander("Hide Document View", expanded=True):
        content = read_file_content(st.session_state['active_file'])
        # Simple scrollable text area for now
        st.text_area("File Content", content, height=400)

    st.divider()

# Chat Area
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Input handling
if prompt := st.chat_input("Enter a Ticker Symbol (e.g., TSLA)..."):
    # Show the user message in chat
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 2. Assistant Action
    with st.chat_message("assistant"):
        ticker = prompt.strip().upper()
        status_placeholder = st.empty()
        status_placeholder.write(f"Contacting SEC Edgar for **{ticker}**...")

        try:
            sec_edgar_fillings_downloader.fetch_latest_10k(ticker)

            # Check if new file was downloaded
            new_files = get_downloaded_files()
            if ticker in new_files:
                success_msg = f"Successfully downloaded 10-K for **{ticker}**. Check the sidebar!"
                status_placeholder.markdown(success_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": success_msg})
                st.rerun()
            else:
                fail_msg = f"Download attempted, but no file found for **{ticker}**. It might not exist or the SEC API failed."
                status_placeholder.markdown(fail_msg)
                st.session_state.messages.append(
                    {"role": "assistant", "content": fail_msg})
        
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            status_placeholder.markdown(error_msg)
            st.session_state.messages.append({"role": "assistant", "content": error_msg})
