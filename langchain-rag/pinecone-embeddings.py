import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

# Load environment variables from a .env file
load_dotenv()

# Load openAI API key from environment
openai_api_key = os.getenv("OPENAI_API_KEY")

# Load document
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

if not documents:
    raise ValueError("No documents found in the specified PDF file")

print(f"Loaded {len(documents)} document(s) from the PDF")