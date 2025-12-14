import os
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Configuration
CHROMA_DB_DIR = "resume_chroma_db"
COLLECTION_NAME = "resume_collection"

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
RESUME_PATH = os.getenv("RESUME_PATH")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables")


def create_resume_index(resume_path: str):
    """
    Loads a resume, chunks it, generates embeddings, and stores them in Chroma DB.
    """
    
    print(f"--- Loading document ---")
    try:
        # Reads the file and creates a list of documents, where each document is a page from the PDF
        loader = PyPDFLoader(resume_path)
        documents = loader.load()
    except Exception as e:
        print(f"Error loading PDF. Make sure the file exists and is a valid PDF: {e}")
        return
    
    print(f"Loaded {len(documents)} pages")

    print(f"--- Splitting document into chunks ---")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len,
        is_separator_regex=False,
    )

    chunks = text_splitter.split_documents(documents)
    print(f"Split into {len(chunks)} chunks for embedding")

    print(f"--- Initializing OpenAI/text-embedding-3-small embedding model ---")
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    print(f"--- Creating Chroma vector store ---")
    vector_db = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=CHROMA_DB_DIR,
        collection_name=COLLECTION_NAME
    )

    vector_db.persist()
    print("--- Indexing Complete with OpenAI Embeddings! ---")


if __name__ == "__main__":
    create_resume_index(RESUME_PATH)

