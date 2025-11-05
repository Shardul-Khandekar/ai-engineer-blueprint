import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    
    # Data directory path
    DATA_DIR = os.path.join(os.getcwd(), "data")

    # Ingestion settings
    CHUNK_SIZE = 500
    CHUNK_OVERLAP = 50

    # Vector index
    INDEX_NAME = "resilient-rag-v1"
    EMBEDDING_MODEL = "text-embedding-3-small"

    # API Keys
    OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
    PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

    # Pinecone Serverless Spec for auto creation
    CLOUD_PROVIDER = "aws" 
    REGION = "us-east-1"

    # LLM Settings
    LLM_MODEL = "gpt-4o-mini"
    TEMPERATURE = 0