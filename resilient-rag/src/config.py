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