import os
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any
import time

from dotenv import load_dotenv
load_dotenv()

# Load Pinecone API key from environment variables
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")

# Index configuration
INDEX_NAME = "sec-10k-rag-index"
# Dimension of the vectors that should match the embedding model text-embedding-ada-002
VECTOR_DIMENSION = 1536
METRIC = "cosine"
CLOUD = "aws"
REGION = "us-east-1"

class PineconeIndexManager:
    """
    Manages connection and index creation for Pinecone.
    """
    def __init__(self, api_key: str, cloud: str, region: str):
        self.pinecone_client = Pinecone(api_key=api_key)
        self.cloud = cloud
        self.region = region

    def create_or_get_index(self, index_name: str, dimension: int, metric: str = "cosine"):
        """
        Creates the Pinecone index if it doesn't exist, otherwise connects to it.
        """
        existing_indexes = [index.name for index in self.pinecone_client.list_indexes()]
        if index_name in existing_indexes:
            print(f"Index '{index_name}' already exists")
            return self.pinecone_client.Index(index_name)

        # Define serverless specification
        spec = ServerlessSpec(cloud=self.cloud, region=self.region)

        print(f"Creating new index '{index_name}'")
        self.pinecone_client.create_index(
            name=index_name,
            dimension=dimension,
            metric=metric,
            spec=spec
        )
        print(f"Index '{index_name}' created successfully with dimension={dimension}, metric='{metric}'")

        # Wait until the index is fully initialized before returning
        while not self.pinecone_client.describe_index(index_name).status['ready']:
            print("... Waiting for index to become active...")
            time.sleep(1)

        return self.pinecone_client.Index(index_name)

    def delete_index(self, index_name: str):
        """
        Deletes the Pinecone index
        """
        if index_name in self.pinecone_client.list_indexes().names:
            print(f"Deleting index '{index_name}'...")
            self.pinecone_client.delete_index(index_name)
            print(f"Index '{index_name}' deleted.")
        else:
            print(f"Index '{index_name}' does not exist")


# Define required metadata fields 
REQUIRED_METADATA_FIELDS: List[Dict[str, Any]] = [
    {"key": "cik", "description": "SEC CIK of the company", "type": "string"},
    {"key": "ticker", "description": "Stock ticker", "type": "string"},
    {"key": "fiscal_year", "description": "The 10-K filing year", "type": "integer"},
    {"key": "section_id", "description": "Item ID (e.g., '1A', '7')", "type": "string"},
    {"key": "section_title", "description": "Full title (e.g., 'Risk Factors')", "type": "string"},
]


if __name__ == "__main__":
    # Initialize Pinecone index manager
    manager = PineconeIndexManager(
        api_key=PINECONE_API_KEY, 
        cloud=CLOUD, 
        region=REGION
    )

    try:
        index = manager.create_or_get_index(
            index_name=INDEX_NAME,
            dimension=VECTOR_DIMENSION,
            metric=METRIC
        )

        print("\nIndex ready. Index stats:")
        print(index.describe_index_stats())

    except Exception as e:
        print(f"\nFATAL ERROR: Could not create or connect to Pinecone index. Error: {e}")