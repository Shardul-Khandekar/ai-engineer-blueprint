import os
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec

# Load environment variables from a .env file
load_dotenv()

# Read Pinecone API key from environment
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Initialize Pinecone client
try:
    pc = Pinecone(api_key = pinecone_api_key)
except Exception as e:
    print(f"Error initializing Pinecone client: {e}")
    exit()

# Define index
index_name = "langchain-rag-index-v1"
# 1536 is the typical dimension for OpenAI's text-embedding-ada-002
dimension = 1536
# Use cosine similarity for text embeddings
metric_type = "cosine"

# Create index if it doesn't exist
if index_name not in pc.list_indexes().names():
    
    print(f"Creating index: {index_name}")
    
    pc.create_index(
        name=index_name,
        dimension=dimension,
        metric=metric_type,
        spec=ServerlessSpec(
            cloud="aws",
            region="us-east-1"
        )
    )

    print(f"Index {index_name} created successfully")
else:
    print(f"Index {index_name} already exists")


# Connect to the index
index = pc.Index(index_name)

# Verify connection by fetching index stats
print(index.describe_index_stats())