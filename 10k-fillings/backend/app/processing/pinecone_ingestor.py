import os
import json
from dotenv import load_dotenv
from pinecone import Pinecone, ServerlessSpec
from typing import List, Dict, Any, Optional
from tqdm import tqdm
from langchain_text_splitters import RecursiveCharacterTextSplitter
from openai import OpenAI

from app.vectorstore.pinecone_setup import PineconeIndexManager, INDEX_NAME, VECTOR_DIMENSION

load_dotenv()
PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not PINECONE_API_KEY or not OPENAI_API_KEY:
    raise ValueError("PINECONE_API_KEY and OPENAI_API_KEY must be set in environment variables")

EXTRACTED_DIR = "extracted_10k_sections"
# Chunking Parameters
CHUNK_SIZE = 512
CHUNK_OVERLAP = 50
# Max vectors per upsert batch
BATCH_SIZE = 100

manager = PineconeIndexManager(
    api_key=PINECONE_API_KEY, 
    cloud=os.getenv("CLOUD", "aws"),
    region=os.getenv("REGION", "us-east-1")
)

openai_client = OpenAI(api_key=OPENAI_API_KEY)
EMBED_MODEL = os.getenv("EMBED_MODEL", "text-embedding-3-small")

index = manager.create_or_get_index(INDEX_NAME, VECTOR_DIMENSION)

# Text Splitter initialization
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=CHUNK_SIZE,
    chunk_overlap=CHUNK_OVERLAP,
    length_function=len,
    is_separator_regex=False
)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Generates embeddings for a list of texts using OpenAI API
    """
    # The OpenAI client returns the list of embedding objects in the 'data' key
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=texts
    )

    # Extracting the 'embedding' list of floats from each data object
    embeddings = [data.embedding for data in response.data]
    return embeddings

def load_text_and_metadata(filepath: str) -> Optional[Dict[str, Any]]:
    """
    Loads metadata and text content from a locally saved file
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        # Read the first line (JSON metadata) and the rest (text)
        metadata_line = f.readline().strip()
        text_content = f.read().strip()

    if not metadata_line:
        print(f"Skipping {filepath}: No metadata found")
        return None
    
    try:
        metadata = json.loads(metadata_line)
        metadata['text'] = text_content
        return metadata
    except json.JSONDecodeError:
        print(f"Skipping {filepath}: Invalid JSON metadata")
        return None


def process_and_upsert_filing():
    """
    Iterates over extracted files, chunks text, and upserts vectors to Pinecone.
    """
    files_to_process = [f for f in os.listdir(EXTRACTED_DIR) if f.endswith('.txt')]

    if not files_to_process:
        print(f"No extracted files found in '{EXTRACTED_DIR}'. Run the extractor first")
        return

    print(f"\nProcessing {len(files_to_process)} extracted section files")

    all_vectors_to_upsert = []

    for filepath in tqdm(files_to_process, desc="Chunking and Preparing"):
        full_path = os.path.join(EXTRACTED_DIR, filepath)
        filing_data = load_text_and_metadata(full_path)

        if not filing_data:
            continue

        text_content = filing_data.pop('text')
        chunks = text_splitter.split_text(text_content)

        for i, chunk in enumerate(chunks):
            # Generate a unique ID: CIK_SECTIONID_CHUNKINDEX
            vector_id = f"{filing_data['cik']}_{filing_data['section_id']}_{i}"

            # Create a shallow copy of the metadata and add the chunk_index
            chunk_metadata = filing_data.copy()
            chunk_metadata['chunk_index'] = i

            # Pinecone Upsert format: (id, text, metadata)
            # Pass the raw text and Pinecone will handle embedding generation
            all_vectors_to_upsert.append(
                (vector_id, chunk, chunk_metadata)
            )

    print(f"\nTotal chunks generated: {len(all_vectors_to_upsert)}. Starting upsert")

    for i in tqdm(range(0, len(all_vectors_to_upsert), BATCH_SIZE), desc="Upserting to Pinecone"):
        batch = all_vectors_to_upsert[i:i + BATCH_SIZE]

        ids   = [it[0] for it in batch]
        texts = [it[1] for it in batch]
        metadata = [it[2] for it in batch]

        # Generate embeddings for the batch of texts
        vectors = embed_texts(texts)
        # The output vector is List[List[float]]

        # Build Pinecone vector objects
        # Format accepted by pinecone: list of dicts or tuples (id, values, metadata)
        pinecone_vectors_to_upsert = []
        
        for j in range(len(ids)):
            pinecone_vectors_to_upsert.append(
                (ids[j], vectors[j], metadata[j])
            )

        index.upsert(vectors=pinecone_vectors_to_upsert, batch_size=BATCH_SIZE)
    
    print(f"\nUpsert complete. Total vectors indexed: {index.describe_index_stats()['total_vector_count']}")


if __name__ == "__main__":
    process_and_upsert_filing()