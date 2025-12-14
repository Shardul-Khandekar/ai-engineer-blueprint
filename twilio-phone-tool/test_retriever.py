import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from dotenv import load_dotenv

# Configuration
CHROMA_DB_DIR = "resume_chroma_db"
COLLECTION_NAME = "resume_collection"

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables")


def test_retrieval(query: str):
    """
    Loads the persisted Chroma DB and performs a similarity search.
    """
    print(f"--- Loading Embeddings and Chroma DB ---")

    # Use the same embedding model as used during indexing
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Load the existing Chroma vector store
    vector_db = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    print(f"--- Performing Similarity Search for query: '{query}' ---")
    retrieved_docs = vector_db.similarity_search(query, k=3)
    print(f"\n Retrieval Successful! Found {len(retrieved_docs)} relevant chunks\n")

    # Print the retrieved documents
    for i, doc in enumerate(retrieved_docs):
        # Extract the source and page number from the metadata
        source = doc.metadata.get('source', 'N/A')
        page = doc.metadata.get('page', 'N/A')

        print("-" * 50)
        print(f"** Chunk {i+1} **")
        print(f"Source: {os.path.basename(source)} (Page {page})")
        print(f"Relevance Score (Internal to Chroma): {doc.metadata.get('_score', 'N/A')}")
        print("Content Snippet (First 300 chars):")
        print(f"'{doc.page_content[:300]}...'")
        print("-" * 50)

if __name__ == "__main__":

    test_queries = [
        "What was the candidate's biggest accomplishment at their last job?",
        "Where did Shardul Khandekar go to school and what degree did he get?",
    ]

    for q in test_queries:
        test_retrieval(q)
        print("\n" + "="*80 + "\n")
