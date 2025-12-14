import os
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.llms import OpenAI
from langchain_community.document_compressors import FlashrankRerank
from dotenv import load_dotenv

# Configuration
CHROMA_DB_DIR = "resume_chroma_db"
COLLECTION_NAME = "resume_collection"
INITIAL_RETRIEVAL_K = 10
RERANKED_K = 3

# Load environment variables
load_dotenv()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

if not OPENAI_API_KEY:
    raise ValueError("OPENAI_API_KEY is not set in environment variables")


def setup_reranking_retriever():
    """
    Sets up a LangChain ContextualCompressionRetriever using Chroma as the
    base retriever and FlashRank as the reranker (compressor).
    """

    # Use the same embedding model as used during indexing
    embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

    # Load the existing Chroma vector store
    vector_db = Chroma(
        persist_directory=CHROMA_DB_DIR,
        embedding_function=embeddings,
        collection_name=COLLECTION_NAME
    )

    # Base retriever from Chroma
    base_retriever = vector_db.as_retriever(
        search_kwargs={"k": INITIAL_RETRIEVAL_K}
    )

    # Initialize BGE reranker model
    reranker = FlashrankRerank(
        model="ms-marco-MiniLM-L-12-v2",
        top_n=RERANKED_K
    )

    def reranking_retriever(query: str):
        """
        Performs retrieval with reranking for the given query.
        """
        initial_docs = vector_db.similarity_search(query, k=INITIAL_RETRIEVAL_K)
        reranked_docs = reranker.compress_documents(query=query, documents=initial_docs)
        return reranked_docs

    return reranking_retriever
