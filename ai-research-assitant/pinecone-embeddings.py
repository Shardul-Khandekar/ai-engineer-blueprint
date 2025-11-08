import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore

# Load environment variables from a .env file
load_dotenv()

# Load openAI and Pinecone API keys from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "ai-assistant"

# Load document
loader = PyPDFLoader("sample.pdf")
documents = loader.load()

if not documents:
    raise ValueError("No documents found in the specified PDF file")

print(f"Loaded {len(documents)} document(s) from the PDF")

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Maximum character size of each chunk
    chunk_size=1000, 
    # Overlap between chunks
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

if not chunks:
    raise ValueError("No chunks were created from the documents")

print(f"Split document into {len(chunks)} chunks")

# Create embedding model
embeddings_model = OpenAIEmbeddings(
    openai_api_key = openai_api_key,
    model="text-embedding-ada-002"
)

# Create Pinecone client
pc = Pinecone(api_key=pinecone_api_key)
index = pc.Index(index_name)

# Create Pinecone vector store from document chunks
vectorstore = PineconeVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings_model,
    index_name=index_name,
)

print("Pinecone vector store created successfully from document chunks")
stats = index.describe_index_stats()
print(f"Index stats: {stats}")