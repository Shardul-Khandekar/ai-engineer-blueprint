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

# Split document into chunks
text_splitter = RecursiveCharacterTextSplitter(
    # Maximum size of each chunk
    chunk_size=1000, 
    # Overlap between chunks
    chunk_overlap=200
)

chunks = text_splitter.split_documents(documents)

if not chunks:
    raise ValueError("No chunks were created from the documents")

print(f"Split document into {len(chunks)} chunks")


# Create embeddings for one of the chunks
embeddings_model = OpenAIEmbeddings(openai_api_key = openai_api_key)

first_chunk_text = chunks[0].page_content
print(f"Creating embedding for the first chunk: {first_chunk_text[:50]}...")

first_chunk_embedding = embeddings_model.embed_query(first_chunk_text)
# print(f"Embedding for the first chunk: {first_chunk_embedding}")