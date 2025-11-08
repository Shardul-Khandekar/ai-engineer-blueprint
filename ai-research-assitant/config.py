import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Load environment variables from .env file
load_dotenv()

# Initialize openai_api_key
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
tavily_api_key = os.getenv("TAVILY_API_KEY")

# Pinecone index
pinecone_index_name = "ai-assistant"

# Initialize the LLM
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0)

# Initialize the embeddings model
embeddings = OpenAIEmbeddings(openai_api_key=openai_api_key, model="text-embedding-3-small")

print("LLM and Embeddings model initialized successfully")