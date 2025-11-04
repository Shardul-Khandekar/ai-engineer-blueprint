import os
from dotenv import load_dotenv

# Load environment variables from a .env file
load_dotenv()

# Read keys from environment
openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")

# Check if the keys are loaded correctly
if  openai_api_key and pinecone_api_key:
    print("OpenAI and Pinecone API keys loaded successfully")
else:
    raise EnvironmentError("API keys are not set in the environment variables.")