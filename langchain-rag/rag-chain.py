import os
from dotenv import load_dotenv

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "langchain-rag-index-v1"

# Initialize the LLM
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo")

# Use the same embeddings model as in pinecone-embeddings.py
embeddings_model = OpenAIEmbeddings(
    openai_api_key=openai_api_key, 
    model="text-embedding-ada-002"
)

# Connect and load the Pinecone vector store
vectorstore = PineconeVectorStore.from_existing_index(
    index_name=index_name,
    embedding=embeddings_model
)

# Create a retriever from the vector store with top 3 most relevant documents
retriever = vectorstore.as_retriever(search_kwargs={"k": 3})


template = """
You are an assistant for question-answering tasks. 
Use the following pieces of retrieved context to answer the question. 
If you don't know the answer, just say that you don't know. 
Be concise.

Context: {context} 

Question: {question} 

Helpful Answer:
"""

prompt = ChatPromptTemplate.from_template(template)


# Create RAG chain
rag_chain = (
    {"context": retriever, "question": RunnablePassthrough()} 
    | prompt 
    | llm 
    | StrOutputParser()
)

question = "What is this document about?"
answer = rag_chain.invoke(question)
print(answer)