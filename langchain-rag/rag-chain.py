import os
import gradio as gr
from dotenv import load_dotenv
from operator import itemgetter

from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_pinecone import PineconeVectorStore

from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

from langchain_core.messages import HumanMessage, AIMessage

load_dotenv()

openai_api_key = os.getenv("OPENAI_API_KEY")
pinecone_api_key = os.getenv("PINECONE_API_KEY")
index_name = "langchain-rag-index-v1"

# Initialize the LLM
llm = ChatOpenAI(openai_api_key=openai_api_key, model="gpt-3.5-turbo", temperature=0)

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

prompt = ChatPromptTemplate.from_messages(
    [
        ("system", template),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{question}"),
    ]
)

# Formats the retrieved documents into a single string
def format_docs(docs):
    return "\n\n".join([doc.page_content for doc in docs])

# Convert Gradio history into LangChain messages
def format_history(chat_history):
    history_messages = []
    for message in chat_history:
        history_messages.append(HumanMessage(content=message[0]))
        history_messages.append(AIMessage(content=message[1]))
    return history_messages

# Create RAG chain
rag_chain = (
    {
        "context": itemgetter("question") | retriever | format_docs, 
        "question": itemgetter("question"),
        "chat_history": itemgetter("chat_history")
    } 
    | prompt 
    | llm 
    | StrOutputParser()
)

# gr.ChatInterface will call this function on every new message
def get_response(message, history):

    formatted_history = format_history(history)

    response = rag_chain.invoke({
        "question": message,
        "chat_history": formatted_history
    })

    return response


# question = "What is this document about?"
# answer = rag_chain.invoke(question)
# print(answer)

print("Launching Gradio interface")

iface = gr.ChatInterface(
    # fn defines the function to be called when a user submits a message
    fn=get_response,
    title="LangChain RAG Chat",
    description="A simple RAG chatbot using LangChain, Pinecone, and OpenAI.",
    theme="soft"
)

iface.launch()