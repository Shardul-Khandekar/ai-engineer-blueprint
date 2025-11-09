from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from config import llm, embeddings, pinecone_index_name
from tools import get_retriever_tool


def create_rag_chain():
    """
    Creates the complete RAG chain for document retrieval and answering.
    """

    retriever = get_retriever_tool()

    # Define the prompt template
    template = """
    You are an assistant for question-answering tasks. 
    Use the following pieces of retrieved context to answer the question. 
    If you don't know the answer, just say that you don't know. 
    Use three sentences maximum and keep the answer concise.

    Question: {question} 

    Context: {context} 

    Answer:
    """

    prompt = ChatPromptTemplate.from_template(template)

    # Convert retrieved documents into a single string
    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)
    
    rag_chain = (
        {"context": retriever | format_docs, "question": RunnablePassthrough()}
        | prompt
        | llm
        | StrOutputParser()
    )
    
    print("RAG chain created successfully")
    return rag_chain


# Verify RAG chain creation
if __name__ == "__main__":
    rag_chain = create_rag_chain()

    # Test the RAG chain
    test_question = "How to use censys.io perform an IPv4 Hosts Search on the network blocks identified?"
    answer = rag_chain.invoke(test_question)
    print(f"Answer: {answer}")