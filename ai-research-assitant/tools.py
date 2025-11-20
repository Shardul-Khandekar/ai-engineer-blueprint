from langchain_pinecone import PineconeVectorStore
from langchain_community.tools.tavily_search.tool import TavilySearchResults
from config import embeddings, pinecone_index_name, tavily_api_key


def get_retriever_tool():
    """
        Creates and returns a retriever for the existing Pinecone index
    """
    vectorstore = PineconeVectorStore(
        index_name=pinecone_index_name,
        embedding=embeddings
    )

    # Create the retriever
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})

    return retriever


def get_search_tool():
    """
        Creates and returns a Tavily search tool
    """
    search_tool = TavilySearchResults(
        api_key=tavily_api_key)
    
    return search_tool


# Verify tools initialization
if __name__ == "__main__":

    retriever = get_retriever_tool()
    search_tool = get_search_tool()

    # # Test retriever
    # test_query = "censys.io"
    # retrieved_docs = retriever.invoke(test_query)
    # print(f"Retrieved: {retrieved_docs[0].page_content}")

    # Test the search tool
    search_query = "What is the weather in New York City?"
    search_result = search_tool.invoke(search_query)
    print(f"Result: {search_result}")