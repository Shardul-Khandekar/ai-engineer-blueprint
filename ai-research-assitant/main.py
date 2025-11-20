from langchain.agents import create_tool_calling_agent, AgentExecutor
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.tools import Tool

from config import llm
from chains import create_rag_chain
from tools import get_search_tool

# Create tools
rag_chain = create_rag_chain()
search_tool = get_search_tool()


# Wrap RAG chain as a tool
rag_tool = Tool(
    name="private_document_search",
    func=rag_chain.invoke,
    description="Use this tool to search for information in the user's private documents. Use it for any questions about what the assignment is and what is covered and how to solve it"
)

# Create a list of tools for the agent to use
tools = [search_tool, rag_tool]

agent_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant. You have access to two tools: a Tavily search tool for web queries and a RAG tool for private documents."),
        MessagesPlaceholder(variable_name="chat_history"),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ]
)

# Create the agent by binding the LLM, tools, and prompt together
agent = create_tool_calling_agent(llm, tools, agent_prompt)

# Create the agent executor
agent_executor = AgentExecutor(
    agent=agent, 
    tools=tools, 
    verbose=True
)

chat_history = []

while True:
    try:
        query = input("\nAsk me anything: ")
        if query.lower() in ["exit", "quit"]:
            break

        result = agent_executor.invoke(
            {
                "input": query,
                "chat_history": chat_history
            }
        )

        print(f"\nAssistant: {result['output']}")
        chat_history.append((query, result['output']))
    
    except Exception as e:
        print(f"An error occurred: {e}")