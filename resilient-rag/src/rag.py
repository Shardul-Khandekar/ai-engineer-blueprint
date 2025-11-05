from langchain_openai import ChatOpenAI
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from src.config import Config
from src.vector_store import VectorDBManager


class RAGEngine:

    def __init__(self):
        self.config = Config()
        self.db_manager = VectorDBManager()

        # Setup LLM
        self.llm = ChatOpenAI(
            model=self.config.LLM_MODEL,
            temperature=self.config.TEMPERATURE,
            api_key=self.config.OPENAI_API_KEY
        )

        # Setup Retriever
        self.retriever = self.db_manager.get_vector_store().as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3}
        )

        # Build RAG Chain
        self.chain = self._build_chain()

    def _build_chain(self):

        prompt = ChatPromptTemplate.from_template("""
        You are a helpful assistant. Answer the user's question based ONLY on the context provided below.
        If the answer is not in the context, say "I don't know" and do not make up facts.

        <context>
        {context}
        </context>

        Question: {input}
        """)

        # It takes a list of documents and embeds them into the prompt template in the context section.
        document_chain = create_stuff_documents_chain(self.llm, prompt)
        # the below functions by taking input, passing it to the retriever to get relevant documents,
        # and then passing those documents to the document_chain to generate a final answer.
        retrieval_chain = create_retrieval_chain(
            self.retriever, document_chain)

        return retrieval_chain

    def query(self, user_input: str):
        """Runs the query and returns a structured response"""
        print(f"Processing query: '{user_input}'")

        response = self.chain.invoke({"input": user_input})

        # Return response and source documents
        return {
            "answer": response["answer"],
            "sources": [doc.page_content for doc in response["context"]],
            "source_metadata": [doc.metadata for doc in response["context"]]
        }
