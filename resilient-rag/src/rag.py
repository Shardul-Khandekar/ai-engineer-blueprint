# src/rag.py
from typing import List
from operator import itemgetter

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_pinecone import PineconeVectorStore

from src.config import Config
from src.vector_store import VectorDBManager


def _join_docs(docs) -> str:
    return "\n\n".join(d.page_content for d in docs)


class RAGEngine:
    def __init__(self):
        self.config = Config()
        self.db_manager = VectorDBManager()

        # LLM
        self.llm = ChatOpenAI(
            model=self.config.LLM_MODEL,
            temperature=self.config.TEMPERATURE,
            api_key=self.config.OPENAI_API_KEY,
        )

        # Retriever
        self.retriever = self.db_manager.get_vector_store().as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

        # Prompt & chain (Runnable/LCEL)
        self.prompt = ChatPromptTemplate.from_template(
            """You are a helpful assistant. Answer the user's question based ONLY on the context provided.
If the answer is not in the context, say "I don't know" and do not make up facts.

<context>
{context}
</context>

Question: {question}"""
        )

        # Build: retrieve -> format -> prompt -> llm -> text
        self.chain = (
            {
                "context": self.retriever | _join_docs,   # docs -> joined context
                "question": RunnablePassthrough(),        # pass the user input through
            }
            | self.prompt
            | self.llm
            | StrOutputParser()
        )

    def query(self, user_input: str) -> dict:
        """
        Run retrieval and generation. Returns fields compatible with RAGAS:
        - answer: str
        - contexts: List[str] (just the page contents)
        - source_metadata: List[dict]
        """
        # Get docs explicitly so we can expose them
        docs = self.retriever.invoke(user_input)
        answer_text = self.chain.invoke(user_input)

        return {
            "answer": answer_text,
            "contexts": [d.page_content for d in docs],
            "source_metadata": [d.metadata for d in docs],
        }
