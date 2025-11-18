from contextlib import asynccontextmanager
from fastapi import FastAPI, HTTPException
from src.models import QueryRequest, QueryResponse, HealthStatus
from src.rag import RAGEngine
from src.vector_store import VectorDBManager
from src.ingestion import IngestionPipeline


ml_models = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """
    Code before yeield runs on startup.
    Code after yield runs on shutdown.
    """

    print("Starting application")
    ml_models["rag_engine"] = RAGEngine()
    ml_models["db_manager"] = VectorDBManager()

    print("Models loaded and ready")

    yield

    print("Shutting down application")

    ml_models.clear()


# Initialize FastAPI app with lifespan
app = FastAPI(title="Resilient RAG API", version="1.0", lifespan=lifespan)


@app.post("/chat", response_model=QueryResponse)
async def chat_endpoint(request: QueryRequest):
    """ Endpoint to handle user queries and return answers with sources """
    try:
        rag_engine = ml_models["rag_engine"]
        result = rag_engine.query(request.query)
        return QueryResponse(
            answer=result["answer"],
            sources=result["sources"]
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
