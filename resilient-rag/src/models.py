from pydantic import BaseModel, Field
from typing import List, Dict, Optional


class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's question", min_length=3)


class Source(BaseModel):
    content: str
    metadata: Dict


class QueryResponse(BaseModel):
    answer: str
    sources: List[str]
    confidence_score: Optional[float] = None


class HealthStatus(BaseModel):
    status: str
    pinecone: str
    openai: str
