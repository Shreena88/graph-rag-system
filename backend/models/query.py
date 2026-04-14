from pydantic import BaseModel, Field
from typing import List, Optional, Literal


class QueryRequest(BaseModel):
    question: str
    doc_ids: Optional[List[str]] = None
    max_hops: int = Field(default=3, ge=1, le=5)
    top_k: int = Field(default=10, ge=1, le=50)
    stream: bool = True


class EntityRef(BaseModel):
    name: str
    type: str
    confidence: float = 1.0


class ParsedQuery(BaseModel):
    original: str
    intent: Literal["factual", "comparative", "causal", "procedural"]
    entities: List[EntityRef]
    keywords: List[str]
    sub_questions: List[str]


class SourceChunk(BaseModel):
    chunk_id: str
    text: str
    score: float
    doc_id: str
    page_number: int


class GraphStep(BaseModel):
    entity: str
    relation: str


class QueryResponse(BaseModel):
    answer: str
    sources: List[SourceChunk]
    reasoning_path: List[GraphStep]
    confidence: float
