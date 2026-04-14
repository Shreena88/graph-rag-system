from pydantic import BaseModel
from typing import List, Optional, Literal
from datetime import datetime


class PageContent(BaseModel):
    page_number: int
    text: str
    tables: List[dict] = []


class ParsedDocument(BaseModel):
    pages: List[PageContent]
    filename: str
    total_pages: int


class DocumentResult(BaseModel):
    doc_id: str
    filename: str
    status: Literal["queued", "processing", "indexed", "failed"]
    chunk_count: int = 0
    entity_count: int = 0
    created_at: datetime = datetime.utcnow()
    error: Optional[str] = None
