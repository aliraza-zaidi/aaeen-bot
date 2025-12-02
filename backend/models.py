from pydantic import BaseModel
from typing import List, Optional

class Query(BaseModel):
    question: str
    top_k: Optional[int] = 3
    max_tokens: Optional[int] = 500

class Source(BaseModel):
    chunk_id: int
    content: str
    distance: float

class Response(BaseModel):
    question: str
    answer: str
    sources: List[Source]