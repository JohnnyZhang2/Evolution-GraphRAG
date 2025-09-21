from pydantic import BaseModel
from typing import Optional

class IngestRequest(BaseModel):
    path: str

class QueryRequest(BaseModel):
    question: str
    stream: bool = True

class QueryChunk(BaseModel):
    id: str
    text: str
    score: Optional[float] = None

