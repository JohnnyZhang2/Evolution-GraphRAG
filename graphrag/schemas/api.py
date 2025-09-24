from pydantic import BaseModel
from typing import Optional, List, Union, Literal

class IngestRequest(BaseModel):
    path: str

class QueryRequest(BaseModel):
    question: str
    stream: bool = True
    # 额外外部上下文，允许纯文本或带 id 的结构
    context: Optional[List[Union[str, "QueryChunk"]]] = None
    # 会话历史，按顺序给出
    history: Optional[List["ChatMessage"]] = None

class QueryChunk(BaseModel):
    id: str
    text: str
    score: Optional[float] = None


class ChatMessage(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str

