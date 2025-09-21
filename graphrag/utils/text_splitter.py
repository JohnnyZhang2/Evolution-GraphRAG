from typing import List
import re
from ..config.settings import get_settings

settings = get_settings()

def clean_text(text: str) -> str:
    text = text.replace('\u3000', ' ')
    text = re.sub(r"[ \t]+", " ", text)
    return text.strip()

def split_text(text: str, chunk_size: int = None, overlap: int = None) -> List[str]:
    if chunk_size is None:
        chunk_size = settings.chunk_size
    if overlap is None:
        overlap = settings.chunk_overlap

    text = clean_text(text)
    if len(text) <= chunk_size:
        return [text]

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        # 尽量在句号或换行处截断
        if end < len(text):
            match = re.search(r'[。.!?]\s', text[start:end][::-1])
            if match:
                cut = len(text[start:end]) - match.start()
                if cut > chunk_size * 0.3:  # 避免太短
                    chunk = text[start:start+cut]
                    end = start + cut
        chunks.append(chunk)
        start = max(end - overlap, end) if overlap < chunk_size else end
    return [c.strip() for c in chunks if c.strip()]
