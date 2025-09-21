import requests
from typing import List
from tenacity import retry, stop_after_attempt, wait_exponential
from ..config.settings import get_settings

settings = get_settings()

HEADERS = {
    "Authorization": f"Bearer {settings.llm_api_key}",
    "Content-Type": "application/json"
}

@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def embed_texts(texts: List[str]) -> List[List[float]]:
    """批量获取文本向量 (OpenAI embeddings 兼容接口)。"""
    url = f"{settings.llm_base_url}/v1/embeddings"
    payload = {
        "model": settings.embedding_model,
        "input": texts
    }
    r = requests.post(url, json=payload, headers=HEADERS, timeout=300)
    try:
        r.raise_for_status()
    except requests.HTTPError as e:
        detail = None
        try:
            detail = r.json()
        except Exception:
            detail = r.text[:500]
        raise RuntimeError(f"Embedding request failed: {e}; detail={detail}") from e
    data = r.json()
    if 'data' not in data:
        raise RuntimeError(f"Embedding response missing 'data' field: {data}")
    vectors = [item.get("embedding") for item in data.get("data", [])]
    if not vectors or any(v is None for v in vectors):
        raise RuntimeError(f"Embedding response contains empty embeddings: {data}")
    return vectors
