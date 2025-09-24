from typing import List, Dict
from ..config.settings import get_settings
from ..embedding.client import embed_texts
import requests, time, math

settings = get_settings()

_CB_STATE = {
    'fails': 0,
    'opened_until': 0.0,
}
_RERANK_CACHE: dict[str, tuple[float, List[float]]] = {}

def _cache_key(question: str, cands: List[Dict]) -> str:
    ids = '|'.join(c.get('id','') for c in cands)
    return f"{hash(question + ids)}"

def _cosine(a: List[float], b: List[float]) -> float:
    if not a or not b:
        return 0.0
    s = sum(x*y for x,y in zip(a,b))
    na = math.sqrt(sum(x*x for x in a)) or 1.0
    nb = math.sqrt(sum(y*y for y in b)) or 1.0
    return s / (na * nb)

def _remote_rerank(question: str, cands: List[Dict]) -> List[float]:
    """尝试调用外部 rerank 服务。
    期望接口：POST JSON {"query": str, "candidates": [str]} -> {"scores": [float]}
    返回与 cands 等长的 score 列表，失败抛异常。
    """
    if not settings.rerank_endpoint:
        raise RuntimeError("no endpoint")
    now = time.time()
    # 熔断检查
    if _CB_STATE['opened_until'] > now:
        raise RuntimeError("circuit_open")
    ttl = settings.rerank_cache_ttl
    ck = _cache_key(question, cands)
    if ttl > 0:
        cached = _RERANK_CACHE.get(ck)
        if cached and (now - cached[0]) < ttl:
            return cached[1]
    payload = {"query": question, "candidates": [c.get('text','')[:800] for c in cands]}
    try:
        resp = requests.post(settings.rerank_endpoint, json=payload, timeout=settings.rerank_timeout)
        resp.raise_for_status()
    except Exception as e:
        _CB_STATE['fails'] += 1
        if _CB_STATE['fails'] >= settings.rerank_cb_fails:
            _CB_STATE['opened_until'] = now + settings.rerank_cb_cooldown
        raise RuntimeError(f"remote_error:{e}")
    data = resp.json()
    scores = data.get('scores')
    if not isinstance(scores, list) or len(scores) != len(cands):
        raise RuntimeError("invalid rerank response shape")
    mn, mx = min(scores), max(scores)
    span = (mx - mn) or 1.0
    norm = [(s - mn)/span for s in scores]
    _CB_STATE['fails'] = 0
    _CB_STATE['opened_until'] = 0.0
    if ttl > 0:
        _RERANK_CACHE[ck] = (now, norm)
    return norm

def rerank_post_score(question: str, candidates: List[Dict]) -> List[Dict]:
    """Lightweight rerank implementation.

    当前无专用 cross-encoder，采用折衷策略：
      1. 为每个候选构造 pseudo pair 文本: "Q: {question}\nA: {chunk_preview}"（截断 chunk）
      2. 使用 embedding 模型对 [question] + 所有 pair 文本做一次批量向量化
      3. 计算 question 向量与 pair 向量余弦相似度作为 rerank_score（0~1）
      4. 与原 composite_score 融合: final = alpha * composite + (1-alpha) * rerank_score

    局限：
      - 语义交互不足，不能建模精细词级对齐
      - 仍偏向原 embedding 语义空间
    后续可替换：
      - 本地 cross-encoder (e.g., bge-reranker / m3e-rerank)
      - LLM judge (成本文更高)
    """
    if not settings.rerank_enabled or not candidates:
        return candidates
    alpha = max(0.0, min(1.0, settings.rerank_alpha))
    # 限制 rerank 计算对象数量，避免长尾开销
    limit = min(len(candidates), settings.rerank_top_n)
    slice_cands = candidates[:limit]
    remote_ok = False
    if settings.rerank_endpoint:
        try:
            remote_scores = _remote_rerank(question, slice_cands)
            for c, sc in zip(slice_cands, remote_scores):
                c['rerank_score'] = round(sc, 4)
            remote_ok = True
        except Exception:
            remote_ok = False
    if not remote_ok:
        # fallback 余弦近似方案
        pair_texts = [f"Q: {question}\nA: {c.get('text','')[:380]}" for c in slice_cands]
        try:
            embeds = embed_texts([question] + pair_texts)
            q_vec = embeds[0]
            pair_vecs = embeds[1:]
            for c, pv in zip(slice_cands, pair_vecs):
                sim = max(0.0, _cosine(q_vec, pv))
                sim01 = (sim + 1.0)/2.0
                c['rerank_score'] = round(sim01, 4)
        except Exception:
            for c in slice_cands:
                base = c.get('composite_score', 0.0)
                c['rerank_score'] = base
    # 未进入 slice 的尾部候选使用其 composite 作为 rerank_score
    for c in candidates[limit:]:
        c['rerank_score'] = c.get('composite_score', 0.0)
    # 融合
    for c in candidates:
        base = c.get('composite_score', 0.0)
        rr = c.get('rerank_score', base)
        c['final_score'] = round(alpha * base + (1 - alpha) * rr, 4)
    candidates.sort(key=lambda x: x.get('final_score', x.get('composite_score', 0)), reverse=True)
    return candidates
