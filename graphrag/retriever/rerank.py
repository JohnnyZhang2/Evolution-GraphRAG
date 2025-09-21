from typing import List, Dict
from ..config.settings import get_settings

settings = get_settings()

def rerank_post_score(question: str, candidates: List[Dict]) -> List[Dict]:
    """Post-score rerank stub.
    Adds a placeholder rerank_score (mirror composite_score) and fusion final_score.
    Only active if settings.rerank_enabled is True.
    """
    if not settings.rerank_enabled or not candidates:
        return candidates
    alpha = max(0.0, min(1.0, settings.rerank_alpha))
    for c in candidates:
        base = c.get('composite_score', 0.0)
        # 占位：真实实现应调用 cross-encoder 计算语义匹配分
        c['rerank_score'] = base  # TODO: replace with model score
        c['final_score'] = round(alpha * base + (1 - alpha) * c['rerank_score'], 4)
    # 以 final_score 排序
    candidates.sort(key=lambda x: x.get('final_score', x.get('composite_score', 0)), reverse=True)
    return candidates
