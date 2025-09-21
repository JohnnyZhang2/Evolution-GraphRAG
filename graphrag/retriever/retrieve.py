# Evolution RAG Retrieval Logic
# Composite hybrid retrieval: vector, BM25 (optional), graph rank, relation weights,
# and optional rerank placeholder.
# Copyright (c) 2025 EvolutionAI Studio
# All Rights Holder: Johnny Zhang
# License: MIT (see LICENSE)

from neo4j import GraphDatabase
from typing import List, Dict, Tuple
import logging
from ..config.settings import get_settings
from ..utils.entity_normalize import normalize_entity, load_synonyms
from collections import OrderedDict
from ..embedding.client import embed_texts
from ..llm.client import chat_completion
from .bm25 import bm25_index
from .rerank import rerank_post_score
from ..utils.graph_rank import get_degree_scores

settings = get_settings()

# ------------- 缓存实现（参数化） -------------
EMBED_CACHE_MAX = settings.embed_cache_max
ANSWER_CACHE_MAX = settings.answer_cache_max

_question_embedding_cache: OrderedDict[str, List[float]] = OrderedDict()
_answer_cache: OrderedDict[str, Dict] = OrderedDict()  # key -> {"text": str, "sources": [...]}

def _lru_get(cache: OrderedDict, key: str):
    if key in cache:
        val = cache.pop(key)
        cache[key] = val
        return val
    return None

def _lru_put(cache: OrderedDict, key: str, value, max_size: int):
    if key in cache:
        cache.pop(key)
    cache[key] = value
    if len(cache) > max_size:
        cache.popitem(last=False)

def get_driver():
    return GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))

VECTOR_SEARCH = f"""
CALL db.index.vector.queryNodes('{settings.vector_index_name}', $topK, $queryEmbedding) YIELD node, score
RETURN node.id AS id, node.text AS text, score
"""

FALLBACK_SEARCH = """
// 当向量索引不可用时的降级：按最近导入顺序或任意顺序取前 topK
MATCH (c:Chunk)
RETURN c.id AS id, c.text AS text
LIMIT $topK
"""

EXPAND_ENTITIES = """
UNWIND $chunkIds AS cid
MATCH (c:Chunk {id: cid})-[:HAS_ENTITY]->(e:Entity)
OPTIONAL MATCH (e)<-[:HAS_ENTITY]-(other:Chunk)
WITH c, e, collect(DISTINCT other) AS others
UNWIND others AS oc
RETURN DISTINCT oc.id AS id, oc.text AS text, 'entity' AS reason
LIMIT $limit
"""

EXPAND_RELATES = """
UNWIND $chunkIds AS cid
MATCH (c:Chunk {id: cid})-[:RELATES_TO]-(o:Chunk)
RETURN DISTINCT o.id AS id, o.text AS text, 'relates' AS reason
LIMIT $limit
"""

EXPAND_LLM_REL = """
UNWIND $chunkIds AS cid
MATCH (c:Chunk {id: cid})-[r:REL]->(o:Chunk)
RETURN DISTINCT o.id AS id, o.text AS text, 'llm_rel' AS reason, r.type AS rel_type, r.confidence AS rel_conf
LIMIT $limit
"""

EXPAND_COOC_ENTITIES = """
UNWIND $chunkIds AS cid
MATCH (c:Chunk {id: cid})-[:HAS_ENTITY]->(e:Entity)-[:CO_OCCURS_WITH]-(e2:Entity)<-[:HAS_ENTITY]-(oc:Chunk)
RETURN DISTINCT oc.id AS id, oc.text AS text, 'cooccur' AS reason
LIMIT $limit
"""

# 若需要多跳，可扩展 Cypher，这里简化 1 hop

def _normalize_query_text(query: str) -> str:
    """若启用实体标准化，对问句中的可能同义词做规范化或追加同义形式（简单启发实现）。
    策略：
    - 加载 synonyms 映射（alt->canonical 或 canonical->... 已在 load_synonyms 展开）
    - 对出现的 alt 词条做替换为 canonical；若 canonical 与原不同，保留原词一次（通过追加括号）以不损伤语义。
    - 为避免中文/英文混写破坏，替换采用分词降级策略：按空白与常见标点粗分。
    """
    if not getattr(settings, 'entity_normalize_enabled', False):
        return query
    mapping = load_synonyms()
    if not mapping:
        return query
    import re
    tokens = re.split(r"(\s+|[,.!?;，。！？；（）()\[\]])", query)
    rebuilt = []
    for t in tokens:
        low = t.lower()
        if low in mapping:
            cano = mapping[low]
            if cano != t:
                rebuilt.append(cano)
                # 若原词重要可保留： rebuilt.append(f"({t})") 目前简化不保留
            else:
                rebuilt.append(t)
        else:
            rebuilt.append(t)
    return ''.join(rebuilt)


def vector_search(query: str, top_k: int) -> Tuple[List[Dict], List[float], bool, str]:
    """执行向量检索，失败时降级为普通 LIMIT 检索。

    Returns:
        hits: 检索到的文档列表
        embedding: 查询向量（若失败则空列表）
        degraded: 是否发生了降级
        warn: 降级原因描述（无降级则为空字符串）
    """
    # 预处理查询（实体规范化）
    norm_query = _normalize_query_text(query)
    embedding = []
    degraded = False
    warn = ""
    try:
        key = query.strip().lower()
        cached = _lru_get(_question_embedding_cache, key)
        if cached is not None:
            embedding = cached
        else:
            embedding = embed_texts([norm_query])[0]
            _lru_put(_question_embedding_cache, key, embedding, EMBED_CACHE_MAX)
    except Exception as e:
        degraded = True
        warn = f"embedding_failed: {e}"
        logging.warning(warn)
    with get_driver().session() as session:
        if not degraded:
            try:
                records = session.run(VECTOR_SEARCH, topK=top_k, queryEmbedding=embedding)
                hits = [{"id": r["id"], "text": r["text"], "score": r["score"]} for r in records]
                if not hits:
                    # 可能索引存在但无结果，保持不降级（允许空返回）
                    return hits, embedding, False, ""
                return hits, embedding, False, ""
            except Exception as e:
                degraded = True
                warn = f"vector_search_failed: {e}"[:300]
                logging.warning(warn)
        # 降级路径
        try:
            records = session.run(FALLBACK_SEARCH, topK=top_k)
            hits = [{"id": r["id"], "text": r["text"], "score": None} for r in records]
        except Exception as e2:
            # fallback 也失败，抛出最终异常
            raise RuntimeError(f"fallback_search_failed: {e2}; prior={warn}")
    return hits, embedding, degraded, warn


def expand_context(chunk_ids: List[str], limit: int = 20, hops: int = 1):
    if not chunk_ids:
        return []
    collected = {}
    with get_driver().session() as session:
        # hop1 实体扩展
        records = session.run(EXPAND_ENTITIES, chunkIds=chunk_ids, limit=limit)
        for r in records:
            collected.setdefault(r["id"], {"id": r["id"], "text": r["text"], "reason": r["reason"]})
        if hops > 1:
            # 关系扩展
            records2 = session.run(EXPAND_RELATES, chunkIds=chunk_ids, limit=limit)
            for r in records2:
                collected.setdefault(r["id"], {"id": r["id"], "text": r["text"], "reason": r["reason"]})
            # LLM 关系扩展
            records_rel = session.run(EXPAND_LLM_REL, chunkIds=chunk_ids, limit=limit)
            for r in records_rel:
                collected.setdefault(r["id"], {"id": r["id"], "text": r["text"], "reason": r["reason"], "rel_type": r.get("rel_type"), "rel_conf": r.get("rel_conf")})
            # 共现实体引出的二跳 chunk
            records3 = session.run(EXPAND_COOC_ENTITIES, chunkIds=chunk_ids, limit=limit)
            for r in records3:
                collected.setdefault(r["id"], {"id": r["id"], "text": r["text"], "reason": r["reason"]})
    return list(collected.values())


def build_prompt(question: str, contexts: List[Dict]) -> list:
    # 用 [S1] [S2] 形式标记，使回答可引用
    context_text_lines = []
    for idx, c in enumerate(contexts, 1):
        tag = f"[S{idx}]"
        context_text_lines.append(f"{tag} {c['text']}")
    context_text = '\n\n'.join(context_text_lines)
    system = (
        "你是一个知识图谱增强的智能问答助手。必须仅依据提供的上下文回答，不得编造。"
        "在引用具体事实、数据、时长或文件段落时，紧跟相应引用标记如 [S1]、[S2]。"
        "如果多个来源支持同一事实，可合并如 [S1][S3]。无法回答时明确说明。保持用户语言。"
    )
    user = f"问题: {question}\n\n上下文(含编号):\n{context_text}\n\n请给出结构化回答，并恰当添加引用标记。"
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user}
    ]


def answer_question(question: str, return_sources: bool = False, stream: bool = True):
    cache_key = question.strip().lower()
    if not stream:  # 仅对非流式做完整回答缓存（流式需实时）
        cached_answer = _lru_get(_answer_cache, cache_key)
        if cached_answer and return_sources:
            from types import GeneratorType
            def gen_once():
                yield cached_answer["text"]
            return gen_once(), cached_answer["sources"]
        elif cached_answer and not return_sources:
            def gen_once():
                yield cached_answer["text"]
            return gen_once()

    hits, _, degraded, warn = vector_search(question, settings.top_k)
    # --- Hybrid: BM25 初选融合（仅在非降级且启用时）---
    bm25_scores = {}
    if settings.bm25_enabled and not degraded:
        try:
            # 若索引未构建则构建一次（使用当前所有 hits + 扩展后的文档集合可能更好，此处先用 hits 做局部快速融合）
            # 简化：当索引空时，拉取一定数量 Chunk 构建全局索引
            if not bm25_index._built:  # type: ignore
                # 全量抓取前 2000 条构建（可配置化）
                with get_driver().session() as session:
                    recs = session.run("MATCH (c:Chunk) RETURN c.id AS id, c.text AS text LIMIT 2000")
                    docs = [{"id": r["id"], "text": r["text"]} for r in recs]
                bm25_index.build(docs)
            bm25_scores = bm25_index.score(question, top_k=settings.top_k * 3)
        except Exception as be:
            logging.warning(f"[BM25 WARN] {be}")
    expand = [] if degraded else expand_context([h['id'] for h in hits], limit=settings.top_k * 2, hops=settings.expand_hops)
    merged = {h['id']: h for h in hits}
    for e in expand:
        merged.setdefault(e['id'], e)
    contexts = list(merged.values())
    # 组合打分：向量分数归一化 + 不同来源权重
    # 规则：
    # vector: 原始 score（越大越相关） -> 先收集最大最小归一化到 [0,1]
    # relates: +0.15, cooccur: +0.1, llm_rel: +0.2 * rel_conf
    vec_scores = [c['score'] for c in contexts if c.get('score') is not None]
    if vec_scores:
        vmin, vmax = min(vec_scores), max(vec_scores)
        span = (vmax - vmin) or 1.0
    else:
        vmin, vmax, span = 0.0, 1.0, 1.0
    degree_scores = get_degree_scores() if settings.graph_rank_enabled else {}
    for c in contexts:
        base = 0.0
        if c.get('score') is not None:
            base = (c['score'] - vmin) / span
        bonus = 0.0
        reason = c.get('reason')
        # BM25 融合：将 bm25 归一得分按权重加入（仅对初始向量命中的文档或所有文档？这里对所有上下文若存在 bm25 分值）
        if settings.bm25_enabled and not degraded:
            bscore = bm25_scores.get(c['id'])
            if bscore is not None:
                bonus += settings.bm25_weight * bscore
        if reason == 'relates':
            bonus += settings.rel_weight_relates
        elif reason == 'cooccur':
            bonus += settings.rel_weight_cooccur
        elif reason == 'llm_rel':
            conf = float(c.get('rel_conf') or 0.5)
            # 关系类型权重映射
            rtype = (c.get('rel_type') or '').upper()
            weight_lookup = {
                'STEP_NEXT': settings.rel_weight_step_next,
                'REFERENCES': settings.rel_weight_references,
                'FOLLOWS': settings.rel_weight_follows,
                'CAUSES': settings.rel_weight_causes,
                'SUPPORTS': settings.rel_weight_supports,
                'PART_OF': settings.rel_weight_part_of,
                'SUBSTEP_OF': settings.rel_weight_substep_of,
                'CONTRASTS': settings.rel_weight_contrasts
            }
            rel_base = weight_lookup.get(rtype, settings.rel_weight_default)
            bonus += rel_base * conf
        # 图中心性加成（degree）
        if settings.graph_rank_enabled:
            dsc = degree_scores.get(c['id'])
            if dsc:
                bonus += settings.graph_rank_weight * dsc
        c['composite_score'] = round(base + bonus, 4)
    # 按 composite_score 降序，然后截取
    contexts.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
    contexts = contexts[: settings.top_k * 3]
    # --- Rerank (post-score) ---
    if settings.rerank_enabled:
        contexts = rerank_post_score(question, contexts)
    messages = build_prompt(question, contexts)
    gen = chat_completion(messages, stream=stream)
    if return_sources:
        sources = []
        for idx, c in enumerate(contexts, 1):
            sources.append({
                "id": c['id'],
                "preview": c['text'][:80].replace('\n', ' ') + ('...' if len(c['text']) > 80 else ''),
                "score": c.get('score'),
                "reason": c.get('reason', 'vector' if c.get('score') is not None else 'context'),
                "composite_score": c.get('composite_score'),
                "final_score": c.get('final_score', c.get('composite_score')),
                "rel_type": c.get('rel_type'),
                "rel_conf": c.get('rel_conf'),
                "rank": idx
            })
        if degraded and warn:
            sources.insert(0, {"id": "__warning__", "preview": f"DEGRADED: {warn}. 请创建/修复向量索引后重试。", "score": None, "reason": "warning", "rank": 0})
        # 若是非流式，收集生成器输出缓存
        if not stream:
            # 预取一次完整文本
            full_text = ''.join(list(gen))
            _lru_put(_answer_cache, cache_key, {"text": full_text, "sources": sources}, ANSWER_CACHE_MAX)
            def replay():
                yield full_text
            return replay(), sources
        return gen, sources
    return gen
