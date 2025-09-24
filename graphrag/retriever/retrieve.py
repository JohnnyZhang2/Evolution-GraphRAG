# Evolution RAG Retrieval Logic
# Composite hybrid retrieval: vector, BM25 (optional), graph rank, relation weights,
# and optional rerank placeholder.
# Copyright (c) 2025 EvolutionAI Studio
# All Rights Holder: Johnny Zhang
# License: MIT (see LICENSE)

from neo4j import GraphDatabase
from typing import List, Dict, Tuple, Any, Optional
import logging
from ..config.settings import get_settings
from ..utils.entity_normalize import normalize_entity, load_synonyms
from collections import OrderedDict
from ..embedding.client import embed_texts
from ..llm.client import chat_completion
from .bm25 import bm25_index
from .rerank import rerank_post_score
from ..utils.graph_rank import get_degree_scores

# 调试模式：可通过环境变量 DEBUG_RETRIEVAL=true 打开更详细日志
import os
DEBUG_RETRIEVAL = os.getenv("DEBUG_RETRIEVAL", "false").lower() in ("1","true","yes")

# 导出最近一次子图统计（调试用）
_last_subgraph_stats = {}

__all__ = [
    'answer_question','debug_retrieve','_extract_subgraph','_last_subgraph_stats'
]

# ---------------- 针对集合/成员类问题的快速路径 -----------------
import re

MEMBERSHIP_PATTERNS = [
    re.compile(r"(.+?)(?:里|中|內|内)?有谁[?？]*$"),
    re.compile(r"(.+?)(?:的)?成员都有谁[?？]*$"),
    re.compile(r"(.+?)(?:有哪些|有哪几|有哪)人[?？]*$"),
]

# 简单问句类型分类：用于后续自适应图策略（若 ADAPTIVE_QUERY_STRATEGY=true 则启用）
# 返回标签之一: membership, list, relation, cause, definition, event, generic
_Q_LIST = re.compile(r"(有哪些|哪几|列表|清单|盘点|都有谁)")
_Q_REL = re.compile(r"(关系|联系|关联|师徒|伴侣|道侣|妻子|丈夫|徒弟|师父|双修|婚|敌对|阵营)")  # 增补: 联系/关联
_Q_CAUSE = re.compile(r"(为什么|为何|原因|怎么会|如何导致|怎样导致|怎么导致)")
_Q_DEFINE = re.compile(r"(是什么|是什麼|指的是|含义|含義|意思|定义是什?么|定义是什么|如何定义|怎样定义|怎么定义|被称为|称为|属于什么|属于哪类|本质是什么)")
_Q_EVENT = re.compile(r"(经过|过程|发生了什么|经历了什么|如何发展|发展历程)")

def _classify_query(q: str) -> str:
    qs = q.strip()
    # 先做 membership
    if _detect_membership_question(qs):
        return 'membership'
    # 归一化简单变体
    qs_n = qs.lower().replace('？','?')
    # definition 特殊增强："X 的定义是什么" / "什么是 X"
    # “什么是X” / “X的定义是什么” 直接判定定义类
    if re.search(r"^[‘'\"\s]*什么是.+", qs_n) or re.search(r".+的定义是什?么[?？]*$", qs_n):
        return 'definition'
    if _Q_REL.search(qs):
        return 'relation'
    if _Q_CAUSE.search(qs):
        return 'cause'
    if _Q_DEFINE.search(qs):
        return 'definition'
    if _Q_EVENT.search(qs):
        return 'event'
    if _Q_LIST.search(qs):
        return 'list'
    return 'generic'

def _detect_membership_question(q: str) -> str | None:
    q_strip = q.strip().replace('\n','')
    # 过滤过短提问
    if len(q_strip) < 2:
        return None
    for pat in MEMBERSHIP_PATTERNS:
        m = pat.match(q_strip)
        if m:
            group = m.group(1).strip()
            # 去掉常见前缀/疑问词
            group = re.sub(r"^(请问|问一下|想知道|那个|这?个)", "", group)
            group = group.strip('的 ')
            if 1 < len(group) <= 40:
                return group
    return None

def _answer_membership(group: str, question: str, return_sources: bool, stream: bool):
    """直接基于图/文本查询某组织/地点/势力包含的成员实体（无需重新 ingest）。

    策略：
      1. 优先用实体节点：若图中存在 Entity(name=group)，通过共享 Chunk 找其他实体。
      2. 否则回退：全文 chunk contains group -> 抽取其 HAS_ENTITY 成员。
      3. 统计支持 chunk 数 support，按 support DESC。
    """
    driver = get_driver()
    members: list[dict] = []
    source_chunks: dict[str, str] = {}
    with driver.session() as session:
        # Path1: group entity exists?
        rec = session.run("MATCH (e:Entity {name:$g}) RETURN e.name AS name LIMIT 1", g=group).single()
        if rec:
            cy = """
            MATCH (eg:Entity {name:$g})<-[:HAS_ENTITY]-(c:Chunk)-[:HAS_ENTITY]->(m:Entity)
            WHERE m.name <> $g
            WITH m, collect(distinct c)[0..5] AS cs, count(distinct c) AS support
            RETURN m.name AS member, support, [x IN cs | x.id][0..5] AS chunk_ids
            ORDER BY support DESC, member ASC
            LIMIT 40
            """
            rows = session.run(cy, g=group)
            for r in rows:
                members.append({
                    'member': r['member'],
                    'support': r['support'],
                    'chunks': r['chunk_ids']
                })
        else:
            # Path2: fallback contains search
            cy = """
            MATCH (c:Chunk) WHERE c.text CONTAINS $g
            OPTIONAL MATCH (c)-[:HAS_ENTITY]->(m:Entity)
            WITH m, c WHERE m IS NOT NULL AND m.name <> $g
            WITH m.name AS member, count(distinct c) AS support, collect(distinct c.id)[0..5] AS chunk_ids
            RETURN member, support, chunk_ids
            ORDER BY support DESC, member ASC
            LIMIT 40
            """
            rows = session.run(cy, g=group)
            for r in rows:
                members.append({
                    'member': r['member'],
                    'support': r['support'],
                    'chunks': r['chunk_ids']
                })
        # 取需要的 chunk 文本作为引用源
        all_chunk_ids = sorted({cid for m in members for cid in (m.get('chunks') or [])})[:60]
        if all_chunk_ids:
            text_rows = session.run("""
            UNWIND $ids AS cid
            MATCH (c:Chunk {id:cid})
            RETURN c.id AS id, c.text AS text
            """, ids=all_chunk_ids)
            for tr in text_rows:
                source_chunks[tr['id']] = tr['text']
    if not members:
        def gen_empty():
            yield f"未能在当前知识中找到“{group}”的成员信息。"
        return (gen_empty(), []) if return_sources else gen_empty()
    # 构造回答（不走 LLM，直接结构化）
    lines = [f"问题: {question}", f"识别到集合主体: {group}", "成员列表(按出现支持度排序):"]
    for idx, m in enumerate(members, 1):
        lines.append(f"{idx}. {m['member']} (支持chunk:{m['support']})")
    answer_text = '\n'.join(lines)
    def gen_answer():
        yield answer_text
    if return_sources:
        # sources: 选取每个成员首个 chunk 作为引用
        sources = []
        used = set()
        for idx, m in enumerate(members, 1):
            cid = (m.get('chunks') or [None])[0]
            if not cid or cid in used:
                continue
            used.add(cid)
            txt = source_chunks.get(cid, '')
            sources.append({
                'id': cid,
                'preview': txt[:80].replace('\n',' ') + ('...' if len(txt) > 80 else ''),
                'reason': 'membership',
                'composite_score': None,
                'rank': idx
            })
        return gen_answer(), sources
    return gen_answer()

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
WHERE r.confidence >= $min_conf
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
            # 使用全局设定的最低置信度过滤
            from ..config.settings import get_settings as _gs
            _st = _gs()
            min_conf = getattr(_st, 'relation_min_confidence', 0.4)
            records_rel = session.run(EXPAND_LLM_REL, chunkIds=chunk_ids, limit=limit, min_conf=min_conf)
            for r in records_rel:
                collected.setdefault(r["id"], {"id": r["id"], "text": r["text"], "reason": r["reason"], "rel_type": r.get("rel_type"), "rel_conf": r.get("rel_conf")})
            # 共现实体引出的二跳 chunk
            records3 = session.run(EXPAND_COOC_ENTITIES, chunkIds=chunk_ids, limit=limit)
            for r in records3:
                collected.setdefault(r["id"], {"id": r["id"], "text": r["text"], "reason": r["reason"]})
    return list(collected.values())


# ---- 子图提取骨架 (后续可选调用) ----
def _extract_subgraph(seed_chunk_ids: List[str], settings_obj=None) -> List[Dict]:
    """基于初始检索命中构建一个局部子图的候选 Chunk 列表。

    初始实现（轻量）：
      1. 取种子 chunk 的实体集合 E
      2. 找共享这些实体的其他 chunk（限制最大数量）
      3. 若启用 LLM 关系且 RELATION_EXTRACTION=true，再取与种子 chunk 有高置信度关系的 chunk
    未来可扩展：
      - 限定关系类型子集
      - 多跳 BFS 带衰减
      - 路径评分 (Path scoring) / random walk
    返回：[{id, text, reason='subgraph'}]
    """
    if not seed_chunk_ids:
        return []
    st = settings_obj or settings
    if not getattr(st, 'subgraph_enable', False):
        return []
    max_nodes = getattr(st, 'subgraph_max_nodes', 120)
    rel_types_filter = getattr(st, 'subgraph_rel_types', '*')
    rel_min_conf = getattr(st, 'relation_min_confidence', 0.4)
    collected: Dict[str, Dict] = {}
    max_depth = getattr(st, 'subgraph_max_depth', 1)
    depth1_cap = getattr(st, 'subgraph_depth1_cap', 0)
    depth1_rel_cap = getattr(st, 'subgraph_depth1_rel_cap', 0)
    deep_reserve = getattr(st, 'subgraph_deep_reserve_nodes', 0)
    per_limit = getattr(st,'subgraph_per_entity_limit',4)
    def _norm_id(x: str) -> str:
        if not getattr(settings, 'id_normalize_enable', True) or not isinstance(x, str):
            return x
        # 去除零宽字符与首尾空白，折叠连续空白
        import re
        x2 = re.sub(r"[\u200B\u200C\u200D\uFEFF]", "", x.strip())
        x2 = re.sub(r"\s+", " ", x2)
        return x2
    norm_seeds = [_norm_id(s) for s in seed_chunk_ids]
    explored_ids = set(norm_seeds)
    frontier_ids = list(norm_seeds)
    # 路径存储: node_id -> list[path_dict]
    path_records: Dict[str, List[Dict[str, Any]]] = {}
    depth = 1
    # 统计调试信息
    global _last_subgraph_stats
    rel_examined = 0
    rel_upgraded = 0
    rel_added = 0
    ent_added = 0
    ent_added_d1 = 0  # depth=1 实体新增（用于严格限流）
    rel_added_d1 = 0  # depth=1 关系新增（用于严格限流）
    with get_driver().session() as session:
        while frontier_ids and depth <= max_depth and len(collected) < max_nodes:
            # 共享实体扩展本深度
            remaining_total = max_nodes - len(collected)
            # 深层预留：如果还在 depth=1 且设置了 deep_reserve，则最多可用 remaining_total - deep_reserve
            if depth == 1 and deep_reserve > 0:
                remaining = max(0, remaining_total - deep_reserve)
            else:
                remaining = remaining_total
            # depth=1 单独限流（可选）
            if depth == 1 and depth1_cap and depth1_cap > 0:
                remain_cap = depth1_cap - ent_added_d1
                remaining = min(remaining, remain_cap)
            # 为关系扩展预留 25% 配额（防止被实体一次性填满），在已限流基础上再次缩减
            ent_budget = remaining
            if ent_budget > 8:  # 只有剩余较多时才预留
                ent_budget = max(2, int(ent_budget * 0.75))
            if ent_budget <= 0:
                next_frontier = []
                # 即便实体扩展无预算，也继续做关系升级扫描
                rows = []
            else:
                rows = session.run(
                    """
                    UNWIND $ids AS cid
                    MATCH (c:Chunk {id:cid})-[:HAS_ENTITY]->(e:Entity)<-[:HAS_ENTITY]-(oc:Chunk)
                    WHERE oc.id <> cid
                    WITH e, oc
                    ORDER BY oc.id
                    WITH e, collect(DISTINCT oc)[0..$per] AS ocs
                    UNWIND ocs AS oc2
                    RETURN DISTINCT oc2.id AS id, oc2.text AS text
                    LIMIT $lim
                    """, ids=frontier_ids, lim=ent_budget, per=per_limit
                )
            next_frontier = []
            for r in rows:
                if len(collected) >= max_nodes:
                    break
                nid_raw = r['id']
                nid = _norm_id(nid_raw)
                if nid in explored_ids:
                    continue
                explored_ids.add(nid)
                collected.setdefault(nid, {'id': nid_raw, 'text': r['text'], 'reason': 'subgraph', 'sg_depth': depth})
                # 路径记录（实体共享）
                pr = path_records.setdefault(nid, [])
                if len(pr) < getattr(settings, 'subgraph_path_max_records', 8):
                    pr.append({'kind':'entity', 'depth': depth})
                next_frontier.append(nid)
                ent_added += 1
                if depth == 1:
                    ent_added_d1 += 1
            # 关系扩展（对新增节点同样施加 depth1 限流，确保 deep_reserve 生效）
            # 即便已接近/达到上限，也运行关系扫描用于升级已有节点
            if st.relation_extraction:
                rel_query = """
                UNWIND $ids AS cid
                MATCH (c:Chunk {id:cid})-[r:REL]-(o:Chunk)
                WHERE r.confidence >= $minc
                RETURN DISTINCT o.id AS id, o.text AS text, r.type AS rtype, r.confidence AS conf
                LIMIT $lim
                """
                # 关系查询限制使用剩余容量 + 适度上浮，避免遗漏升级机会
                rel_limit = max(10, max_nodes)  # 允许扫描更多行做升级
                rows2 = session.run(rel_query, ids=frontier_ids, minc=rel_min_conf, lim=rel_limit)
                rel_type_allow = None
                if rel_types_filter != '*':
                    rel_type_allow = set([t.strip() for t in rel_types_filter.split(',') if t.strip()])
                for r in rows2:
                    rel_examined += 1
                    nid_raw = r['id']
                    nid = _norm_id(nid_raw)
                    rtype = r['rtype']
                    if rel_type_allow and rtype not in rel_type_allow:
                        continue
                    existing = collected.get(nid)
                    if existing:
                        # 即便达到 max_nodes 也允许升级已有节点
                        hits = existing.setdefault('rel_hits', [])
                        hits.append({'type': rtype, 'conf': r['conf'], 'via_depth': depth})
                        if existing.get('reason') != 'subgraph_rel':
                            existing['reason'] = 'subgraph_rel'
                            existing['rel_type'] = rtype
                            existing['rel_conf'] = r['conf']
                            rel_upgraded += 1
                        else:
                            if r['conf'] > (existing.get('rel_conf') or 0):
                                existing['rel_type'] = rtype
                                existing['rel_conf'] = r['conf']
                        # 路径记录添加关系型
                        pr = path_records.setdefault(nid, [])
                        if len(pr) < getattr(settings, 'subgraph_path_max_records', 8):
                            pr.append({'kind':'rel', 'type': rtype, 'conf': r['conf'], 'depth': depth})
                        continue
                    # 新节点受容量约束 + 深度配额约束
                    if len(collected) >= max_nodes:
                        continue
                    if nid in explored_ids:
                        continue
                    # depth=1 关系新增配额控制（独立 or 合并）
                    if depth == 1:
                        # 若设置了总的 depth1_cap，则关系也应受其影响：统计当前 depth1 新增(实体+关系)
                        if depth1_cap and depth1_cap > 0:
                            total_d1_new = ent_added_d1 + rel_added_d1
                            if total_d1_new >= depth1_cap:
                                continue
                        if depth1_rel_cap and depth1_rel_cap > 0 and rel_added_d1 >= depth1_rel_cap:
                            continue
                    explored_ids.add(nid)
                    collected[nid] = {
                        'id': nid_raw,
                        'text': r['text'],
                        'reason': 'subgraph_rel',
                        'rel_type': rtype,
                        'rel_conf': r['conf'],
                        'rel_hits': [{'type': rtype, 'conf': r['conf'], 'via_depth': depth}],
                        'sg_depth': depth
                    }
                    # 路径记录
                    pr = path_records.setdefault(nid, [])
                    if len(pr) < getattr(settings, 'subgraph_path_max_records', 8):
                        pr.append({'kind':'rel', 'type': rtype, 'conf': r['conf'], 'depth': depth})
                    next_frontier.append(nid)
                    rel_added += 1
                    if depth == 1:
                        rel_added_d1 += 1
            depth += 1
            if depth > max_depth:
                break
            frontier_ids = next_frontier
    # 计算路径级评分（可选）
    if getattr(settings, 'subgraph_path_score_enable', True):
        entity_base = getattr(settings, 'subgraph_path_entity_base', 0.5)
        rel_w = getattr(settings, 'subgraph_path_rel_conf_weight', 0.6)
        ent_decay = getattr(settings, 'subgraph_path_entity_decay', 0.7)
        rel_decay = getattr(settings, 'subgraph_path_rel_decay', 0.75)
        weight_scale = getattr(settings, 'subgraph_path_score_weight', 0.4)
        for nid, node in collected.items():
            prs = path_records.get(nid, [])
            if not prs:
                continue
            score_acc = 0.0
            for p in prs:
                d = max(1, p.get('depth',1))
                if p['kind'] == 'entity':
                    score_acc += entity_base * (ent_decay ** (d-1))
                else:  # rel
                    conf = float(p.get('conf') or 0.0)
                    score_acc += rel_w * conf * (rel_decay ** (d-1))
            node['path_score'] = round(score_acc * weight_scale, 6)
            node['path_records'] = prs
    _last_subgraph_stats = {
        'relations_examined': rel_examined,
        'relations_upgraded': rel_upgraded,
        'relations_added': rel_added,
        'relations_added_depth1': rel_added_d1,
        'entity_added': ent_added,
        'entity_added_depth1': ent_added_d1,
        'max_depth_reached': depth-1,
        'total_nodes': len(collected)
    }
    return list(collected.values())


def build_prompt(question: str, contexts: List[Dict], history: Optional[List[Dict]] = None, system_prefix: Optional[str] = None) -> list:
    # 用 [S1] [S2] 形式标记，使回答可引用
    context_text_lines = []
    for idx, c in enumerate(contexts, 1):
        tag = f"[S{idx}]"
        context_text_lines.append(f"{tag} {c['text']}")
    context_text = '\n\n'.join(context_text_lines)
    system = (
        (system_prefix + "\n") if system_prefix else ""
    ) + (
        "你是一个知识图谱增强的智能问答助手。必须仅依据提供的上下文回答，不得编造。"
        "在引用具体事实、数据、时长或文件段落时，紧跟相应引用标记如 [S1]、[S2]。"
        "如果多个来源支持同一事实，可合并如 [S1][S3]。无法回答时明确说明。保持用户语言。"
        "请尽量整合所有相关片段的信息，避免只回答局部；若问题涉及人物/实体之间的关系、身份或属性，请在答案中明确列出并附引用。"
    )
    user = f"问题: {question}\n\n上下文(含编号):\n{context_text}\n\n请给出结构化回答，并恰当添加引用标记。"
    msgs = [{"role": "system", "content": system}]
    # 合并会话历史（可选）
    if history:
        for m in history:
            r = m.get('role')
            c = m.get('content')
            if r in ("system","user","assistant") and isinstance(c, str):
                msgs.append({"role": r, "content": c})
    msgs.append({"role": "user", "content": user})
    return msgs


def answer_question(question: str, return_sources: bool = False, stream: bool = True, extra_contexts: Optional[List[Dict]] = None, history: Optional[List[Dict]] = None, system_prefix: Optional[str] = None):
    # 1) 先做集合/成员类问题快速路径（无需向量召回）
    if os.getenv('ENABLE_MEMBERSHIP_QA', 'true').lower() in ('1','true','yes'):
        group = _detect_membership_question(question)
        if group:
            return _answer_membership(group, question, return_sources, stream)
    # 自适应问句分类（仅用于调参，不改变缓存 key）
    q_type = _classify_query(question) if getattr(settings, 'adaptive_query_strategy', False) else 'generic'
    # 备份原值
    dynamic_top_k = settings.top_k
    dynamic_context_max = getattr(settings, 'context_max', settings.top_k * 3)
    # 根据类型微调策略 + 子图自适应参数
    adaptive_subgraph_cfg = {}
    if getattr(settings, 'adaptive_query_strategy', False):
        if q_type in ('list', 'membership'):
            dynamic_top_k = min(settings.top_k, 14)
            dynamic_context_max = min(dynamic_context_max, 20)
            adaptive_subgraph_cfg = {
                'max_depth': 1,
                'depth1_cap': 0,
                'deep_reserve': 0,
                'path_weight_scale': 0.30,
            }
        elif q_type == 'relation':
            dynamic_top_k = max(settings.top_k, 15)
            # 强制产生二跳：限制第一跳容量并加大预留
            adaptive_subgraph_cfg = {
                'max_depth': 2,
                'depth1_cap': 20,  # 进一步降低，强制保留容量给二跳
                'deep_reserve': max(26, getattr(settings, 'subgraph_deep_reserve_nodes', 20)),
                'path_weight_scale': 0.42,  # 略降防止路径过强
            }
        elif q_type == 'cause':
            dynamic_top_k = max(settings.top_k + 2, 18)
            dynamic_context_max = min(max(dynamic_context_max, 24), 26)
            adaptive_subgraph_cfg = {
                'max_depth': 2,
                'depth1_cap': getattr(settings, 'subgraph_depth1_cap', 0) or 36,
                'deep_reserve': max(24, getattr(settings, 'subgraph_deep_reserve_nodes', 20)),
                'path_weight_scale': 0.50,
            }
        elif q_type == 'event':
            dynamic_top_k = max(settings.top_k + 1, 16)
            adaptive_subgraph_cfg = {
                'max_depth': 2,
                'depth1_cap': getattr(settings, 'subgraph_depth1_cap', 0) or 34,
                'deep_reserve': getattr(settings, 'subgraph_deep_reserve_nodes', 20),
                'path_weight_scale': 0.48,
            }
        elif q_type == 'definition':
            dynamic_top_k = min(settings.top_k, 12)
            dynamic_context_max = min(dynamic_context_max, 18)
            adaptive_subgraph_cfg = {
                'max_depth': 1,
                'depth1_cap': 0,
                'deep_reserve': 0,
                'path_weight_scale': 0.25,
            }
        # generic 保持默认（不设 adaptive_subgraph_cfg）
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

    hits, _, degraded, warn = vector_search(question, dynamic_top_k)
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
    expand = [] if degraded else expand_context([h['id'] for h in hits], limit=dynamic_top_k * 2, hops=settings.expand_hops)
    # 可选子图扩展（基于种子命中）
    subgraph_extra = []
    if not degraded and getattr(settings, 'subgraph_enable', False):
        try:
            # 若之前计算了 adaptive_subgraph_cfg，则临时覆盖 settings
            adaptive_subgraph_cfg_local = locals().get('adaptive_subgraph_cfg', {})  # 读取上文定义
            if adaptive_subgraph_cfg_local:
                _orig = {
                    'max_depth': getattr(settings, 'subgraph_max_depth', None),
                    'depth1_cap': getattr(settings, 'subgraph_depth1_cap', None),
                    'deep_reserve': getattr(settings, 'subgraph_deep_reserve_nodes', None),
                    'path_weight': getattr(settings, 'subgraph_path_score_weight', None),
                }
                try:
                    if 'max_depth' in adaptive_subgraph_cfg_local:
                        settings.subgraph_max_depth = adaptive_subgraph_cfg_local['max_depth']  # type: ignore
                    if 'depth1_cap' in adaptive_subgraph_cfg_local:
                        settings.subgraph_depth1_cap = adaptive_subgraph_cfg_local['depth1_cap']  # type: ignore
                    if 'deep_reserve' in adaptive_subgraph_cfg_local:
                        settings.subgraph_deep_reserve_nodes = adaptive_subgraph_cfg_local['deep_reserve']  # type: ignore
                    if 'path_weight_scale' in adaptive_subgraph_cfg_local:
                        settings.subgraph_path_score_weight = adaptive_subgraph_cfg_local['path_weight_scale']  # type: ignore
                    raw_subgraph = _extract_subgraph([h['id'] for h in hits])
                finally:
                    # 恢复
                    if _orig['max_depth'] is not None:
                        settings.subgraph_max_depth = _orig['max_depth']  # type: ignore
                    if _orig['depth1_cap'] is not None:
                        settings.subgraph_depth1_cap = _orig['depth1_cap']  # type: ignore
                    if _orig['deep_reserve'] is not None:
                        settings.subgraph_deep_reserve_nodes = _orig['deep_reserve']  # type: ignore
                    if _orig['path_weight'] is not None:
                        settings.subgraph_path_score_weight = _orig['path_weight']  # type: ignore
            else:
                raw_subgraph = _extract_subgraph([h['id'] for h in hits])
            # 优先保留关系节点
            rel_min = getattr(settings, 'subgraph_rel_min_keep', 0)
            cap = dynamic_top_k * 3
            if cap <= 0:
                subgraph_extra = raw_subgraph
            else:
                rel_nodes = [n for n in raw_subgraph if n.get('reason') == 'subgraph_rel']
                other_nodes = [n for n in raw_subgraph if n.get('reason') != 'subgraph_rel']
                picked_rel = rel_nodes[:max(rel_min, len(rel_nodes))]
                remain_slots = max(0, cap - len(picked_rel))
                subgraph_extra = picked_rel + other_nodes[:remain_slots]
        except Exception as sg_e:
            logging.warning(f"[Subgraph] failed: {sg_e}")
    merged = {h['id']: h for h in hits}
    for e in expand:
        merged.setdefault(e['id'], e)
    for sg in subgraph_extra:
        merged.setdefault(sg['id'], sg)
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
        # 实体扩展来源加成（默认 0.12，可通过环境变量 REL_WEIGHT_ENTITY 调整）
        if reason == 'entity':
            bonus += settings.rel_weight_entity
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
        elif reason == 'subgraph':
            # 子图扩展来源的额外加成（尚无细粒度；未来可结合节点度/路径长度）
            try:
                bonus += settings.subgraph_weight
            except Exception:
                pass
        elif reason == 'subgraph_rel':
            # 关系型子图节点，应用全局 multiplier + 关系类型 multiplier
            try:
                base_mul = getattr(settings, 'subgraph_rel_multiplier', 1.0)
                type_mul_map = {}
                raw_map = getattr(settings, 'subgraph_rel_type_multipliers', '')
                if raw_map:
                    for seg in raw_map.split(','):
                        if ':' in seg:
                            t, v = seg.split(':',1)
                            try:
                                type_mul_map[t.strip().upper()] = float(v)
                            except ValueError:
                                continue
                rtype = (c.get('rel_type') or '').upper()
                type_mul = type_mul_map.get(rtype, 1.0)
                # 深度衰减稍后统一（已在上层 bonus? 这里沿用原逻辑: subgraph_rel 不使用 depth; depth 衰减在 debug/bonus_detail 中显示）
                base_component = settings.subgraph_weight * base_mul * type_mul
                # 多关系加权：除主显示关系外，对 rel_hits 按置信度排序做衰减加成
                rel_hits = c.get('rel_hits') or []
                if rel_hits:
                    # 以当前显示的 rel_type 作为主关系，剩余关系参与附加加成
                    show_type = rtype
                    extras = [h for h in rel_hits if (h.get('type') or '').upper() != show_type]
                    if extras:
                        extras_sorted = sorted(extras, key=lambda x: x.get('conf') or 0, reverse=True)
                        decay = getattr(settings, 'subgraph_rel_hits_decay', 0.7)
                        scale = getattr(settings, 'subgraph_rel_multi_scale', 0.35)
                        maxn = getattr(settings, 'subgraph_rel_hits_max', 5)
                        acc = 0.0
                        for idx, h in enumerate(extras_sorted[:maxn]):
                            conf = float(h.get('conf') or 0.0)
                            acc += conf * (decay ** idx)
                        base_component += scale * acc
                    # 关系密度抑制
                    cap = getattr(settings, 'subgraph_rel_density_cap', 6)
                    alpha = getattr(settings, 'subgraph_rel_density_alpha', 0.15)
                    total_rel = len(rel_hits)
                    if total_rel > cap:
                        penalty = 1.0 / (1 + alpha * (total_rel - cap))
                        base_component *= penalty
                bonus += base_component
            except Exception:
                pass
        # 图中心性加成（degree）
        if settings.graph_rank_enabled:
            dsc = degree_scores.get(c['id'])
            if dsc:
                bonus += settings.graph_rank_weight * dsc
        c['composite_score'] = round(base + bonus, 4)
    # 按 composite_score 降序，然后截取
    contexts.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
    # ------- 上下文裁剪策略 -------
    max_ctx = dynamic_context_max
    min_per_reason = getattr(settings, 'context_min_per_reason', 1)
    prune_ratio = getattr(settings, 'context_prune_ratio', 2.0)
    prune_gap = getattr(settings, 'context_prune_gap', 0.6)
    initial_len = len(contexts)
    # 一级：若总数 <= max_ctx 直接保留
    if len(contexts) > max_ctx:
        # 二级条件：长度远超 (max_ctx * prune_ratio) 时，先按分数阈值剔除尾部
        if len(contexts) > max_ctx * prune_ratio:
            if contexts:
                top_score = contexts[0].get('composite_score', 0)
                # 保留得分 >= top_score - prune_gap 的优先候选
                filtered = [c for c in contexts if c.get('composite_score', 0) >= top_score - prune_gap]
                if len(filtered) >= max_ctx * 0.6:  # 仍然保留足够数量用于多样性
                    contexts = filtered
        # 保障来源多样性：按 reason bucket 采样
        buckets = {}
        for c in contexts:
            r = c.get('reason') or ('vector' if c.get('score') is not None else 'context')
            buckets.setdefault(r, []).append(c)
        # 各 bucket 按分数排序
        for b in buckets.values():
            b.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
        selected = []
        # 先为每类拿 min_per_reason
        for r, b in buckets.items():
            selected.extend(b[:min_per_reason])
        # 去重
        seen_ids = {c['id'] for c in selected}
        if len(selected) < max_ctx:
            # 合并所有剩余候选（已排序）填满到 max_ctx
            remaining = []
            for r, b in buckets.items():
                for c in b[min_per_reason:]:
                    if c['id'] not in seen_ids:
                        remaining.append(c)
            remaining.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
            for c in remaining:
                if len(selected) >= max_ctx:
                    break
                selected.append(c)
                seen_ids.add(c['id'])
        contexts = selected[:max_ctx]
    # 重新按得分排序（多样化采样后保序）
    contexts.sort(key=lambda x: x.get('composite_score', 0), reverse=True)
    # --- Rerank (post-score) ---
    if settings.rerank_enabled:
        contexts = rerank_post_score(question, contexts)
    # 合并外部上下文（若提供）
    if extra_contexts:
        # 追加到上下文末尾，参与编号
        for ec in extra_contexts:
            if isinstance(ec, dict) and 'text' in ec:
                # 以临时 id 标识，避免与真实 id 冲突
                ec_id = ec.get('id') or f"__ext__{len(contexts)+1}"
                contexts.append({'id': ec_id, 'text': ec['text'], 'reason': 'external'})
            elif isinstance(ec, str):
                contexts.append({'id': f"__ext__{len(contexts)+1}", 'text': ec, 'reason': 'external'})
    messages = build_prompt(question, contexts, history=history, system_prefix=system_prefix)
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
                    "rerank_score": c.get('rerank_score'),
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


# ---------------- 调试检索: 返回各阶段特征分解 -----------------
def debug_retrieve(question: str) -> Dict:
    """执行与 answer_question 相似的检索流程，但不调用 LLM，返回特征分解便于调参。

    返回字典包含：
      degraded: bool 是否降级
      warn:      str  降级原因
      initial_hits: [{id,text,raw_score}]
      expanded: [{id,reason}]
      features: 每个候选的特征 & 构成
      final_rank: 排序后列表（不含 LLM rerank）
    """
    hits, _, degraded, warn = vector_search(question, settings.top_k)
    bm25_scores = {}
    if settings.bm25_enabled and not degraded:
        try:
            if not bm25_index._built:  # type: ignore
                with get_driver().session() as session:
                    recs = session.run("MATCH (c:Chunk) RETURN c.id AS id, c.text AS text LIMIT 2000")
                    docs = [{"id": r["id"], "text": r["text"]} for r in recs]
                bm25_index.build(docs)
            bm25_scores = bm25_index.score(question, top_k=settings.top_k * 5)
        except Exception as e:
            bm25_scores = {}
    expand = [] if degraded else expand_context([h['id'] for h in hits], limit=settings.top_k * 2, hops=settings.expand_hops)
    # 子图扩展（调试中即使主流程已加，也单独加入便于观察来源差异）
    subgraph_extra = []
    if not degraded and getattr(settings, 'subgraph_enable', False):
        try:
            raw_subgraph = _extract_subgraph([h['id'] for h in hits])
            rel_min = getattr(settings, 'subgraph_rel_min_keep', 0)
            cap = settings.top_k * 3
            rel_nodes = [n for n in raw_subgraph if n.get('reason') == 'subgraph_rel']
            other_nodes = [n for n in raw_subgraph if n.get('reason') != 'subgraph_rel']
            picked_rel = rel_nodes[:max(rel_min, len(rel_nodes))]
            remain_slots = max(0, cap - len(picked_rel))
            subgraph_extra = picked_rel + other_nodes[:remain_slots]
        except Exception as sg_e:
            logging.warning(f"[Subgraph][debug] failed: {sg_e}")
    merged = {h['id']: h for h in hits}
    for e in expand:
        merged.setdefault(e['id'], e)
    for sg in subgraph_extra:
        merged.setdefault(sg['id'], sg)
    contexts = list(merged.values())
    vec_scores = [c.get('score') for c in contexts if c.get('score') is not None]
    if vec_scores:
        vmin, vmax = min(vec_scores), max(vec_scores)
        span = (vmax - vmin) or 1.0
    else:
        vmin, vmax, span = 0.0, 1.0, 1.0
    degree_scores = get_degree_scores() if settings.graph_rank_enabled else {}
    feature_rows = []
    for c in contexts:
        raw_vec = c.get('score')
        vec_norm = 0.0
        if raw_vec is not None:
            vec_norm = (raw_vec - vmin)/span if span else 0.0
        bm25 = bm25_scores.get(c['id']) if settings.bm25_enabled and not degraded else None
        reason = c.get('reason', 'vector' if raw_vec is not None else 'context')
        rel_bonus = 0.0
        rel_detail = {}
        if settings.bm25_enabled and not degraded and bm25 is not None:
            rel_bonus += settings.bm25_weight * bm25
            rel_detail['bm25_w'] = settings.bm25_weight * bm25
        if reason == 'entity':
            rel_bonus += settings.rel_weight_entity
            rel_detail['entity'] = settings.rel_weight_entity
        elif reason == 'relates':
            rel_bonus += settings.rel_weight_relates
            rel_detail['relates'] = settings.rel_weight_relates
        elif reason == 'cooccur':
            rel_bonus += settings.rel_weight_cooccur
            rel_detail['cooccur'] = settings.rel_weight_cooccur
        elif reason == 'llm_rel':
            conf = float(c.get('rel_conf') or 0.5)
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
            rel_add = rel_base * conf
            rel_bonus += rel_add
            rel_detail[f'llm_rel:{rtype}'] = rel_add
        elif reason in ('subgraph','subgraph_rel'):
            depth = c.get('sg_depth', 1)
            decay = getattr(settings, 'subgraph_depth_decay', 0.0)
            factor = 1.0 / (1 + decay * (max(1, depth) - 1)) if decay > 0 else 1.0
            base_w = getattr(settings, 'subgraph_weight', 0.0)
            if reason == 'subgraph_rel':
                base_w *= getattr(settings, 'subgraph_rel_multiplier', 1.0)
                # 关系类型 multiplier
                raw_map = getattr(settings, 'subgraph_rel_type_multipliers', '')
                if raw_map:
                    type_mul_map = {}
                    for seg in raw_map.split(','):
                        if ':' in seg:
                            t,v = seg.split(':',1)
                            try:
                                type_mul_map[t.strip().upper()] = float(v)
                            except ValueError:
                                continue
                    rtype = (c.get('rel_type') or '').upper()
                    base_w *= type_mul_map.get(rtype, 1.0)
                # 多关系加权
                rel_hits = c.get('rel_hits') or []
                if rel_hits:
                    show_type = (c.get('rel_type') or '').upper()
                    extras = [h for h in rel_hits if (h.get('type') or '').upper() != show_type]
                    if extras:
                        extras_sorted = sorted(extras, key=lambda x: x.get('conf') or 0, reverse=True)
                        decay_h = getattr(settings, 'subgraph_rel_hits_decay', 0.7)
                        scale = getattr(settings, 'subgraph_rel_multi_scale', 0.35)
                        maxn = getattr(settings, 'subgraph_rel_hits_max', 5)
                        acc = 0.0
                        for idx, h in enumerate(extras_sorted[:maxn]):
                            conf = float(h.get('conf') or 0.0)
                            acc += conf * (decay_h ** idx)
                        add_multi = scale * acc
                        base_w += add_multi
                        rel_detail['rel_multi'] = round(add_multi,4)
                    # 密度惩罚
                    cap = getattr(settings, 'subgraph_rel_density_cap', 6)
                    alpha = getattr(settings, 'subgraph_rel_density_alpha', 0.15)
                    total_rel = len(rel_hits)
                    if total_rel > cap:
                        penalty = 1.0 / (1 + alpha * (total_rel - cap))
                        base_w *= penalty
                        rel_detail['rel_density_penalty'] = round(penalty,4)
            sg_w = base_w * factor
            rel_bonus += sg_w
            rel_detail['subgraph'] = sg_w
        degree_bonus = 0.0
        if settings.graph_rank_enabled:
            dsc = degree_scores.get(c['id'])
            if dsc:
                degree_bonus = settings.graph_rank_weight * dsc
                rel_bonus += degree_bonus
                rel_detail['degree'] = degree_bonus
        # 路径级评分融合
        path_score = 0.0
        if getattr(settings, 'subgraph_path_score_enable', True):
            path_score = float(c.get('path_score') or 0.0)
            if path_score:
                rel_bonus += path_score
                rel_detail['path_score'] = path_score
        composite = round(vec_norm + rel_bonus, 4)
        feature_rows.append({
            'id': c['id'],
            'reason': reason,
            'raw_vector_score': raw_vec,
            'vector_norm': round(vec_norm,4),
            'bm25': bm25,
            'bonus_detail': rel_detail,
            'sg_depth': c.get('sg_depth'),
            'rel_type': c.get('rel_type'),
            'rel_conf': c.get('rel_conf'),
            'rel_hits': c.get('rel_hits'),
            'path_score': path_score if path_score else None,
            'composite_score': composite,
            'text_preview': c.get('text','')[:100].replace('\n',' ')
        })
    # 排序
    feature_rows.sort(key=lambda x: x['composite_score'], reverse=True)
    final_rank = [
        {k: v for k,v in row.items() if k in ('id','composite_score','reason','raw_vector_score','vector_norm')}
        for row in feature_rows
    ]
    # 深度聚合统计
    depth_counts = {}
    depth_path_scores = {}
    for fr in feature_rows:
        if fr.get('reason','').startswith('subgraph'):
            d = fr.get('sg_depth') or 1
            depth_counts[d] = depth_counts.get(d,0)+1
            ps = fr.get('path_score')
            if ps is not None:
                arr = depth_path_scores.setdefault(d,[])
                arr.append(ps)
    depth_metrics = {
        str(d): {
            'count': depth_counts[d],
            'path_score_avg': round(sum(depth_path_scores.get(d,[]))/len(depth_path_scores.get(d,[]) or [1]),6) if depth_path_scores.get(d) else 0.0
        } for d in sorted(depth_counts.keys())
    }
    debug_obj = {
        'question': question,
        'degraded': degraded,
        'warn': warn,
        'vector_span': {'min': vmin, 'max': vmax, 'span': span},
        'initial_hits': [{'id': h['id'], 'raw_score': h.get('score')} for h in hits],
    'expanded': [{'id': e['id'], 'reason': e.get('reason')} for e in expand],
    'subgraph_added': [{'id': e['id'], 'reason': e.get('reason')} for e in subgraph_extra],
        'features': feature_rows[: settings.top_k * 5],
        'final_rank': final_rank[: settings.top_k * 5],
        'subgraph_stats': globals().get('_last_subgraph_stats', {}),
        'subgraph_depth_metrics': depth_metrics
    }
    if DEBUG_RETRIEVAL:
        import json, logging
        logging.warning('[DEBUG_RETRIEVAL]%s', json.dumps(debug_obj, ensure_ascii=False)[:4000])
    return debug_obj
