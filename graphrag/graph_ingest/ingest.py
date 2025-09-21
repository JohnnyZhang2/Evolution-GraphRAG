# Evolution RAG Ingestion Pipeline
# Handles splitting, embedding, entity & relation extraction, and graph writes.
# Copyright (c) 2025 EvolutionAI Studio
# All Rights Holder: Johnny Zhang
# License: MIT (see LICENSE)

import os
from neo4j import GraphDatabase
from typing import List, Dict, Tuple, Set
from ..config.settings import get_settings
from ..utils.text_splitter import split_text
from ..embedding.client import embed_texts
from ..llm.client import extract_entities, extract_relations
from ..utils.entity_normalize import normalize_entity
from ..utils.hash_utils import hash_text
import math

settings = get_settings()

def get_driver():
    return GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))

# Neo4j 5 (GQL) 经常要求一次只执行一个语句，这里拆分
SCHEMA_STATEMENTS = [
    "CREATE CONSTRAINT chunk_id IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
    "CREATE CONSTRAINT entity_name IF NOT EXISTS FOR (e:Entity) REQUIRE e.name IS UNIQUE"
]

VECTOR_INDEX_CYPHER = """
CALL db.index.vector.createNodeIndex(
  'chunk_embedding_index',
  'Chunk',
  'embedding',
  $dimension,
  'cosine'
) YIELD name;"""

UPSERT_CHUNK = """
MERGE (c:Chunk {id: $id})
SET c.text = $text, c.embedding = $embedding, c.source = $source, c.hash = $hash
"""

MERGE_ENTITY = """
UNWIND $entities AS en
MERGE (e:Entity {name: en})
RETURN count(*) as merged
"""

# 为了避免依赖 APOC，这里 aliases 只在首次创建或该属性为空时写入，不做集合并操作
SET_ALIASES_IF_EMPTY = """
UNWIND $rows AS r
MATCH (e:Entity {name: r.name})
WITH e, r WHERE size(r.aliases) > 0
SET e.aliases = CASE WHEN e.aliases IS NULL THEN r.aliases ELSE e.aliases END
"""

LINK_CHUNK_ENTITY = """
UNWIND $pairs AS p
MATCH (c:Chunk {id: p.chunk_id})
MATCH (e:Entity {name: p.entity})
MERGE (c)-[:HAS_ENTITY]->(e)
"""

# 基于同一 Chunk 中实体对生成共现关系（无权重累加示例：可在已有属性上加 count）
LINK_ENTITY_COOC = """
UNWIND $entityPairs AS ep
MERGE (e1:Entity {name: ep.e1})
MERGE (e2:Entity {name: ep.e2})
MERGE (e1)-[r:CO_OCCURS_WITH]-(e2)
ON CREATE SET r.count = 1
ON MATCH SET r.count = r.count + 1
"""

# 基于共享实体的 Chunk-Chuck 相关性（简单：同一实体 -> 建 RELATES_TO，无方向）
LINK_CHUNK_REL = """
UNWIND $chunkPairs AS cp
MATCH (c1:Chunk {id: cp.c1})
MATCH (c2:Chunk {id: cp.c2})
MERGE (c1)-[r:RELATES_TO]-(c2)
ON CREATE SET r.weight = 1
ON MATCH SET r.weight = r.weight + 1
"""

QUERY_INDEX_DIM = """
SHOW INDEXES YIELD name, entityType, labelsOrTypes, properties, type, options
WHERE name = 'chunk_embedding_index'
RETURN options['indexConfig']['vector.dimensions'] AS dim
"""

def ensure_schema_and_index(driver, sample_vector_dim: int):
    with driver.session() as session:
        # 执行各独立约束创建
        for stmt in SCHEMA_STATEMENTS:
            try:
                session.run(stmt)
            except Exception as e:
                # 约束已存在或其它非致命错误打印警告即可
                print(f"[SCHEMA WARN] {stmt} -> {e}")
        # 检查 index 是否存在
        result = session.run(QUERY_INDEX_DIM).single()
        if result is None and sample_vector_dim > 0:
            try:
                session.run(VECTOR_INDEX_CYPHER, dimension=sample_vector_dim)
            except Exception as e:
                print(f"[VECTOR INDEX WARN] create failed: {e}")


def read_file_content(path: str) -> str:
    ext = os.path.splitext(path.lower())[1]
    if ext in ['.md', '.txt']:
        with open(path, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    elif ext == '.pdf':
        from PyPDF2 import PdfReader
        reader = PdfReader(path)
        texts = []
        for page in reader.pages:
            try:
                texts.append(page.extract_text() or '')
            except Exception:
                continue
        return '\n'.join(texts)
    elif ext == '.docx':
        import docx  # python-docx
        d = docx.Document(path)
        return '\n'.join(p.text for p in d.paragraphs)
    elif ext in ['.xls', '.xlsx']:
        import openpyxl
        wb = openpyxl.load_workbook(path, read_only=True, data_only=True)
        texts = []
        for ws in wb.worksheets:
            for row in ws.iter_rows(values_only=True):
                row_vals = [str(v) for v in row if v is not None]
                if row_vals:
                    texts.append(' '.join(row_vals))
        return '\n'.join(texts)
    else:
        return ''


def ingest_path(path: str, incremental: bool = False, refresh: bool = False, refresh_relations: bool = True):
    """Ingest documents from path.

    Args:
        path: 文件或目录
        incremental: 仅插入新增的 chunk（已存在 id 跳过实体/关系）
        refresh: 对已存在 chunk 重新抽取实体与图关系（会先删除旧 HAS_ENTITY/RELATES_TO/CO_OCCURS_WITH，可选 LLM REL）
        refresh_relations: 当 refresh=True 时是否重新跑 LLM 关系抽取
    注意：incremental 与 refresh 不应同时为 True；若同时为 True，则 refresh 优先。
    """
    driver = get_driver()
    all_chunks: List[Tuple[str, str]] = []  # (id, text)
    source = os.path.abspath(path)

    if os.path.isdir(path):
        for root, _, files in os.walk(path):
            for fn in files:
                full = os.path.join(root, fn)
                content = read_file_content(full)
                if not content:
                    continue
                chunks = split_text(content)
                for idx, c in enumerate(chunks):
                    chunk_id = f"{os.path.relpath(full, path)}::chunk_{idx}"
                    all_chunks.append((chunk_id, c))
    else:
        content = read_file_content(path)
        chunks = split_text(content)
        for idx, c in enumerate(chunks):
            chunk_id = f"{os.path.basename(path)}::chunk_{idx}"
            all_chunks.append((chunk_id, c))

    # 预计算所有 chunk hash
    chunk_hashes = {cid: hash_text(txt) for cid, txt in all_chunks}

    # 如果启用 hash 增量，需要预取已存在 chunk 的 hash
    existing_hashes = {}
    existing_ids: Set[str] = set()
    if incremental or refresh or settings.hash_incremental_enabled:
        with get_driver().session() as session:
            res = session.run("MATCH (c:Chunk) RETURN c.id AS id, c.hash AS hash")
            for r in res:
                existing_ids.add(r["id"])
                existing_hashes[r["id"]] = r.get("hash")

    # 需要实际重嵌入的 chunk 列表（hash 新增或变化，或刷新模式）
    to_embed: List[Tuple[str, str]] = []
    unchanged = 0
    if refresh:
        to_embed = all_chunks
    else:
        for cid, txt in all_chunks:
            if incremental or settings.hash_incremental_enabled:
                old_h = existing_hashes.get(cid)
                new_h = chunk_hashes[cid]
                if old_h and old_h == new_h:
                    # 未变化：跳过重嵌入/实体关系
                    unchanged += 1
                    continue
            to_embed.append((cid, txt))

    texts = [c[1] for c in to_embed]
    vectors = embed_texts(texts) if to_embed else []
    dim = len(vectors[0]) if vectors else 0

    ensure_schema_and_index(driver, dim)

    # 校验：已存在向量索引维度与本次嵌入向量维度不一致时提示（仅警告不终止）
    try:
        with driver.session() as session:
            res = session.run("SHOW INDEXES YIELD name, options WHERE name = $n RETURN options", n=settings.vector_index_name).single()
            if res:
                opts = res["options"] or {}
                cfg = opts.get("indexConfig", {}) if isinstance(opts, dict) else {}
                existing_dim = cfg.get("vector.dimensions")
                if existing_dim and dim and existing_dim != dim:
                    print(f"[VECTOR DIM WARN] index '{settings.vector_index_name}' dim={existing_dim} but new embedding dim={dim}; 检索可能失败或结果不正确，建议重建索引或重新嵌入。")
    except Exception as _e:  # 仅打印调试
        print(f"[VECTOR DIM CHECK WARN] {_e}")

    # existing_ids 已在前面（hash 读取阶段）填充

    with driver.session() as session:
        # 如果是 refresh，需要清理旧的图关系（实体相关 & 派生关系）；不删除 Chunk 本身
        if refresh:
            session.run("""
            MATCH (c:Chunk)-[r:HAS_ENTITY]->(e:Entity) WHERE c.id IN $ids DELETE r
            """, ids=[cid for cid, _ in all_chunks if cid in existing_ids])
            # 删除孤立实体（无其它 HAS_ENTITY 连接）
            session.run("MATCH (e:Entity) WHERE NOT (e)<-[:HAS_ENTITY]-() DETACH DELETE e")
            # 删除派生关系（RELATES_TO / CO_OCCURS_WITH / REL）
            session.run("MATCH ()-[r:RELATES_TO|CO_OCCURS_WITH|REL]->() DELETE r")

        # 写入需要重嵌入的 chunk
        for (cid, text), vec in zip(to_embed, vectors):
            session.run(UPSERT_CHUNK, id=cid, text=text, embedding=vec, source=source, hash=chunk_hashes[cid])

        # 对于未变化但尚未存 hash 的旧节点，补写 hash（无 embedding 重算）
        if (incremental or settings.hash_incremental_enabled) and not refresh:
            stale_nohash = [cid for cid, _ in all_chunks if cid in existing_ids and cid not in {c for c, _ in to_embed} and not existing_hashes.get(cid)]
            if stale_nohash:
                for batch_start in range(0, len(stale_nohash), 200):
                    batch = stale_nohash[batch_start: batch_start+200]
                    session.run("""
                    UNWIND $rows AS rid
                    MATCH (c:Chunk {id: rid}) SET c.hash = $h[rid]
                    """, rows=batch, h={cid: chunk_hashes[cid] for cid in batch})

        # 实体抽取（仅对重写入或刷新 chunk）
        if not settings.disable_entity_extract:
            changed_or_new_ids = {cid for cid, _ in to_embed} if not refresh else {cid for cid, _ in all_chunks}
            for (cid, text) in ([c for c in all_chunks if c[0] in changed_or_new_ids]):
                try:
                    raw_ents = extract_entities(text)
                except Exception as ee:
                    print(f"[WARN] entity extraction failed for {cid}: {ee}")
                    continue
                if not raw_ents:
                    continue
                # 噪声过滤：长度 < entity_min_length 或全部是符号/标点则丢弃
                cleaned = []
                import re
                for r in raw_ents:
                    if r is None:
                        continue
                    rs = r.strip()
                    if len(rs) < settings.entity_min_length:
                        continue
                    # 只包含标点/空白的跳过
                    if not re.search(r"[\w\u4e00-\u9fa5]", rs):
                        continue
                    cleaned.append(rs)
                if not cleaned:
                    continue
                # --- 实体标准化（可选）---
                if settings.entity_normalize_enabled:
                    canon_map = {}
                    alias_payload = []  # [{'name': canonical, 'aliases': [...]}]
                    for r in cleaned:
                        cano = normalize_entity(r)
                        bucket = canon_map.setdefault(cano, set())
                        bucket.add(r)
                    ents = list(canon_map.keys())
                    # 仅在首次出现时写入 aliases（属性若已有则保持）
                    for cano, forms in canon_map.items():
                        aliases = sorted([f for f in forms if f != cano])
                        if aliases:
                            alias_payload.append({"name": cano, "aliases": aliases})
                else:
                    ents = cleaned
                    alias_payload = []

                session.run(MERGE_ENTITY, entities=ents)
                if alias_payload:
                    try:
                        session.run(SET_ALIASES_IF_EMPTY, rows=alias_payload)
                    except Exception as ae:
                        print(f"[ALIAS WARN] {ae}")
                # Chunk-Entity 连接 & 共现
                pairs = [{"chunk_id": cid, "entity": e} for e in ents]
                session.run(LINK_CHUNK_ENTITY, pairs=pairs)
                if len(ents) > 1:
                    epairs = []
                    uniq = list({e: None for e in ents}.keys())
                    for i in range(len(uniq)):
                        for j in range(i+1, len(uniq)):
                            epairs.append({"e1": uniq[i], "e2": uniq[j]})
                    if epairs:
                        session.run(LINK_ENTITY_COOC, entityPairs=epairs)
        if not settings.disable_entity_extract:
            # 基于共享实体创建 Chunk-Chuck RELATES_TO
            rel_cypher = """
            MATCH (c1:Chunk)-[:HAS_ENTITY]->(e:Entity)<-[:HAS_ENTITY]-(c2:Chunk)
            WHERE c1.id < c2.id
            WITH c1, c2, count(DISTINCT e) AS overlap
            WHERE overlap >= 1
            CALL {
              WITH c1, c2, overlap
              MERGE (c1)-[r:RELATES_TO]-(c2)
              ON CREATE SET r.weight = overlap
              ON MATCH SET r.weight = coalesce(r.weight,0) + overlap
            } IN TRANSACTIONS OF 500 ROWS
            RETURN count(*) AS rels
            """
            try:
                session.run(rel_cypher)
            except Exception as re:
                print(f"[RELATES_TO WARN] {re}")

        # --- 高级关系抽取（基于 LLM） ---
    if settings.relation_extraction and (not incremental or refresh_relations or refresh or not settings.hash_incremental_enabled):
            print(f"[REL-EXTRACT] starting relation extraction window={settings.relation_window}")
            window = max(1, settings.relation_window)
            REL_MERGE = """
            UNWIND $rels AS r
            MATCH (a:Chunk {id: r.src})
            MATCH (b:Chunk {id: r.dst})
            MERGE (a)-[rel:REL {type: r.type}]->(b)
            ON CREATE SET rel.confidence = r.confidence, rel.evidence = r.evidence, rel.createdAt = timestamp()
            ON MATCH SET rel.confidence = (rel.confidence + r.confidence)/2.0, rel.evidence = r.evidence
            """
            created_rels = 0
            # 关系抽取范围：刷新模式使用全部；hash 增量时仅对发生变化的序列对做（简化：仍遍历全部，但跳过两端均未变化的配对）
            changed_set = {cid for cid, _ in to_embed} if (settings.hash_incremental_enabled and not refresh) else None
            for i, (cid_i, text_i) in enumerate(all_chunks):
                for j in range(i+1, min(i+1+window, len(all_chunks))):
                    cid_j, text_j = all_chunks[j]
                    if changed_set is not None:
                        if cid_i not in changed_set and cid_j not in changed_set:
                            continue
                    if settings.relation_debug:
                        print(f"[REL-EXTRACT PAIR] {cid_i}->{cid_j} calling LLM")
                    try:
                        rels = extract_relations(
                            cid_i, cid_j, text_i, text_j,
                            max_chars=settings.relation_chunk_trunc,
                            temperature=settings.relation_llm_temperature
                        )
                    except Exception as ree:
                        print(f"[REL-EXTRACT WARN] {cid_i}->{cid_j} {ree}")
                        continue
                    if not rels:
                        if settings.relation_debug:
                            print(f"[REL-EXTRACT EMPTY] {cid_i}->{cid_j} no LLM relations")
                        # Heuristic fallback: 如果两个 chunk 都较长且顺序上相邻，则认为 STEP_NEXT
                        if len(text_i) > 80 and len(text_j) > 50:
                            rels = [{
                                "type": "STEP_NEXT",
                                "direction": "forward",
                                "confidence": settings.rel_fallback_confidence,
                                "evidence": "heuristic_fallback"
                            }]
                            if settings.relation_debug:
                                print(f"[REL-EXTRACT DEBUG] Fallback STEP_NEXT {cid_i}->{cid_j}")
                        else:
                            if settings.relation_debug:
                                print(f"[REL-EXTRACT DEBUG] no relation {cid_i}->{cid_j}")
                            continue
                    # 方向处理：forward a->b; backward b->a; undirected 统一 a->b
                    payload = []
                    for r in rels:
                        direction = r.get('direction', 'undirected')
                        if direction == 'backward':
                            payload.append({
                                'src': cid_j,
                                'dst': cid_i,
                                'type': r.get('type', 'REL')[:30],
                                'confidence': r.get('confidence', 0.5),
                                'evidence': r.get('evidence', '')[:200]
                            })
                        else:
                            payload.append({
                                'src': cid_i,
                                'dst': cid_j,
                                'type': r.get('type', 'REL')[:30],
                                'confidence': r.get('confidence', 0.5),
                                'evidence': r.get('evidence', '')[:200]
                            })
                    if payload:
                        try:
                            session.run(REL_MERGE, rels=payload)
                            created_rels += len(payload)
                        except Exception as me:
                            print(f"[REL-MERGE WARN] {me}")
            if created_rels:
                print(f"[REL-EXTRACT INFO] created/updated {created_rels} relations (LLM + fallback)")
            else:
                print("[REL-EXTRACT INFO] no relations produced")

    # --- 共现边清理（prune 低计数）---
    try:
        if settings.cooccur_min_count > 1:
            with driver.session() as session:
                session.run("""
                MATCH ()-[r:CO_OCCURS_WITH]-() WHERE coalesce(r.count,0) < $min DETACH DELETE r
                """, min=settings.cooccur_min_count)
    except Exception as pe:
        print(f"[PRUNE COOCCUR WARN] {pe}")

    driver.close()
    return {
        "chunks_total": len(all_chunks),
        "chunks_embedded": len(to_embed),
        "chunks_unchanged": unchanged,
        "path": path,
        "mode": "refresh" if refresh else ("incremental" if incremental else "full"),
        "hash_incremental_enabled": settings.hash_incremental_enabled,
        "entity_extraction": not settings.disable_entity_extract,
        "relation_extraction": settings.relation_extraction and (not incremental or refresh_relations or refresh or not settings.hash_incremental_enabled),
        "entity_normalize_enabled": settings.entity_normalize_enabled
    }

if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else './docs'
    print(ingest_path(p))
