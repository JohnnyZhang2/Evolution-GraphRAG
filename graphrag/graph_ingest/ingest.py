# Evolution RAG Ingestion Pipeline
# Handles splitting, embedding, entity & relation extraction, and graph writes.
# Copyright (c) 2025 EvolutionAI Studio
# All Rights Holder: Johnny Zhang
# License: MIT (see LICENSE)

import os
import json
from neo4j import GraphDatabase
from typing import List, Dict, Tuple, Set, Callable, Optional
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
ON CREATE SET c.createdAt = timestamp()
SET c.text = $text, c.embedding = $embedding, c.source = $source, c.hash = $hash, c.updatedAt = timestamp()
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


def ingest_path(path: str, incremental: bool = False, refresh: bool = False, refresh_relations: bool = True, collect_progress: bool = False, checkpoint: bool = True, progress_callback: Optional[Callable[[Dict], None]] = None):
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
    progress: List[Dict] = []

    def record(stage: str, detail: str = "", current: int | None = None, total: int | None = None):
        if not collect_progress:
            return
        item = {"stage": stage}
        if detail:
            item["detail"] = detail
        if current is not None:
            item["current"] = current
        if total is not None:
            item["total"] = total
        progress.append(item)
        if progress_callback:
            try:
                progress_callback(item)
            except Exception as cb_e:
                print(f"[PROGRESS CALLBACK WARN] {cb_e}")

    record("scan_input", f"path={path}")
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
    record("hash_computed", current=len(chunk_hashes), total=len(chunk_hashes))

    # 如果启用 hash 增量，需要预取已存在 chunk 的 hash
    existing_hashes = {}
    existing_ids: Set[str] = set()
    if incremental or refresh or settings.hash_incremental_enabled:
        # 避免 Neo4j 在库尚未有任何 hash 属性时返回 UnknownPropertyKeyWarning：
        # 先探测是否存在至少一个 Chunk，再安全读取 hash。
        with get_driver().session() as session:
            probe = session.run("MATCH (c:Chunk) RETURN c.id AS id LIMIT 1").single()
            if probe:
                res = session.run("MATCH (c:Chunk) RETURN c.id AS id, c.hash AS hash")
                for r in res:
                    existing_ids.add(r["id"])
                    existing_hashes[r["id"]] = r.get("hash")
            # 若无任何 Chunk 节点，保持 existing_ids / hashes 为空即可
        record("existing_scan", current=len(existing_ids))

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

    # ---- Checkpoint 文件路径 ----
    ck_path = os.path.join(os.path.dirname(os.path.abspath(path if os.path.exists(path) else '.')), 
                           f".ingest_ck_{os.path.basename(path).replace(os.sep,'_')}.json")
    ck = {"chunks": {}, "entities_done": False, "rels_done": False, "rel_pairs": {}}
    if checkpoint and os.path.exists(ck_path) and not refresh:
        try:
            with open(ck_path, 'r', encoding='utf-8') as cf:
                ck = json.load(cf)
            record("checkpoint_load", detail=ck_path)
        except Exception as ce:
            print(f"[CK WARN] load failed: {ce}")
    # 过滤已完成 embedding 的 chunk（ck['chunks'][cid]['emb']=True）
    pending_pairs = []
    for cid, txt in to_embed:
        meta = ck.get("chunks", {}).get(cid)
        if meta and meta.get("emb") and not refresh:
            continue
        pending_pairs.append((cid, txt))
    texts = [c[1] for c in pending_pairs]
    from ..embedding.client import embed_texts_iter
    dim = 0
    if pending_pairs:
        record("embedding", detail="start", current=0, total=len(to_embed))
        collected = 0
        driver_session = get_driver().session()
        try:
            for start_idx, vec_batch in embed_texts_iter(texts, progress_cb=lambda ev: record("embedding_batch", detail=ev.get("event",""), current=ev.get("batches_done"), total=ev.get("batches_total_initial"))):
                # 将本批写入 Neo4j (UPSERT_CHUNK)
                slice_pairs = pending_pairs[start_idx: start_idx + len(vec_batch)]
                with driver_session.begin_transaction() as tx:
                    for (cid, text), vec in zip(slice_pairs, vec_batch):
                        tx.run(UPSERT_CHUNK, id=cid, text=text, embedding=vec, source=source, hash=chunk_hashes[cid])
                if vec_batch:
                    dim = len(vec_batch[0])
                # 更新 checkpoint
                if checkpoint:
                    for (cid, _), _v in zip(slice_pairs, vec_batch):
                        ck.setdefault("chunks", {}).setdefault(cid, {})["emb"] = True
                    try:
                        with open(ck_path, 'w', encoding='utf-8') as cf:
                            json.dump(ck, cf, ensure_ascii=False, indent=2)
                    except Exception as ws:
                        print(f"[CK WARN] write failed: {ws}")
                collected += len(vec_batch)
                if collected % 50 == 0 or collected == len(texts):
                    record("embedding_progress", current=collected, total=len(to_embed))
            record("embedding", detail="done", current=len(to_embed), total=len(to_embed))
        finally:
            driver_session.close()
    else:
        record("embedding", detail="skip_all_cached", current=len(to_embed), total=len(to_embed))

    ensure_schema_and_index(driver, dim)
    record("schema_index")

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

        # 已改为批次即时写入；这里仅为兼容旧流程统计写入完成事件
        if to_embed:
            done_emb = sum(1 for cid,_ in to_embed if ck.get("chunks", {}).get(cid, {}).get("emb"))
            record("chunks_written", current=done_emb, total=len(to_embed))

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
            # 跳过已做实体抽取的 chunk（checkpoint）
            if checkpoint and not refresh:
                changed_or_new_ids = {cid for cid in changed_or_new_ids if not ck.get("chunks", {}).get(cid, {}).get("ent")}
            total_entity_targets = len(changed_or_new_ids)
            processed_entities = 0
            for (cid, text) in ([c for c in all_chunks if c[0] in changed_or_new_ids]):
                try:
                    raw_ents = extract_entities(text)
                except Exception as ee:
                    print(f"[WARN] entity extraction failed for {cid}: {ee}")
                    continue
                if not raw_ents:
                    continue
                # 兼容 typed 模式：提取原始 name 列表供后续清洗
                if settings.entity_typed_mode and raw_ents and isinstance(raw_ents[0], dict):
                    raw_names = [e.get('name','').strip() for e in raw_ents if isinstance(e, dict)]
                else:
                    raw_names = [ (e.strip() if isinstance(e,str) else '') for e in raw_ents ]
                # 噪声过滤：长度 < entity_min_length 或全部是符号/标点则丢弃
                cleaned = []
                import re
                for rs in raw_names:
                    if not rs:
                        continue
                    if len(rs) < settings.entity_min_length:
                        continue
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
                # 写入实体类型（entity_typed_mode 下 extract_entities 返回 dict 列表，我们已展开为 ents 名称列表且保留映射）
                if settings.entity_typed_mode and raw_ents and isinstance(raw_ents[0], dict):
                    try:
                        name_type_pairs = {}
                        for tr in raw_ents:
                            if not isinstance(tr, dict):
                                continue
                            nm = tr.get('name'); tp = tr.get('type')
                            if not nm or not tp:
                                continue
                            # 若标准化开启，映射到 canonical
                            key_name = normalize_entity(nm) if settings.entity_normalize_enabled else nm
                            if key_name in ents:
                                name_type_pairs.setdefault(key_name, tp)
                        if name_type_pairs:
                            rows = [{'name': k, 'type': v} for k,v in name_type_pairs.items()]
                            session.run("""
                            UNWIND $rows AS r
                            MATCH (e:Entity {name:r.name})
                            SET e.type = COALESCE(e.type, r.type)
                            """, rows=rows)
                    except Exception as te:
                        print(f"[ENTITY TYPE WARN] {te}")
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
                # 标记 checkpoint
                if checkpoint:
                    ck.setdefault("chunks", {}).setdefault(cid, {})["ent"] = True
                    try:
                        with open(ck_path, 'w', encoding='utf-8') as cf:
                            json.dump(ck, cf, ensure_ascii=False, indent=2)
                    except Exception as ws:
                        print(f"[CK WARN] write failed: {ws}")
                processed_entities += 1
                if processed_entities % 50 == 0:
                    record("entity_extraction", current=processed_entities, total=total_entity_targets)
            if total_entity_targets:
                record("entity_extraction", current=total_entity_targets, total=total_entity_targets)
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
                record("relates_to")
            except Exception as re:
                print(f"[RELATES_TO WARN] {re}")

        # --- 高级关系抽取（基于 LLM） ---
        if settings.relation_extraction and (not incremental or refresh_relations or refresh or not settings.hash_incremental_enabled):
            print(f"[REL-EXTRACT] starting relation extraction window={settings.relation_window}")
            record("relation_start", detail=f"window={settings.relation_window}")
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
            # 预估总 pair 数（用于进度显示）
            total_pairs = 0
            # 生成所有候选 pair，并为其分配顺序 index，支持断点
            pairs_list: List[Tuple[str,str,int,int]] = []  # (cid_i, cid_j, i, j)
            for i, (cid_i, _text_i) in enumerate(all_chunks):
                for j in range(i+1, min(i+1+window, len(all_chunks))):
                    cid_j, _ = all_chunks[j]
                    if changed_set is not None and cid_i not in changed_set and cid_j not in changed_set:
                        continue
                    pairs_list.append((cid_i, cid_j, i, j))
            total_pairs = len(pairs_list)
            # 过滤已处理 pair
            remaining_pairs = []
            rel_pairs_ck = ck.get("rel_pairs", {}) if checkpoint else {}
            for (cid_i, cid_j, i, j) in pairs_list:
                key = f"{cid_i}|{cid_j}"
                if checkpoint and rel_pairs_ck.get(key):
                    continue
                remaining_pairs.append((cid_i, cid_j, i, j))
            for (cid_i, cid_j, i, j) in remaining_pairs:
                text_i = all_chunks[i][1]
                text_j = all_chunks[j][1]
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
                    # 标记失败也视为已处理以避免卡住（可选：写入 fail 标志）
                    if checkpoint:
                        ck.setdefault("rel_pairs", {})[f"{cid_i}|{cid_j}"] = {"err": True}
                        try:
                            with open(ck_path, 'w', encoding='utf-8') as cf:
                                json.dump(ck, cf, ensure_ascii=False, indent=2)
                        except Exception as ws:
                            print(f"[CK WARN] write failed: {ws}")
                    continue
                if not rels:
                    if settings.relation_debug:
                        print(f"[REL-EXTRACT EMPTY] {cid_i}->{cid_j} no LLM relations")
                    # Fallback STEP_NEXT
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
                        # 仍要标记已处理
                        if checkpoint:
                            ck.setdefault("rel_pairs", {})[f"{cid_i}|{cid_j}"] = {"empty": True}
                            try:
                                with open(ck_path, 'w', encoding='utf-8') as cf:
                                    json.dump(ck, cf, ensure_ascii=False, indent=2)
                            except Exception as ws:
                                print(f"[CK WARN] write failed: {ws}")
                        continue
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
                # 更新 checkpoint
                if checkpoint:
                    ck.setdefault("rel_pairs", {})[f"{cid_i}|{cid_j}"] = {"rels": len(payload)}
                    try:
                        with open(ck_path, 'w', encoding='utf-8') as cf:
                            json.dump(ck, cf, ensure_ascii=False, indent=2)
                    except Exception as ws:
                        print(f"[CK WARN] write failed: {ws}")
                if total_pairs:
                    processed_pairs = len(rel_pairs_ck) + 1  # 近似：已在 ck 中的数量 + 当前
                    if processed_pairs % 50 == 0:
                        record("relation_extraction", current=processed_pairs, total=total_pairs)
            if created_rels:
                print(f"[REL-EXTRACT INFO] created/updated {created_rels} relations (LLM + fallback)")
                record("relation_extraction_done", detail=f"created={created_rels}")
            else:
                print("[REL-EXTRACT INFO] no relations produced")
                record("relation_extraction_done", detail="none")

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
    result = {
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
    if collect_progress:
        result["progress"] = progress
    return result

if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else './docs'
    print(ingest_path(p))
