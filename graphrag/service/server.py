from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from ..schemas.api import IngestRequest, QueryRequest, QueryChunk, ChatMessage
from ..graph_ingest.ingest import ingest_path
import threading
import queue
import json
import time
from fastapi import Query
from ..retriever.retrieve import answer_question, _question_embedding_cache, _answer_cache
from ..retriever.retrieve import vector_search, expand_context, get_driver
from ..retriever.bm25 import bm25_index
from ..retriever.rerank import rerank_post_score
from ..utils.graph_rank import get_degree_scores
from ..config.settings import get_settings
from ..utils.diagnostics import run_all as run_diagnostics
from ..retriever.retrieve import build_prompt
from ..llm.client import extract_relations

settings = get_settings()

app = FastAPI(title="Evolution RAG QA")

@app.get("/health")
def health():
    return {"status": "ok", "api_version": settings.api_version}

@app.post("/ingest")
def ingest(
    req: IngestRequest,
    incremental: bool = Query(False, description="仅新增文件/Chunk"),
    refresh: bool = Query(False, description="刷新已存在 Chunk 的实体与关系"),
    refresh_relations: bool = Query(True, description="刷新模式下是否重新抽取 LLM 关系"),
    progress: bool = Query(False, description="返回详细进度阶段列表")
):
    try:
        result = ingest_path(
            req.path,
            incremental=incremental,
            refresh=refresh,
            refresh_relations=refresh_relations,
            collect_progress=progress
        )
        return result
    except Exception as e:
        import traceback
        tb = traceback.format_exc(limit=3)
        raise HTTPException(status_code=500, detail=f"{e}; trace={tb}")

@app.get("/ingest/stream")
def ingest_stream(
    path: str = Query(..., description="文件或目录绝对路径 或 用于关系增量补齐的 source 路径"),
    incremental: bool = Query(False, description="常规增量：仅新增/变化 chunk 做嵌入与实体"),
    refresh: bool = Query(False, description="刷新：重做实体/共现/RELATES_TO 及可选 LLM 关系"),
    refresh_relations: bool = Query(True, description="刷新模式下是否重新跑 LLM 关系"),
    checkpoint: bool = Query(True, description="启用断点续跑"),
    inc_rel_only: bool = Query(False, description="仅做新增区间 LLM 语义关系补齐(不嵌入/不抽实体)"),
    new_after: str | None = Query(None, description="旧区间最后一个 chunk id，之后视为新增"),
    detect_after: bool = Query(False, description="自动探测旧/新增分界，需要提供 new_count"),
    new_count: int | None = Query(None, description="最近新增 chunk 数量 (与 detect_after 搭配)"),
    rel_window: int = Query(settings.relation_window, description="关系窗口大小，用于 inc_rel_only"),
    rel_truncate: int = Query(settings.relation_chunk_trunc, description="关系抽取截断字符"),
    rel_temperature: float = Query(settings.relation_llm_temperature, description="关系抽取温度"),
):
    """通过 SSE 流式推送 ingest 进度事件。

    事件格式：data: {"stage":"...", "detail":"...", "current":1, "total":100}\n\n
    结束：发送一条 data: {"stage":"done", ...}\n\n
    错误：发送 data: {"stage":"error", "error":"..."}\n\n 后终止。
    """
    q: "queue.Queue[dict]" = queue.Queue()
    stop_flag = {"stopped": False}

    def progress_cb(ev: dict):
        try:
            q.put(ev)
        except Exception:
            pass

    def worker():
        try:
            # --- 仅关系增量模式 ---
            if inc_rel_only:
                # 校验参数
                if not new_after and not (detect_after and new_count):
                    raise ValueError("inc_rel_only 模式需要提供 new_after 或 (detect_after + new_count)")
                from neo4j import GraphDatabase
                driver = GraphDatabase.driver(settings.neo4j_uri, auth=(settings.neo4j_user, settings.neo4j_password))
                created = 0
                skipped = 0
                pairs_total = 0
                try:
                    with driver.session() as session:
                        # 读取指定 source 的所有 chunk
                        recs = session.run("MATCH (c:Chunk {source:$s}) RETURN c.id AS id, c.text AS text ORDER BY c.id", s=path)
                        chunks = [(r["id"], r["text"]) for r in recs]
                        if not chunks:
                            raise ValueError("未找到指定 source 的任何 Chunk，请确认已先完成 ingest")
                        ids = [c[0] for c in chunks]
                        # 自动探测分界
                        if detect_after:
                            if not new_count or new_count >= len(ids):
                                raise ValueError("detect_after 需要合理的 new_count 且小于总 chunk 数")
                            split_index = len(ids) - new_count - 1
                            new_after_id = ids[split_index]
                        else:
                            if new_after not in ids:
                                raise ValueError(f"提供的 new_after 不在当前 chunk 集合: {new_after}")
                            new_after_id = new_after  # type: ignore
                        new_index_start = ids.index(new_after_id) + 1
                        new_ids = ids[new_index_start:]
                        old_ids = ids[:new_index_start]
                        if not new_ids:
                            q.put({"stage": "relation_incremental_done", "created": 0, "skipped": 0, "pairs": 0, "detail": "无新增 chunk"})
                            q.put({"stage": "result", "result": {"mode": "relations_incremental", "created": 0, "skipped": 0, "pairs": 0, "old": len(old_ids), "new": len(new_ids)}})
                            q.put({"stage": "done"})
                            return
                        window = max(1, rel_window)
                        tail_old = old_ids[-window:]
                        pairs: list[tuple[str,str]] = []
                        # new 内部窗口配对
                        for i, src in enumerate(new_ids):
                            for j in range(i+1, min(i+1+window, len(new_ids))):
                                pairs.append((src, new_ids[j]))
                        # old 尾部 -> 前若干 new（保持与 CLI 一致：仅前 window 个 new）
                        for o in tail_old:
                            for n in new_ids[:window]:
                                pairs.append((o, n))
                        # 去重
                        seen = set()
                        dedup_pairs = []
                        for p in pairs:
                            if p not in seen:
                                seen.add(p)
                                dedup_pairs.append(p)
                        pairs = dedup_pairs
                        pairs_total = len(pairs)
                        q.put({"stage": "relation_incremental_start", "old": len(old_ids), "new": len(new_ids), "window": window, "pairs": pairs_total})
                        # 为快速查 text 建立 map
                        text_map = {cid: txt for cid, txt in chunks}
                        # 查询已存在 (src,dst,type)
                        involved_ids = list({p[0] for p in pairs} | {p[1] for p in pairs})
                        existing_rels = session.run(
                            """
                            UNWIND $lst AS x
                            UNWIND $lst AS y
                            WITH x,y WHERE x<>y
                            MATCH (a:Chunk {id:x})-[r:REL]->(b:Chunk {id:y})
                            RETURN a.id AS src, b.id AS dst, r.type AS type
                            """, lst=involved_ids
                        )
                        existing = {(r["src"], r["dst"], r["type"]) for r in existing_rels}
                        REL_MERGE = """
                        UNWIND $rels AS r
                        MATCH (a:Chunk {id:r.src})
                        MATCH (b:Chunk {id:r.dst})
                        MERGE (a)-[rel:REL {type:r.type}]->(b)
                        ON CREATE SET rel.confidence=r.confidence, rel.evidence=r.evidence, rel.createdAt=timestamp()
                        ON MATCH SET rel.confidence=(rel.confidence + r.confidence)/2.0, rel.evidence=r.evidence
                        """
                        processed = 0
                        for (src, dst) in pairs:
                            try:
                                rels = extract_relations(src, dst, text_map[src], text_map[dst], max_chars=rel_truncate, temperature=rel_temperature)
                            except Exception as ee:
                                q.put({"stage": "relation_incremental_warn", "pair": f"{src}->{dst}", "error": str(ee)})
                                processed += 1
                                continue
                            if not rels:
                                processed += 1
                                if processed % 20 == 0:
                                    q.put({"stage": "relation_incremental_progress", "processed": processed, "total": pairs_total, "created": created, "skipped": skipped})
                                continue
                            payload = []
                            for r in rels:
                                direction = r.get('direction','undirected')
                                if direction == 'backward':
                                    s, d = dst, src
                                else:
                                    s, d = src, dst
                                key = (s, d, r.get('type','REL')[:30])
                                if key in existing:
                                    skipped += 1
                                    continue
                                payload.append({
                                    'src': s,
                                    'dst': d,
                                    'type': r.get('type','REL')[:30],
                                    'confidence': r.get('confidence',0.5),
                                    'evidence': r.get('evidence','')[:200]
                                })
                            if payload:
                                try:
                                    session.run(REL_MERGE, rels=payload)
                                    for p in payload:
                                        existing.add((p['src'], p['dst'], p['type']))
                                    created += len(payload)
                                except Exception as me:
                                    q.put({"stage": "relation_incremental_warn", "pair": f"{src}->{dst}", "error": str(me)})
                            processed += 1
                            if processed % 20 == 0 or processed == pairs_total:
                                q.put({"stage": "relation_incremental_progress", "processed": processed, "total": pairs_total, "created": created, "skipped": skipped})
                    # 汇总
                    result = {"mode": "relations_incremental", "created": created, "skipped": skipped, "pairs": pairs_total, "old": len(old_ids), "new": len(new_ids)}
                    q.put({"stage": "relation_incremental_done", **result})
                    q.put({"stage": "result", "result": result})
                    q.put({"stage": "done"})
                finally:
                    driver.close()
                return
            # --- 正常 ingest 模式 ---
            result = ingest_path(
                path,
                incremental=incremental,
                refresh=refresh,
                refresh_relations=refresh_relations,
                collect_progress=True,
                checkpoint=checkpoint,
                progress_callback=progress_cb
            )
            q.put({"stage": "result", "result": result})
            q.put({"stage": "done"})
        except Exception as e:
            q.put({"stage": "error", "error": str(e)})
        finally:
            stop_flag["stopped"] = True

    thread = threading.Thread(target=worker, daemon=True)
    thread.start()

    def event_stream():
        # 发送初始握手事件
        yield "event: start\ndata: {}\n\n"
        while not stop_flag["stopped"] or not q.empty():
            try:
                ev = q.get(timeout=0.5)
            except Exception:
                continue
            # 区分 result 事件（非进度）
            if ev.get("stage") == "result":
                payload = json.dumps(ev, ensure_ascii=False)
                yield f"event: result\ndata: {payload}\n\n"
            else:
                payload = json.dumps(ev, ensure_ascii=False)
                yield f"data: {payload}\n\n"
        # 结束（若未显式 done）
        if not any(ev.get("stage") == "done" for ev in []):
            yield "data: {\"stage\":\"done\"}\n\n"

    headers = {
        "Cache-Control": "no-cache",
        "Content-Type": "text/event-stream; charset=utf-8",
        "Connection": "keep-alive",
        "X-Accel-Buffering": "no"
    }
    return StreamingResponse(event_stream(), headers=headers, media_type="text/event-stream")

@app.post("/query")
def query(req: QueryRequest):
    try:
        # 解析外部上下文
        extra_ctx = []
        if req.context:
            for c in req.context:
                if isinstance(c, str):
                    extra_ctx.append({'text': c})
                elif isinstance(c, QueryChunk):
                    extra_ctx.append({'id': c.id, 'text': c.text})
                elif isinstance(c, dict) and 'text' in c:
                    extra_ctx.append({'id': c.get('id'), 'text': c['text']})
        # 解析历史
        history = []
        if req.history:
            for m in req.history:
                if isinstance(m, ChatMessage):
                    history.append({'role': m.role, 'content': m.content})
                elif isinstance(m, dict) and 'role' in m and 'content' in m:
                    history.append({'role': m['role'], 'content': m['content']})
        # 先走原始检索/排序 + 注入外部上下文与历史
        answer_gen, sources = answer_question(
            req.question,
            return_sources=True,
            stream=req.stream,
            extra_contexts=extra_ctx or None,
            history=history or None,
            system_prefix=None
        )
    except Exception as e:
        import traceback
        return JSONResponse({
            "error": "query_failed",
            "message": str(e),
            "trace": traceback.format_exc(limit=3)
        }, status_code=500)
    # 将外部上下文与历史集成到提示词（仅在我们自行组装 messages 的非流式分支中可直接控制；
    # 流式分支按原逻辑维持最小变更，后续可扩展为显式 messages 管线）
    if req.stream:
        def plain_stream():
            try:
                for chunk in answer_gen:
                    yield chunk
            except Exception as ie:
                yield f"\n[ERROR] {ie}\n"
            if sources:
                refs = "\n\n[SOURCES]\n" + "\n".join(
                    f"- {i}. {s['id']} (rank={s.get('rank')}, reason={s.get('reason')}, score={s.get('score')})" for i, s in enumerate(sources, 1)
                )
                yield refs
        return StreamingResponse(plain_stream(), media_type="text/plain; charset=utf-8")
    else:
        text = ''.join(list(answer_gen))
        try:
            if sum(ch in text for ch in 'âÂÏ¸') > 3:
                repaired = text.encode('latin-1', errors='ignore')
                text_utf8 = repaired.decode('utf-8', errors='ignore')
                if text_utf8.count('的') + text_utf8.count('流程') > text.count('的') + text.count('流程'):
                    text = text_utf8
        except Exception:
            pass
        # 为非流式回答末尾添加 References 映射（不修改原 answer 主体，单独提供 references 字段）
        # 引用后处理
        import re
        used_tags = set(re.findall(r"\[S(\d+)\]", text))
        references = []
        unused = []
        for s in sources:
            rk = s.get('rank')
            if not rk:
                continue
            label = f"S{rk}"
            entry = {"label": label, "id": s['id'], "reason": s.get('reason'), "score": s.get('score'),
                     "composite_score": s.get('composite_score'), "rel_type": s.get('rel_type'), "rel_conf": s.get('rel_conf')}
            references.append(entry)
            if str(rk) not in used_tags:
                unused.append(label)
        warnings = []
        if unused:
            warnings.append({
                "type": "unused_sources",
                "detail": f"这些来源已提供但未在回答中引用: {', '.join(unused)}"
            })
        # 检测未引用的数字事实（简单启发：出现 >=4 位数字的年份或数值段未被引用附近无 [S#]）
        numeric_spans = list(re.finditer(r"\d{4,}", text))
        for m in numeric_spans:
            # 在数字后 12 个字符内是否有 [S
            tail = text[m.end(): m.end()+12]
            if '[S' not in tail:
                warnings.append({
                    "type": "unreferenced_numeric",
                    "detail": f"数字'{m.group()}' 可能需要引用标记"
                })
        # 统计回答上下文实体频次（基于已检索的 contexts -> sources id 集合）
        entities_stat = []
        try:
            with get_driver().session() as session:
                chunk_ids = [s['id'] for s in sources if not s['id'].startswith('__')]
                if chunk_ids:
                    recs = session.run("""
                    UNWIND $ids AS cid
                    MATCH (c:Chunk {id: cid})-[:HAS_ENTITY]->(e:Entity)
                    RETURN e.name AS name, count(*) AS freq
                    ORDER BY freq DESC, name
                    LIMIT 50
                    """, ids=chunk_ids)
                    entities_stat = [{"name": r["name"], "freq": r["freq"]} for r in recs]
        except Exception as est:
            entities_stat = [{"error": str(est)}]
        payload = {"answer": text, "sources": sources, "references": references, "entities": entities_stat}
        if warnings:
            payload["warnings"] = warnings
        return JSONResponse(payload)

@app.get("/diagnostics")
def diagnostics():
    return run_diagnostics()

# ---- Cache management endpoints ----
@app.get("/cache/stats")
def cache_stats():
    try:
        return {
            "embedding_cache_size": len(_question_embedding_cache),
            "answer_cache_size": len(_answer_cache),
            "embedding_cache_keys": list(_question_embedding_cache.keys())[:20],
            "answer_cache_keys": list(_answer_cache.keys())[:20]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/cache/clear")
def cache_clear():
    try:
        _question_embedding_cache.clear()
        _answer_cache.clear()
        return {"cleared": True}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

@app.post('/ranking/preview')
def ranking_preview(req: QueryRequest):
    """返回检索与扩展阶段的打分拆解（不调用回答 LLM）。"""
    try:
        hits, emb, degraded, warn = vector_search(req.question, settings.top_k)
        bm25_scores = {}
        if settings.bm25_enabled and not degraded:
            try:
                if not bm25_index._built:  # type: ignore
                    with get_driver().session() as session:
                        recs = session.run("MATCH (c:Chunk) RETURN c.id AS id, c.text AS text LIMIT 2000")
                        docs = [{"id": r["id"], "text": r["text"]} for r in recs]
                    bm25_index.build(docs)
                bm25_scores = bm25_index.score(req.question, top_k=settings.top_k * 3)
            except Exception as be:
                import logging
                logging.warning(f"[BM25 WARN] {be}")
        expand = [] if degraded else expand_context([h['id'] for h in hits], limit=settings.top_k * 2, hops=settings.expand_hops)
        merged = {h['id']: h for h in hits}
        for e in expand:
            merged.setdefault(e['id'], e)
        contexts = list(merged.values())
        # 复制 composite 逻辑（与 answer_question 一致）
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
            if settings.bm25_enabled and not degraded:
                bscore = bm25_scores.get(c['id'])
                if bscore is not None:
                    bonus += settings.bm25_weight * bscore
            if reason == 'entity':
                bonus += settings.rel_weight_entity
            if reason == 'relates':
                bonus += settings.rel_weight_relates
            elif reason == 'cooccur':
                bonus += settings.rel_weight_cooccur
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
                bonus += rel_base * conf
            if settings.graph_rank_enabled:
                dsc = degree_scores.get(c['id'])
                if dsc:
                    bonus += settings.graph_rank_weight * dsc
            c['base_norm'] = round(base,4)
            c['bonus'] = round(bonus,4)
            c['composite_score'] = round(base + bonus,4)
        contexts.sort(key=lambda x: x.get('composite_score',0), reverse=True)
        if settings.rerank_enabled:
            contexts = rerank_post_score(req.question, contexts)
        return {
            'question': req.question,
            'degraded': degraded,
            'warn': warn,
            'items': [
                {
                    'id': c['id'],
                    'reason': c.get('reason'),
                    'score_raw': c.get('score'),
                    'base_norm': c.get('base_norm'),
                    'bonus': c.get('bonus'),
                    'composite_score': c.get('composite_score'),
                    'final_score': c.get('final_score', c.get('composite_score')),
                    'rel_type': c.get('rel_type'),
                    'rel_conf': c.get('rel_conf')
                } for c in contexts[: settings.top_k * 3]
            ]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
