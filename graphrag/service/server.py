from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse, JSONResponse
from ..schemas.api import IngestRequest, QueryRequest
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
    path: str = Query(..., description="文件或目录绝对路径"),
    incremental: bool = Query(False),
    refresh: bool = Query(False),
    refresh_relations: bool = Query(True),
    checkpoint: bool = Query(True),
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
        answer_gen, sources = answer_question(req.question, return_sources=True, stream=req.stream)
    except Exception as e:
        import traceback
        return JSONResponse({
            "error": "query_failed",
            "message": str(e),
            "trace": traceback.format_exc(limit=3)
        }, status_code=500)
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
