# Evolution RAG API Reference (English)

> Version: `api_version` is exposed via `/health` and `/diagnostics`.

## Architecture Overview (Quick)

End‑to‑end: Hybrid retrieval + subgraph expansion (≤2 hops with quotas/reserve) + path‑level scoring + optional rerank; `/query` accepts external `context` and conversational `history`. External contexts show up in `sources` (reason=external) and can be cited with `[S#]`.

## Overview

Evolution RAG is a local, explainable Graph-augmented Retrieval QA system built on FastAPI + Neo4j. This document describes REST endpoints, request/response payloads, scoring interpretation, and troubleshooting notes.

Key capabilities:

- Hybrid retrieval (vector + optional BM25 + graph relation bonuses + optional graph degree + future rerank)
- Incremental ingestion via content hashing
- Entity normalization & alias aggregation
- LLM-derived semantic relations plus lightweight co-occurrence / window relations
- Explainable scoring preview (`/ranking/preview`)
- Diagnostics & feature flags (`/diagnostics`)

## Base URL

```text
http://localhost:8010
```

## Authentication

Currently no authentication layer. If you deploy externally, add a reverse proxy (e.g. Traefik / Nginx) plus auth (token header, mTLS, etc.).

## Endpoints Summary

| Method | Path | Purpose |
|--------|------|---------|
| GET | `/health` | Quick liveness + api_version |
| POST | `/ingest` | Ingest file or directory into Neo4j graph |
| POST | `/query` | Ask a question (stream or JSON) |
| GET | `/diagnostics` | Environment, feature flags, weights, relation stats |
| GET | `/cache/stats` | Show embedding & answer cache sizes |
| POST | `/cache/clear` | Clear caches |
| POST | `/ranking/preview` | Show retrieval scoring breakdown (no LLM answer) |
| GET | `/ingest/stream` | Server-Sent Events streaming ingest progress + final result |

## Versioning

- Field: `api_version` (string) appears in `/health` and `/diagnostics`.
- Increment policy:
  - PATCH (x.y.z -> x.y.z+1): Non-breaking additions (new optional fields)
  - MINOR (x.y.z -> x.y+1.0): Backwards-compatible new endpoints/features
  - MAJOR (x.y.z -> x+1.0.0): Breaking changes (field rename/removal, semantics change)

## 1. Health

GET `/health`

Response:

```json
{
  "status": "ok",
  "api_version": "0.1.0"
}
```

## 2. Ingest

POST `/ingest?incremental=true&refresh=false&refresh_relations=true`

Query Params:

- `incremental` (bool, default false): Skip unchanged chunks via hash
- `refresh` (bool, default false): Rebuild entities & relations for already ingested chunks
- `refresh_relations` (bool, default true): When refresh=true, re-run LLM relation extraction

Request Body:

```jsonc
{
  "path": "/absolute/path/to/file/or/dir"
}
```

Typical Response (fields may evolve):

```jsonc
{
  "files_processed": 1,
  "chunks_new": 12,
  "chunks_skipped": 4,
  "entities_created": 23,
  "relations_created": 17,
  "cooccur_edges": 9,
  "llm_rel_edges": 8,
  "elapsed_sec": 2.41,
  "incremental": true
}
```

Error (HTTP 500) example:

```json
{
  "detail": "some error; trace=Traceback ..."
}
```

### 2.1 Streaming Ingest Progress (SSE)

GET `/ingest/stream?path=/abs/file/or/dir&incremental=false&refresh=false&refresh_relations=true&checkpoint=true`

Media Type: `text/event-stream`

Event format examples (each ends with a blank line):

```text
event: start
data: {}

data: {"stage":"scan_input","detail":"path=/data/book.txt"}

data: {"stage":"embedding_batch","detail":"batch_ok","current":3,"total":10}

data: {"stage":"embedding_progress","current":320,"total":1300}

data: {"stage":"relation_extraction","current":150,"total":860}

event: result
data: {"stage":"result","result":{"chunks_total":1300,"chunks_embedded":820,...}}

data: {"stage":"done"}
```

Stages emitted mirror the synchronous `progress=true` array: scan_input, hash_computed, existing_scan, embedding / embedding_batch / embedding_progress, schema_index, entity_extraction, relates_to, relation_start, relation_extraction, relation_extraction_done, plus `done` and a synthetic `result` event with the final JSON.

If an error occurs:

```json
{"stage":"error","error":"<message>"}
```

#### Relations-Only Incremental (semantic patch)

When you only appended new chunks and want to supplement LLM semantic relations without re-running embedding/entity phases you can call the same streaming endpoint in relations-only mode:

```text
GET /ingest/stream?path=/abs/source/file.txt&inc_rel_only=true&new_after=<last_old_chunk_id>&rel_window=2
```

Or automatic boundary detection if you just know how many new chunks were appended:

```text
GET /ingest/stream?path=/abs/source/file.txt&inc_rel_only=true&detect_after=true&new_count=8
```

Parameters (active only when `inc_rel_only=true`):

| Param | Required | Description |
|-------|----------|-------------|
| path | yes | Must equal the original absolute file path used during ingest (matches `Chunk.source`) |
| inc_rel_only | yes | Enable relations-only incremental mode |
| new_after | one of | Explicit last old chunk id (exclusive boundary) |
| detect_after + new_count | one of | Auto pick boundary: treat last `new_count` chunks as new |
| rel_window | optional | Sliding window size (default = global `relation_window`) |
| rel_truncate | optional | Per chunk truncation characters (prompt length control) |
| rel_temperature | optional | LLM temperature override |

Emitted incremental relation stages:

```text
data: {"stage":"relation_incremental_start","old":120,"new":8,"window":2,"pairs":34}
data: {"stage":"relation_incremental_progress","processed":10,"total":34,"created":5,"skipped":3}
data: {"stage":"relation_incremental_warn","pair":"chunk_119::chunk_121","error":"timeout"}
data: {"stage":"relation_incremental_done","created":12,"skipped":9,"pairs":34,"old":120,"new":8}
event: result
data: {"stage":"result","result":{"mode":"relations_incremental","created":12,"skipped":9,"pairs":34,"old":120,"new":8}}
data: {"stage":"done"}
```

Notes:

- Only new-new and tail-old to early-new pairs inside the window are considered.
- Existing (src,dst,type) :REL edges are skipped; no deletions performed.
- Safe to re-run with corrected boundary (idempotent for existing types, confidence averaging still only applies in full ingest path).
- Keep `rel_window` small for large graphs to limit LLM calls.

If boundary was wrong you can re-run relations-only with the right `new_after` or fall back to a full refresh (`refresh=true`).

#### Checkpoint / Resume

When `checkpoint=true` (default) the pipeline writes a JSON file `.ingest_ck_<basename>.json` alongside the source path. It records:

```json
{
  "chunks": { "file::chunk_0": {"emb": true, "ent": true } },
  "rel_pairs": { "file::chunk_0|file::chunk_1": {"rels":1} }
}
```

Re-running the same ingest (without `refresh=true`) skips completed embeddings, entity extraction and processed relation pairs. Delete the checkpoint file or pass `refresh=true` to force a full rebuild.

Notes:

- Relation pair retries are not persisted yet; failed pairs are marked with `err` and skipped on resume.
- For large corpora SSE keeps the client responsive without waiting for final JSON.

Client example (curl will show raw events):

```bash
curl -N 'http://localhost:8010/ingest/stream?path=/data/book.txt&checkpoint=true'
```

## 3. Query

POST `/query`

Request Body:

```jsonc
{
  "question": "Describe the multi-step approval flow.",
  "stream": false,
  "context": [
    {"id":"ext1","text":"An external memo about subgraph depth quotas and path scoring."},
    "A plain text snippet can also be used as extra context."
  ],
  "history": [
    {"role":"user","content":"Do you support hybrid retrieval?"},
    {"role":"assistant","content":"Yes: vector + entities + co-occur + llm-rel (+BM25/+degree)."}
  ]
}
```

Behavior:

- `stream=true`: Server-Sent text chunks (plain text) + final sources list
- `stream=false`: JSON structure including answer, sources (includes external items with `reason=external`), references, entities, warnings

Non-stream JSON Response example:

```jsonc
{
  "answer": "... answer text with inline references like [S1] ...",
  "sources": [
    {
      "id": "chunk:abc123",
      "rank": 1,
      "reason": "vector",
      "score": 0.8421,
      "base_norm": 0.73,
      "bonus": 0.12,
      "composite_score": 0.85,
      "rel_type": null,
      "rel_conf": null
    }
  ],
  "references": [
    { "label": "S1", "id": "chunk:abc123", "score": 0.8421, "composite_score": 0.85 }
  ],
  "entities": [
    { "name": "审批", "freq": 5 }
  ],
  "warnings": [
    { "type": "unused_sources", "detail": "这些来源已提供但未在回答中引用: S3" },
    { "type": "unreferenced_numeric", "detail": "数字'2024' 可能需要引用标记" }
  ]
}
```

Notes:

- `sources`: Raw retrieval + expansion contexts (vector / relates / cooccur / llm_rel)
- `references`: Mapping S# tokens to source entries (only those with `rank`)
- `warnings` may include unused source IDs or unreferenced numeric facts

Error shape:

```json
{
  "error": "query_failed",
  "message": "details",
  "trace": "Traceback ..."
}
```

## 4. Diagnostics

GET `/diagnostics`

Includes (example trimmed):

```jsonc
{
  "api_version": "0.1.0",
  "neo4j": { "neo4j": true, "detail": 1 },
  "vector_index": { "vector_index": true, "name": "chunk_embedding_index", "dimension": 1024 },
  "llm": { "llm_api": true, "model_listed": true },
  "embedding": { "embedding_api": true, "dimension": 1024 },
  "relations": {
    "llm_rel": 120,
    "relates_to": 345,
    "co_occurs_with": 220,
    "total": 685,
    "relation_extraction_enabled": true,
    "relation_debug": false
  },
  "relation_weights": { "STEP_NEXT": 0.12, "CAUSES": 0.22, ... },
  "feature_flags": {
    "rerank_enabled": false,
    "bm25_enabled": false,
    "graph_rank_enabled": false,
    "hash_incremental_enabled": true,
    "entity_normalize_enabled": true
  },
  "noise_control": { "entity_min_length": 2, "cooccur_min_count": 2 },
  "entity_aliases": {
    "total_entities": 430,
    "with_aliases": 120,
    "ratio": 0.2791,
    "sample_top5": [ { "name": "流程", "aliases": ["过程"], "alias_count": 1 } ]
  }
}
```

## 5. Cache Stats

GET `/cache/stats`

Response:

```json
{
  "embedding_cache_size": 12,
  "answer_cache_size": 4,
  "embedding_cache_keys": ["question:..."],
  "answer_cache_keys": ["Describe the flow..."]
}
```

## 6. Cache Clear

POST `/cache/clear`

```json
{ "cleared": true }
```

## 7. Ranking Preview

POST `/ranking/preview`

Request:

```jsonc
{ "question": "How does the approval flow progress?" }
```

Response (abridged):

```jsonc
{
  "question": "How does the approval flow progress?",
  "degraded": false,
  "items": [
    {
      "id": "chunk:abc123",
      "reason": "vector",
      "score_raw": 0.83,
      "base_norm": 0.72,
      "bonus": 0.13,
      "composite_score": 0.85,
      "final_score": 0.85,
      "rel_type": null,
      "rel_conf": null
    }
  ]
}
```

Scoring logic:

- base_norm: Min-max normalized vector similarity within current context set
- bonus: Sum of (BM25 weight × normalized BM25 score) + relation type weight × confidence + optional degree weight
- composite_score: base_norm + bonus
- final_score: After optional rerank (when enabled), otherwise same as composite

## Field Semantics Cheatsheet

| Field | Meaning |
|-------|---------|
| score_raw | Original vector similarity from embedding search |
| base_norm | Normalized (0-1) vector score for comparability |
| bonus | Additive graph / BM25 / degree contributions |
| composite_score | base_norm + bonus (pre-rerank) |
| final_score | Post-rerank score (if rerank enabled) |
| reason | Why the context got included (vector / relates / cooccur / llm_rel) |
| rel_type | LLM semantic relation type (STEP_NEXT, CAUSES, etc.) |
| rel_conf | LLM relation confidence (0-1 approx) |

## Error Handling Patterns

| Scenario | Symptom | Action |
|----------|---------|--------|
| Neo4j down | `/diagnostics` neo4j=false | Check container / service logs |
| Vector index missing | vector_index=false | Recreate index in Neo4j (see README) |
| Embedding API fails | embedding_api=false | Verify model name & LM Studio started |
| Slow ingestion | Large doc & relation_extraction=true | Use `incremental=true` or disable relations temporarily |
| Missing references | warnings unused_sources | Ensure answer generation references necessary sources |

## Version Upgrade Guidance

- Always read `/diagnostics.api_version` before client adaptation.
- Clients SHOULD ignore unknown new fields.
- Breaking changes will increment MAJOR.

## Changelog (inline excerpt)

- 0.1.0: Introduced `api_version`, English API docs, version fields in health/diagnostics.

A more detailed `CHANGELOG.md` is recommended for future iterations.

## Roadmap (non-binding)

- Actual rerank model integration
- PageRank / centrality scoring option
- Adaptive weight tuning endpoint
- Auth layer & rate limiting
- Structured OpenAPI schema for references/warnings

## License

MIT License. Copyright (c) 2025 EvolutionAI Studio.
