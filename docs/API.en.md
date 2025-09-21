# Evolution RAG API Reference (English)

> Version: `api_version` is exposed via `/health` and `/diagnostics`.

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

## 3. Query

POST `/query`

Request Body:

```jsonc
{
  "question": "Describe the multi-step approval flow.",
  "stream": false
}
```

Behavior:

- `stream=true`: Server-Sent text chunks (plain text) + final sources list
- `stream=false`: JSON structure including answer, sources, references, entities, warnings

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
