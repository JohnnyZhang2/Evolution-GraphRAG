# Detailed Feature Guide (English)

> For the Chinese version see `FEATURES.zh-CN.md`.

## Table of Contents

- [Architecture & Goals](#architecture--goals)
- [Data Model](#data-model)
- [Ingest Pipeline](#ingest-pipeline)
- [Ingest Modes](#ingest-modes)
- [Real-time Ingest & Recovery](#real-time-ingest--recovery)
- [Query Flow](#query-flow)
- [Scoring & Weights](#scoring--weights)
- [Caching](#caching)
- [Diagnostics & Tooling](#diagnostics--tooling)
- [Degradation & Resilience](#degradation--resilience)
- [Feature Overview](#feature-overview)
- [Environment Variables](#environment-variables)
- [Relation Construction](#relation-construction)
- [Citations & Post-processing](#citations--post-processing)
- [Query Response Schema](#query-response-schema)
- [Diagnostics Extensions](#diagnostics-extensions)
- [CLI Utilities](#cli-utilities)
- [Noise Control](#noise-control)
- [Relation Property Semantics](#relation-property-semantics)
- [Tuning Guide](#tuning-guide)
- [Weight Tuning Steps](#weight-tuning-steps)
- [Troubleshooting](#troubleshooting)
- [Future Roadmap](#future-roadmap)

## Architecture & Goals

This project provides a locally runnable GraphRAG (Graph‑enhanced Retrieval Augmented Generation) system combining multiple heterogeneous evidence signals:

- Vector similarity (primary semantic recall)
- Graph structure (entity co‑occurrence, shared entities, LLM semantic relations)
- Optional sparse (BM25) lexical signal
- Optional graph centrality (degree for now, extensible to PageRank, etc.)
- Configurable relation‑type weighting (cause / support / step_next / etc.)

Design goals:

- Modular: splitting, embedding, entity extraction, relation extraction, retrieval fusion, ranking, answer post‑processing.
- Highly tunable: almost every parameter exposed as environment variables (weights, windows, expansion hops, cache sizes, feature flags).
- Observable: diagnostics endpoint + dump utilities + env diff + requirement import checker.
- Incremental friendly: content hashing skips unchanged chunks; standalone relation refresh & entity re‑normalization utilities.

### Mermaid Flow

```mermaid
flowchart TD
  subgraph Ingest
    A[Read Files] --> B[Split Chunks]
    B --> C[Embedding]
    C --> D[Write Chunk Nodes]
    D --> E{Entity Extract?}
    E -- No --> G[Relation Build (skip entity graph)]
    E -- Yes --> F[LLM Entities -> Entity/HAS_ENTITY]
    F --> G[RELATES_TO / CO_OCCURS_WITH]
    G --> H[Pairwise LLM Semantic :REL]
    H --> I[Done / Incremental or Refresh]
  end

  subgraph Query
    Q1[User Question] --> Q2[Embed (cache)] --> Q3[Vector TOP_K]
    Q3 --> Q4{EXPAND_HOPS=2?}
    Q4 -- No --> Q6[Merge Candidates]
    Q4 -- Yes --> Q5[Expand via Entities/Relations]
    Q5 --> Q6[Merge]
    Q6 --> Q7[Hybrid Scoring]
    Q7 --> Q8[TopN]
    Q8 --> Q9[Context w/ S#]
    Q9 --> Q10[LLM Answer]
    Q10 --> Q11[Citation Postproc]
    Q11 --> Q12[Return JSON / Stream]
  end

  I --> Q3
```

### ASCII Diagram

```text
[Docs] -> split -> [Chunks] -> embed(hash skip) -> (Chunk nodes)
    -> (LLM entity normalize) -> [Entity] -> RELATES_TO / CO_OCCURS_WITH
    -> (Pairwise LLM) -> :REL(type,confidence,evidence)

Query: Question -> normalize synonyms -> embed(cache)
   -> vector retrieve -> (optional expand by entities/relations)
   -> hybrid score (vector + bm25 + graph rank + relation bonuses)
   -> context -> LLM -> answer + references + warnings
```

## Data Model

Nodes:

- `Chunk { id, text, embedding, hash, source }`
- `Entity { name, aliases? }` (aliases present when normalization enabled)

Relationships:

| Relation | Meaning | Direction | Key Properties |
|----------|---------|-----------|----------------|
| `HAS_ENTITY` | Chunk contains entity | one-way | - |
| `RELATES_TO` | Shared entities between chunks | undirected | `weight` (count of shared entities) |
| `CO_OCCURS_WITH` | Sliding window entity co‑occurrence | undirected | `count` (window frequency, pruned if low) |
| `:REL` | LLM semantic relation | directed | `type, confidence, evidence` |

Normalization merges synonym/alias forms into canonical entity nodes if enabled.

## Ingest Pipeline

1. File load & format parsing (md / pdf / txt / docx / xlsx supported).
2. Text splitting with `CHUNK_SIZE` + `CHUNK_OVERLAP`.
3. Embedding + write `Chunk` node; compute & store content hash.
4. (Optional) Entity extraction via LLM -> filter noise -> normalization -> write `Entity` + `HAS_ENTITY`.
5. Build `RELATES_TO` (shared entity accumulation) and `CO_OCCURS_WITH` (sliding window); prune low `count` edges.
6. Pairwise LLM semantic relations across a window (`RELATION_WINDOW`), each side truncated to `RELATION_CHUNK_TRUNC`.
7. Parse JSON; on failure fallback to `STEP_NEXT` with `REL_FALLBACK_CONFIDENCE` confidence score.

Incremental ingest: unchanged chunks (same hash) skip embedding, entity extraction, and relation regeneration unless a forced refresh is requested.

## Ingest Modes

| Mode | Example Params | Behavior | Use Case |
|------|----------------|----------|---------|
| Full | `incremental=false&refresh=false` | Insert or update; existing unchanged chunks may skip some rebuilding | First load or bulk append |
| Incremental | `incremental=true` | Only new files/new chunks processed (hash skip) | Frequent small updates |
| Refresh | `refresh=true` | Rebuild all HAS_ENTITY / RELATES_TO / CO_OCCURS_WITH / :REL | Rule/weight changes |
| Refresh (no LLM :REL) | `refresh=true&refresh_relations=false` | Rebuild entity & basic graph only | Debug base graph |

If both `incremental` and `refresh` supplied, refresh takes precedence.

## Real-time Ingest & Recovery

### 1. Server-Sent Events (SSE) Progress

For long documents or enabled relation extraction, `/ingest` may block for a while. Use:

```text
GET /ingest/stream?path=/abs/file/or/dir&incremental=false&refresh=false&refresh_relations=true&checkpoint=true
```

It streams `text/event-stream` events, each separated by a blank line:

```text
event: start
data: {}

data: {"stage":"scan_input","detail":"path=/data/book.txt"}
data: {"stage":"embedding_batch","detail":"batch_ok","current":3,"total":12}
data: {"stage":"embedding_progress","current":240,"total":1300}
data: {"stage":"relation_extraction","current":120,"total":860}
event: result
data: {"stage":"result","result":{"chunks_total":1300,"chunks_embedded":820,...}}
data: {"stage":"done"}
```

Common stages: `scan_input`, `hash_computed`, `existing_scan`, `embedding`, `embedding_batch`, `embedding_progress`, `schema_index`, `entity_extraction`, `relates_to`, `relation_start`, `relation_extraction`, `relation_extraction_done`, plus synthetic `result` and terminal `done`. On error:

```json
{"stage":"error","error":"<message>"}
```

Client guidance: consume line by line (EventSource or fetch stream), update progress bars using `(current,total)` in embedding / relation stages, cache final summary from `result`, terminate on `done`. Heartbeat events can be added later (not yet built-in).

### 2. Checkpoint (Resume)

With `checkpoint=true` (default) a JSON file `.ingest_ck_<basename>.json` is written alongside the source:

```json
{
  "chunks": {"file::chunk_0": {"emb": true, "ent": true}},
  "rel_pairs": {"file::chunk_0|file::chunk_1": {"rels": 1}}
}
```

Meaning:

- `chunks[chunk_id].emb`: embedding done
- `chunks[chunk_id].ent`: entity extraction done
- `rel_pairs[a|b].rels`: relation pair processed (may hold >1 relations)
- Optional `err` marks last failure (retry on refresh)

Re-running ingest (without `refresh=true`) skips completed steps. Delete the checkpoint file or set `refresh=true` to force rebuild.

### 3. Embedding Batching & Adaptive Retry

`embed_texts` workflow:

1. Start with `EMBEDDING_BATCH_SIZE`.
2. On timeout / 5xx / 429: halve the batch (binary split) and retry.
3. Single-item batches still failing: exponential backoff until `EMBEDDING_MAX_RETRIES` exceeded.
4. Preserve input order; collect per-batch success metrics and emit `embedding_batch` + aggregate `embedding_progress` events (and optional callback).

### 4. Optional EOS Token Append

Some embedding backends prefer an explicit end token; enable:

- `EMBEDDING_APPEND_EOS=true` to append a token to every chunk.
- `EMBEDDING_EOS_TOKEN=</eos>` (adjust to model vocabulary).

### 5. Progress Callback & Observability

The ingest pipeline records internal stages via `record()`. They are:

- Collected into the synchronous `/ingest?progress=true` response array.
- Forwarded immediately via a provided `progress_callback` (SSE supplies one) to stream events.

### 6. New Environment Variables (Summary)

| Variable | Purpose | Example Default | Notes |
|----------|---------|-----------------|-------|
| EMBEDDING_BATCH_SIZE | Initial embedding batch size | 32 | Auto-splits on failures |
| EMBEDDING_TIMEOUT | Per-batch request timeout (sec) | 30 | Triggers retry/split |
| EMBEDDING_MAX_RETRIES | Max retries for a unit (batch/item) | 4 | Backoff on 429/5xx/timeout |
| EMBEDDING_APPEND_EOS | Append EOS token if true | false | Helps tokenizer stability |
| EMBEDDING_EOS_TOKEN | EOS token string | </eos> | Must exist in vocab |
| checkpoint (query param) | enable checkpoint file | true | Passed via URL param |

See `.env.example` for the authoritative list.

## Query Flow

1. Normalize query (synonyms if entity normalization active).
2. Embed (embedding cache lookup first).
3. Vector retrieval for initial `TOP_K` chunks.
4. Optional expansion (`EXPAND_HOPS=2`) using entities, `RELATES_TO`, `CO_OCCURS_WITH`, and `:REL` neighbors.
5. Hybrid base score: normalized vector + optional BM25 + optional Graph Rank.
6. Apply relation bonuses: shared entity / co‑occurrence / each semantic relation type \* confidence \* weight.
7. (Placeholder) rerank stage if enabled (does not yet reorder significantly).
8. Truncate to TopN context -> tag with `[S#]` markers.
9. LLM answer generation.
10. Post‑processing: citation extraction, unused source warnings, unreferenced numeric detection.

## Scoring & Weights

Base formula (vector similarity normalized to 0~1 first):

```text
composite = norm(vector_score)
         + bm25_weight * bm25_norm (optional)
         + graph_rank_weight * degree_norm (optional)
         + REL_WEIGHT_RELATES (if brought by RELATES_TO)
         + REL_WEIGHT_COOCCUR (if brought by CO_OCCURS_WITH)
         + Σ( REL_WEIGHT_<TYPE> * confidence )   // for each LLM semantic relation type matched
```

Fallback `STEP_NEXT`: uses `REL_FALLBACK_CONFIDENCE * REL_WEIGHT_STEP_NEXT` when semantic relation JSON parse fails.

## Caching

Two LRU caches:

| Cache | Maps | Hit Condition | Eviction |
|-------|------|---------------|----------|
| Embedding | question text -> embedding vector | Identical question repeated | LRU / restart |
| Answer | question text -> final non‑stream answer JSON | Identical question repeated | LRU / restart |

Endpoints: `/cache/stats`, `/cache/clear`.

## Diagnostics & Tooling

- `/diagnostics`: vector index presence, API availability, relation counts, weights, feature flags, noise control, alias stats.
- `python -m graphrag.diagnostics.dump`: inspect weights, index dimension, embedding dimension.
- `scripts/check_env_sync.py`: diff `.env` vs runtime settings.
- `scripts/check_requirements_imports.py`: verify declared dependencies are importable.
- Ranking preview endpoint: `/ranking/preview` for inspecting factor contributions.

## Degradation & Resilience

| Scenario | Degradation | Warning | Recovery |
|----------|-------------|---------|----------|
| Vector index missing | Fallback simple `MATCH ... LIMIT TOP_K` query | `__warning__` source block | Create Neo4j vector index |
| Embedding failure | Skip vector retrieval for that query | `embedding_failed` | Check model load / dimension consistency |
| LLM relation parse fails | Insert fallback `STEP_NEXT` relation | `relation_parse_fallback` | Increase truncation / lower temperature |

## Feature Overview

- BM25 sparse fusion (lazy index build)
- Graph Rank (degree centrality bonus)
- Hash-based incremental ingest (skip unchanged chunks)
- Entity normalization & alias statistics
- Noise control (entity length filter / co-occurrence pruning)
- LLM pairwise semantic relations + structured+fallback parse
- Hybrid ranking (vector + graph + relation + rerank placeholder)

## Environment Variables

Excerpt (see `.env.example` for full list). Defaults shown below are the code defaults from `graphrag/config/settings.py` (Pydantic `Field(default, env=...)`). If an env var is absent at runtime these defaults apply.

| Env Var | Default (settings.py) | Purpose (abridged) |
|---------|-----------------------|--------------------|
| TOP_K | 8 | Initial vector retrieval size |
| EXPAND_HOPS | 1 | Graph expansion hops (set 2 to expand) |
| CHUNK_SIZE | 800 | Split length (characters) |
| CHUNK_OVERLAP | 120 | Overlap when splitting |
| RELATION_WINDOW | 2 | Pairwise relation window width |
| RELATION_CHUNK_TRUNC | 400 | Max chars per chunk for relation LLM |
| REL_FALLBACK_CONFIDENCE | 0.3 | Confidence used for STEP_NEXT fallback |
| REL_WEIGHT_CAUSES | 0.22 | Relation type weight example |
| REL_WEIGHT_STEP_NEXT | 0.12 | Fallback / step-next weight |
| REL_WEIGHT_SUPPORTS | 0.2 | Supports relation weight |
| REL_WEIGHT_REFERENCES | 0.18 | References relation weight |
| REL_WEIGHT_FOLLOWS | 0.16 | Follows relation weight |
| REL_WEIGHT_PART_OF | 0.15 | Part-of relation weight |
| REL_WEIGHT_SUBSTEP_OF | 0.17 | Substep-of relation weight |
| REL_WEIGHT_CONTRASTS | 0.14 | Contrasts relation weight |
| REL_WEIGHT_DEFAULT | 0.15 | Default semantic relation weight |
| REL_WEIGHT_RELATES | 0.15 | Shared-entity edge bonus |
| REL_WEIGHT_COOCCUR | 0.10 | Co-occurrence edge bonus |
| BM25_ENABLED | false | Enable sparse BM25 fusion |
| GRAPH_RANK_ENABLED | false | Enable degree-based bonus |
| HASH_INCREMENTAL_ENABLED | false | Skip unchanged hashes |
| ENTITY_NORMALIZE_ENABLED | false | Enable entity synonym normalization |
| ENTITY_MIN_LENGTH | 2 | Min entity length filter |
| COOCCUR_MIN_COUNT | 2 | Min co-occurrence edge count |
| EMBEDDING_MODEL | text-embedding-qwen3-embedding-0.6b | Embedding model name |
| EMBEDDING_BATCH_SIZE | 64 | Initial embedding batch size |
| EMBEDDING_TIMEOUT | 120 | Per-batch HTTP timeout (seconds) |
| EMBEDDING_MAX_RETRIES | 6 | Max retries after batch splits |
| EMBEDDING_APPEND_EOS | false | Append EOS token if true |
| EMBEDDING_EOS_TOKEN | "" | EOS token (blank -> fallback to newline) |
| GRAPH_RANK_WEIGHT | 0.1 | Degree score weight |
| BM25_WEIGHT | 0.4 | BM25 normalized score weight |
| RERANK_ENABLED | false | Enable rerank placeholder stage |
| RERANK_ALPHA | 0.5 | Blend factor if rerank enabled |
| EMBED_CACHE_MAX | 128 | Embedding cache capacity |
| ANSWER_CACHE_MAX | 64 | Answer cache capacity |
| VECTOR_INDEX_NAME | chunk_embedding_index | Neo4j vector index name |
| RELATION_LLM_TEMPERATURE | 0.0 | Temperature for relation extraction LLM |
| RELATION_EXTRACTION | true | Enable pairwise relation extraction |
| DISABLE_ENTITY_EXTRACT | false | Disable entity extraction stage |

Footnotes:

1. Some weights (REL_WEIGHT_*) may be tuned; keep them within a small band (0.05–0.3) to avoid dominance.
2. Setting `EMBEDDING_APPEND_EOS=true` with empty `EMBEDDING_EOS_TOKEN` falls back to a newline character at runtime.
3. Raising `EMBEDDING_BATCH_SIZE` can improve throughput but increases timeout risk; adaptive splitting will mitigate failures.
4. `HASH_INCREMENTAL_ENABLED` only influences skip logic when `incremental=true` is passed to ingest.
5. `RELATION_EXTRACTION=false` disables all :REL creation, reducing ingest latency substantially.

Adjusting these impacts recall, precision, latency, and hallucination risk. See Tuning Guide.

## Relation Construction

| Type | Generation | Direction | Properties | Notes |
|------|------------|-----------|-----------|-------|
| HAS_ENTITY | Entity list from LLM extraction | one-way | - | Noise filtered (length / symbol) |
| RELATES_TO | Shared entity intersection count | undirected | weight | Accumulates across shared entities |
| CO_OCCURS_WITH | Sliding window co‑occurrence | undirected | count | Edges pruned if `count < COOCCUR_MIN_COUNT` |
| :REL | Pairwise LLM semantic relation | directed | type, confidence, evidence | Fallback to STEP_NEXT when parse fails |

## Citations & Post-processing

1. Scan answer text for `[S\d+]` markers -> map to sources.
2. Detect unused retrieved sources -> produce `unused_sources` warning.
3. Detect numeric facts (years, large numbers) lacking citation -> `unreferenced_numeric` warning suggestion.

## Query Response Schema

```jsonc
{
  "answer": "...",
  "sources": [...],
  "references": [...],
  "entities": [{"name": "EntityA", "freq": 5}],
  "warnings": [...]
}
```

## Diagnostics Extensions

Example extended diagnostics JSON:

```jsonc
{
  "feature_flags": {"bm25_enabled": true, "entity_normalize_enabled": true},
  "entity_aliases": {"total_entities": 96, "with_aliases": 16, "ratio": 0.1667},
  "noise_control": {"entity_min_length":2, "cooccur_min_count":2},
  "relation_counts": {"llm_rel":10, "relates_to":25, "co_occurs_with":6}
}
```

## CLI Utilities

| Command | Purpose | Key Args |
|---------|---------|---------|
| python -m graphrag.cli.normalize_entities --dry-run | Re-normalize legacy entities (when normalization was off) | --apply, --limit |
| python -m graphrag.cli.refresh_relations --source \<file\> | Rebuild :REL semantic relations for one file or all | --window, --truncate, --all |

## Noise Control

- Entity filtering: length < `ENTITY_MIN_LENGTH` or mostly symbol -> discard.
- Co‑occurrence pruning: delete edges whose `count < COOCCUR_MIN_COUNT`.

Effect: reduces graph noise, improves interpretability and reliability of relation bonuses.

## Relation Property Semantics

- `RELATES_TO.weight` = shared entity count (integer)
- `CO_OCCURS_WITH.count` = co‑occurrence window count (after pruning)

## Tuning Guide

| Scenario | Symptom | Primary Adjustments | Secondary | Notes |
|----------|---------|---------------------|-----------|-------|
| Low recall | Missing key info | Increase TOP_K | EXPAND_HOPS=2, raise RELATION_CHUNK_TRUNC | May raise noise |
| Too much noise | Irrelevant citations | Decrease TOP_K, lower weak relation weights | Reduce CHUNK_SIZE | Can disable expansion |
| Missing citations | Facts lack [S#] tags | Increase TOP_K / CHUNK_OVERLAP | Lower relation weights | Ensure prompt intact |
| Relation hallucination | Unreal causal/steps | Lower RELATION_LLM_TEMPERATURE | Lower REL_FALLBACK_CONFIDENCE | Inspect RAW debug |
| Sparse relations | Low bonus impact | Increase RELATION_CHUNK_TRUNC | Raise key relation weights | Or enlarge window |
| Co-occurrence bias | Co-occurrence dominates | Lower REL_WEIGHT_COOCCUR | Increase TOP_K | Or raise vector weight |
| Slow answers | Latency high | Reduce TOP_K / set EXPAND_HOPS=1 | Disable relation extraction | Combine with caching |
| High memory | Process footprint large | Shrink caches | Reduce CHUNK_SIZE | Externalize cache |
| Dimension mismatch | Errors / fallback | Rebuild index / verify dims | Use dump script | Embedding dims must align |

## Weight Tuning Steps

1. Validate vector retrieval quality (index healthy, TOP_K not too small).
2. Enable `RELATION_DEBUG` and sample distribution of relation types.
3. Use `/ranking/preview` to observe base_norm vs accumulated bonuses.
4. Down‑weight over‑dominant relation types (especially co‑occurrence) if skewed.
5. Iterate on key Q&A scenarios, logging changes for reproducibility.

## Troubleshooting

| Issue | Possible Cause | Investigation | Resolution |
|-------|----------------|---------------|-----------|
| `__warning__` source block | Vector index missing | `/diagnostics` -> vector_index | Create Neo4j vector index |
| All STEP_NEXT | LLM parse failures | Enable debug inspect RAW JSON | Adjust truncation / temperature |
| Many fallbacks | Truncation too short / unstable model | Raise RELATION_CHUNK_TRUNC | Lower temperature |
| Co-occurrence explosion | Threshold too low | Check diagnostics noise_control | Raise COOCCUR_MIN_COUNT |
| Fragmented entities | Normalization disabled | Inspect feature_flags | Enable normalization & re-run normalize_entities |

## Future Roadmap

- Real rerank model (Cross-Encoder)
- Richer graph features (PageRank / Betweenness)
- Deletion / file move cascade cleanup
- BM25 adaptive refresh
- Automatic weight optimization (feedback loop)
- More robust alias merge (remove APOC dependency)

---

> Keep this file synchronized with `FEATURES.zh-CN.md` when updating capabilities.
