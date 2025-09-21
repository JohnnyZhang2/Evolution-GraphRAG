# Detailed Feature Guide (English)

> For the Chinese version see `FEATURES.zh-CN.md`.

## Table of Contents

- [Architecture & Goals](#architecture--goals)
- [Data Model](#data-model)
- [Ingest Pipeline](#ingest-pipeline)
- [Ingest Modes](#ingest-modes)
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

(Port the high-level architecture / mermaid / ASCII diagram summary.)

## Data Model

(Describe nodes & relations: Chunk, Entity, HAS_ENTITY, RELATES_TO, CO_OCCURS_WITH, :REL.)

## Ingest Pipeline

(7 steps from file read to LLM semantic relations + incremental hash logic.)

## Ingest Modes

(full / incremental / refresh / refresh_relations and precedence.)

## Query Flow

(vector retrieval -> optional expansion -> composite scoring -> context assembly -> LLM -> citations.)

## Scoring & Weights

(Formula, relation weight types, fallback STEP_NEXT semantics.)

## Caching

(Question embedding LRU + answer LRU; stats & clear endpoints.)

## Diagnostics & Tooling

(`/diagnostics`, dump script, env sync checker, requirements import checker.)

## Degradation & Resilience

(Vector index fallback, LLM parse fallback -> STEP_NEXT, warning injection.)

## Feature Overview

- BM25 sparse fusion (lazy index build)
- Graph Rank (degree centrality bonus)
- Hash-based incremental ingest (skip unchanged chunks)
- Entity normalization & alias statistics
- Noise control (entity length filter / co-occurrence pruning)
- LLM pairwise semantic relations + structured+fallback parse
- Hybrid ranking (vector + graph + relation + rerank placeholder)

## Environment Variables

(Replicate key .env section + parameter explanations.)

## Relation Construction

(Generation logic for HAS_ENTITY, RELATES_TO, CO_OCCURS_WITH, :REL; confidence handling.)

## Citations & Post-processing

(References extraction, unused sources warning, unreferenced numeric detection.)

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

- feature_flags
- entity_aliases
- noise_control
- relation_weights
- vector_index status

## CLI Utilities

| Command | Purpose | Key Args |
|---------|---------|---------|
| python -m graphrag.cli.normalize_entities --dry-run | Re-normalize legacy entities (when normalization was off) | --apply, --limit |
| python -m graphrag.cli.refresh_relations --source \<file\> | Rebuild :REL semantic relations for one file or all | --window, --truncate, --all |

## Noise Control

(ENTITY_MIN_LENGTH filter + COOCCUR_MIN_COUNT prune rationale & impact.)

## Relation Property Semantics

- RELATES_TO.weight = number of shared entities
- CO_OCCURS_WITH.count = co-occurrence count (post-prune)

## Tuning Guide

(Translate original tuning scenarios table.)

## Weight Tuning Steps

(Five-step iterative adjustment process.)

## Troubleshooting

(Index missing, excessive STEP_NEXT, fallback storms, degraded warnings.)

## Future Roadmap

- Real rerank model (Cross-Encoder)
- Richer graph features (PageRank / Betweenness)
- Deletion / file move cascade cleanup
- BM25 adaptive refresh
- Automatic weight optimization (feedback loop)
- More robust alias merge (remove APOC dependency)

---

> Keep this file synchronized with `FEATURES.zh-CN.md` when updating capabilities.
