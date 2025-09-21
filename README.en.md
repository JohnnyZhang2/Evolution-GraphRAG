# Evolution RAG Q&A (Local LLM)

[中文说明](./README.md) | **English** | [Detailed Features](./docs/FEATURES.en.md)

## ✨ Overview

A locally deployable GraphRAG (Graph‑enhanced Retrieval Augmented Generation) system that fuses multiple retrieval signals:

- Vector similarity (Neo4j native vector index)
- Graph structure (shared entities, co-occurrence, semantic relations)
- Optional BM25 sparse scoring
- Optional graph degree centrality bonus
- Entity normalization + alias statistics
- Hash-based incremental ingestion

Designed for traceable, tunable, and maintainable knowledge QA with full transparency of ranking signals and references.

## 🔑 Key Capabilities

- Multi-format ingest: Markdown / PDF / TXT / DOCX / XLSX
- Modular pipeline: split → embed → (entity normalize) → graph build → semantic relations → hybrid retrieval → answer
- Transparent references: answer includes `[S#]` markers + structured `references` + `warnings`
- Hybrid scoring: vector + (BM25) + (graph rank) + static & semantic relation bonuses
- Fast iteration: `/ranking/preview` shows score decomposition
- Diagnostics rich: `/diagnostics` exposes feature flags, alias metrics, noise control, relation counts
- Noise control: entity length filter + co-occurrence pruning
- Incremental efficiency: hash skipping untouched chunks

## 🧱 Architecture Snapshot

```text
Docs -> split -> embed(hash skip) -> entities(normalize) -> RELATES_TO / CO_OCCURS_WITH
                 -> pairwise LLM -> :REL(type,confidence)
Query -> synonym normalize -> embed(cache) -> vector retrieve -> (expand) -> hybrid score -> answer + references
```

Mermaid / full data model & flow diagrams are in the feature guide.

## 🚀 Quick Start

Clone & enter directory:

```bash
git clone https://github.com/your-org/evolution-rag.git
cd evolution-rag
```

```bash
python -m venv .venv
source .venv/bin/activate  # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
uvicorn graphrag.service.server:app --reload --port 8000
```

Ingest documents:

```bash
curl -X POST 'http://localhost:8000/ingest?incremental=false' \
  -H 'Content-Type: application/json' \
  -d '{"path":"./docs"}'
```

Ask a question:

```bash
curl -X POST http://localhost:8000/query \
  -H 'Content-Type: application/json' \
  -d '{"question":"Explain the system architecture"}'
```

## 🛠 Feature Switches (Selected)

| Feature | Env | Status | Notes |
|---------|-----|--------|-------|
| BM25 fusion | BM25_ENABLED | implemented | lazy in-memory index |
| Graph rank bonus | GRAPH_RANK_ENABLED | implemented | degree centrality |
| Hash incremental | HASH_INCREMENTAL_ENABLED | implemented | skip unchanged chunks |
| Entity normalization | ENTITY_NORMALIZE_ENABLED | implemented | ingest + query synonyms |
| Noise control | ENTITY_MIN_LENGTH / COOCCUR_MIN_COUNT | implemented | filter + prune |
| Rerank placeholder | RERANK_ENABLED | placeholder | no reordering yet |

Full variable list: see `.env` + feature guide.

## 📦 Query Response (Non‑stream)

```jsonc
{
  "answer": "... [S1][S3] ...",
  "sources": [...],
  "references": [...],
  "entities": [{"name":"EntityA","freq":5}],
  "warnings": [...]
}
```

## 🔍 Diagnostics

Highlights from `/diagnostics`:

- feature_flags
- entity_aliases (total / with_aliases / ratio / sample)
- noise_control (entity_min_length, cooccur_min_count)
- relation_counts
- vector index status

## 📚 Detailed Docs

For deep dive (scoring math, tuning tables, CLI utilities, failure modes) read:

- [Chinese Feature Guide](./docs/FEATURES.zh-CN.md)
- [English Feature Guide](./docs/FEATURES.en.md)

## 🧪 Testing & Health

- `scripts/check_env_sync.py` ensures `.env` vs `settings.py` consistency
- `scripts/check_requirements_imports.py` validates runtime imports
- `python -m graphrag.diagnostics.dump` prints weight & embedding/index dimensions

## 🗺 Roadmap (Excerpt)

- Real reranker (Cross‑Encoder / ColBERT)
- Richer graph metrics (PageRank, Betweenness)
- Deletion / orphan cleanup
- Adaptive BM25 refresh
- Automated weight tuning (feedback / RL)

## 🔖 Versioning & Release

API version is declared in `graphrag/config/settings.py` (`api_version`) and surfaced via `/health` & `/diagnostics`.

Use the helper script to bump and roll changelog:

```bash
python scripts/bump_version.py --type patch   # or minor / major
python scripts/bump_version.py --set 0.2.0    # explicit
python scripts/bump_version.py --dry-run --type minor
```

Actions performed:

- Updates `api_version` in settings
- Moves Unreleased notes into a dated release block in `CHANGELOG.md`
- Recreates an empty Unreleased template
- Appends compare link placeholders

Conventions:

- MAJOR: breaking response contract / semantics change
- MINOR: backward compatible endpoint / field additions
- PATCH: non-breaking fixes / doc only or internal quality updates

## 🤝 Contributing

PRs welcome: please keep both Chinese & English docs in sync. For new config keys: update `settings.py`, `.env`, both READMEs, and feature guides.

## 📄 License & Copyright

Released under the MIT License.  
Copyright © 2025 EvolutionAI Studio (All Rights Holder: Johnny Zhang).  
Additional contributions by community contributors (see commit history).

When redistributing, retain both `LICENSE` and `NOTICE` files.

---
> Built for transparent, controllable GraphRAG experimentation.
