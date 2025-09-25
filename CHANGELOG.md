# Changelog

All notable changes to this project will be documented in this file.

Format adapted from Keep a Changelog and adheres to Semantic Versioning (MAJOR.MINOR.PATCH).

See also: [Chinese API Docs](docs/API.zh-CN.md) | [English API Docs](docs/API.en.md) | README (bilingual variants in repository root).

## [Unreleased]

### Added (since 0.1.0)

- Frontend React admin (Docs viewer + Local graph egonet + QA chat with streaming, citations, markdown, syntax highlight, retry/copy, conversation titles).
- Prompt templates store (`prompts.json`) with REST endpoints `/prompts` (GET/POST) supporting active template + multiple variants.
- Configurable system prompt & multi-template priority resolution (prompts.json > active template in settings > single answer_system_prompt > built-in default).
- Subgraph path-level scoring (`SUBGRAPH_PATH_SCORE_ENABLE` & related weights) integrated into hybrid ranking bonus.
- External context & conversation history injection into `/query` (context items appear as cite-able sources).

### Changed (since 0.1.0)

- `build_prompt` now prefers prompts.json active template before settings fields.
- README updated to reflect default port 8010 and frontend startup, rerank status wording, and prompt management.

### Fixed (since 0.1.0)

- Cytoscape local graph not rendering when modal initially hidden (force resize/layout on visibility change).
- Streaming chat occasional stall by introducing requestAnimationFrame throttling and typewriter buffer logic.

### Deprecated (since 0.1.0)

- Environment variable based multi-template storage (`ANSWER_PROMPT_TEMPLATES` / `ANSWER_PROMPT_ACTIVE`) planned for removal in a future major once all users migrate to prompts.json.

### Security (since 0.1.0)

- CORS liberal dev default (note: production hardening guidance to be added).

### Deprecated

- (placeholder)

### Removed

- (placeholder)

### Fixed (release)

- (placeholder)

### Security

- (placeholder)

## [0.1.0] - 2025-09-21

### Added

- Initial public API version exposure (`api_version` in `/health` & `/diagnostics`).
- English API reference `docs/API.en.md` and Chinese API reference `docs/API.zh-CN.md` (version notes updated).
- Hybrid retrieval scoring preview endpoint `/ranking/preview` (vector base_norm + relation/BM25/graph bonuses + rerank placeholder).
- Entity normalization & alias aggregation with optional synonym file.
- Incremental ingestion via chunk content hashing.
- Diagnostics endpoint enhanced: feature flags, relation weights, relation counts, alias stats, noise control parameters.
- Noise control: minimum entity length & co-occurrence edge pruning threshold.
- Project-wide rebranding to “Evolution RAG” with unified copyright headers (MIT license).

### Changed

- README split into bilingual documentation sets; branding & license notices updated.

### Fixed

- Markdown formatting issues (headings, fenced code blocks, list spacing) in documentation.
- Encoding repair heuristic for Latin-1 garbled characters in non-stream query responses.

### Notes

- Rerank logic currently a placeholder integration point.
- Graph centrality (degree) bonus optional; future PageRank planned.

[Unreleased]: https://example.com/compare/0.1.0...HEAD
[0.1.0]: https://example.com/releases/0.1.0
