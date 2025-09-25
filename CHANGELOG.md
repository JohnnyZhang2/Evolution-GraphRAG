# Changelog

All notable changes to this project will be documented in this file.

Format adapted from Keep a Changelog and adheres to Semantic Versioning (MAJOR.MINOR.PATCH).

See also: [Chinese API Docs](docs/API.zh-CN.md) | [English API Docs](docs/API.en.md) | README (bilingual variants in repository root).

## [Unreleased]

### Added (Unreleased)

### Changed (Unreleased)

- Migrate PDF parsing dependency from deprecated PyPDF2 to actively maintained pypdf (ingest supports fallback import for compatibility).

### Fixed (Unreleased)

### Deprecated (Unreleased)

### Removed (Unreleased)

### Security (Unreleased)

## [0.2.0] - 2025-09-25

### Added (0.2.0)

- Frontend React admin (Docs viewer, local egonet graph, QA chat with streaming + citations + markdown + syntax highlight + retry/copy + conversation titles).
- Prompt template store (`prompts.json`) with `/prompts` (GET/POST) for active + multiple templates.
- Multi-template priority resolution (prompts.json > settings active > single system prompt > built-in default).
- Subgraph path-level scoring (`SUBGRAPH_PATH_SCORE_ENABLE` & weights) integrated into hybrid ranking.
- External context & conversation history in `/query` (external context becomes cite-able sources).

### Changed (0.2.0)

- `build_prompt` selects prompts.json active template before legacy settings fields.
- README updates: default port 8010, frontend startup instructions, rerank placeholder clarification, prompt management highlight, path-level scoring highlight.

### Fixed (0.2.0)

- Cytoscape graph not rendering on initially hidden modal (trigger resize/layout on visibility).
- Occasional streaming stalls mitigated via rAF throttling + typewriter buffering.

### Deprecated (0.2.0)

- Env-based multi-template variables (`ANSWER_PROMPT_TEMPLATES`, `ANSWER_PROMPT_ACTIVE`) slated for removal after migration period.

### Security (0.2.0)

- Current CORS configuration permissive for dev; production hardening guidance pending.

## [0.1.0] - 2025-09-21

### Added (0.1.0)

- Initial public API version exposure (`api_version` in `/health` & `/diagnostics`).
- English API reference `docs/API.en.md` and Chinese API reference `docs/API.zh-CN.md` (version notes updated).
- Hybrid retrieval scoring preview endpoint `/ranking/preview` (vector base_norm + relation/BM25/graph bonuses + rerank placeholder).
- Entity normalization & alias aggregation with optional synonym file.
- Incremental ingestion via chunk content hashing.
- Diagnostics endpoint enhanced: feature flags, relation weights, relation counts, alias stats, noise control parameters.
- Noise control: minimum entity length & co-occurrence edge pruning threshold.
- Project-wide rebranding to “Evolution RAG” with unified copyright headers (MIT license).

### Changed (0.1.0)

- README split into bilingual documentation sets; branding & license notices updated.

### Fixed (0.1.0)

- Markdown formatting issues (headings, fenced code blocks, list spacing) in documentation.
- Encoding repair heuristic for Latin-1 garbled characters in non-stream query responses.

### Notes

- Rerank logic currently a placeholder integration point.
- Graph centrality (degree) bonus optional; future PageRank planned.

[Unreleased]: https://example.com/compare/0.2.0...HEAD
[0.2.0]: https://example.com/releases/0.2.0
[0.1.0]: https://example.com/releases/0.1.0
