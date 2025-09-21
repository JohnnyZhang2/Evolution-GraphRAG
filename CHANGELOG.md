# Changelog

All notable changes to this project will be documented in this file.

Format adapted from Keep a Changelog and adheres to Semantic Versioning (MAJOR.MINOR.PATCH).

See also: [Chinese API Docs](docs/API.zh-CN.md) | [English API Docs](docs/API.en.md) | README (bilingual variants in repository root).

## [Unreleased]

### Added (release)

- (placeholder) Add new items here when preparing next release.

### Changed (release)

- (placeholder)

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
