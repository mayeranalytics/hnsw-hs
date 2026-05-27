# Changelog

## 0.1.0.0 — 2026-05-23

Initial 0.1.0.0 development entry.

### Added
- HNSW index creation, insertion, search, batch convenience search, update, remove, save/load.
- Runtime `setEf`.
- Metadata v3 with live/reserved labels.
- Criterion benchmark suite.
- Atomic save behavior and failure-injection tests.
- Haddock API documentation.

### Decisions
- `searchRadius` unsupported.
- `getEf` deferred until concrete use case.
- Optimized/native `searchBatch` no-ship after prototype (Slices 33–35).
- Cosine policy: pre-normalize vectors and use `InnerProduct` (Slice 36).

### Packaging
- Vendored hnswlib headers included in source distribution.
- `stack sdist --test-tarball` passes.