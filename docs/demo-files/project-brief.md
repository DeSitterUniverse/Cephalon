# Project Brief

Cephalon is a local-first document workbench. It stores document metadata and evidence in SQLite, keeps dense vectors in LanceDB, and answers with cited local sources.

## Goals

- Keep document search inspectable and local.
- Use the fixed Jina Nano Retrieval Q8_0 embedder (normalized 768 dimensions) through a dedicated llama.cpp server.
- Use Jina Reranker v3.5 listwise ranking in an isolated Transformers worker.
- Preserve page/layout/table provenance and show source, retrieval, and evidence diagnostics.
- Keep chat generation on a separate, user-operated llama.cpp server.

## Demo Query

Ask: "What is Cephalon designed to do?"
