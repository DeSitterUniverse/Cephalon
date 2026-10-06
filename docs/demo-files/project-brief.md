# Project Brief

Cephalon is a local-first document workbench. It stores document metadata and evidence in SQLite, keeps dense vectors in LanceDB, and answers with cited local sources.

## Goals

- Keep document search inspectable and local.
- Use EmbeddingGemma 2 Text 270M Q8_0 GGUF (normalized 768 dimensions, asymmetric retrieval query/document prompts) through a dedicated offline llama.cpp Vulkan server.
- Use Jina Reranker v3.5 Q8_0 listwise cosine ranking through the pinned PR #26286 Vulkan runtime.
- Preserve page/layout/table provenance and show source, retrieval, and evidence diagnostics.
- Keep chat generation on a separate, user-operated llama.cpp server.

## Demo Query

Ask: "What is Cephalon designed to do?"
