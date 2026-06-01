# Project Brief

Cephalon is a local-first document workbench. It indexes files on the user's machine, keeps metadata in SQLite, stores dense vectors in LanceDB, and answers with cited local sources when the answer relies on retrieved documents.

## Goals

- Keep document search local and inspectable.
- Use ONNX Runtime for the embedder and reranker.
- Load GGUF chat models explicitly from the desktop app.
- Show sources, retrieval scores, and answer support instead of hiding the retrieval pipeline.

## Demo Query

Ask: "What is Cephalon designed to do?"
