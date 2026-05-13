# Cephalon Self Context

Cephalon is a local-first desktop document search and answer app. It runs a Tauri/React frontend, a Rust launcher, and a local FastAPI backend. Prefer local data and local models, avoid cloud assumptions, and keep answers grounded in retrieved document evidence.

## Runtime Shape

- Frontend: dense dark React workbench with document library, chat, sources, job state, model controls, and settings.
- Desktop shell: Tauri launches the backend sidecar or dev backend and opens a local UI window.
- Backend: FastAPI app in `cephalon_core`, split into config, schemas, routes, storage, ingestion, retrieval, generation, jobs, metrics, and model services.
- Storage: SQLite stores documents, chunks, FTS5 lexical chunks, jobs, events, tags, settings, and metrics metadata. LanceDB stores dense embeddings only.
- Models: GGUF chat models load through `llama-cpp-python`; use Vulkan/GPU offload when the installed backend supports it. ONNX Runtime is used for embeddings and reranking.

## Model Defaults

- Chat models are `.gguf` LLM files in the model directory. Embedding and reranker GGUF files are not chat models and should not appear in the chat model picker.
- Embedder: `jinaai/jina-embeddings-v5-text-small` ONNX, 1024 dimensions, retrieval adapter, normalized pooled output.
- Reranker: `jinaai/jina-reranker-v3` ONNX with validated score metadata.

## Retrieval Behavior

Ingestion extracts text, imports text-like unsupported files when safe, chunks content, embeds chunks, writes SQLite metadata and FTS rows, and writes LanceDB dense vectors with embedding model id, dimension, content hash, chunk length, indexed timestamp, stale state, and extraction mode.

Query flow: plan subqueries for compound questions, run LanceDB dense retrieval and SQLite FTS5 BM25 retrieval independently, fuse candidates with reciprocal rank fusion, rerank, stream typed events, generate grounded answer text, calculate confidence, and write numeric metrics. Structured stream events are `subquery`, `source`, `answer_meta`, `token`, `error`, and `done`.

Answer behavior: cite retrieved sources, use document filenames/chunks when useful, state uncertainty when evidence is weak, and prefer "not enough evidence" plus closest matches over unsupported claims.

## Local Data

Generated indexes can be backed up and rebuilt when model dimensions or schema state change. Source documents and user GGUF models should be preserved unless explicitly obsolete generated duplicates. Metrics are numeric-first and written outside the repo for later drift analysis.
