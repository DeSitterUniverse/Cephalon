# Cephalon Architecture Deep Dive

Cephalon is a local-first RAG desktop app. The architecture keeps local metadata durable, keeps model-heavy work in the backend process, and gives the frontend a small typed API surface.

## Runtime Topology

Cephalon runs as three cooperating layers:

1. **Tauri shell** owns the native window, file dialogs, drag/drop, backend process launch, release resources, and runtime environment setup.
2. **React workbench** owns the visible app: library, chat, sources, jobs, selected document, model controls, settings, and status.
3. **FastAPI backend** owns local data, ingestion, retrieval, ONNX inference, GGUF loading, generation, metrics, and HTTP/SSE contracts.

In development, Tauri launches `python/main.py` from `.venv`. In packaged builds, Tauri launches the PyInstaller sidecar under `src-tauri/backend/engine`. If `CEPHALON_EXTERNAL_BACKEND=1` is set, Tauri skips local launch so a future backend on another machine can be used.

## Backend Package

The uvicorn entrypoint is `python/main.py`; application logic is under `python/cephalon_core`.

- `config`: environment and defaults for data paths, model paths, API host/port, metrics, CORS, retrieval, and generation.
- `schemas`: Pydantic API models and setting validation.
- `app_factory`: FastAPI construction, startup state, ONNX loading, storage initialization, retrieval health, event bus, and job worker.
- `routes`: HTTP and SSE endpoints for health, models, settings, documents, ingestion, jobs, events, metrics, and queries.
- `storage`: SQLite connection, migrations, locking, FTS5, document payloads, tags, app settings, LanceDB table selection, and generated vector cleanup.
- `services.documents`: file discovery and extraction, including safe text import for unknown extensions.
- `services.ingestion`: chunking, embedding, duplicate handling, document state, FTS writes, LanceDB writes, reindex, and delete cleanup.
- `services.retrieval`: query embeddings, dense search, FTS search, RRF fusion, reranking, source shaping, confidence, deterministic numeric scan, and retrieval metrics.
- `services.generation`: prompt assembly and llama.cpp token streaming.
- `services.jobs`: durable single-worker ingestion queue and job events.
- `services.metrics`: JSONL retrieval events and CSV corpus snapshots.
- `services.models`: GGUF discovery, Vulkan/backend diagnostics, context selection, and explicit llama.cpp model loading.

## Storage Architecture

SQLite is the source of truth for app metadata:

- documents
- chunks
- chunks_fts virtual table
- document_tags
- jobs
- job_events
- app_settings
- schema_migrations

SQLite is protected by a process-local reentrant lock because the backend shares a connection across async request and worker paths.

LanceDB stores dense vectors in the active embedding table. It does not own document metadata, lexical search, tags, or job state. Current active vector state is for Jina v5 small 1024-dimensional embeddings. Older generated vector tables can be backed up and dropped because documents and SQLite metadata are the rebuild source.

## Retrieval Architecture

Hybrid retrieval is explicit and testable:

```text
query
  -> optional deterministic numeric analyzer
  -> subquery planner
  -> dense retriever: LanceDB
  -> lexical retriever: SQLite FTS5 BM25
  -> reciprocal rank fusion
  -> ONNX reranker
  -> source selection
  -> confidence metadata
  -> citation-aware generation
```

Lexical state stays transactional with document metadata, while dense vectors stay isolated in LanceDB.

The reranker score is bounded before combining it with retrieval evidence. Exact lexical matches and strong fused candidates cannot be discarded solely because an ONNX logit is poorly calibrated. Core-memory prompt echoes are excluded from document dense retrieval so repeated failed questions do not reinforce themselves.

Exact max-style questions over indexed numeric rows use deterministic analysis before generation. This keeps row arithmetic and simple comparisons out of the generative path.

## Query Stream

`POST /query` requires the selected GGUF model to already be loaded through `POST /models/load`. This makes the model allocation phase explicit in the UI and prevents the first query from appearing to hang while llama.cpp loads.

The query endpoint streams server-sent events:

- `subquery`: planned retrieval unit.
- `source`: source metadata with document id, filename, chunk id, rank, vector score, lexical score, fusion score, rerank score, snippet, and subquery id.
- `answer_meta`: confidence, uncertainty, no-answer flag, search modes, latency, and metrics path.
- `token`: generated answer text.
- `error`: stream failure.
- `done`: terminal marker.

Sources and answer tokens are separate by design so the UI can inspect evidence even when the answer is low confidence.

## Model Runtime

Embedding and reranking use ONNX Runtime. Export tooling uses Transformers, Optimum, and model-specific wrappers only outside normal runtime.

Defaults:

- Embedder: `jinaai/jina-embeddings-v5-text-small`, 1024 dimensions.
- Reranker: `jinaai/jina-reranker-v3`, validation metadata records score mode.
- Chat models: local `.gguf` files loaded by llama.cpp.

Runtime tokenizer loading uses `fix_mistral_regex=True` for Jina reranker compatibility. Startup fails visibly if ONNX models are missing, mismatched, unvalidated, or dimension-incompatible.

GGUF loading keeps `n_gpu_layers=-1`, `offload_kqv=True`, configurable context length, optional full model context, and Vulkan diagnostics. Health/model endpoints expose active model, loaded context, model metadata context, loaded llama library path, and Vulkan availability.

## Frontend Architecture

The frontend is intentionally lightweight:

- React for rendering.
- TanStack Query for backend server state and cache invalidation.
- Zustand for selected model, selected document, sources, panel selection, and event-stream state.
- A typed API module for HTTP calls.
- A typed SSE hook for live events with reconnect/fallback behavior.
- Lucide icons only; no heavy UI framework.

Layout:

- Left panel: library, import actions, search, filters, reindex/delete.
- Center: fixed top bar, explicit model load control, chat stream, contained composer.
- Right panel: jobs, sources, document details, or settings.

The root uses `100dvh`, fixed grid rows, and isolated scroll regions so long chats do not push away the top bar.

## Startup And Model Loading UX

The app shows a simple boot screen while the frontend waits for `/health`. After boot, the model picker selects a GGUF filename but does not load it. The user presses **Load**, the backend loads the model into llama.cpp, and chat is enabled only when the selected model matches `/models.active_model`.

This state gives model loading a clear failure boundary with visible errors.

## Metrics

Metrics are numeric-first and written outside the repo:

- query/retrieval JSONL events
- corpus/index CSV snapshots
- source counts
- score distributions
- retrieval latency
- confidence/no-answer values
- document/chunk/embedding counts
- stale/orphan/never-retrieved indicators

Metrics failures are non-fatal. `/health` exposes the last metrics write error.

## Multi-OS And Future Offload Path

The current tested target is local Windows/Tauri with a local FastAPI backend. The code keeps a narrow future path for other environments:

- Tauri backend host and port come from `CEPHALON_HOST` and `CEPHALON_PORT`.
- `CEPHALON_EXTERNAL_BACKEND=1` skips local backend launch.
- The frontend API base can be overridden with `VITE_CEPHALON_API_URL` or `localStorage["cephalon.apiBaseUrl"]`.
- Model and data paths are environment-driven.

Remote/offloaded serving is not a release target yet, but these seams avoid hardcoding localhost assumptions throughout the app.

## Release Boundaries

Release packaging must include:

- frontend `dist`
- Tauri resources
- PyInstaller backend sidecar
- ONNX embedder/reranker directories and license notices
- llama.cpp runtime libraries, including Vulkan DLLs on Windows

Release smoke checks should verify `/health`, `/models`, explicit model load, ingestion, query streaming, sources, metrics export, and shutdown cleanup.
