# Cephalon

Cephalon is a local-first desktop RAG workbench for indexing files, asking cited questions over them, and running local GGUF chat models through llama.cpp. The app is built for offline use: metadata stays in SQLite, dense vectors stay in LanceDB, embedding/reranking runs through ONNX Runtime, and generation runs through an explicitly loaded local model.

## Features

- Tauri desktop app with a dense dark React workbench.
- Document library with import, text-safe unknown file ingestion, reindexing, tags, delete, and document details.
- Durable ingestion jobs with live SSE progress.
- Explicit GGUF model picker and **Load** action before querying.
- Vulkan/GPU backend diagnostics for llama.cpp.
- Hybrid retrieval without Tantivy: SQLite FTS5 BM25 + LanceDB dense vectors + reciprocal rank fusion.
- Jina ONNX embedder and reranker with strict validation and no silent model fallback.
- Hierarchical indexing with summary nodes, parent chunks, and child chunks.
- Structured query stream events for subqueries, sources, metadata, tokens, errors, and completion.
- Source drawer with dense, lexical, fusion, rerank, confidence, and citation metadata.
- Persistent chat history stored in SQLite.
- Numeric-first metrics under the user Documents metrics directory.

## Architecture

- `src-tauri`: desktop shell, backend sidecar launch, window config, and release resources.
- `src`: React/Vite frontend, TanStack Query server cache, Zustand UI state, typed API client, and compact workbench panels.
- `python/main.py`: FastAPI/uvicorn entrypoint.
- `python/cephalon_core`: backend package for config, routes, schemas, storage, ingestion, retrieval, generation, jobs, metrics, documents, and model loading.

Read the architecture guide in [CEPHALON_ARCHITECTURE_DEEP_DIVE.html](CEPHALON_ARCHITECTURE_DEEP_DIVE.html).

## Local Data And Models

Default paths:

```powershell
~/cephalon-data
~/cephalon-data/models
~/Documents/Cephalon Metrics
```

Expected model layout:

```text
~/cephalon-data/models/
  embedder/model.onnx
  embedder/tokenizer files...
  reranker/model.onnx
  reranker/tokenizer files...
  chat-model.gguf
```

Only chat-capable `.gguf` files appear in the model picker. GGUF files whose names indicate embedding, retrieval, reranking, or cross-encoder use are reported as auxiliary assets and are hidden from chat selection.

## Install

```powershell
npm install
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
```

Export ONNX models only when missing or intentionally replacing them:

```powershell
python -m venv .venv-export
.\.venv-export\Scripts\python.exe -m pip install -r requirements-export.txt
.\.venv-export\Scripts\python.exe export_onnx.py
.\.venv\Scripts\python.exe scripts\validate_onnx_models.py --mark
```

## Run

Backend only:

```powershell
.\.venv\Scripts\python.exe python\main.py
```

Desktop development app:

```powershell
npm run tauri dev
```

Frontend only:

```powershell
npm run dev
```

In the app, select a chat GGUF model and press **Load** before running a query.

## Build

Frontend:

```powershell
npm.cmd run build
```

Backend sidecar:

```powershell
.\.venv\Scripts\python.exe build_backend.py
```

Tauri package:

```powershell
npm run tauri build
```

Full Windows release pipeline:

```powershell
.\scripts\build_release.ps1
```

Generated folders such as `.venv`, `.venv-export`, `dist`, `build`, `src-tauri/target`, `src-tauri/backend`, and local data directories should not be committed.

## Configuration

Common runtime variables:

```powershell
$env:CEPHALON_DATA_DIR="C:\path\to\data"
$env:CEPHALON_MODEL_DIR="C:\path\to\models"
$env:CEPHALON_METRICS_DIR="$HOME\Documents\Cephalon Metrics"
$env:CEPHALON_HOST="127.0.0.1"
$env:CEPHALON_PORT="8765"
$env:CEPHALON_REQUIRE_VULKAN="1"
$env:CEPHALON_LLAMA_VERBOSE="0"
$env:CEPHALON_CONTEXT_TOKENS="32768"
$env:CEPHALON_FULL_CONTEXT="0"
```

Future remote/offloaded backend mode:

```powershell
$env:CEPHALON_EXTERNAL_BACKEND="1"
$env:CEPHALON_HOST="192.168.1.20"
$env:CEPHALON_PORT="8765"
```

For frontend-only remote testing, set `VITE_CEPHALON_API_URL` at build/dev time or set `cephalon.apiBaseUrl` in browser local storage.

## API

- `GET /health`: startup status, paths, model diagnostics, Vulkan status, retrieval state, and embedding metadata.
- `GET /models`: available chat GGUF models, auxiliary GGUF assets, and active model state.
- `POST /models/load`: load the selected GGUF into llama.cpp.
- `GET/PUT /settings`: RAG and generation defaults.
- `POST /ingest`: queue file/folder ingestion.
- `GET /jobs`: recent ingestion jobs.
- `GET /events`: SSE job/document/settings stream.
- `GET/PATCH/DELETE /documents/{id}`: document details, rename, and delete.
- `POST /documents/{id}/reindex`: reindex while preserving display name and tags.
- `GET/POST/PATCH/DELETE /conversations`: chat history management.
- `POST /query`: typed SSE query stream. The selected model must already be loaded.
- `POST /metrics/export`: write a numeric corpus snapshot CSV.

## Model Licenses

The default Jina embedder and reranker models are distributed under CC BY-NC 4.0. When bundling those model files with a release, include their license notices with the packaged artifacts. For commercial distribution or commercial use, verify that the selected embedding and reranking models are licensed for that use case, or replace them with models that are.
