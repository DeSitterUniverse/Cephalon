# Cephalon

Cephalon is a local-first desktop RAG workbench for searching, analyzing, and citing files on your machine. It uses a Tauri shell, a React workbench, a FastAPI backend, SQLite FTS5, LanceDB vectors, ONNX Runtime for embedding/reranking, and llama.cpp for local GGUF chat models.

## Current Features

- Dense dark desktop workbench with document library, chat, sources, jobs, settings, and document details.
- Explicit GGUF model loading from the UI before querying, with Vulkan/context diagnostics.
- Local ingestion for common document types plus text-safe unknown file extensions.
- Durable ingestion jobs with document/job state and SSE live updates.
- SQLite metadata, FTS5 lexical search, LanceDB dense vectors, RRF fusion, ONNX reranking, confidence metadata, and structured query streams.
- Source inspection with document id, file name, chunk id, vector score, lexical score, fusion score, rerank score, snippet, and subquery id.
- Deterministic numeric scan for traffic-style maximum questions where exact row math is better than generative inference.
- Numeric metrics export to the user Documents metrics directory for later analysis.

## Architecture

- `src-tauri`: native shell, backend launch, window config, file dialogs, release resources, and environment setup.
- `src`: React/Vite frontend with TanStack Query server state, Zustand UI state, typed API calls, SSE hook, and compact panels.
- `python/main.py`: uvicorn entrypoint.
- `python/cephalon_core`: backend package for config, app factory, routes, schemas, storage, ingestion, retrieval, generation, jobs, metrics, and model loading.
- SQLite is the metadata source of truth and provides FTS5 BM25 lexical search.
- LanceDB stores dense vectors only.
- ONNX Runtime handles `jinaai/jina-embeddings-v5-text-small` and `jinaai/jina-reranker-v3`.
- llama.cpp loads selectable local `.gguf` chat models with GPU offload enabled when the installed backend supports Vulkan.

See [CEPHALON_ARCHITECTURE_DEEP_DIVE.md](CEPHALON_ARCHITECTURE_DEEP_DIVE.md) for a deeper system walkthrough.

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
  granite-4.1-8b-Q4_K_S.gguf
  NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf
```

Only chat-capable `.gguf` files appear in the model picker. Embedder, retrieval, reranker, and cross-encoder GGUF files are intentionally hidden from chat selection.

## Install

```powershell
npm install
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
```

Export ONNX models only when missing or intentionally replacing them:

```powershell
python -m venv .venv-export
.venv-export\Scripts\python.exe -m pip install -r requirements-export.txt
.venv-export\Scripts\python.exe export_onnx.py
.venv\Scripts\python.exe scripts\validate_onnx_models.py --mark
```

The reranker tokenizer is loaded with `fix_mistral_regex=True`. The backend refuses unvalidated or mismatched ONNX artifacts instead of silently falling back to another model.

## Run

Backend only:

```powershell
.venv\Scripts\python.exe python\main.py
```

Desktop development app:

```powershell
npm run tauri dev
```

Frontend only:

```powershell
npm run dev
```

The desktop app starts the backend automatically in dev and packaged builds unless `CEPHALON_EXTERNAL_BACKEND=1` is set. In the UI, choose a GGUF model and press **Load** before running a query.

## Build And Release

Frontend:

```powershell
npm.cmd run build
```

Backend sidecar:

```powershell
.venv\Scripts\python.exe build_backend.py
```

Full Windows release pipeline:

```powershell
.\scripts\build_release.ps1
```

Tauri package:

```powershell
npm run tauri build
```

Generated folders such as `.venv`, `.venv-export`, `dist`, `build`, `src-tauri/target`, `src-tauri/backend`, and local data directories should not be committed.

## Test

```powershell
.venv\Scripts\python.exe -m py_compile python\main.py python\test_backend_stabilization.py python\test_ingest_query.py python\test_query_only.py
.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp
.venv\Scripts\python.exe scripts\validate_onnx_models.py
npx.cmd tsc --noEmit
npm.cmd run test:frontend
npm.cmd run build
cd src-tauri
cargo check
```

Manual backend smoke test with a running backend:

```powershell
$env:CEPHALON_TEST_MODEL="NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf"
.venv\Scripts\python.exe python\test_ingest_query.py
```

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

For frontend-only remote testing, set `VITE_CEPHALON_API_URL` at build/dev time or put `cephalon.apiBaseUrl` in browser local storage. This path is present for future multi-PC/offloaded deployments, but release smoke testing currently targets the local backend.

## API Overview

- `GET /health`: startup status, paths, model diagnostics, Vulkan status, retrieval index state, and embedding metadata.
- `GET /models`: available chat GGUF models and current loaded model.
- `POST /models/load`: explicitly load the selected GGUF into llama.cpp.
- `GET/PUT /settings`: RAG and generation defaults.
- `POST /ingest`: queue file/folder ingestion.
- `GET /jobs`: recent ingestion jobs.
- `GET /events`: SSE job/document/settings stream.
- `GET/PATCH/DELETE /documents/{id}`: document details, rename, and delete.
- `POST /documents/{id}/reindex`: reindex a document while preserving display name and tags.
- `POST /query`: typed SSE query stream. The selected model must already be loaded.
- `POST /metrics/export`: write a numeric corpus snapshot CSV.

## Licensing Notes

Jina embedding and reranking model artifacts are CC BY-NC 4.0. Keep model notices with packaged artifacts and verify downstream usage is compatible with that license.
