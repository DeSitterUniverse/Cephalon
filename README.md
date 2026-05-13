# Cephalon

Cephalon is a local-first desktop RAG workbench built with Tauri, React, FastAPI, LanceDB, SQLite, ONNX Runtime, and llama.cpp. It ingests local files, embeds and reranks them locally, then answers with source metadata exposed in the UI.

## Architecture

- **Desktop shell:** Tauri v2 starts the React frontend. In dev it launches the current source backend from `.venv`; in packaged builds it launches the PyInstaller backend sidecar.
- **Frontend:** React + Vite dense workbench UI with TanStack Query for server state, Zustand for UI state, and SSE for live job/document updates.
- **Backend:** FastAPI package under `python/cephalon_core` with config, migrations, storage, routes, ingestion jobs, retrieval, generation, and model services.
- **Storage:** SQLite tracks metadata, jobs, tags, settings, migrations, and FTS5 lexical chunks. LanceDB stores dense vectors only.
- **Inference:** ONNX Runtime handles only embeddings/reranking. `.gguf` chat models in `~/cephalon-data/models` are loaded through `llama-cpp-python`; Tauri dev requires the Vulkan llama.cpp backend and reuses the bundled Vulkan DLLs when `src-tauri/backend/engine/_internal` is present.

## Local Data

Default local data path:

```powershell
~/cephalon-data
```

Expected model layout:

```text
~/cephalon-data/models/
  reranker/model.onnx
  embedder/model.onnx
  *.gguf
```

Useful overrides:

```powershell
$env:CEPHALON_DATA_DIR="C:\path\to\data"
$env:CEPHALON_MODEL_DIR="C:\path\to\models"
$env:CEPHALON_MAX_TOKENS="512"
$env:CEPHALON_TOP_K="20"
$env:CEPHALON_RERANK_TOP_N="3"
$env:CEPHALON_CONTEXT_TOKENS="32768"
$env:CEPHALON_FULL_CONTEXT="0"
$env:CEPHALON_METRICS_DIR="$HOME\Documents\Cephalon Metrics"
$env:CEPHALON_REQUIRE_VULKAN="1"
$env:CEPHALON_LLAMA_VERBOSE="0"
```

## Install

```powershell
npm install
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r requirements.txt -r requirements-dev.txt
```

If ONNX models are missing:

```powershell
python -m venv .venv-export
.venv-export\Scripts\python.exe -m pip install -r requirements-export.txt
.venv-export\Scripts\python.exe export_onnx.py
.venv\Scripts\python.exe scripts\validate_onnx_models.py --mark
```

The Jina v5 embedder uses a custom PEFT/Qwen architecture, so `export_onnx.py` exports it through Cephalon's direct Torch ONNX wrapper instead of Optimum's generic feature-extraction exporter. The exported embedder records a fixed ONNX sequence length in `cephalon_onnx_meta.json`; the backend pads to that length automatically. The Jina v3 reranker export records the validated scoring mode in the same metadata file and must pass `scripts\validate_onnx_models.py --mark` before startup. Runtime tokenizer loading passes `fix_mistral_regex=True` so the Jina reranker tokenizer uses the corrected regex behavior.

## Run

Backend:

```powershell
.venv\Scripts\python.exe python\main.py
```

Desktop development app:

```powershell
npm run tauri dev
```

`npm run tauri dev` now starts the FastAPI backend automatically. It uses `.venv\Scripts\python.exe`, points the backend at `~/cephalon-data`, discovers GGUF files from `~/cephalon-data/models`, and requires the Vulkan llama.cpp backend for model loading. If a stale backend is already using port `8765`, stop it first so Tauri can launch the current source backend.

Frontend-only development:

```powershell
npm run dev
```

## Build

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

Tauri app:

```powershell
npm run tauri build
```

## Test

```powershell
.venv\Scripts\python.exe -m py_compile python\main.py python\test_backend_stabilization.py python\test_ingest_query.py python\test_query_only.py
.venv\Scripts\python.exe -m pytest -q --basetemp .pytest-tmp
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

## API Notes

- `GET /health` returns startup diagnostics, Vulkan status, active embedding table, and retrieval index health.
- `GET /settings` and `PUT /settings` manage RAG defaults, including context token cap and full-context mode.
- `POST /ingest` queues a durable job and returns `job_id`; known document types use native extractors, and unknown file types are imported as text when binary guards allow it. Binary unknown files fail with a clear reason instead of being silently accepted.
- `GET /jobs` lists recent jobs.
- `GET /events` streams SSE job/document/settings updates.
- `POST /query` streams typed SSE events: `subquery`, `source`, `answer_meta`, `token`, `error`, and `done`.
- `POST /metrics/export` writes a numeric corpus snapshot CSV under the configured metrics directory. If the directory is unavailable, the endpoint returns `status: "failed"` and `/health` exposes `last_metrics_error`; chat/query still works.
- Document APIs support detail, rename, delete, reindex, and tag management.

## Reproducibility Notes

- Runtime dependencies are pinned in `requirements.txt`; export-only dependencies are isolated in `requirements-export.txt`.
- Retrieval uses SQLite FTS5 plus LanceDB dense vectors with reciprocal rank fusion. Tantivy is not part of the runtime path.
- Jina embedding/reranker models are CC BY-NC 4.0. Keep license notices with packaged artifacts.
- The release pipeline validates ONNX models before packaging. Unvalidated or mismatched embedder/reranker exports stop startup with a clear error instead of falling back to a different model.
