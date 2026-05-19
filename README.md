# Cephalon

Cephalon is a local-first desktop RAG workbench for indexing files, asking questions over them and getting cited answers when needed, and running local GGUF chat models through Vulkan-enabled llama.cpp. The app is built for offline use: metadata stays in SQLite, dense vectors stay in LanceDB, embedding/reranking runs through ONNX Runtime, and generation runs through an explicitly loaded local model.

## Features

- Lightweight OLED friendly Tauri v2 desktop app with a default #000000 background and #FFE5CC text. Optional Graphite theme uses a #171717 dark background with #FFFFFF text.
- Document library with import, text-safe unknown file ingestion, reindexing, tags, delete, and document details.
- Durable ingestion jobs with live SSE progress.
- Explicit GGUF model picker and **Load** action before querying.
- Vulkan-enabled llama.cpp backend diagnostics and explicit local model loading.
- Hybrid retrieval: SQLite FTS5 BM25 + LanceDB dense vectors + reciprocal rank fusion.
- Jina ONNX embedder and reranker setup from Settings.
- Hierarchical indexing with summary nodes, parent chunks, and child chunks.
- Structured query stream events for subqueries, sources, metadata, tokens, errors, and completion.
- Source drawer with dense, lexical, fusion, rerank, confidence, and citation metadata.
- Retrieval Trace panel for inspecting vector, BM25, fused, reranked, unused, and final context candidates.
- Index Health panel for chunk counts, stale state, duplicate rates, retrieval counts, and embedding distribution.
- Minimal local eval runner with Recall@k and MRR.
- Answer Support panel with deterministic citation trust labels.
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
~/Documents/Obsidian Vault
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

Packaged installers do not bundle the embedder/reranker ONNX artifacts. If they are missing, Cephalon opens Settings and shows the configured download sources. Install the configured ONNX engines or browse to local exported folders. A valid embedder/reranker folder contains `model.onnx`, `tokenizer.json`, `tokenizer_config.json`, `onnx_profile.json`, and any external ONNX data files referenced by the model. Older exported folders are still accepted automatically.

Current prepared ONNX repos:

- Embedder: [s-lorin/jina-embeddings-v5-small-onnx](https://huggingface.co/s-lorin/jina-embeddings-v5-small-onnx)
- Reranker: [s-lorin/jina-reranker-v3-onnx](https://huggingface.co/s-lorin/jina-reranker-v3-onnx)

Use different prepared repos with:

```powershell
$env:CEPHALON_EMBEDDER_ONNX_REPO="s-lorin/jina-embeddings-v5-small-onnx"
$env:CEPHALON_RERANKER_ONNX_REPO="s-lorin/jina-reranker-v3-onnx"
```

Upload the prepared local model repos with:

```powershell
.\scripts\upload_onnx_models_to_hf.ps1 -Namespace "s-lorin"
```

## Install

```powershell
npm install
.\scripts\setup_local_python.ps1
```

The setup script uses `python` or `py -3` from PATH, disables user-site package leakage with `PYTHONNOUSERSITE=1`, installs the packages in `requirements.txt`, force-rebuilds `llama-cpp-python` with Vulkan enabled, and runs the runtime preflight without requiring ONNX artifacts.

Export ONNX models only when missing or intentionally replacing them:

```powershell
python -m pip install --upgrade -r requirements-export.txt
python export_onnx.py
python scripts\validate_onnx_models.py --mark
```

## Run

Backend only:

```powershell
python python\main.py
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

The selected Python environment must contain a Vulkan-enabled `llama-cpp-python` package with `ggml-vulkan.dll`. Rebuild it when needed:

```powershell
$env:CMAKE_ARGS="-DGGML_VULKAN=on"
$env:FORCE_CMAKE="1"
python -m pip install --upgrade --force-reinstall --no-cache-dir --no-binary llama-cpp-python llama-cpp-python
python scripts\preflight_runtime.py --skip-onnx
```

## Build

Frontend:

```powershell
npm.cmd run build
```

Backend sidecar:

```powershell
python build_backend.py
```

The sidecar build packages the backend only. It does not include embedder/reranker model folders; those are installed into the user model directory from the app Settings screen.

Tauri package:

```powershell
npm run tauri build
```

Full Windows release pipeline:

```powershell
.\scripts\build_release.ps1
```

## Configuration

Common runtime variables:

```powershell
$env:CEPHALON_DATA_DIR="C:\path\to\data"
$env:CEPHALON_MODEL_DIR="C:\path\to\models"
$env:CEPHALON_METRICS_DIR="$HOME\Documents\Cephalon Metrics"
$env:CEPHALON_OBSIDIAN_VAULT_DIR="$HOME\Documents\Obsidian Vault"
$env:CEPHALON_HOST="127.0.0.1"
$env:CEPHALON_PORT="8765"
$env:CEPHALON_LLAMA_VERBOSE="0"
$env:CEPHALON_CONTEXT_TOKENS="32768"
$env:CEPHALON_FULL_CONTEXT="0"
$env:CEPHALON_EMBEDDER_ONNX_REPO="s-lorin/jina-embeddings-v5-small-onnx"
$env:CEPHALON_RERANKER_ONNX_REPO="s-lorin/jina-reranker-v3-onnx"
```

Future remote/offloaded backend mode:

```powershell
$env:CEPHALON_EXTERNAL_BACKEND="1"
$env:CEPHALON_HOST="192.168.1.20"
$env:CEPHALON_PORT="8765"
```

For frontend-only remote testing, set `VITE_CEPHALON_API_URL` at build/dev time or set `cephalon.apiBaseUrl` in browser local storage.

## API

- `GET /health`: startup status, paths, model diagnostics, backend status, retrieval state, and embedding metadata.
- `GET /models`: available chat GGUF models, auxiliary GGUF assets, and active model state.
- `POST /models/load`: load the selected GGUF into llama.cpp.
- `GET /models/onnx/status`: inspect embedder/reranker setup state.
- `POST /models/onnx/download`: download configured prepared ONNX artifacts into the model directory.
- `POST /models/onnx/install-local`: install a local exported ONNX folder for the embedder or reranker.
- `GET/PUT /settings`: RAG and generation defaults.
- `POST /ingest`: queue file/folder ingestion.
- `GET /vaults/obsidian`: configured Obsidian vault path and existence check.
- `POST /vaults/obsidian/ingest`: queue the configured Obsidian vault, skipping `.obsidian` and other internal folders.
- `GET /jobs`: recent ingestion jobs.
- `GET /retrieval/traces`: recent retrieval traces.
- `GET /retrieval/traces/{query_id}`: full retrieval trace with candidate stages, scores, context, and latency.
- `GET /observability/index-health`: document/chunk/index health summary.
- `GET/POST /eval/runs`: run and inspect small JSON eval sets.
- `POST /feedback`: store answer or citation feedback locally.
- `GET /events`: SSE job/document/settings stream.
- `GET/PATCH/DELETE /documents/{id}`: document details, rename, and delete.
- `POST /documents/{id}/reindex`: reindex while preserving display name and tags.
- `GET/POST/PATCH/DELETE /conversations`: chat history management.
- `POST /query`: typed SSE query stream. The selected model must already be loaded.
- `POST /metrics/export`: write a numeric corpus snapshot CSV.
