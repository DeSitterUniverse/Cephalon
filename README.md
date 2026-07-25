# Cephalon

Cephalon is a local-first desktop RAG workbench for indexing files, asking questions over them and getting cited answers when needed, and connecting to a user-operated external llama.cpp server for generation. The app is built for offline use: metadata stays in SQLite, dense vectors stay in LanceDB, embedding/reranking runs through ONNX Runtime, and generation stays on the llama.cpp server you run.

Recommended for use with LLMs fine-tuned for a specific domain, task, or behavior, such as, but not limited to, models fine-tuned on personal writing styles, programming conventions, academic subjects, technical terminology, document-extraction formats, accessibility preferences, creative genres, or specialized knowledge.

## In Use

Cephalon keeps the document library, cited answer, retrieval controls, and external-server status together in one workspace. This example is from the included live RAG validation corpus, with documents from the World Bank, NIST, arXiv, NOAA, USGS, Our World in Data, and W3C indexed locally.

![A cited answer to a question about the RATE paper](docs/screenshots/rag-cited-answer.png)

Long-form answers retain citations and distinguish the roles of retrieval, reranking, source provenance, and refusing to answer when the evidence is weak.

![A cited long-form explanation of retrieval evidence, reranking, citations, and weak-evidence refusal](docs/screenshots/long-evidence-answer.png)

## Quick Start

Cephalon supports Windows and Linux. It requires Node.js, Python 3.14, an externally running `llama-server`, and its two ONNX retrieval engines.

1. Install the frontend and Python dependencies for your platform:

   Windows PowerShell:

   ```powershell
   npm.cmd install
   py -3.14 scripts\setup_python.py
   ```

   Linux shell:

   ```bash
   npm install
   python3.14 scripts/setup_python.py
   ```

2. In a separate terminal, start your own llama.cpp server with the GGUF you want to use. Set its context size to match the value you will give Cephalon:

   Windows PowerShell:

   ```powershell
   & "C:\path\to\llama-server.exe" -m "D:\models\your-model.gguf" --host 127.0.0.1 --port 8080 -c 32768 -ngl 99 --reasoning-budget 2048
   ```

   Linux shell:

   ```bash
   llama-server -m "$HOME/models/your-model.gguf" --host 127.0.0.1 --port 8080 -c 32768 -ngl 99 --reasoning-budget 2048
   ```

3. Start Cephalon, then open **Settings** and use **Download both defaults** to install the embedder and reranker. Restart Cephalon after installation.

   Windows PowerShell:

   ```powershell
   npm.cmd run tauri dev
   ```

   Linux shell:

   ```bash
   npm run tauri dev
   ```

4. In **Settings**, enter the llama.cpp URL (including its port), an optional display label, and its context window. Save, press **Connect**, import documents, then ask a question.


## Features

- Lightweight OLED friendly Tauri v2 desktop app with a default #000000 background and #FFE5CC text. Optional Graphite theme uses a #171717 dark background with #FFFFFF text.
- Document library with import, text-safe unknown file ingestion, reindexing, tags, delete, and document details.
- Durable ingestion jobs with live SSE and stage-level extraction, chunking, embedding, and persistence progress.
- Recoverable ingestion jobs with cancel/retry controls and interrupted-job recovery after restart.
- Explicit external llama.cpp server connection before querying.
- No bundled llama.cpp runtime or chat model; use your preferred llama.cpp build and acceleration backend.
- Hybrid retrieval: SQLite FTS5 BM25 + LanceDB dense vectors + reciprocal rank fusion.
- Jina ONNX embedder and reranker setup from Settings.
- Hierarchical indexing with summary nodes, parent chunks, and child chunks.
- Structured query stream events for subqueries, sources, metadata, tokens, errors, and completion.
- Low/Medium/High retrieval scope control for narrowing or broadening source review without changing model intelligence.
- Quick/Balanced/Thorough response effort with final-answer limits of 2,048 / 4,096 / 4,096 tokens. Thorough uses a 4,096-token draft and a separate 4,096-token final pass.
- Labeled workbench navigation, collapsible panels, keyboard dismissal, and a narrow-window details drawer.
- Source drawer with dense, lexical, fusion, rerank, confidence, and citation metadata.
- Retrieval Trace panel for inspecting vector, BM25, fused, reranked, unused, and final context candidates.
- Index Health panel for chunk counts, stale state, duplicate rates, retrieval counts, and embedding distribution.
- Minimal local eval runner with Recall@k and MRR.
- Answer Support panel with deterministic citation trust labels.
- Persistent chat history stored in SQLite.
- Searchable and renameable chat history with paginated older-message loading.
- Numeric-first metrics under the user Documents metrics directory.

## Architecture

- `src-tauri`: desktop shell, backend sidecar launch, window config, and release resources.
- `src`: React/Vite frontend, TanStack Query server cache, Zustand UI state, typed API client, and compact workbench panels.
- `python/main.py`: FastAPI/uvicorn entrypoint.
- `python/cephalon_core`: backend package for config, routes, schemas, storage, ingestion, retrieval, generation, jobs, metrics, documents, and external-server connection.

Read the architecture guide in [CEPHALON_ARCHITECTURE_DEEP_DIVE.html](CEPHALON_ARCHITECTURE_DEEP_DIVE.html).

## Local Data And Models

Default paths:

```powershell
$HOME\cephalon-data
$HOME\Documents\Cephalon Metrics
```

For example, the default Windows data directory is `C:\Users\<your-user>\cephalon-data`; on Linux it is `/home/<your-user>/cephalon-data`.

Expected model layout:

```text
~/cephalon-data/models/
  embedder/model.onnx
  embedder/tokenizer files...
  reranker/model.onnx
  reranker/tokenizer files...
```

The `models` directory contains only the embedder and reranker used by Cephalon. Your llama.cpp server owns the chat GGUF path and model-loading options.

Packaged installers do not bundle the embedder/reranker ONNX artifacts. If they are missing, Cephalon opens Settings and shows the configured download sources.

- **Download default** fetches the configured Hugging Face ONNX export and installs it into Cephalon's model directory.
- **Use local folder** copies a compatible Transformer embedder or cross-encoder reranker ONNX folder from your computer. It must contain an ONNX model, `tokenizer.json`, and `tokenizer_config.json`; Cephalon creates a local profile when one is absent and retains any external ONNX data files.
- Replacing an engine removes the previous installation before placing the new engine in its destination.
- Restart Cephalon after installing or replacing either engine. The running backend does not reload ONNX engines automatically.

### Reranker ONNX batch compatibility

During RAG, the reranker scores each candidate as a query/document pair. Some ONNX reranker exports support multiple pairs in one inference, while others are exported with an internal singleton reshape and only support one. Cephalon safely uses one pair at a time unless the model metadata explicitly declares a validated `max_batch_size`. This prevents the ONNX Runtime reshape error that looks like `Input shape: {6,133}, requested shape: {1,1,133}`.

In that message, `6` is the number of candidate pairs submitted together and `133` is the padded token length. The model holds `6 × 133 = 798` values, but its fixed reshape only accepts `1 × 1 × 133 = 133`; it is a batch-size limitation of that ONNX export, not an issue with the document text, llama.cpp, or the chat model.

Cephalon is not restricted to the prepared Jina exports. A local embedder may use a different model ID and vector dimension; Cephalon creates a separate LanceDB table for that embedding profile so it cannot mix incompatible vectors with an existing index. Reindex documents after changing embedders. Local rerankers may use different score shapes, but should be validated before use:

```powershell
py -3.14 scripts\validate_onnx_models.py --mark
```

For a reranker known to support a larger batch, explicitly validate the desired size before setting or retaining it in its `onnx_profile.json`:

```powershell
py -3.14 scripts\validate_onnx_models.py --mark --reranker-batch-size 4
```

Current prepared ONNX repos:

- Embedder: [s-lorin/jina-embeddings-v5-small-onnx](https://huggingface.co/s-lorin/jina-embeddings-v5-small-onnx)
- Reranker: [s-lorin/jina-reranker-v3-onnx](https://huggingface.co/s-lorin/jina-reranker-v3-onnx)

Use different prepared repos with:

```powershell
$env:CEPHALON_EMBEDDER_ONNX_REPO="s-lorin/jina-embeddings-v5-small-onnx"
$env:CEPHALON_RERANKER_ONNX_REPO="s-lorin/jina-reranker-v3-onnx"
```

Use `CEPHALON_EMBEDDER_ONNX_SUBFOLDER` or `CEPHALON_RERANKER_ONNX_SUBFOLDER` when the selected Hugging Face repository keeps its export below a subfolder.

## Maintaining Prepared ONNX Repositories

This is only needed by contributors preparing the default Hugging Face exports, not by ordinary users installing engines from Settings.

Upload the prepared local ONNX repositories with:

```powershell
.\scripts\upload_onnx_models_to_hf.ps1 -Namespace "s-lorin"
```

## Platform Setup

### Windows

```powershell
npm.cmd install
py -3.14 scripts\setup_python.py
```

The existing PowerShell wrapper remains available as `.\scripts\setup_local_python.ps1`. Both paths require Python 3.14 and run the same platform-neutral setup script.

### Linux

Install Python 3.14, Node.js, Rust, and the Tauri WebKitGTK build dependencies for your distribution. On Ubuntu/Debian-based systems:

```bash
sudo apt update
sudo apt install build-essential curl file libayatana-appindicator3-dev libgtk-3-dev libssl-dev libwebkit2gtk-4.1-dev librsvg2-dev libxdo-dev
npm install
python3.14 scripts/setup_python.py
```

The setup script disables user-site package leakage with `PYTHONNOUSERSITE=1`, installs `requirements.txt`, and runs the runtime preflight without requiring ONNX artifacts. It does not install or build llama.cpp.

Export ONNX models only when missing or intentionally replacing them:

Windows PowerShell:

```powershell
py -3.14 -m pip install --upgrade -r requirements-export.txt
py -3.14 export_onnx.py
py -3.14 scripts\validate_onnx_models.py --mark
```

Linux shell:

```bash
python3.14 -m pip install --upgrade -r requirements-export.txt
python3.14 export_onnx.py
python3.14 scripts/validate_onnx_models.py --mark
```

## Run

Start your own `llama-server` with its chosen GGUF model before launching Cephalon. For example, with a llama.cpp build that includes `llama-server`:

Windows PowerShell:

```powershell
& "C:\path\to\llama-server.exe" -m "D:\models\your-model.gguf" --host 127.0.0.1 --port 8080 -c 32768 -ngl 99 --reasoning-budget 2048
```

Linux shell:

```bash
llama-server -m "$HOME/models/your-model.gguf" --host 127.0.0.1 --port 8080 -c 32768 -ngl 99 --reasoning-budget 2048
```

In the terminal used to launch Cephalon, configure a non-default server address, display label, or context size as needed:

```powershell
$env:CEPHALON_LLAMA_SERVER_URL="http://127.0.0.1:8080"
$env:CEPHALON_LLAMA_SERVER_MODEL="Your model name" # display label only
$env:CEPHALON_LLAMA_SERVER_CONTEXT_TOKENS="32768" # match llama-server's -c value
```

```bash
export CEPHALON_LLAMA_SERVER_URL="http://127.0.0.1:8080"
export CEPHALON_LLAMA_SERVER_MODEL="Your model name" # display label only
export CEPHALON_LLAMA_SERVER_CONTEXT_TOKENS="32768" # match llama-server's -c value
```

Then launch one of the following:

Backend only:

```powershell
py -3.14 python\main.py
```

```bash
python3.14 python/main.py
```

Desktop development app:

```powershell
npm.cmd run tauri dev
```

```bash
npm run tauri dev
```

Frontend development (the established Vite-only command; it uses the local backend already running on port 8765):

```powershell
npm.cmd run dev
```

```bash
npm run dev
```

Use `npm run dev:full` only when you specifically want one terminal to launch both the local backend and Vite. Then open Cephalon, set the server URL and port in **Settings**, and press **Connect**. Cephalon checks the server health endpoint, shows the model actually loaded by llama.cpp, and uses its OpenAI-compatible `/v1/chat/completions` streaming API. It does not start, download, compile, or package llama.cpp, and it does not load a GGUF itself. Keep the server bound to `127.0.0.1` unless you deliberately intend remote access.

## Build

Frontend:

```powershell
npm.cmd run build
```

```bash
npm run build
```

Backend sidecar:

```powershell
py -3.14 build_backend.py
```

```bash
python3.14 build_backend.py
```

The sidecar build packages the backend only. It does not include llama.cpp, chat GGUF files, or embedder/reranker model folders; ONNX assets are installed into the user model directory from the app Settings screen.

Tauri package:

```powershell
npm.cmd run tauri build
```

```bash
npm run tauri build
```

Full native release pipeline:

Windows PowerShell:

```powershell
.\scripts\build_release.ps1
```

Linux shell:

```bash
python3.14 scripts/build_release.py
```

Each release must be built on its target operating system. Windows produces Windows artifacts and Linux produces native Linux artifacts such as `.deb` and AppImage; do not reuse a packaged backend sidecar across operating systems.

## Configuration

Common runtime variables:

```powershell
$env:CEPHALON_DATA_DIR="C:\path\to\data"
$env:CEPHALON_MODEL_DIR="C:\path\to\models"
$env:CEPHALON_METRICS_DIR="$HOME\Documents\Cephalon Metrics"
$env:CEPHALON_HOST="127.0.0.1"
$env:CEPHALON_PORT="8765"
$env:CEPHALON_LLAMA_SERVER_URL="http://127.0.0.1:8080"
$env:CEPHALON_LLAMA_SERVER_MODEL="External llama.cpp server" # display label only
$env:CEPHALON_LLAMA_SERVER_CONTEXT_TOKENS="32768" # match llama-server's -c/--ctx-size value
$env:CEPHALON_CONTEXT_TOKENS="32768" # fallback only when server context is not supplied
$env:CEPHALON_FULL_CONTEXT="0"
$env:CEPHALON_EMBEDDER_ONNX_REPO="s-lorin/jina-embeddings-v5-small-onnx"
$env:CEPHALON_RERANKER_ONNX_REPO="s-lorin/jina-reranker-v3-onnx"
$env:CEPHALON_EMBEDDER_ONNX_SUBFOLDER=""
$env:CEPHALON_RERANKER_ONNX_SUBFOLDER=""
```

`CEPHALON_HOST` and `CEPHALON_PORT` configure Cephalon's own backend. `CEPHALON_LLAMA_SERVER_URL` configures the separately running llama.cpp server. When supplied, `CEPHALON_LLAMA_SERVER_CONTEXT_TOKENS` takes precedence for generation prompt budgeting; set it to the context window used to start llama-server.

Cephalon reserves 2,048 generated tokens for reasoning and sends llama.cpp enough total completion capacity to preserve the visible-answer limits: Quick 2,048, Balanced 4,096, and Thorough 4,096 for both its draft and final pass. Start a reasoning-capable llama.cpp server with `--reasoning-budget 2048` so that allocation is enforced. Non-reasoning models simply use the extra completion capacity without a thinking phase.

On Linux, set the same variables with `export NAME="value"` instead of PowerShell's `$env:NAME="value"` syntax.

For frontend-only remote testing, set `VITE_CEPHALON_API_URL` at build/dev time or set `cephalon.apiBaseUrl` in browser local storage.

## API

- `GET /health`: startup status, paths, model diagnostics, backend status, retrieval state, and embedding metadata.
- `GET /models`: configured external llama.cpp server model and active connection state.
- `POST /models/load`: connect to the configured external llama.cpp server after its model is already loaded.
- `GET/PUT /models/server`: read or save the external llama.cpp URL, display label, and optional context window.
- `GET /models/onnx/status`: inspect embedder/reranker setup state.
- `POST /models/onnx/download`: download configured prepared ONNX artifacts into the model directory; restart required after installation.
- `POST /models/onnx/install-local`: install a local exported ONNX folder for the embedder or reranker; restart required after installation.
- `GET/PUT /settings`: RAG and generation defaults.
- `POST /ingest`: queue file/folder ingestion, including note vaults imported as normal folders.
- `GET /jobs`: recent ingestion jobs.
- `POST /jobs/{id}/cancel` and `POST /jobs/{id}/retry`: control recoverable ingestion jobs.
- `GET /retrieval/traces`: recent retrieval traces.
- `GET /retrieval/traces/{query_id}`: full retrieval trace with candidate stages, scores, context, and latency.
- `GET /observability/index-health`: document/chunk/index health summary.
- `GET/POST /eval/runs`: run and inspect small JSON eval sets.
- `POST /feedback`: store answer or citation feedback locally.
- `GET /events`: SSE job/document/settings stream.
- `GET/PATCH/DELETE /documents/{id}`: document details, rename, and delete.
- `POST /documents/{id}/reindex`: reindex while preserving display name and tags.
- `GET/POST/PATCH/DELETE /conversations`: chat history management.
- `POST /query`: typed SSE query stream. The configured external llama.cpp server must be connected.
- `POST /metrics/export`: write a numeric corpus snapshot CSV.

## Troubleshooting

- **Connect fails:** confirm llama-server is still running, its URL and port match `CEPHALON_LLAMA_SERVER_URL`, and opening `<server-url>/health` succeeds. Cephalon does not start the server for you.
- **ONNX engines are installed but unavailable:** restart Cephalon after any download or local-folder replacement.
- **The backend does not start:** confirm `py -3.14 --version` (Windows) or `python3.14 --version` (Linux) succeeds, then rerun the platform setup command above.
- **The Linux desktop build cannot find WebKitGTK:** install your distribution's Tauri/WebKitGTK development packages, including `libwebkit2gtk-4.1-dev` on current Ubuntu/Debian releases.
