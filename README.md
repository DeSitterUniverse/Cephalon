# Cephalon

Cephalon is a local-first document search and answer workbench. It imports documents, retains PDF layout/provenance, indexes text locally, and produces answers with stable citations such as `[[src:S1]]`.

Generation and retrieval are deliberately separate. A user-operated `llama-server` provides chat completion on port 8080. Cephalon uses a second, dedicated `llama-server` for embeddings on port 8090 and a separate Python process for reranking.

## Fixed retrieval stack

Cephalon supports one retrieval stack only:

| Role | Fixed model | Runtime | Output |
| --- | --- | --- | --- |
| Embedder | Jina Embeddings v5 Nano Retrieval, `Q8_0` GGUF | dedicated llama.cpp embeddings server | normalized 768-dimensional vectors |
| Reranker | Jina Reranker v3.5 | isolated Transformers worker with `trust_remote_code=True` | listwise rank and raw relevance score |

The Nano vector table uses normalized 768-dimensional embeddings. Reindex documents after installing the retrieval stack.

## Requirements

- Windows or Linux
- Node.js and npm
- Python 3.14 (`py -3.14` on Windows)
- A recent `llama-server` build with the required backend (Vulkan for the AMD GPU setup described below)
- A chat GGUF chosen and started by the user
- Internet access only when downloading the two fixed retrieval models from Hugging Face

Install project dependencies:

```powershell
py -3.14 -m pip install -r requirements.txt
npm.cmd install
```

The desktop package bundles the application backend, but not llama.cpp, a chat GGUF, or the retrieval model files.

## Model locations

By default, Cephalon stores retrieval models below `~/cephalon-data/models`:

```text
~/cephalon-data/models/
  jina-v5-nano-retrieval-q8_0/
    v5-nano-retrieval-Q8_0.gguf
  jina-reranker-v3.5/
    config.json
    tokenizer.json
    ... official Hugging Face files and manifest
```

Use `CEPHALON_DATA_DIR` or `CEPHALON_MODEL_DIR` to change those locations. The embedder checksum is checked against the fixed official SHA-256. Reranker verification compares local files to the official revision manifest; a mismatch fails verification.

## First-time setup

1. Start your chat-generation server (separate from retrieval).
2. Start Cephalon.
3. Open **Settings → Fixed retrieval stack** and download the embedder and reranker.
4. Restart Cephalon so it can start its owned embedder and reranker processes.
5. Run **Reindex all documents** after installing the fixed retrieval stack.

The Settings panel provides download, integrity verification, folder opening, confirmed cache deletion, process health, model paths, the fixed 768-dimension value, and reindex status.

## Start the chat server

Cephalon never loads the chat GGUF itself. Start the server separately and configure `CEPHALON_LLAMA_SERVER_URL` if it is not `http://127.0.0.1:8080`.

Windows example with Vulkan GPU offload:

```powershell
& "C:\AI\llama.cpp\build\bin\Release\llama-server.exe" `
  -m "C:\AI\models\your-chat-model.gguf" `
  --device Vulkan0 --gpu-layers 999 `
  --ctx-size 8192 --host 127.0.0.1 --port 8080 --no-webui
```

Linux example:

```bash
llama-server -m "$HOME/models/your-chat-model.gguf" \
  --ctx-size 8192 --host 127.0.0.1 --port 8080 --no-webui
```

Keep the server bound to `127.0.0.1` unless remote access is deliberate. Confirm it is ready with `http://127.0.0.1:8080/health`.

## Embedding server

After the Nano GGUF is installed, Cephalon normally starts its dedicated server automatically when the backend starts. It uses:

```text
--embedding --pooling last --embd-normalize 2
--device Vulkan0 --gpu-layers 999
--batch-size 4096 --ubatch-size 4096
--host 127.0.0.1 --port 8090 --no-webui
```

It calls `/v1/embeddings` in batches during ingestion and for query embedding, validates response ordering, and rejects any output that is not exactly 768-dimensional.

### Vulkan GPU embedder (Windows)

Cephalon starts its managed embedding server with `--device Vulkan0 --gpu-layers 999`. Override those values with `CEPHALON_EMBEDDER_DEVICE` and `CEPHALON_EMBEDDER_GPU_LAYERS` when your llama.cpp device has a different name or offload target. You can also start the dedicated server yourself before Cephalon; it detects the healthy endpoint and reuses it:

```powershell
& "C:\AI\llama.cpp\build\bin\Release\llama-server.exe" `
  -m "$env:USERPROFILE\cephalon-data\models\jina-v5-nano-retrieval-q8_0\v5-nano-retrieval-Q8_0.gguf" `
  --embedding --pooling last --embd-normalize 2 `
  --device Vulkan0 --gpu-layers 999 `
  --batch-size 4096 --ubatch-size 4096 `
  --host 127.0.0.1 --port 8090 --no-webui
```

Set `CEPHALON_EMBEDDER_SERVER_URL` and `CEPHALON_EMBEDDER_SERVER_PORT` if using a different localhost endpoint. Set `CEPHALON_LLAMA_SERVER_BIN` if Cephalon should auto-start the dedicated server from a different llama.cpp path.

The chat and embedding servers must remain separate; never point the embedding client at the chat server.

## Reranker worker and degraded mode

When the official v3.5 model directory is installed, Cephalon starts `jina_reranker_worker` in a separate Python process. It loads the model through the official Transformers custom-code interface and calls `model.rerank(query, documents, top_n=None)` once for the complete fused candidate set.

The worker returns the original candidate index, raw v3.5 relevance score, and listwise order. Cephalon maps those results back to source IDs, chunk IDs, document IDs, provenance, and the original dense/lexical/fusion scores.

- Missing or failed embedder: retrieval is disabled and queries are blocked.
- Missing or failed reranker: retrieval continues in visible degraded mode with dense-plus-FTS5/RRF results only.
- The installed Windows PyTorch runtime may be CPU-only. A CPU-only reranker is valid but will be shown by the process status; do not describe it as GPU accelerated.

## Start Cephalon

### Browser development

Start backend and Vite together:

```powershell
npm.cmd run dev:full
```

Open `http://127.0.0.1:1420`.

Or run separately:

```powershell
py -3.14 python\main.py
```

```powershell
npm.cmd run dev
```

### Tauri desktop development

```powershell
npm.cmd run tauri dev
```

### Installed desktop release

Launch Cephalon from its shortcut. The bundled sidecar starts at `127.0.0.1:8765`; chat generation remains the separately operated server on 8080.

## Runtime configuration

```powershell
$env:CEPHALON_DATA_DIR="$env:USERPROFILE\cephalon-data"
$env:CEPHALON_MODEL_DIR="$env:USERPROFILE\cephalon-data\models"
$env:CEPHALON_LLAMA_SERVER_BIN="C:\AI\llama.cpp\build\bin\Release\llama-server.exe"
$env:CEPHALON_LLAMA_SERVER_URL="http://127.0.0.1:8080"
$env:CEPHALON_LLAMA_SERVER_CONTEXT_TOKENS="8192"
$env:CEPHALON_EMBEDDER_SERVER_URL="http://127.0.0.1:8090"
$env:CEPHALON_EMBEDDER_SERVER_PORT="8090"
$env:CEPHALON_EMBEDDER_DEVICE="Vulkan0"
$env:CEPHALON_EMBEDDER_GPU_LAYERS="999"
$env:CEPHALON_EMBEDDER_BATCH_SIZE="16"
$env:CEPHALON_EMBEDDER_PHYSICAL_BATCH_SIZE="4096"
```

`CEPHALON_HOST` and `CEPHALON_PORT` control the Cephalon API itself (defaults: `127.0.0.1:8765`).

Common ports:

| Service | Default port |
| --- | --- |
| Cephalon backend | 8765 |
| Vite development server | 1420 |
| Chat llama.cpp server | 8080 |
| Dedicated Nano embeddings server | 8090 |

## Retrieval and evidence flow

1. Documents are parsed with page/layout/table/caption/bounding-box provenance preserved where available. Malformed pages use a text-safe fallback.
2. Chunks are embedded by Nano Retrieval and stored only in `vectors_jina_v5_nano_retrieval_768`.
3. Dense LanceDB search and SQLite FTS5 BM25 search remain independent.
4. Results are combined by reciprocal-rank fusion.
5. The full fused set is listwise-reranked by Jina v3.5 when available.
6. Evidence validation, claim coverage, citation accounting, retrieval traces, and frontend diagnostics retain source/chunk/document identities and provenance.

Trace candidates expose `vector_score`, `lexical_score`, `fusion_score`, `reranker_raw_score`, `listwise_rank`, `final_score`, and retrieval rank. Raw v3.5 scores and listwise rank are distinct fields.

## Model and reindex APIs

All endpoints are local backend endpoints.

| Endpoint | Purpose |
| --- | --- |
| `GET /models/status` | Fixed model identity, paths, integrity/install state, runtime state, and reindex requirement |
| `POST /models/download` | Download `{ "kind": "embedder" }` or `{ "kind": "reranker" }`; restart required |
| `POST /models/verify` | Verify model files; use the same `kind` body |
| `POST /models/delete` | Delete a model cache; requires `{ "kind": "…", "confirmed": true }` |
| `POST /models/open` | Open the model directory for a `kind` |
| `GET /runtime/embedder/status` | Embedder port, PID, request/health times, and failure state |
| `GET /runtime/reranker/status` | Worker PID, queue size, health time, and last failure |
| `POST /reindex/full` | Queue every document for reindexing |
| `POST /reindex/stale` | Queue only stale documents |
| `GET /reindex/progress` | Current or last persisted run totals and stale-index state |

Reindex progress remains available after a run completes. A successful reindex does not delete source documents or their PDF provenance.

## Validation

Run the backend and frontend checks:

```powershell
py -3.14 -m pytest python\test_backend_stabilization.py -q
npm.cmd run test:frontend
npm.cmd run build
```

For live checks, start the desired chat and embedding servers, start Cephalon, then inspect `GET /models/status`, `GET /runtime/embedder/status`, and `GET /runtime/reranker/status`. After local testing, stop only the servers you started and confirm their ports are closed.
