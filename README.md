# Cephalon

Cephalon is a local-first desktop workbench for asking grounded questions about your own documents. Import a library, connect the local chat model you already use, and get answers with stable citations such as `[[src:S1]]`.

It is designed for private research, technical documentation, notes, PDFs, and specialised collections where knowing *which source supports an answer* matters as much as the answer itself. Your documents, metadata, chat history, and retrieval index stay on your computer.

Cephalon is particularly useful with models fine-tuned for specialised knowledge domains, tasks, writing styles, programming conventions, academic subjects, specialised knowledge base, or creative work.

I built this originally for running LLM inference on a large corpus of scientific and technical papers. I improved the architecture by incorporating the best RAG techniques I found:

- Hybrid retrieval with independent signals
- True full-set listwise reranking
- Citation accounting and fail-safe behavior
- Exact evidence and provenance
- RATE-style validation
- High document fidelity: PDF ingestion preserves page, layout, tables, captions, bounding boxes, and asset provenance.
- Auditable retrieval

` `

![A cited answer to a question about the RATE paper](docs/screenshots/rag-cited-answer.png)

` ` 

![A cited long-form explanation with evidence and citations](docs/screenshots/long-evidence-answer.png)

## What it does

- Imports common office, text, data, and PDF files into a searchable local library.
- Preserves useful PDF provenance, including pages, layout, tables, captions, and bounding boxes when available.
- Combines semantic and keyword search, then shows sources and retrieval diagnostics alongside answers.
- Validates evidence and claim coverage before presenting citations.
- Keeps chat generation separate from document retrieval, so you stay in control of the chat GGUF and llama.cpp server.
- Provides a focused desktop workflow: library, chat, sources, trace, health, evaluations, settings, and local model management.

## The retrieval stack

Cephalon intentionally uses one local retrieval stack:

| Role | Model | Runtime |
| --- | --- | --- |
| Embedder | Jina Embeddings v5 Nano Retrieval `Q8_0` | dedicated llama.cpp embeddings server, normalized 768-dimensional vectors |
| Reranker | Jina Reranker v3.5 BF16 GGUF | verified llama.cpp/Vulkan listwise worker, with the previous Transformers CPU worker as compatibility fallback |

Dense LanceDB retrieval and SQLite FTS5 keyword retrieval stay independent, are fused with reciprocal-rank fusion, and the full fused candidate set is listwise reranked when the reranker is available. If the reranker is unavailable, Cephalon continues in clearly marked degraded mode; if the embedder is unavailable, retrieval is safely disabled.

## Jina AI models

Cephalon uses Jina AI's [Jina Embeddings v5 Nano Retrieval](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF) and [Jina Reranker v3.5 GGUF](https://huggingface.co/jinaai/jina-reranker-v3.5-GGUF). Thank you to Jina AI for making these retrieval models available.

Both models are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); use of the model files is subject to that license.

## Quick start

Cephalon supports Windows and Linux. The commands below use Windows PowerShell; see [LOCAL_STARTUP_NOTES.md](LOCAL_STARTUP_NOTES.md) for Linux and detailed setup notes.

1. Install Node.js, Python 3.14, and a recent llama.cpp build with `llama-server`.

2. Install Cephalon dependencies:

   ```powershell
   py -3.14 scripts\setup_python.py
   npm.cmd install
   ```

3. Start the chat server with the GGUF you want to use. Cephalon does not choose or load this model for you:

   ```powershell
   & "C:\AI\llama.cpp\build\bin\Release\llama-server.exe" `
     -m "C:\AI\models\your-chat-model.gguf" `
     --device Vulkan0 --gpu-layers 999 `
     --ctx-size 8192 --host 127.0.0.1 --port 8080 --no-webui
   ```

4. Start the desktop app:

   ```powershell
   npm.cmd run tauri dev
   ```

5. Build `llama-embedding` from llama.cpp revision
   `80c940e5a80555167c4ec37652deca6528810f91` with Vulkan enabled, then set
   `CEPHALON_RERANKER_LLAMA_EMBEDDING_BIN` to that executable. In
   **Settings → Fixed retrieval stack**, download the embedder and reranker.
   Restart Cephalon, then run **Reindex all documents**.

6. Press **Connect** for the chat server, import a few documents, and ask a question. Open **Sources** or **Trace** whenever you want to inspect the supporting evidence.

## Everyday use

For the best results:

1. Import the documents that should be treated as your reference set.
2. Reindex after installing the retrieval models or after intentionally changing the document collection.
3. Ask focused questions and request citations when the wording must be auditable.
4. Use the source drawer to jump from an answer to the exact document chunk and provenance.
5. Treat a weak-support indicator as a prompt to refine the question or inspect the evidence rather than as a confident answer.

Saved chats are local searchable memory, not model training data.

## Runtime separation

Cephalon uses separate llama.cpp processes for chat generation and document embeddings. Each process loads a different model and serves a different role, so they cannot share a single server instance.

Cephalon manages the embedding process automatically after the retrieval model is installed. The chat server remains external and user-controlled, allowing any compatible GGUF model to be used without coupling it to the retrieval stack.

On Windows, the managed embedding process defaults to Vulkan0 with full GPU layer offload. Override these values when your llama.cpp installation uses a different device name or offload configuration:

$env:CEPHALON_EMBEDDER_DEVICE="Vulkan0"
$env:CEPHALON_EMBEDDER_GPU_LAYERS="999"

## Everyday use

1. Import the documents you want Cephalon to use as its reference library.
2. Reindex documents after changing the retrieval models, chunking configuration, or source collection.
3. Ask focused questions when source precision matters.
4. Open **Sources** to inspect the evidence behind an answer, or **Trace** to examine the retrieval process.
5. Treat weak or unsupported results as a reason to refine the question or review the retrieved evidence.

Saved conversations remain on your computer and are used only as searchable chat history and optional conversation context.

## Model processes

Cephalon manages the fixed embedding and reranking models used for document retrieval. The chat model remains user-controlled and runs through an external llama.cpp server, allowing you to choose any compatible GGUF without changing the retrieval index.

Chat generation and document embedding use separate llama.cpp server processes because they load different models and operate with different runtime settings.

## Local files and configuration

Cephalon stores its database, retrieval index, extracted document assets, and model files under `~/cephalon-data` by default. On Windows, this normally resolves to:

```text
C:\Users\<you>\cephalon-data
```

The retrieval models are stored under:

```text
~/cephalon-data/models/
  jina-v5-nano-retrieval-q8_0/
    v5-nano-retrieval-Q8_0.gguf
  jina-reranker-v3.5-gguf-bf16/
    jina-reranker-v3.5-BF16.gguf
    projector.safetensors
    tokenizer.json
    ...
```

The Settings page can download, verify, open, and remove these model installations. The embedder GGUF is checked against its expected SHA-256, while the reranker is checked against its pinned Hugging Face revision manifest.

Common configuration overrides:

| Variable                               | Purpose                                         | Default                   |
| -------------------------------------- | ----------------------------------------------- | ------------------------- |
| `CEPHALON_DATA_DIR`                    | Database, indexes, assets, and application data | `~/cephalon-data`         |
| `CEPHALON_MODEL_DIR`                   | Retrieval-model directory                       | `<data directory>/models` |
| `CEPHALON_LLAMA_SERVER_BIN`            | Path to the `llama-server` executable           | Platform-specific         |
| `CEPHALON_LLAMA_SERVER_URL`            | Chat-generation server                          | `http://127.0.0.1:8080`   |
| `CEPHALON_LLAMA_SERVER_CONTEXT_TOKENS` | Chat model context limit override               | Detected when available   |
| `CEPHALON_EMBEDDER_DEVICE`             | llama.cpp device used by the managed embedder   | `Vulkan0` on Windows      |
| `CEPHALON_EMBEDDER_GPU_LAYERS`         | Number of embedder layers offloaded to the GPU  | `999`                     |

Set environment variables before launching Cephalon. For example:

```powershell
$env:CEPHALON_EMBEDDER_DEVICE="Vulkan0"
$env:CEPHALON_EMBEDDER_GPU_LAYERS="999"
npm.cmd run tauri dev
```

### Default ports

| Service                    | Port |
| -------------------------- | ---: |
| Cephalon local API         | 8765 |
| Browser development server | 1420 |
| Chat llama.cpp server      | 8080 |
| Managed embedding server   | 8090 |

See [LOCAL_STARTUP_NOTES.md](LOCAL_STARTUP_NOTES.md) for the complete environment-variable reference, Linux commands, manual server operation, release builds, and troubleshooting.

## Development and packaging

| Task                                    | Windows command                                    |
| --------------------------------------- | -------------------------------------------------- |
| Run the desktop application             | `npm.cmd run tauri dev`                            |
| Run the browser development environment | `npm.cmd run dev:full`                             |
| Run only the Python backend             | `py -3.14 python\main.py`                          |
| Build the frontend                      | `npm.cmd run build`                                |
| Check the Tauri project                 | `cargo check --manifest-path src-tauri\Cargo.toml` |
| Build the desktop package               | `npm.cmd run tauri build`                          |

The packaged application includes Cephalon’s Python backend. It does not include llama.cpp, a chat GGUF, or the retrieval-model weights. Retrieval models can be installed from the Settings page after Cephalon is launched.

## Diagnostics and local API

The Settings page reports model installation state, integrity checks, runtime health, process IDs, active paths, and reindex progress. Cephalon also exposes retrieval evidence, claim validation, candidate rankings, latency measurements, and index-health information through its diagnostic views.

The local API can be used for scripting and troubleshooting:

| Endpoint                          | Purpose                                                     |
| --------------------------------- | ----------------------------------------------------------- |
| `GET /health`                     | Overall backend and retrieval-stack health                  |
| `GET /models/status`              | Installed models, integrity state, and reindex requirements |
| `GET /runtime/embedder/status`    | Embedding-server process and health                         |
| `GET /runtime/reranker/status`    | Reranker worker and queue health                            |
| `POST /reindex/full`              | Reindex the complete document library                       |
| `POST /reindex/stale`             | Reindex only documents whose indexes are stale              |
| `GET /reindex/progress`           | Current or most recently completed reindex operation        |
| `GET /observability/index-health` | Document, chunk, index, and retrieval statistics            |
| `GET /retrieval/traces`           | Recently persisted retrieval traces                         |

All endpoints are local by default and are served from `http://127.0.0.1:8765`.

## Validation

Run the complete validation suite before packaging or submitting substantial changes:

```powershell
py -3.14 -m pytest python -q
npm.cmd run test:frontend
npm.cmd run build
cargo check --manifest-path src-tauri\Cargo.toml
```

## License

Cephalon’s source code is available under the [MIT License](LICENSE). Retrieval-model files remain subject to their respective licenses described above.

