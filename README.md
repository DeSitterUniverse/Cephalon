# Cephalon

Cephalon is a local-first desktop workbench for asking grounded questions about your own documents. Import a library, connect the local chat model you already use, and get answers with stable citations such as `[[src:S1]]`.

It is designed for private research, technical documentation, notes, PDFs, and specialised collections where source traceability matters.

Cephalon is particularly useful with models fine-tuned for specialised knowledge domains, tasks, writing styles, programming conventions, academic subjects, specialised knowledge bases, or creative work.

- **Local-first operation:** Files in the Library are local. You choose the chat model, which runs through an external llama.cpp server. Cephalon manages Jina Reranker v3.5 and Jina Embeddings v5 Nano for document retrieval.

I built this originally for running LLM inference on a large corpus of scientific and technical papers. I improved the architecture by incorporating the following RAG techniques:

- **Hybrid retrieval:** Semantic search finds passages with similar meaning, while SQLite FTS5 finds exact terms. Cephalon keeps the result sets independent, combines their ranks, and preserves strong candidates from either path.
- **Full-set listwise reranking:** Jina Reranker v3.5 compares the complete fused candidate set in one pass. Cephalon then selects the final context without discarding a useful passage too early.
- **Hierarchical context assembly:** Cephalon retrieves precise child chunks first, then adds bounded sibling or parent context when it improves completeness. The original child remains the citation anchor. Source: [HiChunk](https://arxiv.org/abs/2509.11552).
- **Layout-aware PDF evidence:** Text can be expanded to related headings, captions, tables, figures, and cross-page continuations. This avoids treating every block as unrelated. Based on: [LAD-RAG: Layout-Aware Dynamic RAG framework](https://arxiv.org/abs/2510.07233).
- **Coverage-aware evidence control:** Cephalon breaks a question into concrete evidence needs and selects sources that cover them. When Thorough mode is selected, it can run one targeted follow-up search for missing evidence. Source: [S2G-RAG](https://arxiv.org/abs/2604.23783).
- **Verified answers:** Cited claims are checked for missing support, contradictions, negation errors, and incorrect numbers or units. Thorough mode can audit the draft and repair it once. Source: [OpenScholar](https://arxiv.org/abs/2411.14199) and [RAGChecker](https://arxiv.org/abs/2408.08067).
- **Structured table reasoning:** PDF, CSV, and XLSX tables retain row, column, cell, and location data. Cephalon can run bounded lookups, filters, comparisons, and arithmetic, then cite and recheck the exact cells used. Source: [T-RAG](https://arxiv.org/abs/2203.16714) and [T²-RAGBench](https://arxiv.org/abs/2506.12071).
- **Exact provenance and retrieval traces:** Citations point to stable source chunks with page, layout, bounding-box, table-cell, and asset details when available. The Sources and Trace views show what was retrieved, reranked, selected, and verified.

![Cephalon's chat workspace with a connected local model](docs/screenshots/rag-cited-answer.png)

![A cited long-form explanation with evidence and citations](docs/screenshots/long-evidence-answer.png)

## What it does

- Imports common office, text, data, and PDF files into a searchable local library.
- Preserves useful PDF provenance, including pages, layout, tables, captions, and bounding boxes when available.
- Combines semantic and keyword search, then shows sources and retrieval diagnostics alongside answers.
- Validates evidence and claim coverage before presenting citations.
- Keeps chat generation separate from document retrieval, so you stay in control of the chat GGUF and llama.cpp server.
- Provides a focused desktop workflow: library, chat, sources, trace, health, evaluations, settings, and local model management.

## The retrieval stack

Cephalon uses one local retrieval stack:

| Role | Model | Runtime |
| --- | --- | --- |
| Embedder | Jina Embeddings v5 Nano Retrieval `Q8_0` | dedicated llama.cpp embeddings server, normalized 768-dimensional vectors |
| Reranker | Jina Reranker v3.5 `Q8_0` GGUF | verified llama.cpp/Vulkan listwise worker; CPU fallback is disabled |

Dense LanceDB and SQLite FTS5 results remain independent and are fused with reciprocal-rank fusion. When the reranker is available, the full fused candidate set is reranked listwise. If the reranker is unavailable, Cephalon continues in clearly marked degraded mode. If the embedder is unavailable, retrieval is safely disabled.

## Jina AI models

Cephalon uses Jina AI's [Jina Embeddings v5 Nano Retrieval](https://huggingface.co/jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF) and [Jina Reranker v3.5 GGUF](https://huggingface.co/jinaai/jina-reranker-v3.5-GGUF). Thank you to Jina AI for making these retrieval models available.

Both models are licensed under [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/); use of the model files is subject to that license.

## Quick start

Cephalon supports Windows and Linux. The commands below use Windows PowerShell. See [LOCAL_STARTUP_NOTES.md](LOCAL_STARTUP_NOTES.md) for Linux and detailed setup notes.

1. Install Rust, Python 3.14, and a recent llama.cpp build with `llama-server`.

2. Install Cephalon dependencies:

   ```powershell
   py -3.14 scripts\setup_python.py
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
   cargo run
   ```

5. Build `llama-embedding` from llama.cpp revision
   `80c940e5a80555167c4ec37652deca6528810f91` with Vulkan enabled.
   Set `CEPHALON_RERANKER_LLAMA_EMBEDDING_BIN` to that executable. In
   **Settings → Fixed retrieval stack**, download the embedder and reranker.
   Restart Cephalon, then run **Reindex all documents**.

6. Press **Connect** for the chat server, import a few documents, and ask a question. Open **Sources** or **Trace** whenever you want to inspect the supporting evidence.

## Everyday use

For the best results:

1. Import the documents that should be treated as your reference set.
2. Reindex after installing or changing the retrieval models, or after changing the chunking configuration or document collection.
3. Ask focused questions when source precision matters. Request citations when the wording must be auditable.
4. Use **Sources** or the source drawer to jump from an answer to the exact document chunk and provenance. Open **Trace** to examine the retrieval process.
5. Treat weak or unsupported results as a prompt to refine the question or inspect the retrieved evidence rather than as a confident answer.

Saved conversations remain on your computer and are used only as searchable chat history and optional conversation context, not as model training data.

## Runtime separation

Cephalon manages the fixed embedding and reranking models used for document retrieval. Chat generation remains user-controlled and runs through an external llama.cpp server, so you can choose any compatible GGUF without changing the retrieval index or coupling the chat server to the retrieval stack.

Chat generation and document embedding use separate llama.cpp server processes because they load different models and operate with different runtime settings. The processes serve different roles and cannot share a single server instance.

Cephalon manages the embedding process automatically after the retrieval model is installed.

On Windows, the managed embedding process defaults to Vulkan0 with full GPU layer offload. If your llama.cpp installation uses a different device name or offload configuration, override these values:

$env:CEPHALON_EMBEDDER_DEVICE="Vulkan0"
$env:CEPHALON_EMBEDDER_GPU_LAYERS="999"

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
  jina-reranker-v3.5-gguf-q8_0/
    jina-reranker-v3.5-Q8_0.gguf
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
cargo run
```

### Default ports

| Service                    | Port |
| -------------------------- | ---: |
| Cephalon local API         | 8765 |
| Chat llama.cpp server      | 8080 |
| Managed embedding server   | 8090 |

See [LOCAL_STARTUP_NOTES.md](LOCAL_STARTUP_NOTES.md) for the complete environment-variable reference, Linux commands, manual server operation, release builds, and troubleshooting.

## Development and packaging

The desktop shell is a native Rust application built with the pinned GPUI
Community Edition (GPUI-CE) runtime and its `gpui_elements` input primitives.
Cephalon's existing Python HTTP/SSE backend remains a separate local service.
The frontend does not embed retrieval or model logic. See
[native-frontend.md](docs/native-frontend.md) for the dependency and
packaging boundary.

| Task                                    | Windows command                                    |
| --------------------------------------- | -------------------------------------------------- |
| Run the native desktop application      | `cargo run`                                         |
| Run only the Python backend             | `py -3.14 python\main.py`                          |
| Check the native GPUI-CE application    | `cargo check`                                       |
| Build the native release                | `cargo build --release`                             |
| Build the packaged desktop directory    | `py -3.14 scripts\build_release.py`                |

The packaged application includes Cephalon’s Python backend. It does not include llama.cpp, a chat GGUF, or the retrieval-model weights. Retrieval models can be installed from the Settings page after Cephalon is launched.

## Diagnostics and local API

The Settings page reports model installation state, integrity checks, runtime health, process IDs, active paths, and reindex progress. Diagnostic views expose retrieval evidence, claim validation, candidate rankings, latency measurements, and index-health information.

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
cargo fmt --check
cargo check
cargo test
```

## License

Cephalon’s source code is available under the [MIT License](LICENSE). Retrieval-model files remain subject to their respective licenses described above.

