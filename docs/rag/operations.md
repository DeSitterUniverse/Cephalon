# RAG operations

Cephalon uses three local services: a user-operated llama.cpp chat server (default port 8080), a managed EmbeddingGemma 2 Q8_0 GGUF embedding server (8090), and a managed Jina v3.5 listwise GGUF worker (8091). The backend defaults to 8765. Retrieval processes bind to loopback and use installed files offline.

## Retrieval setup

The embedder uses `embeddinggemma-2-Q8_0.gguf` from [Unsloth's EmbeddingGemma 2 conversion](https://huggingface.co/unsloth/embeddinggemma-2-GGUF), pinned to commit `ba3888272494be64ed88c9eb536ddc61a1be73d5` and SHA-256 `6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc`. It is the 270M text model, with 768 output dimensions and an 8192-token input limit. The source Google model revision is `914f7f89142e33e77833254d9c9b90c3cef7303b`. No multimodal projector or Transformers embedding environment is required.

Install the GGUF and a separate [llama.cpp b11456 Vulkan runtime](https://github.com/ggml-org/llama.cpp/releases/tag/b11456):

```powershell
py -3.14 scripts\setup_retrieval_models.py embeddinggemma `
  --model-dir "$env:USERPROFILE\cephalon-data\models\embeddinggemma-2-text-270m-q8_0"

py -3.14 scripts\setup_retrieval_models.py llama `
  --runtime-dir "$env:USERPROFILE\cephalon-data\runtimes\llama-b11456-vulkan"
```

The installer verifies the pinned model and runtime archive checksums. Automatic runtime setup supports Windows AMD64 and Linux x86_64. On Linux, use `python3` and the corresponding `~/cephalon-data` paths; install the system Vulkan loader and the GPU driver. Linux inference has not been validated in this change. On other platforms, install the same build manually and set `CEPHALON_EMBEDDER_LLAMA_SERVER_BIN`. The runtime installer requires an empty versioned directory or an unchanged installation whose manifest verifies; it does not replace another runtime. The application never downloads weights or binaries at startup.

The default embedding executable is `data/runtimes/llama-b11456-vulkan/llama-server.exe` on Windows and `llama-server` on Linux. `CEPHALON_EMBEDDER_LLAMA_SERVER_BIN` overrides it; the backend rejects a different build number. Chat-server settings and `LLAMA_ARG_*` environment overrides are kept independent from the managed embedder.

The server uses mean pooling including prompt tokens, full 768-dimensional output, and L2 normalization. Queries use `task: search result | query: `; documents use `title: none | text: `. The backend tokenizes with the GGUF tokenizer, treats special-token spellings in document text as ordinary text, truncates at 8192 tokens, and sends token arrays without re-tokenization. Embedding cache keys include the role and the complete vector-space fingerprint.

One server slot, flash attention, and fixed context/logical/physical batch limits of 8192 bound concurrent attention allocation while allowing a complete bidirectional sequence in one physical batch. `CEPHALON_EMBEDDER_BATCH_SIZE` defaults to 16 texts per backend request; the single server slot processes sequences in order without padded batches. `CEPHALON_EMBEDDER_GPU_LAYERS` defaults to 99 (all text layers); `CEPHALON_EMBEDDER_DEVICE` defaults to automatic device selection. For this workstation, an explicit GPU selection is:

```powershell
$env:CEPHALON_EMBEDDER_DEVICE="Vulkan0"
```

For CPU inference, set `CEPHALON_EMBEDDER_GPU_LAYERS=0` and `CEPHALON_EMBEDDER_DEVICE=none`. Q8_0 is fixed; there is no embedding dtype or embedding Python setting.

Jina v3.5 uses the [official Q8_0 GGUF artifacts](https://huggingface.co/jinaai/jina-reranker-v3.5-GGUF) at commit `884f7c67aa3ac24edb89064da8c7bfd03f4a90f5`. Install all three files: `jina-reranker-v3.5-Q8_0.gguf`, `projector.safetensors`, and `tokenizer.json`. The explicit installer verifies their pinned SHA-256 hashes and installs the worker. The worker needs only `numpy==2.5.1` and `tokenizers==0.22.2`; the GGUF trunk runs in Vulkan, while the small BF16 projector weights are promoted exactly to FP32 for NumPy scoring. No Transformers, PyTorch or ROCm installation is needed.

```powershell
py -3.14 scripts\setup_retrieval_models.py jina `
  --model-dir "$env:USERPROFILE\cephalon-data\models\jina-reranker-v3.5-gguf-q8_0"
py -3.14 scripts\setup_retrieval_models.py jina-worker `
  --runtime-dir "$env:USERPROFILE\cephalon-data\runtimes\jina35-python"
py -3.14 scripts\setup_retrieval_models.py jina-llama `
  --source-dir "C:\AI\bench\cephalon-jina35-pr26286\llama.cpp" `
  --runtime-dir "$env:USERPROFILE\cephalon-data\runtimes\jina35-pr26286-491219a2"
```

The Jina runtime is built from [llama.cpp PR #26286](https://github.com/ggml-org/llama.cpp/pull/26286), pinned to its current head `491219a2bfa0a5a51acfc6a233bea911515b2134`. Build prerequisites are Git, CMake, a C++ compiler, the Vulkan SDK, and `uv` for the isolated worker environment. The installer fetches the exact commit into a dedicated checkout and rejects a changed checkout or a different revision. It builds only `llama-embedding` and records hashes of the helper and its libraries. The backend checks the manifest, hashes, and reported source commit before startup. The application does not fetch a moving PR branch. PR #26286 is still open; retain this pin until a separately validated replacement is available.

Jina uses the Qwen3 model's causal attention, pooling `none`, and `--output-token-ids` to obtain only the document and query marker states, matching the published `rerank.py` inference path. It scores the full candidate set using the official dual-marker prompt and weighted block-query fusion. Inputs are bounded to 125 documents per block, 8191 tokens per document, 1984 query tokens, and a default 32768-token context including prompt markup. Control markers in source text are removed. The worker returns raw cosine scores in [-1, 1], which remain separate from the fused retrieval score. Its runtime identity includes the model manifest, helper checksum, exact PR head, device, GPU layers, context limit, and package versions.

Default paths match the commands above. Overrides are `CEPHALON_RERANKER_PYTHON_BIN`, `CEPHALON_RERANKER_LLAMA_EMBEDDING_BIN`, `CEPHALON_RERANKER_DEVICE` (default `Vulkan0`), `CEPHALON_RERANKER_GPU_LAYERS` (99), and `CEPHALON_RERANKER_MAX_CONTEXT_TOKENS` (32768, accepted range 16384–131072). The helper loads the model for each bounded block and releases its VRAM after the block finishes. Rerank caching includes the model revision, precision, PR head, worker checksum, runtime settings, exact query, and candidate text.

The backend checks the local GGUF checksum and manifest before starting the embedding server, then checks `/health` and `/props` for its model path, precision, alias, modalities, slot count, and context. Both retrieval runtimes refuse occupied ports and stop only their owned processes; stopping Jina also stops an active helper child. Logs are `data/logs/embeddinggemma2-q8-embedder.log` and `data/logs/jina35-q8-reranker.log`.

## Reindex and degraded behavior

After setup, restart Cephalon and run **Reindex all documents**. The new LanceDB table is `vectors_embeddinggemma2_text_270m_q8_0_768`; earlier BF16 and other model vectors remain in separate tables and are never mixed into it. The vector-space fingerprint records Q8_0, the GGUF hash/revision, runtime build, prompts, pooling, and normalization. Migration 025 marks incompatible documents stale and removes incompatible conversation retrieval memories while preserving saved messages and conversations. Reindex preserves the old document on a failed replacement. Check `/reindex/progress`, `/observability/index-health`, and `/models/status` before treating the index as current.

Switching only the reranker does not require reindexing an existing EmbeddingGemma 2 Q8 index. If Jina is unavailable, dense plus FTS5 retrieval continues with explicit degraded status and no reranker score. If EmbeddingGemma 2 is unavailable, document retrieval is disabled. The user-operated chat server remains independent. Q8 performance and retrieval quality have not been benchmarked.

## Feature controls

All controls are booleans, default `true`, have no unit, and accept only
`true/false` in API settings or `1/0` in environment defaults.

| `RagSettings` field | Environment default | Effect when false | Reindex |
|---|---|---|---|
| `hierarchical_context` | `CEPHALON_HIERARCHICAL_CONTEXT` | Exact selected children only | No |
| `layout_evidence` | `CEPHALON_LAYOUT_EVIDENCE` | No structural graph expansion | No |
| `evidence_ledger` | `CEPHALON_EVIDENCE_LEDGER` | Empty disabled ledger; gap round cannot start | No |
| `coverage_selection` | `CEPHALON_COVERAGE_SELECTION` | Pre-A6 selection and compression | No |
| `gap_retrieval` | `CEPHALON_GAP_RETRIEVAL` | Thorough stays single retrieval pass | No |
| `verified_answer_repair` | `CEPHALON_VERIFIED_ANSWER_REPAIR` | Thorough always uses its legacy repair completion | No |
| n/a | `CEPHALON_TYPED_TABLES` | Typed table rows are not written or routed; table text remains retrievable | Yes when re-enabled |
| n/a | `CEPHALON_TABLE_EXECUTION` | No typed planning, deterministic execution, or named-document unit scan; hybrid text retrieval remains | No |

A2 parent-summary v2 is the only Stack A change that requires reindexing. Its
rollback requires checking out the earlier ingestion code and reindexing again.
Migration 016 indexes may be dropped for rollback with a performance cost;
migrations 015/017 only extend evaluation/trace JSON storage.

Migration 018 adds the typed table schema without rewriting documents at
startup. Explicitly reindex the library to populate it. Disabling
`CEPHALON_TYPED_TABLES` is the runtime rollback and leaves dense/FTS table text
available; re-enable and reindex to refresh structured rows.

Migration 019 adds only `retrieval_queries.table_execution_json`; it does not
rewrite the index. `CEPHALON_TABLE_EXECUTION=0` is the B2 rollback and requires
no reindex. The trace records the validated plan, bounds, fallback reason,
candidate sources, execution latency, and completion-call count.

B3 adds no migration, flag, or index format. Its optional cell-citation fields
travel inside the existing source, ledger, trace, message-source, and
answer-citation JSON contracts. Rolling back B3 is therefore a code checkout;
old and new conversations remain readable and no reindex is required.

## Shutdown verification

Benchmark-owned processes must be stopped after a run. Verify both process and
port state rather than assuming a close request succeeded:

```powershell
Get-NetTCPConnection -State Listen -ErrorAction SilentlyContinue |
  Where-Object LocalPort -in 8080,8090,8767 |
  Select-Object LocalAddress,LocalPort,OwningProcess
```

Do not stop the user's normal backend on port 8765 when cleaning an isolated
benchmark. Private artifacts and logs remain under
`C:\tmp\cephalon-private-rag`; the repository should return clean.

## Troubleshooting

- `active_context_tokens=None`: the request's validated `context_tokens` is the
  authoritative fallback; do not cast the nullable server report directly.
- Embedding startup failure: verify the Q8_0 manifest, pinned llama.cpp build,
  Vulkan driver, loopback port, and embedding log. For Jina, check its isolated
  Python executable, pinned packages, PR #26286 helper manifest, and worker log.
- CP1252 output errors: set `PYTHONIOENCODING=utf-8` before the private runner.
- Stale A2 index: use explicit reindex; do not silently mix summary versions.
- Gap timeout/error: the trace records the stop reason and the initial evidence
  remains valid.
