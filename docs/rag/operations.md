# RAG operations

Cephalon uses external llama.cpp chat and embedding servers. The Jina v3.5
reranker is a managed Vulkan GGUF worker launched by the backend from the pinned
feature-compatible llama.cpp build. Retrieval tests do not load Ling; complete
answer tests require the chat server.

## Fixed local services

The scientific benchmark uses:

| Service | Port | Required role |
|---|---:|---|
| Ling 3.0 tiny Q6_K llama.cpp | 8080 | Draft, semantic audit, repair, final generation |
| Jina v5 Nano embedder | 8090 | Query and document embeddings |
| Isolated Cephalon backend | 8767 | Public ingestion/query/evaluation API |

Use `py -3.14` and set the isolated backend's data directory explicitly. The
Ling chat runtime uses llama.cpp PR 26608 at commit
`d8d862521e9ad842f2b47f3b392b039317782aa0`; this is temporary until its
BailingMoE3 support is merged. The Q6_K artifact is stored at
`C:\Users\Fluttershy\cephalon-data\models\ling-3.0-tiny-q6_k`.

The default reranker path expects:

```text
C:\tmp\llama.cpp-jina-v35\build\bin\Release\llama-embedding.exe
llama.cpp commit 80c940e5a80555167c4ec37652deca6528810f91
```

Do not add `--attention non-causal`; that changes Jina v3.5 ranking quality.
Keep the Q8_0 GGUF, projector, tokenizer, model revision, and hashes pinned by
`config.py`.

## Isolated benchmark startup

The following PowerShell example matches the documented ports and keeps logs
with the private benchmark artifacts. Adjust paths only when recording the
change in the paired benchmark report:

```powershell
$repo = 'C:\Projects\Active\Cephalon'
$privateLogs = 'C:\tmp\cephalon-private-rag\a8-live-logs'
$llamaServer = 'C:\tmp\llama.cpp-ling-pr26608\build-vs18\bin\Release\llama-server.exe'
$python314 = 'C:\Users\Fluttershy\AppData\Local\Python\pythoncore-3.14-64\python.exe'

Start-Process -FilePath $llamaServer -WindowStyle Hidden -ArgumentList @(
  '-m', 'C:\Users\Fluttershy\cephalon-data\models\ling-3.0-tiny-q6_k\Ling-3.0-tiny-Q6_K.gguf',
  '--device', 'Vulkan0', '--gpu-layers', '999', '--ctx-size', '8192',
  '--parallel', '1', '--split-mode', 'none', '--no-flash-attn',
  '--seed', '20260812', '--temp', '1.0', '--top-k', '20', '--top-p', '0.95',
  '--min-p', '0.05', '--reasoning-format', 'deepseek', '--reasoning-budget', '2048',
  '--host', '127.0.0.1', '--port', '8080', '--no-webui'
) -RedirectStandardOutput "$privateLogs\chat.stdout.log" `
  -RedirectStandardError "$privateLogs\chat.stderr.log"

Start-Process -FilePath $llamaServer -WindowStyle Hidden -ArgumentList @(
  '-m', 'C:\Users\Fluttershy\cephalon-data\models\jina-v5-nano-retrieval-q8_0\v5-nano-retrieval-Q8_0.gguf',
  '--embedding', '--pooling', 'last', '--embd-normalize', '2',
  '--device', 'Vulkan0', '--gpu-layers', '999', '--batch-size', '4096',
  '--ubatch-size', '4096', '--host', '127.0.0.1', '--port', '8090', '--no-webui'
) -RedirectStandardOutput "$privateLogs\embed.stdout.log" `
  -RedirectStandardError "$privateLogs\embed.stderr.log"

$env:PYTHONPATH = "$repo\python"
$env:CEPHALON_DATA_DIR = 'C:\tmp\cephalon-private-rag\a2-live-data'
$env:CEPHALON_MODEL_DIR = 'C:\Users\Fluttershy\cephalon-data\models'
$env:CEPHALON_METRICS_DIR = 'C:\tmp\cephalon-private-rag\a8-metrics'
$env:CEPHALON_HOST = '127.0.0.1'
$env:CEPHALON_PORT = '8767'
$env:CEPHALON_LLAMA_SERVER_URL = 'http://127.0.0.1:8080'
$env:CEPHALON_EMBEDDER_SERVER_URL = 'http://127.0.0.1:8090'
$env:CEPHALON_RERANKER_LLAMA_EMBEDDING_BIN = `
  'C:\tmp\llama.cpp-jina-v35\build\bin\Release\llama-embedding.exe'
Start-Process -FilePath $python314 -ArgumentList 'python\main.py' `
  -WorkingDirectory $repo -WindowStyle Hidden `
  -RedirectStandardOutput "$privateLogs\backend.stdout.log" `
  -RedirectStandardError "$privateLogs\backend.stderr.log"
```

Wait for `/health` to report `engines_ready=true`, `selected_backend=gguf_vulkan`,
and reranker device `Vulkan0` before timing any request. A CPU fallback is a
valid availability path but is not comparable to the fixed Vulkan benchmark.

The generation boundary is also bounded by the configured context window:
history, retrieved evidence, and repair directives are reduced before a chat
request is sent. The repair directive contains only short claim/status/source
metadata; full deterministic evidence remains in trace metadata. This avoids
over-context HTTP 400 responses from an external server with a smaller loaded
window. The semantic audit requests a JSON schema, while deterministic final
verification remains the safety fallback.

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
- Reranker startup failure: verify the pinned binary revision, GGUF/projector/
  tokenizer hashes, Vulkan device, and backend log.
- CP1252 output errors: set `PYTHONIOENCODING=utf-8` before the private runner.
- Stale A2 index: use explicit reindex; do not silently mix summary versions.
- Gap timeout/error: the trace records the stop reason and the initial evidence
  remains valid.
