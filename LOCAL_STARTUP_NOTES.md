# Local startup notes

Cephalon uses three independent local runtimes:

| Runtime | Default port | Owner |
| --- | --- | --- |
| Chat generation llama.cpp server | 8080 | You |
| Jina Nano embeddings llama.cpp server | 8090 | Cephalon or you |
| Cephalon backend | 8765 | Cephalon |

The chat and embedding servers must never be the same process.

## Installed desktop release

Launch Cephalon from its shortcut. Its backend sidecar starts automatically at `127.0.0.1:8765`.

Start the chat server separately. This Vulkan example is appropriate for a llama.cpp build that lists `Vulkan0`:

```powershell
& "C:\AI\llama.cpp\build\bin\Release\llama-server.exe" `
  -m "C:\AI\models\your-chat-model.gguf" `
  --device Vulkan0 --gpu-layers 999 `
  --ctx-size 8192 --host 127.0.0.1 --port 8080 --no-webui
```

Download the fixed Jina Nano Q8_0 embedder and Jina Reranker v3.5 BF16 GGUF
from **Settings**, then restart Cephalon. It starts the dedicated embedding
server on port 8090 with `--device Vulkan0 --gpu-layers 999`, and starts the
isolated reranker worker when its verified helper and model files are present.

The reranker needs Jina's selected-token and irregular-Qwen3-attention
llama.cpp changes. Build the pinned helper and set its path before launch:

```powershell
git clone https://github.com/ggml-org/llama.cpp.git C:\AI\llama.cpp-jina-reranker
git -C C:\AI\llama.cpp-jina-reranker fetch origin pull/26286/head
git -C C:\AI\llama.cpp-jina-reranker checkout --detach 80c940e5a80555167c4ec37652deca6528810f91
cmake -S C:\AI\llama.cpp-jina-reranker -B C:\AI\llama.cpp-jina-reranker\build -DGGML_VULKAN=ON -DLLAMA_CURL=OFF
cmake --build C:\AI\llama.cpp-jina-reranker\build --config Release --target llama-embedding -j 4
$env:CEPHALON_RERANKER_LLAMA_EMBEDDING_BIN="C:\AI\llama.cpp-jina-reranker\build\bin\Release\llama-embedding.exe"
```

The backend intentionally has no CPU reranker fallback. If the pinned helper
cannot be verified, startup reports a retrieval warning instead of silently
changing ranking semantics. Switching the Vulkan helper does not require
reindexing.

## GPU Nano embedding server (optional external ownership)

The managed embedding server uses Vulkan GPU offload automatically. To operate it yourself, run this before launching Cephalon; it will reuse the healthy endpoint:

```powershell
& "C:\AI\llama.cpp\build\bin\Release\llama-server.exe" `
  -m "$env:USERPROFILE\cephalon-data\models\jina-v5-nano-retrieval-q8_0\v5-nano-retrieval-Q8_0.gguf" `
  --embedding --pooling last --embd-normalize 2 `
  --device Vulkan0 --gpu-layers 999 `
  --batch-size 4096 --ubatch-size 4096 `
  --host 127.0.0.1 --port 8090 --no-webui
```

The Nano endpoint must return normalized 768-dimensional vectors. Confirm `http://127.0.0.1:8090/health` before starting the app.

## Native GPUI development

```powershell
cargo run
```

The GPUI shell starts the Python backend automatically in development mode.
Use `CEPHALON_EXTERNAL_BACKEND=1` when the API is already running elsewhere.

## Health and shutdown

Use Settings or these endpoints to inspect runtime health:

```text
GET /models/status
GET /runtime/embedder/status
GET /runtime/reranker/status
GET /reindex/progress
```

After testing, stop only the chat and embedding servers you started. Verify their listeners are gone before considering the test complete.
