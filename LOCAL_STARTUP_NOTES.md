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

Download the fixed Jina Nano Q8_0 embedder and Jina Reranker v3.5 from **Settings**, then restart Cephalon. It starts the dedicated embedding server on port 8090 with `--device Vulkan0 --gpu-layers 999`, and starts the isolated reranker worker when their files are installed.

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

## Browser development

```powershell
npm.cmd run dev:full
```

Open `http://127.0.0.1:1420`.

Or run the components separately:

```powershell
py -3.14 python\main.py
```

```powershell
npm.cmd run dev
```

## Tauri development

```powershell
npm.cmd run tauri dev
```

## Health and shutdown

Use Settings or these endpoints to inspect runtime health:

```text
GET /models/status
GET /runtime/embedder/status
GET /runtime/reranker/status
GET /reindex/progress
```

After testing, stop only the chat and embedding servers you started. Verify their listeners are gone before considering the test complete.
