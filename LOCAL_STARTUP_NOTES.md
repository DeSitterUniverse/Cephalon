# Local startup notes

## Installed MSI release

Launch Cephalon from its installed shortcut. The packaged app starts its bundled backend sidecar automatically at `127.0.0.1:8765`; no local Python command is needed.

Start llama.cpp separately, then connect from Cephalon Settings:

```powershell
& "C:\path\to\llama-server.exe" `
  -m "C:\AI\models\your-model.gguf" `
  --host 127.0.0.1 --port 8080 -c 8192 -ngl 99 `
  --reasoning-budget 2048
```

Set the server URL to `http://127.0.0.1:8080` in Settings if needed, then press **Connect**.

## Browser development

Start both the Cephalon backend and Vite frontend in one terminal:

```powershell
npm.cmd run dev:full
```

Open `http://127.0.0.1:1420`.

Alternatively, run them separately:

```powershell
py -3.14 python\main.py
```

In a second terminal:

```powershell
npm.cmd run dev
```

## Tauri desktop development

```powershell
npm.cmd run tauri dev
```

This starts the source backend automatically with Python 3.14 and opens the desktop app. The llama.cpp server remains separate in every development mode.

## Common ports

- Cephalon backend: `8765`
- Vite frontend: `1420`
- llama.cpp server: `8080`

If a Cephalon launch reports that port `8765` is in use, stop the old backend process before launching again.
