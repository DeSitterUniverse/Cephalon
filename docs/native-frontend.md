# Native frontend

Cephalon's desktop frontend is a native Rust application. It uses the pinned
GPUI Community Edition (GPUI-CE) runtime, with GPUI-CE's maintained
`gpui_elements` editable-text primitives for native inputs and text areas.
The frontend talks to the existing local Python service over typed HTTP and
SSE; retrieval, reranking, ingestion, model behavior, and backend process
ownership remain in Python.

The current GPUI-CE source revision is pinned in `native/Cargo.toml` and the
workspace `Cargo.lock`:

```text
c738623ffbcec2aeddc44a645cc6b74646d5cf97
```

`gpui-component` was evaluated separately. Its current upstream manifest still
targets Zed's GPUI repository, so it is not a direct dependency until its
GPUI-CE patch path is validated against the same pinned revision. This avoids
silently bringing a second GPUI implementation into Cephalon. The current
native input, scrolling, and diagnostic UI therefore use GPUI-CE primitives
and Cephalon-specific components.

For local development:

```powershell
cargo run
```

The managed Python backend remains the source of truth for application data
and is launched or connected by `BackendService` according to the normal
development, packaged, and external-backend modes.

The desktop identity is kept with the native shell: `assets/cephalon.png` is
passed to GPUI for the X11 window icon, while `assets/cephalon.ico` is embedded
as the Windows executable resource so the title bar and taskbar use the same
Cephalon mark. The portable Linux package keeps the SVG beside its relocatable
`.desktop` entry.
