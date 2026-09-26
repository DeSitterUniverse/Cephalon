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
c39bf5abfa81e3851367be0830c6caf7360f6af3
```

`gpui` provides the UI runtime, `gpui_platform::application()` selects the
native Windows/Linux platform, and `gpui_elements` provides the maintained
editable-text state and elements. All three direct dependencies use the same
exact commit. The GPUI-CE renderer uses WGPU/WGSL internally; Cephalon does
not enable its optional custom GPU, profiler, capture, or embedded-asset APIs.
The platform's defaults retain Wayland and X11 on Linux. Its Windows manifest
feature remains enabled by an upstream platform dependency, alongside
Cephalon's own executable icon resource.

Editable input text lives in `EditableTextState`. Cephalon's `TextInput` keeps
only form styling and form-specific submit, Tab traversal, and selection-only
Cut behavior. The app reads text directly from each editor when submitting or
saving. Each wrapper observes its supplied editor state to redraw itself;
only live library search also notifies the app when its text changes.

Other direct dependencies serve distinct needs: `image` decodes the window
icon, `async-channel` and `smol` carry backend events, `reqwest` and Serde
handle the typed HTTP/SSE API, `pulldown-cmark` renders answers, `libc` manages
the Unix backend process group, and `winresource` embeds the Windows icon.

Native development and CI require Rust 1.95 or newer. The existing Rust 2021
edition and workspace resolver 2 remain in place; changing them would add
unrelated API and process-manager Clippy churn to this migration.

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
