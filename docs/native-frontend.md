# Native frontend

Cephalon's desktop frontend is a native Rust application built with GPUI Kit
`=0.6.6`. Kit supplies the GPUI runtime, platform application, assets, and
component layer through one direct dependency. The lockfile resolves Kit's
GPUI family to `gpui-pre` 0.3.6. The previous GPUI-CE pin was
`c39bf5abfa81e3851367be0830c6caf7360f6af3`.

Kit owns editable inputs and text areas, buttons, switches, forms, dialogs,
notifications, and their native interaction behavior. Cephalon still owns its
research and RAG surfaces: the workspace layout, document/library/history
presentation, citation-aware chat Markdown, source and support panels, and
the Graphite palette. The palette is applied to Kit's theme at startup and
when the user changes the theme.

Each field's `InputState` or `TextareaState` is the editable text source of
truth. `InputEntities` stores those Kit entities; `NativeApp` reads values
when submitting or saving. Only live library search subscribes to input
changes to redraw results. The composer is a Kit auto-growing text area.
Ordinary form fields use Kit inputs, and numeric settings use `NumberInput`.

Cephalon binds Tab and Shift-Tab to form focus traversal because Kit 0.6.6
still binds Tab to editor indentation. Traversal respects Kit's active dialog
focus trap. Ctrl/Cmd+Enter submits from an input. Kit's Cut action leaves a
field unchanged when nothing is selected, so the old Cephalon Cut override is
gone. Input value updates are deferred to a window update because Kit's
`set_value` requires both the window and app contexts.

Confirmation prompts use Kit alert dialogs, and transient messages use Kit
notifications. The old `TextInput` wrapper, input text mirrors and sync
subscriptions, notice timer, and custom confirmation state were removed.
Cephalon retains only application-specific callbacks for actions such as
deleting a conversation or document.

The frontend talks to the existing local Python service over typed HTTP and
SSE. Retrieval, reranking, ingestion, model behavior, and backend process
ownership remain in Python. The query stream uses a bounded channel and
combines nearby token events before redrawing chat. Source and support panels
resolve the selected answer by message ID. Stop also sends a cancellation
request to the Python service.

Other direct dependencies serve distinct needs: `image` decodes the window
icon; `async-channel` and `smol` carry backend events; `reqwest` and Serde
handle HTTP/SSE; `pulldown-cmark` renders citation-aware answers; `libc`
manages the Unix backend process group; and `winresource` embeds the Windows
icon. Kit's default component and asset features are used. Optional
profiler, inspector, code editor, and language parser features are not
enabled. Linux retains Kit's native X11 and Wayland platform support.

Native development and CI use Rust 1.95 or newer; Rust 1.98.1 was used for
this migration. The workspace remains on Rust 2021 and resolver 2 to keep
the framework migration focused. For local development:

```powershell
cargo run
```

The managed Python backend is launched or connected by `BackendService`
according to the normal development, packaged, and external-backend modes.
`assets/cephalon.png` is passed to the native window as its icon, while
`assets/cephalon.ico` is embedded as the Windows executable resource. The
portable Linux package keeps the SVG beside its relocatable `.desktop` entry.
