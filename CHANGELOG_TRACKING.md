# Cephalon Changelog And Tracking

This file tracks development work that should remain with the repo. It is separate from generated release reports and external metrics.

## Current Branch: Retrieval Runtime Hardening

### Completed

- Replaced LanceDB/Tantivy hybrid dependence with explicit SQLite FTS5 lexical search plus LanceDB dense retrieval and RRF fusion.
- Kept SQLite as metadata source of truth and LanceDB as dense-vector storage only.
- Added strict Jina ONNX metadata validation and tokenizer regex fixes.
- Added safe unknown-extension text import with binary guards.
- Added explicit GGUF model loading endpoint and UI Load button.
- Changed query behavior so `/query` requires the selected model to already be loaded.
- Removed automatic query prompt storage from dense retrieval to avoid self-reinforcing prompt echoes.
- Excluded core-memory rows from document dense search.
- Added bounded reranker scoring and retrieval evidence priors so exact lexical matches survive poorly calibrated logits.
- Added deterministic numeric scan for traffic maximum questions.
- Updated startup UI, model picker state, and dense dark UI details.
- Made Tauri backend host/port environment-driven and added external-backend skip mode for future offload work.
- Promoted README, AI system awareness, and architecture deep dive to first-class repo docs.

### Verification Commands Used

- `python -m py_compile` over backend entrypoints and touched backend scripts.
- `pytest -q --basetemp .pytest-tmp-full`
- `scripts\validate_onnx_models.py`
- `npx.cmd tsc --noEmit`
- `cargo check`
- Non-mutating live retrieval traces against the current local index for stress supplement and traffic maximum queries.

### Known Limits

- Full `npm run build` and standalone Tauri smoke may require execution outside the current sandbox when Vite/esbuild cannot read its config path.
- Remote/off-machine backend mode is intentionally a future compatibility seam; it is documented but not release-tested.
- Jina reranker ONNX logits are not treated as absolute probabilities. Runtime combines bounded reranker output with retrieval evidence.

### Next Tracking Slice

- Add frontend unit coverage for explicit model loading and disabled chat state.
- Add a packaged-app smoke script that loads a small GGUF, ingests fixtures, queries, exports metrics, and exits.
- Add UI for API base override if remote/offloaded backend becomes a real release target.
- Add model unload/reload action if repeated model switching becomes common.
