import json
import os
import shutil
import time
from pathlib import Path

from ..config import EMBEDDING_DIMENSION, EMBEDDING_MODEL_ID, RERANKER_MODEL_ID, Settings


KINDS = {"embedder", "reranker"}


def target_dir(settings: Settings, kind: str) -> Path:
    _validate_kind(kind)
    return Path(settings.model_dir) / kind


def status(settings: Settings) -> dict:
    return {
        "model_dir": settings.model_dir,
        "download_sources": {
            "embedder": {
                "repo_id": settings.embedder_onnx_repo,
                "subfolder": settings.embedder_onnx_subfolder,
            },
            "reranker": {
                "repo_id": settings.reranker_onnx_repo,
                "subfolder": settings.reranker_onnx_subfolder,
            },
        },
        "embedder": inspect_model_dir(target_dir(settings, "embedder"), "embedder"),
        "reranker": inspect_model_dir(target_dir(settings, "reranker"), "reranker"),
    }


def inspect_model_dir(model_dir: Path, kind: str) -> dict:
    _validate_kind(kind)
    model_file = _find_model_file(model_dir)
    missing = []
    if not model_dir.exists():
        missing.append("folder")
    if not model_file:
        missing.append("model.onnx")
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        if not (model_dir / filename).exists():
            missing.append(filename)
    meta = _read_meta(model_dir)
    meta_error = _validate_meta(kind, meta)
    ok = not missing and meta_error is None
    return {
        "kind": kind,
        "path": str(model_dir),
        "display_path": _display_path(model_dir),
        "ok": ok,
        "missing": missing,
        "meta_error": meta_error,
        "model_file": str(model_file) if model_file else None,
        "model_id": meta.get("model_id"),
        "validated": meta.get("validated") is True,
        "dimension": meta.get("dimension"),
        "score_mode": meta.get("score_mode"),
    }


def runtime_status(app_state) -> dict:
    payload = status(app_state.settings)
    payload["engines_ready"] = getattr(app_state, "startup_error", None) is None
    payload["startup_error"] = getattr(app_state, "startup_error", None)
    payload["embedder"]["runtime_loaded"] = getattr(app_state, "embedder", None) is not None
    payload["reranker"]["runtime_loaded"] = getattr(app_state, "reranker", None) is not None
    payload["embedder"]["active_model_id"] = getattr(app_state, "embedding_model_id", None)
    payload["reranker"]["active_model_id"] = getattr(app_state, "reranker_model_id", None)
    payload["reranker"]["score_mode"] = getattr(app_state, "reranker_score_mode", payload["reranker"].get("score_mode"))
    return payload


def install_local(settings: Settings, kind: str, source_path: str) -> dict:
    _validate_kind(kind)
    source = Path(source_path).expanduser().resolve()
    if not source.exists() or not source.is_dir():
        raise ValueError(f"{kind} folder does not exist: {source}")
    prepared = _prepare_source_folder(source, kind)
    return _replace_target(settings, kind, prepared)


def download(settings: Settings, kind: str, repo_id: str | None = None, subfolder: str | None = None) -> dict:
    _validate_kind(kind)
    try:
        from huggingface_hub import snapshot_download
    except Exception as exc:
        raise RuntimeError("huggingface_hub is required for first-launch model downloads. Install requirements.txt and try again.") from exc

    resolved_repo = repo_id or (settings.embedder_onnx_repo if kind == "embedder" else settings.reranker_onnx_repo)
    resolved_subfolder = subfolder if subfolder is not None else (
        settings.embedder_onnx_subfolder if kind == "embedder" else settings.reranker_onnx_subfolder
    )
    if not resolved_repo:
        raise ValueError(f"No Hugging Face ONNX repo is configured for {kind}.")

    cache_root = Path(settings.data_dir) / "downloads" / "onnx"
    cache_root.mkdir(parents=True, exist_ok=True)
    local_dir = cache_root / f"{kind}-{int(time.time())}"
    snapshot_download(
        repo_id=resolved_repo,
        local_dir=str(local_dir),
        allow_patterns=["*.onnx", "*.onnx.data", "*.onnx_data", "*.json", "*.txt", "*.model", "*.jinja", "*.vocab", "vocab.*", "merges.txt", "onnx/*"],
    )
    source = local_dir / resolved_subfolder if resolved_subfolder else local_dir
    prepared = _prepare_source_folder(source, kind, repo_id=resolved_repo)
    result = _replace_target(settings, kind, prepared)
    result["source_repo"] = resolved_repo
    result["source_subfolder"] = resolved_subfolder
    return result


def install_all_from_download(settings: Settings) -> dict:
    return {
        "embedder": download(settings, "embedder"),
        "reranker": download(settings, "reranker"),
    }


def _prepare_source_folder(source: Path, kind: str, repo_id: str | None = None) -> Path:
    model_file = _find_model_file(source)
    if not model_file:
        raise ValueError(f"{kind} folder must contain an ONNX model file.")
    for filename in ("tokenizer.json", "tokenizer_config.json"):
        if not (source / filename).exists():
            raise ValueError(f"{kind} folder is missing {filename}.")

    meta = _read_meta(source)
    if not meta:
        meta = _default_meta(kind, repo_id)
    meta_error = _validate_meta(kind, meta)
    if meta_error:
        raise ValueError(meta_error)

    if model_file.name != "model.onnx":
        shutil.copy2(model_file, source / "model.onnx")
    if model_file.parent != source:
        for sibling in model_file.parent.glob(f"{model_file.name}*"):
            if sibling.is_file() and sibling.name != model_file.name:
                shutil.copy2(sibling, source / sibling.name)
        for sibling in model_file.parent.glob("model.onnx*"):
            if sibling.is_file() and sibling.name != "model.onnx":
                shutil.copy2(sibling, source / sibling.name)
    _write_meta(source, meta)
    return source


def _replace_target(settings: Settings, kind: str, source: Path) -> dict:
    destination = target_dir(settings, kind)
    destination.parent.mkdir(parents=True, exist_ok=True)
    installing = destination.parent / f"{kind}.installing"
    backup = destination.parent / f"{kind}.backup-{int(time.time())}"
    if installing.exists():
        shutil.rmtree(installing)
    shutil.copytree(source, installing)
    if destination.exists():
        shutil.move(str(destination), str(backup))
    shutil.move(str(installing), str(destination))
    status_payload = inspect_model_dir(destination, kind)
    status_payload["restart_required"] = True
    status_payload["backup_path"] = str(backup) if backup.exists() else None
    return status_payload


def _find_model_file(model_dir: Path) -> Path | None:
    if not model_dir.exists():
        return None
    preferred = model_dir / "model.onnx"
    if preferred.exists():
        return preferred
    candidates = sorted(model_dir.glob("*.onnx")) + sorted((model_dir / "onnx").glob("*.onnx")) if (model_dir / "onnx").exists() else sorted(model_dir.glob("*.onnx"))
    return candidates[0] if candidates else None


def _read_meta(model_dir: Path) -> dict:
    for filename in ("onnx_profile.json", "cephalon_onnx_meta.json"):
        meta_file = model_dir / filename
        if not meta_file.exists():
            continue
        try:
            data = json.loads(meta_file.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except Exception:
            return {}
    return {}


def _write_meta(model_dir: Path, meta: dict) -> None:
    payload = json.dumps(meta, indent=2)
    (model_dir / "onnx_profile.json").write_text(payload, encoding="utf-8")


def _default_meta(kind: str, repo_id: str | None = None) -> dict:
    if kind == "embedder":
        return {
            "model_id": EMBEDDING_MODEL_ID,
            "kind": "embedder",
            "pooling": "last_token",
            "normalized": True,
            "dimension": EMBEDDING_DIMENSION,
            "fixed_sequence_length": 512,
            "validated": True,
            "validation_key": "downloaded_onnx_artifact",
            "source_repo": repo_id,
        }
    return {
        "model_id": RERANKER_MODEL_ID,
        "kind": "reranker",
        "validated": True,
        "validation_key": "downloaded_onnx_artifact",
        "score_mode": "logit_margin_0_minus_1",
        "source_repo": repo_id,
    }


def _validate_meta(kind: str, meta: dict) -> str | None:
    if kind == "embedder":
        if meta.get("model_id") != EMBEDDING_MODEL_ID:
            return f"Embedder model mismatch: expected {EMBEDDING_MODEL_ID}, got {meta.get('model_id') or 'unknown'}."
        if int(meta.get("dimension") or 0) != EMBEDDING_DIMENSION:
            return f"Embedder dimension mismatch: expected {EMBEDDING_DIMENSION}, got {meta.get('dimension') or 'unknown'}."
        if meta.get("validated") is not True:
            return "Jina embedder ONNX export exists but has not passed validation."
        return None
    if meta.get("model_id") != RERANKER_MODEL_ID:
        return f"Reranker model mismatch: expected {RERANKER_MODEL_ID}, got {meta.get('model_id') or 'unknown'}."
    if meta.get("validated") is not True:
        return "Jina reranker ONNX export exists but has not passed validation."
    if not meta.get("score_mode"):
        return "Jina reranker validation metadata is missing score_mode."
    return None


def _validate_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"kind must be one of: {', '.join(sorted(KINDS))}")


def _display_path(path: Path) -> str:
    try:
        resolved = path.expanduser().resolve()
        home = Path.home().resolve()
        relative = resolved.relative_to(home)
        return "~/" + str(relative).replace("\\", "/")
    except Exception:
        return str(path)
