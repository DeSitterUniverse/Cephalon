"""Fixed local retrieval runtime: Jina Nano through llama.cpp and Jina v3.5.

The old ONNX implementation remains in ``onnx_setup.py`` as migration
reference only.  It must not be selected at runtime: this module owns the
single supported 768-dimensional retrieval profile.
"""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from urllib.error import URLError
from urllib.request import Request, urlopen

from huggingface_hub import HfApi, snapshot_download

from ..config import (
    EMBEDDER_GGUF_FILE,
    EMBEDDER_GGUF_REPO,
    EMBEDDER_GGUF_SHA256,
    EMBEDDING_DIMENSION,
    EMBEDDING_MODEL_ID,
    RERANKER_MODEL_ID,
    RERANKER_REPO,
)

MANIFEST_FILENAME = ".cephalon-hf-manifest.json"


def _now() -> int:
    return int(time.time())


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _git_blob_sha1(path: Path) -> str:
    size = path.stat().st_size
    digest = hashlib.sha1(f"blob {size}\0".encode("utf-8"))
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _official_manifest(repo_id: str, revision: str | None = None) -> dict:
    """Return checksum metadata from the immutable Hub commit we download."""
    info = HfApi().model_info(repo_id, revision=revision, files_metadata=True)
    files: dict[str, dict[str, str | None]] = {}
    for sibling in info.siblings:
        lfs = getattr(sibling, "lfs", None)
        lfs_sha256 = lfs.get("sha256") if isinstance(lfs, dict) else getattr(lfs, "sha256", None)
        files[sibling.rfilename] = {
            # An LFS blob ID identifies the small pointer file in Git, not the
            # checked-out weight bytes.  Verify those bytes only against the
            # Hub LFS SHA-256; ordinary Git files use their blob SHA-1.
            "git_blob_sha1": None if lfs_sha256 else getattr(sibling, "blob_id", None),
            "sha256": lfs_sha256,
        }
    return {"repo_id": repo_id, "revision": info.sha, "files": files}


def _write_manifest(directory: Path, manifest: dict) -> None:
    (directory / MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _read_manifest(directory: Path) -> dict | None:
    path = directory / MANIFEST_FILENAME
    if not path.is_file():
        return None
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    return value if isinstance(value, dict) and isinstance(value.get("files"), dict) else None


def _json_get(url: str, path: str, timeout: float = 2.0) -> dict:
    request = Request(f"{url.rstrip('/')}{path}", method="GET")
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def _json_post(url: str, path: str, body: dict, timeout: float = 60.0) -> dict:
    request = Request(
        f"{url.rstrip('/')}{path}",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    with urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read().decode("utf-8"))
    return payload if isinstance(payload, dict) else {}


def _embedder_path(settings) -> Path:
    return Path(settings.embedder_model_dir) / EMBEDDER_GGUF_FILE


def _reranker_path(settings) -> Path:
    return Path(settings.reranker_model_dir)


def _llama_server_executable(settings) -> str | None:
    configured = Path(settings.llama_server_bin)
    if configured.is_file():
        return str(configured)
    discovered = shutil.which("llama-server") or shutil.which("llama-server.exe")
    return discovered


def _base_model_status(settings, kind: str) -> dict:
    if kind == "embedder":
        path = _embedder_path(settings)
        installed = path.is_file()
        return {
            "kind": kind,
            "name": "Jina Embeddings v5 Nano Retrieval",
            "model_id": EMBEDDING_MODEL_ID,
            "revision": "main",
            "path": str(path.parent),
            "model_file": str(path),
            "installed": installed,
            "dimension": EMBEDDING_DIMENSION,
            "sha256_expected": EMBEDDER_GGUF_SHA256,
            "sha256": _sha256(path) if installed else None,
        }
    path = _reranker_path(settings)
    required = ("config.json", "tokenizer.json")
    manifest = _read_manifest(path)
    return {
        "kind": kind,
        "name": "Jina Reranker v3.5",
        "model_id": RERANKER_MODEL_ID,
        "revision": str(manifest.get("revision")) if manifest and manifest.get("revision") else "main",
        "path": str(path),
        "model_file": None,
        "installed": path.is_dir() and all((path / item).is_file() for item in required),
        "dimension": None,
        "trust_remote_code": True,
        "required": list(required),
    }


def model_status(app_state) -> dict:
    settings = app_state.settings
    embedder = _base_model_status(settings, "embedder")
    reranker = _base_model_status(settings, "reranker")
    return {
        "fixed_stack": True,
        "embedder": {**embedder, "runtime": embedder_status(app_state)},
        "reranker": {**reranker, "runtime": reranker_status(app_state)},
        "reindex_required": bool(getattr(app_state, "reindex_required", True)),
    }


def _process_status(process: subprocess.Popen | None) -> tuple[str, int | None]:
    if process is None:
        return "stopped", None
    if process.poll() is None:
        return "running", process.pid
    return "error", process.pid


def embedder_status(app_state, probe: bool = True) -> dict:
    status, pid = _process_status(getattr(app_state, "embedder_process", None))
    runtime = getattr(app_state, "embedder_runtime_status", {})
    starting = status == "running" and runtime.get("status") == "starting"
    payload = {
        "status": "starting" if starting else status if status != "stopped" else runtime.get("status", status),
        "port": app_state.settings.embedder_server_port,
        "url": app_state.settings.embedder_server_url,
        "pid": pid,
        "last_request_at": runtime.get("last_request_at"),
        "last_health_check": runtime.get("last_health_check"),
        "last_error": runtime.get("last_error"),
        "owned_process": bool(getattr(app_state, "embedder_process_owned", False)),
    }
    if probe and status == "running":
        try:
            _json_get(app_state.settings.embedder_server_url, "/health")
            payload["status"] = "running"
            runtime["last_health_check"] = _now()
            runtime["last_error"] = None
        except (URLError, OSError, ValueError) as exc:
            # Model loading is asynchronous.  A live owned process with no
            # response yet is starting, not failed; a later check promotes it
            # to running or reports an actual exited process as an error.
            if starting:
                payload["status"] = "starting"
            else:
                payload["status"] = "error"
                runtime["last_error"] = f"Embedder health check failed: {exc}"
        app_state.embedder_runtime_status = runtime
        payload["last_health_check"] = runtime.get("last_health_check")
        payload["last_error"] = runtime.get("last_error")
    return payload


def reranker_status(app_state) -> dict:
    status, pid = _process_status(getattr(app_state, "reranker_process", None))
    runtime = getattr(app_state, "reranker_runtime_status", {})
    return {
        "status": status if status != "stopped" else runtime.get("status", status),
        "pid": pid,
        "queue_size": runtime.get("queue_size", 0),
        "last_health_check": runtime.get("last_health_check"),
        "last_failure": runtime.get("last_failure"),
        "trust_remote_code": True,
    }


def _set_retrieval_error(app_state, message: str | None) -> None:
    app_state.retrieval_error = message
    app_state.reindex_required = True if message else bool(getattr(app_state, "reindex_required", False))


def start(app_state) -> None:
    """Configure and start owned local retrieval processes when models exist."""
    app_state.embedding_model_id = EMBEDDING_MODEL_ID
    app_state.embedding_dim = EMBEDDING_DIMENSION
    app_state.embedding_normalized = True
    app_state.embedding_batch_size = app_state.settings.embedder_batch_size
    # A marker for ingestion's batched production path.  Lightweight test
    # states omit it and can still inject a deterministic single-text embedder.
    app_state.embedder = True
    app_state.reranker_model_id = RERANKER_MODEL_ID
    app_state.embedder_process = None
    app_state.embedder_process_owned = False
    app_state.reranker_process = None
    app_state.embedder_runtime_status = {"status": "not_installed", "last_error": None}
    app_state.reranker_runtime_status = {"status": "not_installed", "queue_size": 0, "last_failure": None}

    embedder = _base_model_status(app_state.settings, "embedder")
    reranker = _base_model_status(app_state.settings, "reranker")
    if not embedder["installed"]:
        _set_retrieval_error(app_state, "Jina Nano Retrieval Q8_0 is not installed; document retrieval is disabled.")
        return
    if embedder["sha256"] != EMBEDDER_GGUF_SHA256:
        _set_retrieval_error(app_state, "Jina Nano Retrieval integrity check failed; document retrieval is disabled.")
        return
    try:
        _start_embedder(app_state)
    except RuntimeError as exc:
        _set_retrieval_error(app_state, str(exc))
        return
    if reranker["installed"]:
        try:
            _start_reranker(app_state)
        except RuntimeError as exc:
            app_state.reranker_runtime_status["status"] = "error"
            app_state.reranker_runtime_status["last_failure"] = str(exc)
    else:
        app_state.reranker_runtime_status["status"] = "not_installed"
    _set_retrieval_error(app_state, None)


def _start_embedder(app_state) -> None:
    executable = _llama_server_executable(app_state.settings)
    if not executable:
        raise RuntimeError("llama-server.exe is unavailable; configure CEPHALON_LLAMA_SERVER_BIN for the dedicated embedder.")
    try:
        _json_get(app_state.settings.embedder_server_url, "/health")
        app_state.embedder_runtime_status = {"status": "running", "last_health_check": _now(), "last_error": None}
        return
    except (URLError, OSError, ValueError):
        pass
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    command = [
        executable, "-m", str(_embedder_path(app_state.settings)),
        "--embedding", "--pooling", "last", "--embd-normalize", "2",
        "--device", app_state.settings.embedder_device,
        "--gpu-layers", str(app_state.settings.embedder_gpu_layers),
        "--batch-size", str(app_state.settings.embedder_physical_batch_size),
        "--ubatch-size", str(app_state.settings.embedder_physical_batch_size),
        "--host", "127.0.0.1", "--port", str(app_state.settings.embedder_server_port),
        "--no-webui",
    ]
    app_state.embedder_process = subprocess.Popen(
        command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        creationflags=creationflags,
    )
    app_state.embedder_process_owned = True
    app_state.embedder_runtime_status = {"status": "starting", "last_error": None}


def _start_reranker(app_state) -> None:
    command = [sys.executable, "-m", "cephalon_core.services.jina_reranker_worker", str(_reranker_path(app_state.settings))]
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command, cwd=str(Path(__file__).resolve().parents[2]), text=True,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=1, creationflags=creationflags,
    )
    app_state.reranker_process = process
    app_state.reranker_runtime_lock = threading.RLock()
    app_state.reranker_runtime_status = {"status": "starting", "queue_size": 0, "last_failure": None}


def stop(app_state) -> None:
    for attribute, owned in (("reranker_process", True), ("embedder_process", getattr(app_state, "embedder_process_owned", False))):
        process = getattr(app_state, attribute, None)
        if process is not None and owned and process.poll() is None:
            process.terminate()
            try:
                process.wait(timeout=8)
            except subprocess.TimeoutExpired:
                process.kill()
    app_state.embedder_process = None
    app_state.reranker_process = None


def embed(app_state, texts: list[str]) -> list[list[float]]:
    if getattr(app_state, "retrieval_error", None):
        raise RuntimeError(app_state.retrieval_error)
    response = _json_post(
        app_state.settings.embedder_server_url,
        "/v1/embeddings",
        {"input": texts, "model": EMBEDDING_MODEL_ID, "encoding_format": "float"},
    )
    data = response.get("data")
    if not isinstance(data, list) or len(data) != len(texts):
        raise RuntimeError("Dedicated embedder returned an invalid batch response.")
    ordered = sorted(data, key=lambda item: int(item.get("index", 0)))
    vectors = [item.get("embedding") for item in ordered]
    if any(not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSION for vector in vectors):
        raise RuntimeError(f"Jina Nano embedder must return exactly {EMBEDDING_DIMENSION}-dimensional vectors.")
    app_state.embedder_runtime_status["status"] = "running"
    app_state.embedder_runtime_status["last_request_at"] = _now()
    return [[float(value) for value in vector] for vector in vectors]


def rerank(app_state, query: str, documents: list[str]) -> list[dict]:
    process = getattr(app_state, "reranker_process", None)
    if process is None or process.poll() is not None:
        raise RuntimeError("Jina Reranker v3.5 worker is unavailable.")
    runtime = app_state.reranker_runtime_status
    lock = app_state.reranker_runtime_lock
    request_id = str(uuid.uuid4())
    with lock:
        runtime["queue_size"] = int(runtime.get("queue_size", 0)) + 1
        try:
            assert process.stdin is not None and process.stdout is not None
            process.stdin.write(json.dumps({"id": request_id, "query": query, "documents": documents}) + "\n")
            process.stdin.flush()
            line = process.stdout.readline()
            response = json.loads(line) if line else {}
            if response.get("id") != request_id or response.get("error"):
                raise RuntimeError(str(response.get("error") or "Reranker worker exited without a response."))
            results = response.get("results")
            if not isinstance(results, list):
                raise RuntimeError("Reranker worker returned malformed listwise results.")
            runtime.update({"status": "running", "last_health_check": _now(), "last_failure": None})
            return results
        except Exception as exc:
            runtime.update({"status": "error", "last_failure": str(exc)})
            raise
        finally:
            runtime["queue_size"] = max(0, int(runtime.get("queue_size", 1)) - 1)


def download_model(app_state, kind: str) -> dict:
    if kind not in {"embedder", "reranker"}:
        raise ValueError("kind must be embedder or reranker")
    if kind == "embedder":
        destination = Path(app_state.settings.embedder_model_dir)
        snapshot_download(
            EMBEDDER_GGUF_REPO, local_dir=str(destination),
            allow_patterns=[EMBEDDER_GGUF_FILE, "README.md", ".gitattributes"],
        )
    else:
        destination = Path(app_state.settings.reranker_model_dir)
        manifest = _official_manifest(RERANKER_REPO)
        snapshot_download(RERANKER_REPO, revision=manifest["revision"], local_dir=str(destination))
        _write_manifest(destination, manifest)
    return verify_model(app_state, kind)


def verify_model(app_state, kind: str) -> dict:
    payload = _base_model_status(app_state.settings, kind)
    if kind == "embedder":
        payload["verified"] = bool(payload["installed"] and payload["sha256"] == EMBEDDER_GGUF_SHA256)
        if not payload["verified"]:
            payload["error"] = "GGUF SHA-256 does not match the official Jina manifest."
        return payload
    directory = _reranker_path(app_state.settings)
    payload["files"] = {}
    if not payload["installed"]:
        payload["verified"] = False
        payload["error"] = "Reranker config/tokenizer files are missing."
        return payload
    manifest = _read_manifest(directory)
    if manifest is None:
        try:
            manifest = _official_manifest(RERANKER_REPO)
            _write_manifest(directory, manifest)
        except Exception as exc:
            payload["verified"] = False
            payload["error"] = f"Official reranker manifest is unavailable: {exc}"
            return payload
    mismatches: list[str] = []
    for relative_path, expected in manifest["files"].items():
        item = directory / relative_path
        if not item.is_file():
            mismatches.append(f"missing {relative_path}")
            continue
        actual_sha256 = _sha256(item)
        actual_blob = _git_blob_sha1(item)
        payload["files"][relative_path] = {
            "sha256": actual_sha256,
            "git_blob_sha1": actual_blob,
        }
        if expected.get("sha256") and actual_sha256 != expected["sha256"]:
            mismatches.append(f"sha256 mismatch for {relative_path}")
        elif not expected.get("sha256") and expected.get("git_blob_sha1") and actual_blob != expected["git_blob_sha1"]:
            mismatches.append(f"git blob mismatch for {relative_path}")
    payload["revision"] = manifest.get("revision")
    payload["verified"] = not mismatches
    if mismatches:
        payload["error"] = "; ".join(mismatches)
    return payload


def delete_model(app_state, kind: str) -> dict:
    if kind not in {"embedder", "reranker"}:
        raise ValueError("kind must be embedder or reranker")
    target = Path(app_state.settings.embedder_model_dir if kind == "embedder" else app_state.settings.reranker_model_dir)
    process_attr = "embedder_process" if kind == "embedder" else "reranker_process"
    process = getattr(app_state, process_attr, None)
    if process is not None and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
    setattr(app_state, process_attr, None)
    if target.exists():
        shutil.rmtree(target)
    if kind == "embedder":
        _set_retrieval_error(app_state, "Jina Nano Retrieval Q8_0 was removed; document retrieval is disabled.")
    return {"status": "deleted", "kind": kind, "path": str(target), "exists": target.exists()}


def open_model_directory(app_state, kind: str) -> dict:
    if kind not in {"embedder", "reranker"}:
        raise ValueError("kind must be embedder or reranker")
    target = Path(app_state.settings.embedder_model_dir if kind == "embedder" else app_state.settings.reranker_model_dir)
    target.mkdir(parents=True, exist_ok=True)
    if hasattr(os, "startfile"):
        os.startfile(str(target))  # type: ignore[attr-defined]  # Windows desktop app
    else:
        subprocess.Popen(["xdg-open", str(target)])
    return {"status": "opened", "kind": kind, "path": str(target)}
