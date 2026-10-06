"""Managed, offline Jina v3.5 Q8 GGUF reranking with PR #26286."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import shutil
import signal
import socket
import subprocess
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse
from urllib.error import HTTPError
from urllib.request import Request, urlopen

from ..config import (
    RERANKER_FILE_SHA256, RERANKER_GGUF_FILE, RERANKER_LLAMA_CPP_REVISION,
    RERANKER_MODEL_ID, RERANKER_PACKAGES, RERANKER_PRECISION, RERANKER_REPO,
    RERANKER_REVISION, RERANKER_SCORE_TYPE, RERANKER_WORKER_SHA256,
)
from . import embedding_runtime

MANIFEST_FILE = "cephalon-model-manifest.json"
WORKER_FILE = "cephalon-jina-worker.py"


@lru_cache(maxsize=64)
def _cached_hash(path: Path, size: int, modified_ns: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _hash(path: Path) -> str:
    stat = path.stat()
    return _cached_hash(path, stat.st_size, stat.st_mtime_ns)


def _get(url: str, endpoint: str, timeout: float = 2.0) -> dict:
    with urlopen(Request(url + endpoint), timeout=timeout) as response:
        value = json.load(response)
    return value if isinstance(value, dict) else {}


def verify_model(app_state) -> dict:
    directory = Path(app_state.settings.reranker_model_dir)
    manifest_path = directory / MANIFEST_FILE
    result = {
        "kind": "reranker", "name": "Jina Reranker v3.5",
        "model_id": RERANKER_MODEL_ID, "revision": RERANKER_REVISION,
        "path": str(directory), "model_file": str(directory / RERANKER_GGUF_FILE),
        "installed": (directory / RERANKER_GGUF_FILE).is_file(),
        "selected_backend": "llama_cpp_vulkan", "score_type": RERANKER_SCORE_TYPE,
        "precision": RERANKER_PRECISION, "projector_precision": "fp32",
        "llama_cpp_revision": RERANKER_LLAMA_CPP_REVISION,
        "verified": False,
    }
    if not result["installed"] or not manifest_path.is_file():
        result["error"] = "Run explicit Jina setup; the GGUF, projector, tokenizer and manifest are required."
        return result
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        files = manifest["files"]
    except (OSError, ValueError, KeyError, TypeError) as exc:
        result["error"] = f"Invalid reranker manifest: {exc}"
        return result
    if any(manifest.get(key) != expected for key, expected in {
        "model_id": RERANKER_MODEL_ID, "repo": RERANKER_REPO,
        "revision": RERANKER_REVISION, "precision": RERANKER_PRECISION,
        "packages": RERANKER_PACKAGES,
    }.items()):
        result["error"] = "Reranker manifest identifies a different model revision."
        return result
    expected_files = {**RERANKER_FILE_SHA256, WORKER_FILE: RERANKER_WORKER_SHA256}
    if files != expected_files:
        result["error"] = "Reranker manifest does not match the pinned artifact and worker checksums."
        return result
    for name, expected in files.items():
        item = directory / name
        if not item.is_file() or _hash(item) != expected:
            result["error"] = f"Reranker snapshot checksum mismatch: {name}"
            return result
    result["verified"] = True
    return result


@lru_cache(maxsize=16)
def _runtime_version(path: Path, size: int, modified_ns: int) -> str:
    result = subprocess.run([str(path), "--version"], capture_output=True, text=True, timeout=15, check=True,
                            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    return result.stdout + result.stderr


def verify_runtime(settings) -> None:
    executable = Path(settings.reranker_llama_embedding_bin)
    directory = executable.parent.resolve()
    try:
        manifest = json.loads((directory / "cephalon-runtime-manifest.json").read_text(encoding="utf-8"))
        files = manifest["files"]
        if manifest.get("source_commit") != RERANKER_LLAMA_CPP_REVISION or manifest.get("backend") != "vulkan" or not isinstance(files, dict) or executable.name not in files:
            raise ValueError("runtime is not the pinned PR #26286 head")
        for name, expected in files.items():
            path = (directory / name).resolve()
            if not path.is_relative_to(directory) or not path.is_file() or _hash(path) != expected:
                raise ValueError(f"runtime checksum mismatch: {name}")
        stat = executable.stat()
        if not re.search(r"\bcommit " + RERANKER_LLAMA_CPP_REVISION[:7] + r"\b", _runtime_version(executable, stat.st_size, stat.st_mtime_ns)):
            raise ValueError("llama-embedding reports a different source commit")
    except (OSError, ValueError, KeyError, TypeError, subprocess.SubprocessError) as exc:
        raise RuntimeError(f"Jina needs the pinned llama.cpp PR #26286 runtime: {exc}") from exc


def _worker_ready(settings) -> bool:
    try:
        model = _get(settings.reranker_worker_url, "/model")
    except (OSError, ValueError):
        return False
    manifest = Path(settings.reranker_model_dir) / MANIFEST_FILE
    return (
        model.get("model_id") == RERANKER_MODEL_ID
        and model.get("revision") == RERANKER_REVISION
        and model.get("score_type") == RERANKER_SCORE_TYPE
        and model.get("precision") == RERANKER_PRECISION
        and model.get("llama_cpp_revision") == RERANKER_LLAMA_CPP_REVISION
        and Path(settings.reranker_llama_embedding_bin).is_file()
        and model.get("llama_bin_sha256") == _hash(Path(settings.reranker_llama_embedding_bin))
        and model.get("device") == settings.reranker_device
        and model.get("gpu_layers") == settings.reranker_gpu_layers
        and model.get("max_context_tokens") == settings.reranker_max_context_tokens
        and model.get("packages") == RERANKER_PACKAGES
        and manifest.is_file()
        and model.get("manifest_sha256") == _hash(manifest)
        and os.path.normcase(os.path.normpath(str(model.get("model_path", "")))) == os.path.normcase(os.path.normpath(settings.reranker_model_dir))
    )


def start(app_state) -> None:
    app_state.reranker_model_id = RERANKER_MODEL_ID
    app_state.reranker_process = None
    app_state.reranker_process_owned = False
    app_state.reranker_runtime_status = {"status": "not_installed", "last_failure": None}
    check = verify_model(app_state)
    if not check["verified"]:
        app_state.reranker_runtime_status["last_failure"] = check["error"]
        return
    settings = app_state.settings
    endpoint = urlparse(settings.reranker_worker_url)
    if endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost"} or endpoint.port != settings.reranker_worker_port or endpoint.path or endpoint.username or endpoint.query or endpoint.fragment:
        app_state.reranker_runtime_status = {"status": "error", "last_failure": "Reranker worker must bind to configured loopback port."}
        return
    python_bin = settings.reranker_python_bin
    if not python_bin or not Path(python_bin).is_file():
        app_state.reranker_runtime_status = {"status": "error", "last_failure": "Install the isolated Jina Python environment with scripts/setup_retrieval_models.py jina-worker."}
        return
    try:
        verify_runtime(settings)
    except RuntimeError as exc:
        app_state.reranker_runtime_status = {"status": "error", "last_failure": str(exc)}
        return
    try:
        with socket.create_connection(("127.0.0.1", settings.reranker_worker_port), timeout=0.3):
            app_state.reranker_runtime_status = {"status": "error", "last_failure": "Another service occupies the reranker port; choose a free loopback port."}
            return
    except OSError:
        pass
    environment = os.environ.copy()
    for name in list(environment):
        if name.startswith("LLAMA_ARG_"):
            del environment[name]
    environment.update({"HF_HUB_OFFLINE": "1", "TOKENIZERS_PARALLELISM": "false", "PYTHONUTF8": "1"})
    command = [python_bin, str(Path(settings.reranker_model_dir) / WORKER_FILE),
               "--model-dir", settings.reranker_model_dir, "--port", str(settings.reranker_worker_port),
               "--llama-bin", settings.reranker_llama_embedding_bin, "--device", settings.reranker_device,
               "--gpu-layers", str(settings.reranker_gpu_layers),
               "--max-context-tokens", str(settings.reranker_max_context_tokens)]
    log_path = Path(settings.data_dir) / "logs" / "jina35-q8-reranker.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("ab") as log:
            process = subprocess.Popen(command, env=environment, stdout=log, stderr=subprocess.STDOUT,
                                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0), start_new_session=os.name != "nt")
    except OSError as exc:
        app_state.reranker_runtime_status = {"status": "error", "last_failure": str(exc)}
        return
    app_state.reranker_process = process
    app_state.reranker_process_owned = True
    app_state.reranker_runtime_status = {"status": "starting", "last_failure": None, "log_path": str(log_path)}


def stop(app_state) -> None:
    process = getattr(app_state, "reranker_process", None)
    if process is not None and getattr(app_state, "reranker_process_owned", False) and process.poll() is None:
        # The owned worker can have an active llama-embedding child using VRAM.
        if os.name == "nt":
            subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True,
                           timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        else:
            os.killpg(process.pid, signal.SIGTERM)
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
    app_state.reranker_process = None
    app_state.reranker_process_owned = False
    app_state.reranker_runtime_status = {"status": "stopped", "last_failure": None}


def status(app_state) -> dict:
    process = getattr(app_state, "reranker_process", None)
    runtime = getattr(app_state, "reranker_runtime_status", {})
    state = runtime.get("status", "stopped")
    ready = process is not None and process.poll() is None and _worker_ready(app_state.settings)
    if process is not None:
        state = "error" if process.poll() is not None else ("running" if ready else "starting")
    try:
        identity = _get(app_state.settings.reranker_worker_url, "/model") if ready else {}
    except (OSError, ValueError):
        identity = {}
        state = "error"
    last_failure = runtime.get("last_failure")
    if state == "error" and process is not None and process.poll() is not None and not last_failure:
        last_failure = f"Jina worker exited with code {process.returncode}; see {runtime.get('log_path', 'worker log')}."
    return {
        "status": state, "pid": process.pid if process is not None else None,
        "url": app_state.settings.reranker_worker_url, "backend": "llama_cpp_vulkan",
        "device": identity.get("device"),
        "packages": identity.get("packages"),
        "model_precision": RERANKER_PRECISION, "projector_precision": "fp32",
        "score_type": RERANKER_SCORE_TYPE, "llama_cpp_revision": RERANKER_LLAMA_CPP_REVISION,
        "max_context_tokens": app_state.settings.reranker_max_context_tokens,
        "last_failure": last_failure, "log_path": runtime.get("log_path"),
        "owned_process": bool(getattr(app_state, "reranker_process_owned", False)),
    }


def rerank(app_state, query: str, documents: list[str]) -> list[dict]:
    if not documents:
        return []
    process = getattr(app_state, "reranker_process", None)
    if process is None or process.poll() is not None or not _worker_ready(app_state.settings):
        raise RuntimeError("Expected Jina GGUF worker is unavailable or reports the wrong model/runtime.")
    request = Request(
        app_state.settings.reranker_worker_url + "/rerank",
        data=json.dumps({"query": query, "documents": documents}).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    try:
        with urlopen(request, timeout=max(120, min(600, len(documents) * 1.5))) as response:
            payload = json.load(response)
    except HTTPError as exc:
        try:
            detail = json.loads(exc.read()).get("error", str(exc))
        except (ValueError, AttributeError):
            detail = str(exc)
        raise RuntimeError(f"Jina worker request failed: {str(detail)[-2000:]}") from exc
    except (OSError, ValueError) as exc:
        raise RuntimeError(f"Jina worker request failed: {exc}") from exc
    results = payload.get("results") if isinstance(payload, dict) else None
    if not isinstance(results, list) or len(results) != len(documents) or any(not isinstance(item, dict) for item in results):
        raise RuntimeError("Jina worker returned an invalid response.")
    if any(type(item.get("index")) is not int for item in results) or sorted(item["index"] for item in results) != list(range(len(documents))):
        raise RuntimeError("Jina worker returned invalid candidate indexes.")
    if any(type(item.get("score")) not in (int, float) or not math.isfinite(item["score"]) or not -1 <= item["score"] <= 1 for item in results):
        raise RuntimeError("Jina worker returned an invalid cosine score.")
    return sorted(results, key=lambda item: (-item["score"], item["index"]))


def model_status(app_state) -> dict:
    return {
        "fixed_stack": True,
        "embedder": {**embedding_runtime.verify_model(app_state), "runtime": embedding_runtime.status(app_state)},
        "reranker": {**verify_model(app_state), "runtime": status(app_state)},
        "reindex_required": bool(getattr(app_state, "reindex_required", True)),
    }


def download_model(_app_state, _kind: str) -> dict:
    raise RuntimeError("Retrieval models require explicit pinned offline setup; run scripts/setup_retrieval_models.py.")


def delete_model(app_state, kind: str) -> dict:
    if kind not in {"embedder", "reranker"}:
        raise ValueError("kind must be embedder or reranker")
    if kind == "embedder":
        embedding_runtime.stop(app_state)
    else:
        stop(app_state)
    target = Path(app_state.settings.embedder_model_dir if kind == "embedder" else app_state.settings.reranker_model_dir)
    if target.exists():
        shutil.rmtree(target)
    if kind == "embedder":
        app_state.retrieval_error = "EmbeddingGemma 2 was removed; document retrieval is disabled."
    return {"status": "deleted", "kind": kind, "path": str(target), "exists": target.exists()}


def open_model_directory(app_state, kind: str) -> dict:
    if kind not in {"embedder", "reranker"}:
        raise ValueError("kind must be embedder or reranker")
    target = Path(app_state.settings.embedder_model_dir if kind == "embedder" else app_state.settings.reranker_model_dir)
    target.mkdir(parents=True, exist_ok=True)
    if hasattr(os, "startfile"):
        os.startfile(str(target))
    else:
        subprocess.Popen(["xdg-open", str(target)])
    return {"status": "opened", "kind": kind, "path": str(target)}
