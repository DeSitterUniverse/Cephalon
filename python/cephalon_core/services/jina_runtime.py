"""Fixed local retrieval runtime: Jina Nano through llama.cpp and Jina v3.5."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import uuid
from functools import lru_cache
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
    RERANKER_FILE_SHA256,
    RERANKER_GGUF_FILE,
    RERANKER_LLAMA_CPP_REVISION,
    RERANKER_MODEL_ID,
    RERANKER_PROJECTOR_FILE,
    RERANKER_REPO,
    RERANKER_REVISION,
    RERANKER_TOKENIZER_FILE,
    RERANKER_TRANSFORMERS_REPO,
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


def _legacy_reranker_path(settings) -> Path:
    return Path(settings.legacy_reranker_model_dir)


def _reranker_required_files() -> tuple[str, ...]:
    return (RERANKER_GGUF_FILE, RERANKER_PROJECTOR_FILE, RERANKER_TOKENIZER_FILE)


@lru_cache(maxsize=8)
def _reranker_binary_capabilities(executable: str) -> dict:
    """Verify the exact llama.cpp graph revision used by Jina's GGUF helper.

    Jina v3.5 needs selected-token output and its irregular Qwen3 SWA graph.
    The selected-token option is intentionally hidden from common help, so
    feature-name probing is neither sufficient nor reliable. Pinning the tested
    PR revision guards both behaviors and prevents a superficially compatible
    binary from silently changing rankings.
    """

    path = Path(executable)
    if not path.is_file():
        return {"compatible": False, "path": str(path), "error": "llama-embedding executable is missing."}
    try:
        version_result = subprocess.run(
            [str(path), "--version"],
            capture_output=True,
            text=True,
            timeout=15,
            creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
        )
    except (OSError, subprocess.TimeoutExpired) as exc:
        return {"compatible": False, "path": str(path), "error": str(exc)}
    version_output = f"{version_result.stdout}\n{version_result.stderr}"
    revision_match = re.search(r"\b([0-9a-f]{7,40})\b", version_output, flags=re.IGNORECASE)
    revision = revision_match.group(1).lower() if revision_match else None
    # PR 26286 changes the Qwen3 graph as well as the CLI. An arbitrary build
    # can expose similarly named flags while still treating this checkpoint's
    # irregular SWA pattern incorrectly. Pinning the tested source revision is
    # therefore a ranking-quality guard, not merely a reproducibility detail.
    revision_matches = bool(
        revision
        and RERANKER_LLAMA_CPP_REVISION.lower().startswith(revision)
    )
    compatible = (
        version_result.returncode == 0
        and revision_matches
    )
    errors: list[str] = []
    if version_result.returncode != 0:
        errors.append(f"version probe exited {version_result.returncode}")
    if not revision_matches:
        errors.append(
            f"requires llama.cpp {RERANKER_LLAMA_CPP_REVISION[:7]}, "
            f"found {revision or 'unknown'}"
        )
    return {
        "compatible": compatible,
        "path": str(path),
        "revision": revision,
        "required_revision": RERANKER_LLAMA_CPP_REVISION,
        "selected_token_output": revision_matches,
        "missing_features": [],
        "error": "; ".join(errors) if errors else None,
    }


def _select_reranker_backend(settings, gguf_installed: bool, legacy_installed: bool) -> tuple[str | None, dict]:
    """Resolve the configured backend while preserving a safe CPU rollback."""

    preference = settings.reranker_backend
    if preference not in {"auto", "gguf", "transformers"}:
        preference = "auto"
    capabilities = _reranker_binary_capabilities(settings.reranker_llama_embedding_bin)
    gguf_ready = gguf_installed and bool(capabilities["compatible"])
    if preference == "gguf":
        return ("gguf_vulkan" if gguf_ready else None), capabilities
    if preference == "transformers":
        return ("transformers_cpu" if legacy_installed else None), capabilities
    if gguf_ready:
        return "gguf_vulkan", capabilities
    if legacy_installed:
        return "transformers_cpu", capabilities
    return None, capabilities


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
    legacy_path = _legacy_reranker_path(settings)
    required = _reranker_required_files()
    gguf_installed = path.is_dir() and all((path / item).is_file() for item in required)
    legacy_required = ("config.json", "tokenizer.json", "model.safetensors")
    legacy_installed = legacy_path.is_dir() and all((legacy_path / item).is_file() for item in legacy_required)
    backend, capabilities = _select_reranker_backend(settings, gguf_installed, legacy_installed)
    return {
        "kind": kind,
        "name": "Jina Reranker v3.5",
        "model_id": RERANKER_MODEL_ID,
        "repo_id": RERANKER_REPO,
        "revision": RERANKER_REVISION,
        "path": str(path),
        "model_file": str(path / RERANKER_GGUF_FILE),
        "installed": gguf_installed or legacy_installed,
        "gguf_installed": gguf_installed,
        "legacy_installed": legacy_installed,
        "legacy_path": str(legacy_path),
        "selected_backend": backend,
        "llama_embedding": capabilities,
        "dimension": None,
        "precision": "BF16",
        "trust_remote_code": backend == "transformers_cpu",
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
        "backend": runtime.get("backend"),
        "device": runtime.get("device"),
        "model_precision": runtime.get("model_precision"),
        "queue_size": runtime.get("queue_size", 0),
        "last_health_check": runtime.get("last_health_check"),
        "last_failure": runtime.get("last_failure"),
        "trust_remote_code": runtime.get("backend") == "transformers_cpu",
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
    app_state.reranker_runtime_status = {
        "status": "not_installed",
        "backend": None,
        "device": None,
        "model_precision": None,
        "queue_size": 0,
        "last_failure": None,
    }

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
    status = _base_model_status(app_state.settings, "reranker")
    backend = status.get("selected_backend")
    if backend == "gguf_vulkan":
        verification = verify_model(app_state, "reranker")
        if not verification.get("verified"):
            if status.get("legacy_installed") and app_state.settings.reranker_backend == "auto":
                backend = "transformers_cpu"
            else:
                raise RuntimeError(str(verification.get("error") or "Jina GGUF integrity verification failed."))
    if backend == "gguf_vulkan":
        worker_arguments = [
            str(_reranker_path(app_state.settings)),
            app_state.settings.reranker_llama_embedding_bin,
            app_state.settings.reranker_device,
            str(app_state.settings.reranker_gpu_layers),
            str(app_state.settings.reranker_max_context_tokens),
        ]
        command = (
            [sys.executable, "--cephalon-jina-gguf-worker", *worker_arguments]
            if getattr(sys, "frozen", False)
            else [
                sys.executable,
                "-m",
                "cephalon_core.services.jina_reranker_worker",
                *worker_arguments,
            ]
        )
        device = app_state.settings.reranker_device
        precision = "BF16"
    elif backend == "transformers_cpu":
        worker_arguments = [str(_legacy_reranker_path(app_state.settings))]
        command = (
            [sys.executable, "--cephalon-jina-transformers-worker", *worker_arguments]
            if getattr(sys, "frozen", False)
            else [
                sys.executable,
                "-m",
                "cephalon_core.services.jina_reranker_transformers_worker",
                *worker_arguments,
            ]
        )
        device = "CPU"
        precision = "BF16"
    else:
        capabilities = status.get("llama_embedding", {})
        missing = ", ".join(capabilities.get("missing_features", []))
        reason = capabilities.get("error") or (f"missing llama.cpp features: {missing}" if missing else "model assets are missing")
        raise RuntimeError(f"No usable Jina Reranker v3.5 backend: {reason}.")
    creationflags = getattr(subprocess, "CREATE_NO_WINDOW", 0)
    process = subprocess.Popen(
        command, cwd=str(Path(__file__).resolve().parents[2]), text=True,
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
        bufsize=1, creationflags=creationflags,
    )
    app_state.reranker_process = process
    app_state.reranker_runtime_lock = threading.RLock()
    app_state.reranker_runtime_status = {
        "status": "starting",
        "backend": backend,
        "device": device,
        "model_precision": precision,
        "queue_size": 0,
        "last_failure": None,
    }


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
        manifest = {
            "repo_id": RERANKER_REPO,
            "revision": RERANKER_REVISION,
            "files": {
                filename: {"sha256": sha256, "git_blob_sha1": None}
                for filename, sha256 in RERANKER_FILE_SHA256.items()
            },
        }
        snapshot_download(
            RERANKER_REPO,
            revision=RERANKER_REVISION,
            local_dir=str(destination),
            allow_patterns=[*_reranker_required_files(), "README.md", ".gitattributes"],
        )
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
    if not payload["gguf_installed"]:
        # Preserve verification for pre-GGUF installations and embedders that
        # supplied only `reranker_model_dir` before the split cache paths were
        # introduced. This is a genuine rollback contract: Settings must not
        # claim that an intact CPU fallback is corrupt merely because the new
        # preferred cache is absent.
        configured_legacy = _legacy_reranker_path(app_state.settings)
        inline_legacy = directory
        legacy_directory = (
            inline_legacy
            if (inline_legacy / "config.json").is_file()
            else configured_legacy
        )
        if not all((legacy_directory / item).is_file() for item in ("config.json", "tokenizer.json")):
            payload["verified"] = False
            payload["error"] = "BF16 GGUF assets and Transformers fallback files are missing."
            return payload
        manifest = _read_manifest(legacy_directory)
        if manifest is None:
            try:
                manifest = _official_manifest(RERANKER_TRANSFORMERS_REPO)
                _write_manifest(legacy_directory, manifest)
            except Exception as exc:
                payload["verified"] = False
                payload["error"] = f"Official Transformers reranker manifest is unavailable: {exc}"
                return payload
        mismatches: list[str] = []
        for relative_path, expected in manifest["files"].items():
            item = legacy_directory / relative_path
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
            elif (
                not expected.get("sha256")
                and expected.get("git_blob_sha1")
                and actual_blob != expected["git_blob_sha1"]
            ):
                mismatches.append(f"git blob mismatch for {relative_path}")
        payload["path"] = str(legacy_directory)
        payload["revision"] = manifest.get("revision")
        payload["verified_backend"] = "transformers_cpu"
        payload["verified"] = not mismatches
        if mismatches:
            payload["error"] = "; ".join(mismatches)
        return payload
    mismatches: list[str] = []
    for relative_path, expected_sha256 in RERANKER_FILE_SHA256.items():
        item = directory / relative_path
        if not item.is_file():
            mismatches.append(f"missing {relative_path}")
            continue
        actual_sha256 = _sha256(item)
        payload["files"][relative_path] = {
            "sha256": actual_sha256,
            "expected_sha256": expected_sha256,
        }
        if actual_sha256 != expected_sha256:
            mismatches.append(f"sha256 mismatch for {relative_path}")
    payload["revision"] = RERANKER_REVISION
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
