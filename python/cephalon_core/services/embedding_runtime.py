"""Managed offline llama.cpp service for EmbeddingGemma 2 Text 270M Q8_0."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import socket
import subprocess
import time
from functools import lru_cache
from pathlib import Path
from urllib.parse import urlparse
from urllib.request import Request, urlopen

from ..config import (
    EMBEDDER_MODEL_FILE, EMBEDDING_DIMENSION, EMBEDDING_GGUF_REPO,
    EMBEDDING_GGUF_REVISION, EMBEDDING_GGUF_SHA256, EMBEDDING_LLAMA_RELEASE,
    EMBEDDING_MAX_TOKENS, EMBEDDING_MODEL_ID, EMBEDDING_MODEL_REVISION,
    EMBEDDING_POOLING, EMBEDDING_PROMPT_SCHEME, EMBEDDING_QUANTIZATION,
    embedding_config_hash, embedding_vector_space,
)

MANIFEST_FILE = "cephalon-model-manifest.json"
PROMPTS = {"query": "task: search result | query: ", "document": "title: none | text: "}
MODEL_ALIAS = "cephalon-embeddinggemma2-q8-" + embedding_config_hash()[:16]


@lru_cache(maxsize=32)
def _cached_hash(path: Path, size: int, modified_ns: int) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _file_hash(path: Path) -> str:
    stat = path.stat()
    return _cached_hash(path, stat.st_size, stat.st_mtime_ns)


def verify_model(app_state) -> dict:
    directory = Path(app_state.settings.embedder_model_dir)
    result = {
        "kind": "embedder", "name": "EmbeddingGemma 2 Text 270M Q8_0",
        "model_id": EMBEDDING_MODEL_ID, "revision": EMBEDDING_MODEL_REVISION,
        "path": str(directory), "model_file": str(directory / EMBEDDER_MODEL_FILE),
        "installed": (directory / EMBEDDER_MODEL_FILE).is_file(), "dimension": EMBEDDING_DIMENSION,
        "precision": EMBEDDING_QUANTIZATION, "pooling": EMBEDDING_POOLING,
        "prompt_scheme": EMBEDDING_PROMPT_SCHEME, "selected_backend": "llama.cpp_cpu" if app_state.settings.embedder_gpu_layers == 0 else "llama.cpp_vulkan",
        "modalities": ["text"], "verified": False,
    }
    manifest_path = directory / MANIFEST_FILE
    if not result["installed"] or not manifest_path.is_file():
        result["error"] = "Run scripts/setup_retrieval_models.py embeddinggemma; the pinned Q8_0 GGUF and manifest are required."
        return result
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("vector_space") != embedding_vector_space():
            raise ValueError("embedding vector space differs from the pinned Q8_0 configuration")
        if manifest.get("gguf_repo") != EMBEDDING_GGUF_REPO or manifest.get("gguf_revision") != EMBEDDING_GGUF_REVISION:
            raise ValueError("GGUF manifest identifies a different repository or revision")
        files = manifest.get("files")
        if not isinstance(files, dict) or files.get(EMBEDDER_MODEL_FILE) != EMBEDDING_GGUF_SHA256:
            raise ValueError("manifest does not identify the pinned Q8_0 GGUF checksum")
        for name, expected in files.items():
            item = directory / name
            if not item.resolve().is_relative_to(directory.resolve()) or not item.is_file() or _file_hash(item) != expected:
                raise ValueError(f"GGUF checksum mismatch: {name}")
    except (OSError, ValueError, TypeError, AttributeError) as exc:
        result["error"] = f"Invalid EmbeddingGemma 2 GGUF: {exc}"
        return result
    result["sha256"] = EMBEDDING_GGUF_SHA256
    result["verified"] = True
    return result


def _get(url: str, endpoint: str, timeout: float = 2.0) -> dict:
    with urlopen(Request(url + endpoint), timeout=timeout) as response:
        value = json.load(response)
    return value if isinstance(value, dict) else {}


def _post(settings, endpoint: str, payload: dict) -> dict:
    request = Request(
        settings.embedder_server_url + endpoint, data=json.dumps(payload).encode(),
        headers={"Content-Type": "application/json"}, method="POST",
    )
    with urlopen(request, timeout=120) as response:
        value = json.load(response)
    if not isinstance(value, dict):
        raise RuntimeError("Embedding server returned a non-object response.")
    return value


def _server_ready(settings) -> bool:
    try:
        if _get(settings.embedder_server_url, "/health").get("status") != "ok":
            return False
        props = _get(settings.embedder_server_url, "/props")
        expected = Path(settings.embedder_model_dir) / EMBEDDER_MODEL_FILE
        return (
            props.get("model_alias") == MODEL_ALIAS
            and props.get("model_ftype") == EMBEDDING_QUANTIZATION
            and props.get("modalities") == {"vision": False, "video": False, "audio": False}
            and os.path.normcase(os.path.abspath(str(props.get("model_path", "")))) == os.path.normcase(str(expected.resolve()))
            and props.get("default_generation_settings", {}).get("n_ctx") == EMBEDDING_MAX_TOKENS
            and props.get("total_slots") == 1
        )
    except (OSError, ValueError, TypeError, AttributeError):
        return False


def _port_occupied(port: int) -> bool:
    try:
        with socket.create_connection(("127.0.0.1", port), timeout=0.3):
            return True
    except OSError:
        return False


@lru_cache(maxsize=8)
def _runtime_version(path: str, size: int, modified_ns: int) -> str:
    result = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    output = result.stdout + result.stderr
    if result.returncode or not re.search(r"\bbuild\s+" + EMBEDDING_LLAMA_RELEASE[1:] + r"\b", output):
        raise ValueError(f"EmbeddingGemma 2 requires the pinned llama.cpp {EMBEDDING_LLAMA_RELEASE} build; run the explicit llama setup.")
    return output.strip()


def server_command(settings) -> list[str]:
    # Bidirectional embeddings must fit an entire sequence in one physical
    # batch. One slot bounds attention memory and retains the full 8192 limit.
    command = [
        settings.embedder_llama_server_bin, "--model", str(Path(settings.embedder_model_dir) / EMBEDDER_MODEL_FILE),
        "--alias", MODEL_ALIAS, "--host", "127.0.0.1", "--port", str(settings.embedder_server_port),
        "--embedding", "--pooling", "mean", "--ctx-size", str(EMBEDDING_MAX_TOKENS),
        "--batch-size", str(EMBEDDING_MAX_TOKENS), "--ubatch-size", str(EMBEDDING_MAX_TOKENS),
        "--parallel", "1", "--flash-attn", "on", "--gpu-layers", str(settings.embedder_gpu_layers),
        "--fit", "off",
        "--no-warmup", "--no-webui", "--offline", "--no-mmproj", "--poll", "0", "--poll-batch", "0",
    ]
    if settings.embedder_device:
        command.extend(["--device", settings.embedder_device])
    return command


def start(app_state) -> None:
    app_state.embedding_model_id = EMBEDDING_MODEL_ID
    app_state.embedding_dim = EMBEDDING_DIMENSION
    app_state.embedding_normalized = True
    app_state.embedding_batch_size = app_state.settings.embedder_batch_size
    app_state.embedder = True
    app_state.embedder_process = None
    app_state.embedder_process_owned = False
    app_state.embedder_runtime_status = {"status": "not_installed", "last_error": None}
    check = verify_model(app_state)
    if not check["verified"]:
        app_state.retrieval_error = check["error"]
        app_state.embedder_runtime_status["last_error"] = check["error"]
        return
    settings = app_state.settings
    endpoint = urlparse(settings.embedder_server_url)
    if endpoint.scheme != "http" or endpoint.hostname not in {"127.0.0.1", "localhost"} or endpoint.port != settings.embedder_server_port or endpoint.path or endpoint.query or endpoint.fragment or endpoint.username:
        error = "Embedding server must use the configured loopback HTTP port."
    elif _port_occupied(settings.embedder_server_port):
        error = "Another service occupies the embedding port; Cephalon requires its own Q8_0 embedding server."
    elif not Path(settings.embedder_llama_server_bin).is_file():
        error = f"Install llama.cpp {EMBEDDING_LLAMA_RELEASE} with scripts/setup_retrieval_models.py llama or set CEPHALON_EMBEDDER_LLAMA_SERVER_BIN."
    else:
        error = None
    if error:
        app_state.retrieval_error = error
        app_state.embedder_runtime_status = {"status": "error", "last_error": error}
        return
    try:
        stat = Path(settings.embedder_llama_server_bin).stat()
        _runtime_version(settings.embedder_llama_server_bin, stat.st_size, stat.st_mtime_ns)
    except (OSError, ValueError, subprocess.TimeoutExpired) as exc:
        app_state.retrieval_error = str(exc)
        app_state.embedder_runtime_status = {"status": "error", "last_error": str(exc)}
        return
    # The embedding process must not inherit chat-server LLAMA_ARG overrides.
    environment = {name: value for name, value in os.environ.items() if not name.startswith("LLAMA_ARG_") and name != "LLAMA_API_KEY"}
    log_path = Path(settings.data_dir) / "logs" / "embeddinggemma2-q8-embedder.log"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        with log_path.open("ab") as log:
            app_state.embedder_process = subprocess.Popen(server_command(settings), env=environment, stdout=log, stderr=subprocess.STDOUT, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    except OSError as exc:
        app_state.retrieval_error = f"Could not start EmbeddingGemma 2 Q8_0: {exc}"
        app_state.embedder_runtime_status = {"status": "error", "last_error": app_state.retrieval_error}
        return
    app_state.embedder_process_owned = True
    app_state.embedder_runtime_status = {"status": "starting", "last_error": None, "log_path": str(log_path)}
    app_state.retrieval_error = None


def stop(app_state) -> None:
    process = getattr(app_state, "embedder_process", None)
    if process is not None and getattr(app_state, "embedder_process_owned", False) and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=8)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=8)
    app_state.embedder_process = None
    app_state.embedder_process_owned = False
    app_state.embedder_runtime_status = {"status": "stopped", "last_error": None}


def status(app_state) -> dict:
    process = getattr(app_state, "embedder_process", None)
    runtime = getattr(app_state, "embedder_runtime_status", {})
    ready = _server_ready(app_state.settings) if runtime.get("status") in {"starting", "running"} else False
    state = "running" if ready else runtime.get("status", "stopped")
    if process is not None:
        state = "error" if process.poll() is not None else ("running" if ready else "starting")
    elif state == "running" and not ready:
        state = "error"
    last_error = runtime.get("last_error")
    if state == "error" and not last_error:
        last_error = f"EmbeddingGemma 2 server exited with code {process.returncode}; see {runtime.get('log_path', 'embedder log')}." if process is not None else "EmbeddingGemma 2 server is unavailable."
    return {
        "status": state, "pid": process.pid if process is not None else None,
        "url": app_state.settings.embedder_server_url, "port": app_state.settings.embedder_server_port,
        "backend": "llama.cpp_cpu" if app_state.settings.embedder_gpu_layers == 0 else "llama.cpp_vulkan", "device": app_state.settings.embedder_device or ("cpu" if app_state.settings.embedder_gpu_layers == 0 else "auto"),
        "model_precision": EMBEDDING_QUANTIZATION, "runtime_release": EMBEDDING_LLAMA_RELEASE,
        "owned_process": bool(getattr(app_state, "embedder_process_owned", False)),
        "last_error": last_error, "log_path": runtime.get("log_path"),
    }


def embed(app_state, role: str, texts: list[str]) -> list[list[float]]:
    if role not in PROMPTS:
        raise ValueError("embedding role must be query or document")
    if not texts:
        return []
    if getattr(app_state, "retrieval_error", None):
        raise RuntimeError(app_state.retrieval_error)
    if not _server_ready(app_state.settings):
        raise RuntimeError("Expected EmbeddingGemma 2 Q8_0 model is not available on the embedding port.")
    inputs = []
    for text in texts:
        # Tokenize with the GGUF's own tokenizer, add the role prompt once,
        # then truncate before inference. Token arrays prevent re-tokenization.
        tokens = _post(app_state.settings, "/tokenize", {"content": PROMPTS[role] + text, "add_special": True, "parse_special": False}).get("tokens")
        if not isinstance(tokens, list) or not tokens or not all(type(token) is int and token >= 0 for token in tokens):
            raise RuntimeError("Embedding server returned invalid token IDs.")
        inputs.append(tokens[:EMBEDDING_MAX_TOKENS])
    payload = _post(app_state.settings, "/v1/embeddings", {"input": inputs, "model": MODEL_ALIAS, "encoding_format": "float"})
    data = payload.get("data")
    if not isinstance(data, list) or len(data) != len(texts) or any(not isinstance(item, dict) or type(item.get("index")) is not int for item in data):
        raise RuntimeError("Embedding server returned an invalid batch response.")
    ordered = sorted(data, key=lambda item: item["index"])
    if [item.get("index") for item in ordered] != list(range(len(texts))):
        raise RuntimeError("Embedding server returned invalid batch indexes.")
    vectors = [item.get("embedding") for item in ordered]
    if any(not isinstance(vector, list) or len(vector) != EMBEDDING_DIMENSION or not all(type(value) in (int, float) and math.isfinite(value) for value in vector) or not math.isfinite(math.hypot(*vector)) or not math.hypot(*vector) for vector in vectors):
        raise RuntimeError(f"EmbeddingGemma 2 must return finite nonzero {EMBEDDING_DIMENSION}-dimensional vectors.")
    app_state.embedder_runtime_status = {**getattr(app_state, "embedder_runtime_status", {}), "status": "running", "last_request_at": int(time.time())}
    return vectors
