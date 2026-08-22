import json
from contextlib import nullcontext
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import HTTPException

from ..schemas import LlamaServerSettings


SERVER_LOADING_STATUSES = {"loading", "starting", "initializing", "busy"}


def server_settings(app_state) -> LlamaServerSettings:
    configured = getattr(app_state, "llama_server_settings", None)
    if configured is not None:
        return configured
    settings = getattr(app_state, "settings", None)
    if settings is not None:
        return LlamaServerSettings(
            server_url=settings.llama_server_url,
            model_name=settings.llama_server_model,
            context_tokens=settings.llama_server_context_tokens,
        )
    return LlamaServerSettings()


def _server_request(app_state, path: str, *, timeout: float = 2.0) -> dict:
    config = server_settings(app_state)
    request = Request(f"{config.server_url}{path}", method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace").strip()[:500]
        message = f"llama.cpp server returned HTTP {exc.code}."
        if detail:
            message = f"{message} {detail}"
        raise RuntimeError(message) from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach llama.cpp server at {config.server_url}: {exc.reason}") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not reach llama.cpp server at {config.server_url}: {exc}") from exc
    try:
        parsed = json.loads(body) if body else {}
    except json.JSONDecodeError:
        parsed = {}
    return parsed if isinstance(parsed, dict) else {}


def _loaded_model_name(app_state) -> str:
    configured = server_settings(app_state).model_name
    try:
        props = _server_request(app_state, "/props")
    except RuntimeError:
        return configured
    for key in ("model_alias", "model_name", "model", "model_path"):
        value = props.get(key)
        if isinstance(value, str) and value.strip():
            return Path(value).stem or value.strip()
    return configured


def _clear_active_model(app_state) -> None:
    app_state.active_model_name = None
    app_state.active_context_tokens = None
    app_state.active_model_context_tokens = None


def mark_model_error(app_state, message: str, *, disconnect: bool = False) -> None:
    """Record a model/runtime error without hiding it from the UI.

    Transport failures invalidate the client-side connection. Inference HTTP
    errors and output validation failures do not necessarily mean the server
    is gone, so callers can keep the model connected while showing the error.
    """

    clean = str(message or "Model request failed.").strip()[:1000]
    app_state.last_model_error = clean
    if disconnect:
        _clear_active_model(app_state)
        app_state.last_model_load_error = clean


def clear_model_error(app_state) -> None:
    app_state.last_model_error = None


def _connection_status(
    app_state,
    server_available: bool | None,
    server_status: str = "",
) -> str:
    if server_available is False:
        return "offline"
    normalized_status = server_status.strip().lower()
    if normalized_status in SERVER_LOADING_STATUSES:
        return "connecting"
    if getattr(app_state, "active_model_name", None):
        return "connected"
    if server_available is True:
        return "connecting"
    return "offline"


def llama_backend_info(app_state, *, probe: bool = False) -> dict:
    config = server_settings(app_state)
    info = {
        "provider": "external_llama_server",
        "backend_label": "External llama.cpp server",
        "server_url": config.server_url,
        "model_name": config.model_name,
        "server_available": None,
        "server_status": None,
        "connection_status": _connection_status(app_state, None),
        "server_error": None,
        "vulkan_available": None,
        "vulkan_required": False,
        "gpu_backend_available": None,
    }
    if probe:
        try:
            health = _server_request(app_state, "/health")
            info["server_available"] = True
            info["server_status"] = str(health.get("status") or "")
            info["loaded_model_name"] = _loaded_model_name(app_state)
        except RuntimeError as exc:
            info["server_available"] = False
            info["server_error"] = str(exc)
    info["connection_status"] = _connection_status(
        app_state,
        info["server_available"],
        str(info.get("server_status") or ""),
    )
    return info


def python_runtime_info() -> dict:
    import sys

    return {"executable": sys.executable, "version": sys.version, "prefix": sys.prefix, "base_prefix": getattr(sys, "base_prefix", sys.prefix)}


def model_inventory(app_state) -> dict:
    config = server_settings(app_state)
    return {"chat_models": [config.model_name], "chat_model_details": [], "auxiliary_gguf": []}


def ensure_server_connected(app_state, requested_model: str = "") -> None:
    config = server_settings(app_state)
    try:
        _server_request(app_state, "/health")
    except RuntimeError as exc:
        mark_model_error(app_state, str(exc), disconnect=True)
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    active = getattr(app_state, "active_model_name", None)
    if not active:
        raise HTTPException(status_code=409, detail="Connect to the configured external llama.cpp server before querying.")
    # The server owns model loading. The client label is only a connection
    # target, so a model path reported by llama.cpp must not invalidate chat.
    if config.context_tokens:
        app_state.active_context_tokens = config.context_tokens
        app_state.active_model_context_tokens = config.context_tokens


def load_llm(app_state, _model_name: str = "") -> None:
    runtime = getattr(app_state, "model_runtime", None)
    guard = runtime.exclusive() if runtime is not None else nullcontext()
    with guard:
        try:
            _server_request(app_state, "/health")
        except RuntimeError as exc:
            mark_model_error(app_state, str(exc), disconnect=True)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        app_state.llm = None
        app_state.active_model_name = _loaded_model_name(app_state)
        config = server_settings(app_state)
        app_state.active_context_tokens = config.context_tokens
        app_state.active_model_context_tokens = config.context_tokens
        app_state.last_model_load_error = None
        clear_model_error(app_state)
