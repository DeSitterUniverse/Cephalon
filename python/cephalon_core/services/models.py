import json
from contextlib import nullcontext
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import HTTPException

from ..schemas import LlamaServerSettings


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
        raise RuntimeError(f"llama.cpp server returned HTTP {exc.code}.") from exc
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


def llama_backend_info(app_state, *, probe: bool = False) -> dict:
    config = server_settings(app_state)
    info = {
        "provider": "external_llama_server",
        "backend_label": "External llama.cpp server",
        "server_url": config.server_url,
        "model_name": config.model_name,
        "server_available": None,
        "server_error": None,
        "vulkan_available": None,
        "vulkan_required": False,
        "gpu_backend_available": None,
    }
    if probe:
        try:
            _server_request(app_state, "/health")
            info["server_available"] = True
            info["loaded_model_name"] = _loaded_model_name(app_state)
        except RuntimeError as exc:
            info["server_available"] = False
            info["server_error"] = str(exc)
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
        app_state.active_model_name = None
        app_state.last_model_load_error = str(exc)
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
        ensure_server_connected(app_state, "") if getattr(app_state, "active_model_name", None) else None
        try:
            _server_request(app_state, "/health")
        except RuntimeError as exc:
            app_state.last_model_load_error = str(exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        app_state.llm = None
        app_state.active_model_name = _loaded_model_name(app_state)
        config = server_settings(app_state)
        app_state.active_context_tokens = config.context_tokens
        app_state.active_model_context_tokens = config.context_tokens
        app_state.last_model_load_error = None
