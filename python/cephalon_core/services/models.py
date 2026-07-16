import json
import os
from contextlib import nullcontext
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

from fastapi import HTTPException


DEFAULT_SERVER_URL = "http://127.0.0.1:8080"
DEFAULT_MODEL_NAME = "External llama.cpp server"


def llama_server_url() -> str:
    return os.getenv("CEPHALON_LLAMA_SERVER_URL", DEFAULT_SERVER_URL).rstrip("/")


def llama_server_model_name() -> str:
    return os.getenv("CEPHALON_LLAMA_SERVER_MODEL", DEFAULT_MODEL_NAME).strip() or DEFAULT_MODEL_NAME


def llama_server_context_tokens() -> int | None:
    value = os.getenv("CEPHALON_LLAMA_SERVER_CONTEXT_TOKENS", "").strip()
    if not value:
        return None
    try:
        return max(4096, int(value))
    except ValueError:
        return None


def _server_request(path: str, *, timeout: float = 2.0) -> dict:
    request = Request(f"{llama_server_url()}{path}", method="GET")
    try:
        with urlopen(request, timeout=timeout) as response:
            body = response.read().decode("utf-8", errors="replace")
    except HTTPError as exc:
        raise RuntimeError(f"llama.cpp server returned HTTP {exc.code}.") from exc
    except URLError as exc:
        raise RuntimeError(f"Could not reach llama.cpp server at {llama_server_url()}: {exc.reason}") from exc
    except OSError as exc:
        raise RuntimeError(f"Could not reach llama.cpp server at {llama_server_url()}: {exc}") from exc
    try:
        parsed = json.loads(body) if body else {}
    except json.JSONDecodeError:
        parsed = {"response": body}
    return parsed if isinstance(parsed, dict) else {"response": parsed}


def llama_backend_info(*, probe: bool = False) -> dict:
    info = {
        "provider": "external_llama_server",
        "backend_label": "External llama.cpp server",
        "server_url": llama_server_url(),
        "model_name": llama_server_model_name(),
        "server_available": None,
        "server_error": None,
        "vulkan_available": None,
        "vulkan_required": False,
        "gpu_backend_available": None,
    }
    if probe:
        try:
            _server_request("/health")
            info["server_available"] = True
        except RuntimeError as exc:
            info["server_available"] = False
            info["server_error"] = str(exc)
    return info


def python_runtime_info() -> dict:
    import sys

    return {
        "executable": sys.executable,
        "version": sys.version,
        "prefix": sys.prefix,
        "base_prefix": getattr(sys, "base_prefix", sys.prefix),
    }


def list_models(_settings) -> list[str]:
    return [llama_server_model_name()]


def model_inventory(_settings) -> dict:
    return {
        "chat_models": [llama_server_model_name()],
        "chat_model_details": [],
        "auxiliary_gguf": [],
    }


def load_llm(app_state, model_name: str) -> None:
    if model_name != llama_server_model_name():
        raise HTTPException(status_code=400, detail="Select the configured external llama.cpp server model before connecting.")
    runtime = getattr(app_state, "model_runtime", None)
    guard = runtime.exclusive() if runtime is not None else nullcontext()
    with guard:
        try:
            _server_request("/health")
        except RuntimeError as exc:
            app_state.last_model_load_error = str(exc)
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        app_state.llm = None
        app_state.active_model_name = model_name
        app_state.active_context_tokens = llama_server_context_tokens()
        app_state.active_model_context_tokens = llama_server_context_tokens()
        app_state.last_model_load_error = None
