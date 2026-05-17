import gc
import os
import sys
from contextlib import contextmanager
from pathlib import Path

from fastapi import HTTPException


def _configure_llama_dll_search() -> None:
    dll_dir = os.getenv("CEPHALON_LLAMA_DLL_DIR")
    if dll_dir and os.path.isdir(dll_dir):
        os.environ.setdefault("LLAMA_CPP_LIB_PATH", dll_dir)
        if hasattr(os, "add_dll_directory"):
            os.add_dll_directory(dll_dir)


def _discover_packaged_llama_dll_dir() -> str | None:
    module_path = Path(__file__).resolve()
    repo_root = module_path.parents[3]
    candidates = [
        repo_root / "src-tauri" / "backend" / "engine" / "_internal" / "llama_cpp" / "lib",
        repo_root / "src-tauri" / "target" / "debug" / "backend" / "engine" / "_internal" / "llama_cpp" / "lib",
        repo_root / "src-tauri" / "target" / "release" / "backend" / "engine" / "_internal" / "llama_cpp" / "lib",
    ]
    for candidate in candidates:
        if any(candidate.glob("ggml*.dll")) or any(candidate.glob("libggml*")):
            return str(candidate)
    return None


_configure_llama_dll_search()

from llama_cpp import Llama  # noqa: E402
import llama_cpp  # noqa: E402

from .. import storage
from ..config import Settings
from ..validators import validate_model_filename


def llama_backend_info() -> dict:
    package_dir = Path(llama_cpp.__file__).resolve().parent
    loaded_base_path = getattr(getattr(llama_cpp, "llama_cpp", None), "_base_path", None)
    lib_dirs = []
    env_dll_dir = os.getenv("CEPHALON_LLAMA_DLL_DIR") or os.getenv("LLAMA_CPP_LIB_PATH")
    if env_dll_dir:
        lib_dirs.append(Path(env_dll_dir))
    lib_dirs.extend([package_dir / "lib", package_dir])
    pyinstaller_root = getattr(sys, "_MEIPASS", None)
    if pyinstaller_root:
        lib_dirs.extend([Path(pyinstaller_root) / "llama_cpp" / "lib", Path(pyinstaller_root)])

    seen = set()
    lib_dirs = [path for path in lib_dirs if not (str(path) in seen or seen.add(str(path)))]
    backend_lib = next((candidate for path in lib_dirs for candidate in path.glob("ggml*.dll")), None)
    backend_available = backend_lib is not None or loaded_base_path is not None
    return {
        "package": str(package_dir),
        "lib_dir": str(package_dir / "lib"),
        "loaded_lib_base_path": str(loaded_base_path) if loaded_base_path else None,
        "override_lib_path": os.getenv("LLAMA_CPP_LIB_PATH"),
        "dll_search_paths": [str(path) for path in lib_dirs],
        "gpu_backend_available": backend_available,
        "backend_label": "llama.cpp backend" if backend_available else "CPU backend",
        "vulkan_required": False,
        "vulkan_available": False,
        "vulkan_dll": None,
        "backend_library": str(backend_lib) if backend_lib else str(loaded_base_path) if loaded_base_path else None,
    }


def list_models(settings: Settings) -> list[str]:
    return model_inventory(settings)["chat_models"]


def model_inventory(settings: Settings) -> dict[str, list[str]]:
    os.makedirs(settings.model_dir, exist_ok=True)
    chat_models: list[str] = []
    chat_model_details: list[dict] = []
    auxiliary_gguf: list[str] = []
    for entry in os.scandir(settings.model_dir):
        if not entry.is_file() or not entry.name.lower().endswith(".gguf"):
            continue
        if _looks_like_chat_model(entry.name):
            chat_models.append(entry.name)
            chat_model_details.append({"name": entry.name, "size_bytes": entry.stat().st_size})
        else:
            auxiliary_gguf.append(entry.name)
    return {
        "chat_models": sorted(chat_models),
        "chat_model_details": sorted(chat_model_details, key=lambda item: item["name"]),
        "auxiliary_gguf": sorted(auxiliary_gguf),
    }


def _looks_like_chat_model(filename: str) -> bool:
    lowered = filename.lower()
    return not any(marker in lowered for marker in ("embed", "retrieval", "reranker", "cross-encoder"))


@contextmanager
def _quiet_llama_stderr():
    if os.getenv("CEPHALON_LLAMA_VERBOSE", "0") != "0":
        yield
        return
    stderr_fd = None
    saved_fd = None
    devnull_fd = None
    try:
        stderr_fd = sys.stderr.fileno()
        saved_fd = os.dup(stderr_fd)
        devnull_fd = os.open(os.devnull, os.O_WRONLY)
        os.dup2(devnull_fd, stderr_fd)
    except Exception:
        yield
        return
    try:
        yield
    finally:
        if stderr_fd is not None and saved_fd is not None:
            os.dup2(saved_fd, stderr_fd)
            os.close(saved_fd)
        if devnull_fd is not None:
            os.close(devnull_fd)


def _model_context_length(model_path: str) -> int | None:
    try:
        with _quiet_llama_stderr():
            metadata_model = Llama(model_path=model_path, vocab_only=True, verbose=False)
        metadata = getattr(metadata_model, "metadata", {}) or {}
        del metadata_model
        gc.collect()
    except Exception:
        return None

    for key, value in metadata.items():
        if key.endswith(".context_length"):
            try:
                return int(value)
            except (TypeError, ValueError):
                return None
    return None


def load_llm(app_state, model_filename: str) -> None:
    if not _looks_like_chat_model(model_filename):
        raise HTTPException(status_code=400, detail="Selected GGUF is an embedding/reranker asset, not a chat model.")
    model_path = validate_model_filename(model_filename, app_state.settings.model_dir)
    rag_settings = storage.get_rag_settings(app_state.sqlite)
    model_context_tokens = _model_context_length(model_path)
    context_tokens = model_context_tokens if rag_settings.full_context and model_context_tokens else rag_settings.context_tokens
    backend = llama_backend_info()
    if getattr(app_state, "llm", None) is not None:
        print("Deallocating active VRAM model...")
        del app_state.llm
        gc.collect()

    print(
        f"Loading {model_filename} with llama.cpp "
        f"({backend['backend_label']})."
    )
    try:
        with _quiet_llama_stderr():
            app_state.llm = Llama(
                model_path=model_path,
                n_gpu_layers=-1,
                n_ctx=context_tokens,
                main_gpu=int(os.getenv("CEPHALON_MAIN_GPU", "0")),
                offload_kqv=True,
                verbose=os.getenv("CEPHALON_LLAMA_VERBOSE", "0") != "0",
            )
        app_state.active_model_name = model_filename
        app_state.active_context_tokens = context_tokens
        app_state.active_model_context_tokens = model_context_tokens
        print(f"Model '{model_filename}' loaded successfully.")
    except Exception as e:
        app_state.llm = None
        app_state.active_model_name = None
        app_state.active_context_tokens = None
        app_state.active_model_context_tokens = None
        raise HTTPException(status_code=500, detail=f"Failed to load model: {e}") from e
