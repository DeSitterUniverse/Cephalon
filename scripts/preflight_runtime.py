import argparse
import importlib
import json
import os
import platform
import sys
from pathlib import Path


RUNTIME_IMPORTS = [
    "fastapi",
    "uvicorn",
    "lancedb",
    "pyarrow",
    "docx",
    "pptx",
    "openpyxl",
    "pypdf",
    "onnxruntime",
    "transformers",
    "huggingface_hub",
    "numpy",
    "llama_cpp",
]


def package_version(module_name: str) -> dict:
    try:
        module = importlib.import_module(module_name)
    except Exception as exc:
        return {"name": module_name, "ok": False, "error": str(exc)}
    return {
        "name": module_name,
        "ok": True,
        "version": getattr(module, "__version__", None),
        "path": str(Path(getattr(module, "__file__", "")).resolve()) if getattr(module, "__file__", None) else None,
    }


def llama_info() -> dict:
    try:
        import llama_cpp
    except Exception as exc:
        return {"ok": False, "error": str(exc)}

    package_dir = Path(llama_cpp.__file__).resolve().parent
    lib_dir = package_dir / "lib"
    vulkan_dll = lib_dir / "ggml-vulkan.dll"
    return {
        "ok": vulkan_dll.exists(),
        "package": str(package_dir),
        "lib_dir": str(lib_dir),
        "vulkan_dll": str(vulkan_dll) if vulkan_dll.exists() else None,
        "dlls": sorted(path.name for path in lib_dir.glob("*.dll")) if lib_dir.exists() else [],
    }


def onnx_assets(model_root: Path) -> dict:
    assets = {}
    for name in ("embedder", "reranker"):
        model_dir = model_root / name
        required = ["model.onnx", "tokenizer.json", "tokenizer_config.json"]
        missing = [filename for filename in required if not (model_dir / filename).exists()]
        assets[name] = {
            "path": str(model_dir),
            "ok": not missing,
            "missing": missing,
        }
    return assets


def gguf_assets(model_root: Path) -> dict:
    ggufs = sorted(path.name for path in model_root.glob("*.gguf")) if model_root.exists() else []
    chat_models = [
        name for name in ggufs
        if not any(marker in name.lower() for marker in ("embed", "retrieval", "reranker", "cross-encoder"))
    ]
    return {"model_dir": str(model_root), "gguf_count": len(ggufs), "chat_models": chat_models}


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Cephalon local Python runtime.")
    parser.add_argument("--model-dir", default=os.getenv("CEPHALON_MODEL_DIR", str(Path.home() / "cephalon-data" / "models")))
    parser.add_argument("--skip-onnx", action="store_true", help="Skip embedder/reranker asset checks for app-only release packaging.")
    args = parser.parse_args()
    model_root = Path(args.model_dir).expanduser().resolve()

    imports = [package_version(name) for name in RUNTIME_IMPORTS]
    report = {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
            "user_site_enabled": any("Roaming\\Python" in path for path in sys.path),
        },
        "imports": imports,
        "llama_cpp": llama_info(),
        "onnx_assets": {"skipped": True} if args.skip_onnx else onnx_assets(model_root),
        "gguf_assets": gguf_assets(model_root),
    }
    print(json.dumps(report, indent=2))

    failed_imports = [item["name"] for item in imports if not item["ok"]]
    failed_onnx = [] if args.skip_onnx else [name for name, item in report["onnx_assets"].items() if not item["ok"]]
    failures = []
    if failed_imports:
        failures.append(f"missing imports: {', '.join(failed_imports)}")
    if not report["llama_cpp"]["ok"]:
        failures.append("llama-cpp-python is not Vulkan-enabled or ggml-vulkan.dll is missing")
    if failed_onnx:
        failures.append(f"missing ONNX assets: {', '.join(failed_onnx)}")
    if not report["gguf_assets"]["chat_models"]:
        failures.append("no chat GGUF models found")
    if failures:
        raise SystemExit("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
