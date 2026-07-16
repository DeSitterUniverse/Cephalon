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
        "onnx_assets": {"skipped": True} if args.skip_onnx else onnx_assets(model_root),
    }
    print(json.dumps(report, indent=2))

    failed_imports = [item["name"] for item in imports if not item["ok"]]
    failed_onnx = [] if args.skip_onnx else [name for name, item in report["onnx_assets"].items() if not item["ok"]]
    failures = []
    if failed_imports:
        failures.append(f"missing imports: {', '.join(failed_imports)}")
    if failed_onnx:
        failures.append(f"missing ONNX assets: {', '.join(failed_onnx)}")
    if failures:
        raise SystemExit("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
