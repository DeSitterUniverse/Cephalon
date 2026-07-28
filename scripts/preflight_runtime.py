import argparse
import importlib
import json
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


def main() -> int:
    parser = argparse.ArgumentParser(description="Validate the Cephalon local Python runtime.")
    parser.parse_args()

    imports = [package_version(name) for name in RUNTIME_IMPORTS]
    report = {
        "python": {
            "executable": sys.executable,
            "version": sys.version,
            "platform": platform.platform(),
            "user_site_enabled": any("Roaming\\Python" in path for path in sys.path),
        },
        "imports": imports,
    }
    print(json.dumps(report, indent=2))

    failed_imports = [item["name"] for item in imports if not item["ok"]]
    failures = []
    if failed_imports:
        failures.append(f"missing imports: {', '.join(failed_imports)}")
    if failures:
        raise SystemExit("; ".join(failures))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
