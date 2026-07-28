"""Build a native Cephalon release on Windows or Linux."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]


def require_python_314() -> None:
    if sys.version_info[:2] != (3, 14):
        found = f"{sys.version_info.major}.{sys.version_info.minor}"
        raise SystemExit(f"Cephalon requires Python 3.14; this script is running under Python {found}.")


def run(*args: str, cwd: Path = REPO_ROOT) -> None:
    subprocess.run(args, cwd=cwd, check=True)


def main() -> None:
    require_python_314()
    os.environ["PYTHONNOUSERSITE"] = "1"
    npm = "npm.cmd" if os.name == "nt" else "npm"

    run(sys.executable, "-m", "pip", "install", "--upgrade", "-r", "requirements.txt")
    run(sys.executable, "scripts/preflight_runtime.py")
    run(sys.executable, "-m", "py_compile", "python/main.py", "scripts/smoke_ingest_query.py", "scripts/smoke_query.py")
    run(sys.executable, "-m", "pytest")
    run(npm, "ci")
    run(npm, "run", "build")
    run("cargo", "check", cwd=REPO_ROOT / "src-tauri")
    run(sys.executable, "build_backend.py")
    run(npm, "run", "tauri", "build")

    print(f"Cephalon release build completed for {sys.platform}.")


if __name__ == "__main__":
    main()
