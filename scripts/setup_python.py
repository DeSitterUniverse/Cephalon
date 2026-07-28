"""Install Cephalon's Python runtime dependencies on Windows or Linux."""

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


def run(*args: str) -> None:
    subprocess.run(args, cwd=REPO_ROOT, check=True)


def main() -> None:
    require_python_314()
    os.environ["PYTHONNOUSERSITE"] = "1"
    run(sys.executable, "-m", "pip", "install", "--upgrade", "pip")
    run(sys.executable, "-m", "pip", "install", "--upgrade", "-r", "requirements.txt")
    run(sys.executable, "scripts/preflight_runtime.py")


if __name__ == "__main__":
    main()
