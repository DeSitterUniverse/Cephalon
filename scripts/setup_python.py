"""Install Cephalon's Python runtime dependencies on Windows or Linux."""

from __future__ import annotations

import argparse
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--with-export-tools", action="store_true", help="also install the optional ONNX export toolchain")
    args = parser.parse_args()

    require_python_314()
    os.environ["PYTHONNOUSERSITE"] = "1"
    run(sys.executable, "-m", "pip", "install", "--upgrade", "pip")
    run(sys.executable, "-m", "pip", "install", "--upgrade", "-r", "requirements.txt")
    if args.with_export_tools:
        run(sys.executable, "-m", "pip", "install", "--upgrade", "-r", "requirements-export.txt")
    run(sys.executable, "scripts/preflight_runtime.py", "--skip-onnx")


if __name__ == "__main__":
    main()
