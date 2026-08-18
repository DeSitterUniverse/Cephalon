"""Build a native Cephalon release on Windows or Linux."""

from __future__ import annotations

import os
import shutil
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

    run(sys.executable, "-m", "pip", "install", "--upgrade", "-r", "requirements.txt")
    run(sys.executable, "scripts/preflight_runtime.py")
    run(sys.executable, "-m", "py_compile", "python/main.py", "scripts/smoke_ingest_query.py", "scripts/smoke_query.py")
    run(sys.executable, "-m", "pytest")
    run(sys.executable, "build_backend.py")
    run("cargo", "build", "--release")

    package_dir = REPO_ROOT / "dist" / "cephalon-native"
    if package_dir.exists():
        shutil.rmtree(package_dir)
    package_dir.mkdir(parents=True)

    executable_name = "cephalon.exe" if os.name == "nt" else "cephalon"
    shutil.copy2(REPO_ROOT / "target" / "release" / executable_name, package_dir / executable_name)
    shutil.copytree(REPO_ROOT / "backend" / "engine", package_dir / "backend" / "engine")
    shutil.copytree(REPO_ROOT / "assets", package_dir / "assets")
    for filename in ("README.md", "LICENSE", "LOCAL_STARTUP_NOTES.md"):
        shutil.copy2(REPO_ROOT / filename, package_dir / filename)

    if os.name == "nt":
        (package_dir / "cephalon-launch.cmd").write_text(
            "@echo off\r\ncd /d \"%~dp0\"\r\nstart \"Cephalon\" \"%~dp0cephalon.exe\"\r\n",
            encoding="utf-8",
        )
    else:
        launcher = package_dir / "cephalon-launch.sh"
        launcher.write_text(
            "#!/bin/sh\n"
            "set -eu\n"
            "cd \"$(dirname \"$0\")\"\n"
            "exec ./cephalon \"$@\"\n",
            encoding="utf-8",
        )
        launcher.chmod(0o755)
        desktop_entry = "\n".join(
            [
                "[Desktop Entry]",
                "Type=Application",
                "Name=Cephalon",
                "Comment=Native Cephalon RAG workbench",
                f"Exec={launcher}",
                f"Icon={package_dir / 'assets' / 'cephalon.svg'}",
                "Terminal=false",
                "Categories=Office;Utility;",
                "",
            ]
        )
        (package_dir / "cephalon.desktop").write_text(desktop_entry, encoding="utf-8")
        (package_dir / executable_name).chmod(0o755)

    print(f"Cephalon release build completed for {sys.platform}: {package_dir}")


if __name__ == "__main__":
    main()
