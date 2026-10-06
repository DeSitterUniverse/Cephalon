"""Explicit, pinned retrieval model setup. Never called during app startup."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import shutil
import subprocess
import sys
import tarfile
import tempfile
import zipfile
from pathlib import Path
from urllib.request import urlopen

from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "python"))
from cephalon_core.config import (  # noqa: E402
    EMBEDDER_MODEL_FILE, EMBEDDING_GGUF_REPO, EMBEDDING_GGUF_REVISION,
    EMBEDDING_GGUF_SHA256, EMBEDDING_LLAMA_RELEASE, EMBEDDING_MODEL_ID,
    EMBEDDING_MODEL_REVISION, RERANKER_FILE_SHA256, RERANKER_LLAMA_CPP_REVISION,
    RERANKER_MODEL_ID, RERANKER_PACKAGES, RERANKER_PRECISION, RERANKER_REPO,
    RERANKER_REVISION, RERANKER_WORKER_SHA256,
    embedding_vector_space,
)

MANIFEST_FILE = "cephalon-model-manifest.json"
LLAMA_ASSETS = {
    ("Windows", "AMD64"): ("llama-b11456-bin-win-vulkan-x64.zip", "7c17593521854b6567d0c8037d13a207909bad7557905b4d8c01563031c575a1"),
    ("Linux", "x86_64"): ("llama-b11456-bin-ubuntu-vulkan-x64.tar.gz", "d96de8abba616bb3159b728087bc078e146aa99d178c826303c6b861ef83c691"),
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def setup_embeddinggemma(args) -> None:
    destination = args.model_dir.resolve()
    path = Path(hf_hub_download(EMBEDDING_GGUF_REPO, EMBEDDER_MODEL_FILE, revision=EMBEDDING_GGUF_REVISION, local_dir=str(destination)))
    if digest(path) != EMBEDDING_GGUF_SHA256:
        raise RuntimeError("Downloaded GGUF does not match the pinned Q8_0 checksum")
    manifest = {
        "model_id": EMBEDDING_MODEL_ID, "revision": EMBEDDING_MODEL_REVISION,
        "gguf_repo": EMBEDDING_GGUF_REPO, "gguf_revision": EMBEDDING_GGUF_REVISION,
        "vector_space": embedding_vector_space(), "files": {EMBEDDER_MODEL_FILE: EMBEDDING_GGUF_SHA256},
    }
    (destination / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"model_id": EMBEDDING_MODEL_ID, "gguf_revision": EMBEDDING_GGUF_REVISION, "sha256": EMBEDDING_GGUF_SHA256, "model_file": str(path)}, indent=2))


def setup_llama(args) -> None:
    key = (platform.system(), platform.machine())
    if key not in LLAMA_ASSETS:
        raise RuntimeError("Automatic Vulkan runtime setup supports Windows AMD64 and Linux x86_64; install the pinned llama.cpp build manually on other platforms.")
    asset, expected = LLAMA_ASSETS[key]
    destination = args.runtime_dir.resolve()
    # A versioned, separate directory leaves the chat runtime untouched.
    if destination.exists() and any(destination.iterdir()):
        manifest_path = destination / "cephalon-runtime-manifest.json"
        if manifest_path.is_file():
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            files = manifest.get("files", {})
            if manifest.get("release") == EMBEDDING_LLAMA_RELEASE and manifest.get("archive_sha256") == expected and files and all(
                (destination / name).resolve().is_relative_to(destination) and (destination / name).is_file() and digest(destination / name) == value
                for name, value in files.items()
            ):
                print(f"Verified existing llama.cpp {EMBEDDING_LLAMA_RELEASE}: {destination}")
                return
        raise RuntimeError(f"Runtime directory is already occupied or changed; choose an empty directory: {destination}")
    with tempfile.TemporaryDirectory(prefix="cephalon-llama-setup-") as temporary:
        scratch = Path(temporary)
        archive = args.archive.resolve() if args.archive else scratch / asset
        if not args.archive:
            url = f"https://github.com/ggml-org/llama.cpp/releases/download/{EMBEDDING_LLAMA_RELEASE}/{asset}"
            with urlopen(url, timeout=60) as response, archive.open("wb") as output:
                shutil.copyfileobj(response, output)
        if digest(archive) != expected:
            raise RuntimeError("llama.cpp archive checksum differs from the pinned release")
        extracted = scratch / "extracted"
        extracted.mkdir()
        if asset.endswith(".zip"):
            with zipfile.ZipFile(archive) as package:
                for item in package.infolist():
                    if not (extracted / item.filename).resolve().is_relative_to(extracted.resolve()):
                        raise RuntimeError("Unsafe path in runtime archive")
                package.extractall(extracted)
        else:
            with tarfile.open(archive) as package:
                package.extractall(extracted, filter="data")
        executable = "llama-server.exe" if os.name == "nt" else "llama-server"
        candidates = list(extracted.rglob(executable))
        if len(candidates) != 1:
            raise RuntimeError("Pinned archive did not contain exactly one llama-server")
        shutil.copytree(candidates[0].parent, destination, dirs_exist_ok=True)
        files = {str(path.relative_to(destination)).replace("\\", "/"): digest(path) for path in destination.rglob("*") if path.is_file()}
        manifest = {"release": EMBEDDING_LLAMA_RELEASE, "archive": asset, "archive_sha256": expected, "files": files}
        (destination / "cephalon-runtime-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        print(f"Installed llama.cpp {EMBEDDING_LLAMA_RELEASE}: {destination / executable}")


def setup_jina(args) -> None:
    destination = args.model_dir.resolve()
    destination.mkdir(parents=True, exist_ok=True)
    for name, expected in RERANKER_FILE_SHA256.items():
        path = destination / name
        if not path.is_file() or digest(path) != expected:
            path = Path(hf_hub_download(RERANKER_REPO, name, revision=RERANKER_REVISION, local_dir=str(destination)))
        if digest(path) != expected:
            raise RuntimeError(f"Jina artifact checksum mismatch: {name}")
    worker = ROOT / "scripts" / "jina_reranker_worker.py"
    if digest(worker) != RERANKER_WORKER_SHA256:
        raise RuntimeError("Jina worker source differs from its pinned checksum")
    shutil.copy2(worker, destination / "cephalon-jina-worker.py")
    files = {**RERANKER_FILE_SHA256, "cephalon-jina-worker.py": RERANKER_WORKER_SHA256}
    manifest = {
        "model_id": RERANKER_MODEL_ID, "repo": RERANKER_REPO, "revision": RERANKER_REVISION,
        "precision": RERANKER_PRECISION, "packages": RERANKER_PACKAGES, "files": files,
    }
    (destination / MANIFEST_FILE).write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({"model_id": RERANKER_MODEL_ID, "revision": RERANKER_REVISION, "file_count": len(files)}, indent=2))


def setup_jina_worker(args) -> None:
    destination = args.runtime_dir.resolve()
    executable = destination / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    if not executable.is_file():
        if destination.exists() and any(destination.iterdir()):
            raise RuntimeError(f"Worker environment directory is occupied: {destination}")
        subprocess.run(["uv", "venv", "--python", "3.14", str(destination)], check=True)
    # Local Rust tokenization and NumPy projection need no Hub or tensor runtime.
    subprocess.run(["uv", "pip", "install", "--python", str(executable), "--no-deps",
                    *(f"{name}=={version}" for name, version in RERANKER_PACKAGES.items())], check=True)
    print(f"Installed isolated Jina worker environment: {executable}")


def setup_jina_llama(args) -> None:
    destination = args.runtime_dir.resolve()
    if destination.exists() and any(destination.iterdir()):
        from types import SimpleNamespace
        from cephalon_core.services.reranker_runtime import verify_runtime
        executable = destination / ("llama-embedding.exe" if os.name == "nt" else "llama-embedding")
        verify_runtime(SimpleNamespace(reranker_llama_embedding_bin=str(executable)))
        print(f"Verified existing PR #26286 runtime: {destination}")
        return
    source = args.source_dir.resolve()
    if not source.exists():
        subprocess.run(["git", "init", str(source)], check=True)
        subprocess.run(["git", "-C", str(source), "fetch", "--depth", "1",
                        "https://github.com/ggml-org/llama.cpp.git", RERANKER_LLAMA_CPP_REVISION], check=True)
        subprocess.run(["git", "-C", str(source), "checkout", "--detach", RERANKER_LLAMA_CPP_REVISION], check=True)
    revision = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    changed = subprocess.check_output(["git", "-C", str(source), "status", "--porcelain"], text=True).strip()
    if revision != RERANKER_LLAMA_CPP_REVISION or changed:
        raise RuntimeError("Jina runtime source must be clean and at the pinned PR #26286 commit")
    build = source / "build"
    configure = ["cmake", "-S", str(source), "-B", str(build), "-DGGML_VULKAN=ON",
                 "-DCMAKE_BUILD_TYPE=Release", "-DLLAMA_CURL=OFF", "-DLLAMA_BUILD_TESTS=OFF",
                 "-DLLAMA_BUILD_EXAMPLES=ON", "-DLLAMA_BUILD_SERVER=OFF"]
    subprocess.run(configure, check=True)
    subprocess.run(["cmake", "--build", str(build), "--config", "Release", "--target", "llama-embedding",
                    "--parallel", str(args.jobs)], check=True)
    bin_dir = build / "bin" / "Release" if os.name == "nt" else build / "bin"
    executable = "llama-embedding.exe" if os.name == "nt" else "llama-embedding"
    if not (bin_dir / executable).is_file():
        raise RuntimeError(f"Build did not produce {bin_dir / executable}")
    destination.mkdir(parents=True, exist_ok=True)
    for path in bin_dir.iterdir():
        if path.is_file() and (path.name == executable or path.suffix == ".dll" or ".so" in path.name or path.suffix == ".dylib"):
            shutil.copy2(path, destination / path.name)
    manifest = {
        "source_commit": revision, "source_pr": "https://github.com/ggml-org/llama.cpp/pull/26286",
        "backend": "vulkan", "files": {path.name: digest(path) for path in destination.iterdir() if path.is_file()},
    }
    (destination / "cephalon-runtime-manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    print(f"Installed Jina PR #26286 {revision}: {destination / executable}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="command", required=True)
    embedder = subcommands.add_parser("embeddinggemma")
    embedder.add_argument("--model-dir", type=Path, required=True)
    jina = subcommands.add_parser("jina")
    jina.add_argument("--model-dir", type=Path, required=True)
    worker = subcommands.add_parser("jina-worker")
    worker.add_argument("--runtime-dir", type=Path, required=True)
    jina_llama = subcommands.add_parser("jina-llama")
    jina_llama.add_argument("--source-dir", type=Path, required=True, help="Dedicated llama.cpp checkout outside Cephalon")
    jina_llama.add_argument("--runtime-dir", type=Path, required=True)
    jina_llama.add_argument("--jobs", type=int, default=8)
    llama = subcommands.add_parser("llama")
    llama.add_argument("--runtime-dir", type=Path, required=True)
    llama.add_argument("--archive", type=Path, help="Reuse a local archive with the pinned release checksum")
    args = parser.parse_args()
    if args.command == "embeddinggemma":
        setup_embeddinggemma(args)
    elif args.command == "jina":
        setup_jina(args)
    elif args.command == "jina-worker":
        setup_jina_worker(args)
    elif args.command == "jina-llama":
        setup_jina_llama(args)
    else:
        setup_llama(args)


if __name__ == "__main__":
    main()
