import os
import sys
import shutil
import subprocess
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parent


def add_data_arg(source: str, destination: str) -> str:
    """Return PyInstaller's platform-specific SOURCE<sep>DEST value."""
    return f"{source}{os.pathsep}{destination}"

def build():
    print("Building FastAPI backend with PyInstaller (--onedir)...")
    
    hidden_imports = [
        "lancedb",
        "numpy",
        "uvicorn",
        "docx",
        "openpyxl",
        "pypdf",
        "pdfplumber",
        "pdfminer",
    ]
    excluded_modules = [
        "transformers",
        "torch",
        "tensorflow",
        "jax",
        "flax",
        "optimum",
        "accelerate",
        "sentence_transformers",
        "tokenizers",
        "huggingface_hub",
        "langchain",
        "langchain_core",
        "langchain_text_splitters",
    ]
    
    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--name", "engine",
        "--add-data", add_data_arg("AI_SYSTEM_AWARENESS.md", "."),
        "python/main.py",
    ]
    
    for imp in hidden_imports:
        cmd.extend(["--hidden-import", imp])
    for module in excluded_modules:
        cmd.extend(["--exclude-module", module])
        
    subprocess.run(cmd, cwd=REPO_ROOT, check=True)
    
    print("Build complete. Moving to backend/...")
    
    source_dir = REPO_ROOT / "dist" / "engine"
    target_dir = REPO_ROOT / "backend"
    destination_dir = target_dir / "engine"
    
    if target_dir.exists():
        shutil.rmtree(target_dir)
        
    shutil.copytree(source_dir, destination_dir)
    
    print(f"Backend successfully staged at {destination_dir}")

if __name__ == "__main__":
    build()
