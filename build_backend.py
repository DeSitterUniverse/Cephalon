import os
import sys
import shutil
import subprocess

def build():
    print("Building FastAPI backend with PyInstaller (--onedir)...")
    
    hidden_imports = [
        "lancedb",
        "onnxruntime",
        "transformers",
        "numpy",
        "llama_cpp",
        "huggingface_hub",
        "uvicorn",
        "docx",
        "openpyxl",
        "pypdf"
    ]
    excluded_modules = [
        "torch",
        "tensorflow",
        "jax",
        "flax",
        "optimum",
        "accelerate",
    ]
    
    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--name", "engine",
        "--collect-all=llama_cpp",
        "--add-data", "AI_SYSTEM_AWARENESS.md;.",
        "--add-data", "CEPHALON_ARCHITECTURE_DEEP_DIVE.html;.",
        "python/main.py"
    ]
    
    for imp in hidden_imports:
        cmd.extend(["--hidden-import", imp])
    for module in excluded_modules:
        cmd.extend(["--exclude-module", module])
        
    subprocess.run(cmd, check=True)
    
    print("Build complete. Moving to src-tauri/backend...")
    
    source_dir = os.path.join("dist", "engine")
    target_dir = os.path.join("src-tauri", "backend")
    
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)
        
    shutil.copytree(source_dir, os.path.join(target_dir, "engine"))
    
    print("Backend successfully staged at src-tauri/backend/engine/")

if __name__ == "__main__":
    build()
