import os
import sys
import shutil
import subprocess

def build():
    print("Building FastAPI backend with PyInstaller (--onedir)...")
    
    hidden_imports = [
        "lancedb",
        "tantivy",
        "onnxruntime",
        "transformers",
        "numpy",
        "llama_cpp",
        "uvicorn",
        "docx",
        "openpyxl",
        "pypdf"
    ]
    
    onnx_cross = os.path.expanduser("~/cephalon-data/models/reranker")
    onnx_embed = os.path.expanduser("~/cephalon-data/models/embedder")
    
    if not os.path.exists(onnx_cross) or not os.path.exists(onnx_embed):
        print("ERROR: ONNX models not found. Run 'python export_onnx.py' first.")
        sys.exit(1)
    
    cmd = [
        sys.executable,
        "-m", "PyInstaller",
        "--noconfirm",
        "--onedir",
        "--name", "engine",
        "--collect-all=llama_cpp",
        "--add-data", "AI_SYSTEM_AWARENESS.md;.",
        "--add-data", f"{onnx_cross};onnx_models/reranker",
        "--add-data", f"{onnx_embed};onnx_models/embedder",
        "python/main.py"
    ]
    
    for imp in hidden_imports:
        cmd.extend(["--hidden-import", imp])
        
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
