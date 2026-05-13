import os

from fastapi import HTTPException

from .config import DOCUMENT_ID_PATTERN, SUPPORTED_EXTENSIONS


def is_supported_file(path: str) -> bool:
    return os.path.splitext(path)[1].lower() in SUPPORTED_EXTENSIONS


def validate_model_filename(model_filename: str, model_dir: str) -> str:
    if not model_filename or model_filename != os.path.basename(model_filename):
        raise HTTPException(status_code=400, detail="Model name must be a local .gguf filename.")
    if not model_filename.lower().endswith(".gguf"):
        raise HTTPException(status_code=400, detail="Model name must end with .gguf.")
    model_path = os.path.abspath(os.path.join(model_dir, model_filename))
    model_root = os.path.abspath(model_dir)
    if os.path.commonpath([model_root, model_path]) != model_root:
        raise HTTPException(status_code=400, detail="Model path escapes the configured model directory.")
    if not os.path.isfile(model_path):
        raise HTTPException(status_code=404, detail=f"Model not found: {model_filename}")
    return model_path


def validate_document_id(doc_id: str) -> str:
    if not DOCUMENT_ID_PATTERN.fullmatch(doc_id):
        raise HTTPException(status_code=400, detail="Invalid document id.")
    return doc_id


def normalize_existing_path(path: str) -> str:
    requested_path = path.strip()
    if not requested_path:
        raise HTTPException(status_code=400, detail="Path is required.")
    target_path = os.path.abspath(os.path.expanduser(requested_path))
    if not os.path.exists(target_path):
        raise HTTPException(status_code=404, detail="Path not found.")
    return target_path


def validate_tag(tag: str) -> str:
    normalized = tag.strip().lower()
    if not normalized:
        raise HTTPException(status_code=400, detail="Tag is required.")
    if len(normalized) > 40:
        raise HTTPException(status_code=400, detail="Tag must be 40 characters or shorter.")
    if any(ch in normalized for ch in "'\"\\/"):
        raise HTTPException(status_code=400, detail="Tag contains unsupported characters.")
    return normalized
