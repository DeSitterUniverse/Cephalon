import csv
import hashlib
import os
import time
import uuid

import docx
import openpyxl
import pptx
from pypdf import PdfReader

from .. import storage
from ..validators import is_supported_file


def get_file_hash(path: str) -> str:
    hasher = hashlib.sha256()
    with open(path, "rb") as f:
        hasher.update(f.read())
    return hasher.hexdigest()


def file_metadata(path: str) -> tuple[int, int]:
    stat = os.stat(path)
    return stat.st_size, int(stat.st_mtime)


def find_existing_doc_by_hash(sqlite_conn, content_hash: str):
    return storage.fetchone(
        sqlite_conn,
        "SELECT id, path, status, chunk_count FROM documents WHERE content_hash = ? AND type = 'file' AND status IN ('ready', 'ingesting') LIMIT 1",
        (content_hash,),
    )


def looks_like_text(path: str, sample_size: int = 8192) -> bool:
    with open(path, "rb") as f:
        sample = f.read(sample_size)
    if not sample:
        return True
    if b"\x00" in sample:
        return False
    control_bytes = sum(1 for byte in sample if byte < 32 and byte not in {9, 10, 13})
    return control_bytes / len(sample) < 0.08


def read_text_fallback(path: str) -> str:
    for encoding in ("utf-8-sig", "utf-16", "utf-8", "latin-1"):
        try:
            with open(path, "r", encoding=encoding) as f:
                return f.read()
        except UnicodeError:
            continue
    with open(path, "r", encoding="latin-1", errors="replace") as f:
        return f.read()


def extract_text(path: str, force_text: bool = False) -> tuple[str, str]:
    ext = os.path.splitext(path)[1].lower()

    if force_text and not looks_like_text(path):
        raise ValueError("File appears to be binary and cannot be safely imported as text.")
    if force_text:
        return read_text_fallback(path), "text"

    if ext == ".pdf":
        return "\n".join([page.extract_text() for page in PdfReader(path).pages if page.extract_text()]), "native"
    if ext == ".docx":
        doc = docx.Document(path)
        return "\n".join([para.text for para in doc.paragraphs]), "native"
    if ext == ".pptx":
        prs = pptx.Presentation(path)
        text_runs = []
        for slide in prs.slides:
            for shape in slide.shapes:
                if hasattr(shape, "text"):
                    text_runs.append(shape.text)
        return "\n".join(text_runs), "native"
    if ext == ".xlsx":
        wb = openpyxl.load_workbook(path, data_only=True)
        text_runs = []
        for sheet in wb.worksheets:
            text_runs.append(f"--- Sheet: {sheet.title} ---")
            for row in sheet.iter_rows(values_only=True):
                row_text = "\t".join([str(cell) if cell is not None else "" for cell in row])
                if row_text.strip():
                    text_runs.append(row_text)
        return "\n".join(text_runs), "native"
    if ext == ".csv":
        text_runs = []
        with open(path, newline="", encoding="utf-8") as f:
            reader = csv.reader(f)
            for row in reader:
                text_runs.append("\t".join(row))
        return "\n".join(text_runs), "native"

    if not looks_like_text(path):
        raise ValueError("Unknown file type appears to be binary and cannot be safely imported as text.")
    return read_text_fallback(path), "text"


def collect_supported_files(path: str, force_text: bool = False) -> list[str]:
    if os.path.isfile(path):
        return [path]

    files: list[str] = []
    for root, _, names in os.walk(path):
        for name in names:
            full_path = os.path.join(root, name)
            if is_supported_file(full_path) or looks_like_text(full_path) or force_text:
                files.append(full_path)
    return sorted(files)


def register_ingesting_document(sqlite_conn, path: str, content_hash: str, extraction_mode: str = "native", doc_id: str | None = None, embedding_metadata: dict | None = None) -> str:
    doc_id = doc_id or str(uuid.uuid4())
    embedding_metadata = embedding_metadata or storage.active_embedding_metadata()
    size_bytes, modified_at = file_metadata(path)
    storage.execute(
        sqlite_conn,
        """
        INSERT INTO documents
            (id, path, display_name, content_hash, ingested_at, chunk_count, status, type, size_bytes, modified_at, last_indexed_at, extraction_mode, embedding_model_id, embedding_dim, stale_embedding)
        VALUES (?, ?, COALESCE((SELECT display_name FROM documents WHERE id = ?), ?), ?, ?, ?, ?, 'file', ?, ?, ?, ?, ?, ?, 0)
        ON CONFLICT(id) DO UPDATE SET
            path = excluded.path,
            content_hash = excluded.content_hash,
            ingested_at = excluded.ingested_at,
            chunk_count = excluded.chunk_count,
            status = excluded.status,
            size_bytes = excluded.size_bytes,
            modified_at = excluded.modified_at,
            last_indexed_at = excluded.last_indexed_at,
            extraction_mode = excluded.extraction_mode,
            embedding_model_id = excluded.embedding_model_id,
            embedding_dim = excluded.embedding_dim,
            stale_embedding = 0,
            last_error = NULL
        """,
        (
            doc_id,
            path,
            doc_id,
            os.path.basename(path),
            content_hash,
            int(time.time()),
            0,
            "ingesting",
            size_bytes,
            modified_at,
            int(time.time()),
            extraction_mode,
            embedding_metadata["embedding_model_id"],
            embedding_metadata["embedding_dim"],
        ),
    )
    return doc_id


def mark_document_ready(sqlite_conn, doc_id: str, chunk_count: int) -> None:
    storage.execute(
        sqlite_conn,
        "UPDATE documents SET status = 'ready', chunk_count = ?, last_error = NULL, last_indexed_at = ? WHERE id = ?",
        (chunk_count, int(time.time()), doc_id),
    )


def mark_document_failed(sqlite_conn, doc_id: str, error: str) -> None:
    storage.execute(
        sqlite_conn,
        "UPDATE documents SET status = ?, chunk_count = 0, last_error = ? WHERE id = ?",
        (f"failed: {error[:50]}", error, doc_id),
    )
