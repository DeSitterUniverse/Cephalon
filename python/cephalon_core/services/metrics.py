import csv
import json
import os
import time
from collections import Counter
from pathlib import Path
from typing import Any

from .. import storage


def _metrics_dir(app_state) -> Path:
    path = Path(app_state.settings.metrics_dir)
    path.mkdir(parents=True, exist_ok=True)
    return path


def append_retrieval_event(app_state, event: dict[str, Any]) -> str:
    target = _metrics_dir(app_state) / "retrieval_events.jsonl"
    payload = {"timestamp": int(time.time()), **event}
    with target.open("a", encoding="utf-8") as f:
        f.write(json.dumps(payload, ensure_ascii=False) + "\n")
    return str(target)


def export_corpus_snapshot(app_state) -> str:
    target = _metrics_dir(app_state) / "corpus_snapshots.csv"
    docs = storage.fetchall(app_state.sqlite, "SELECT * FROM documents WHERE type = 'file'")
    chunks = storage.fetchall(app_state.sqlite, "SELECT chunk_length, embedding_dim FROM chunks")
    fts_rows = storage.fetchall(app_state.sqlite, "SELECT count(*) AS count FROM chunks_fts")
    top_retrieved = storage.fetchall(
        app_state.sqlite,
        "SELECT id, retrieval_count FROM documents WHERE type = 'file' ORDER BY retrieval_count DESC, id LIMIT 10",
    )
    doc_type_mix = Counter(os.path.splitext(row["path"])[1].lower() or "none" for row in docs)
    chunk_lengths = [row["chunk_length"] or 0 for row in chunks]
    ready_docs = [row for row in docs if row["status"] == "ready"]
    stale_docs = [row for row in docs if row["stale_embedding"]]
    never_retrieved = [row for row in docs if not row["last_retrieved_at"]]
    index_path = Path(app_state.settings.data_dir) / "lancedb"
    index_size = sum(p.stat().st_size for p in index_path.rglob("*") if p.is_file()) if index_path.exists() else 0

    row = {
        "timestamp": int(time.time()),
        "document_count": len(docs),
        "ready_document_count": len(ready_docs),
        "chunk_count": len(chunks),
        "embedding_count": len(chunks),
        "fts_chunk_count": fts_rows[0]["count"] if fts_rows else 0,
        "average_chunk_length": round(sum(chunk_lengths) / len(chunk_lengths), 2) if chunk_lengths else 0,
        "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0,
        "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0,
        "stale_embedding_count": len(stale_docs),
        "documents_never_retrieved": len(never_retrieved),
        "index_size_bytes": index_size,
        "top_retrieved_docs": json.dumps({row["id"]: row["retrieval_count"] or 0 for row in top_retrieved}, separators=(",", ":")),
        "duplicate_chunk_rate": _duplicate_chunk_rate(app_state),
        "near_duplicate_rate": 0,
        "pdf_count": doc_type_mix[".pdf"],
        "docx_count": doc_type_mix[".docx"],
        "xlsx_count": doc_type_mix[".xlsx"],
        "csv_count": doc_type_mix[".csv"],
        "markdown_count": doc_type_mix[".md"],
        "text_count": doc_type_mix[".txt"],
        "code_count": sum(doc_type_mix[ext] for ext in [".py", ".js", ".ts", ".html", ".json"]),
        "other_count": sum(count for ext, count in doc_type_mix.items() if ext not in {".pdf", ".docx", ".xlsx", ".csv", ".md", ".txt", ".py", ".js", ".ts", ".html", ".json"}),
    }

    exists = target.exists()
    with target.open("a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(row.keys()))
        if not exists:
            writer.writeheader()
        writer.writerow(row)
    return str(target)


def _duplicate_chunk_rate(app_state) -> float:
    rows = storage.fetchall(app_state.sqlite, "SELECT text FROM chunks")
    if not rows:
        return 0.0
    counts = Counter(row["text"] for row in rows)
    duplicate_count = sum(count - 1 for count in counts.values() if count > 1)
    return round(duplicate_count / len(rows), 6)
