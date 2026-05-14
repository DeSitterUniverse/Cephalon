import csv
import json
import os
import time
import re
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


def quality_metrics(
    *,
    query: str,
    answer: str,
    context: str,
    relevant_sentence_count: int,
    total_sentence_count: int,
    supported_statement_count: int,
    total_statement_count: int,
    answer_query_similarity: float,
) -> dict[str, float]:
    """Return bounded numeric QA metrics so later analysis jobs can compare runs."""
    return {
        "context_relevance": _ratio(relevant_sentence_count, total_sentence_count),
        "groundedness": _ratio(supported_statement_count, total_statement_count),
        "answer_relevance": _bounded(answer_query_similarity),
    }


def estimate_answer_quality(query: str, answer: str, context: str) -> dict[str, float]:
    context_sentences = _sentences(context)
    answer_statements = _sentences(answer)
    query_terms = _terms(query)
    relevant_sentences = sum(1 for sentence in context_sentences if _overlap(_terms(sentence), query_terms) > 0)
    context_terms = _terms(context)
    supported_statements = sum(1 for statement in answer_statements if _overlap(_terms(statement), context_terms) >= 0.35)
    answer_similarity = _overlap(_terms(answer), query_terms)
    return quality_metrics(
        query=query,
        answer=answer,
        context=context,
        relevant_sentence_count=relevant_sentences,
        total_sentence_count=len(context_sentences),
        supported_statement_count=supported_statements,
        total_statement_count=len(answer_statements),
        answer_query_similarity=answer_similarity,
    )


def _sentences(text: str) -> list[str]:
    return [part.strip() for part in re.split(r"(?<=[.!?])\s+|\n+", text) if part.strip()]


def _terms(text: str) -> set[str]:
    return {term for term in re.findall(r"[\w]+", text.lower(), flags=re.UNICODE) if len(term) >= 3}


def _overlap(left: set[str], right: set[str]) -> float:
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return _bounded(numerator / denominator)


def _bounded(value: float) -> float:
    return round(max(0.0, min(1.0, float(value))), 6)


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
