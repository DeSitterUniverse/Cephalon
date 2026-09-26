from __future__ import annotations

import hashlib
import json
import os
import statistics
from typing import Any

from .. import storage
from ..schemas import SourceChunk


DEFAULT_NO_ANSWER_THRESHOLDS = {
    "min_confidence": 0.35,
    "min_rerank_score": 0.15,
    "min_vector_score": 0.05,
    "min_source_count": 1,
}


def detect_stale_state(stored: dict[str, Any], current: dict[str, Any]) -> dict[str, Any]:
    checks = [
        ("content_hash", "file_hash_changed"),
        ("chunking_config_hash", "chunking_config_changed"),
        ("parser_version", "parser_version_changed"),
        ("embedding_model_id", "embedding_model_changed"),
        ("embedding_config_hash", "embedding_config_changed"),
    ]
    reasons = [reason for key, reason in checks if stored.get(key) != current.get(key)]
    return {"stale": bool(reasons), "reasons": reasons}


def chunking_config_hash(profile: str, settings: dict[str, Any]) -> str:
    payload = json.dumps({"profile": profile, "settings": settings}, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8", errors="replace")).hexdigest()


def no_answer_diagnostics(sources: list[SourceChunk], thresholds: dict[str, Any] | None = None) -> dict[str, Any]:
    active = {**DEFAULT_NO_ANSWER_THRESHOLDS, **(thresholds or {})}
    if not sources:
        return {
            "confidence": 0.0,
            "uncertainty": "high",
            "no_answer": True,
            "reason": "No retrieved sources.",
            "reasons": ["no_sources"],
            "thresholds": active,
            "agreement": {"hybrid_overlap": False, "source_diversity": 0},
            "top_scores": {},
        }

    rerank_scores = [source.rerank_score for source in sources if source.rerank_score is not None]
    vector_scores = [source.vector_score for source in sources if source.vector_score is not None]
    bm25_scores = [source.lexical_score for source in sources if source.lexical_score is not None]
    final_scores = [source.score for source in sources]
    top_rerank = max(rerank_scores) if rerank_scores else max(final_scores)
    top_vector = max(vector_scores) if vector_scores else None
    top_bm25 = min(bm25_scores) if bm25_scores else None
    ranked = sorted(final_scores, reverse=True)
    score_gap = ranked[0] - ranked[1] if len(ranked) > 1 else ranked[0]
    source_diversity = len({source.doc_id for source in sources})
    hybrid_overlap = any(source.vector_score is not None and source.lexical_score is not None for source in sources)

    confidence = 0.25
    confidence += min(max(top_rerank, 0.0), 1.5) * 0.22
    if top_vector is not None:
        confidence += min(max(top_vector, 0.0), 1.0) * 0.18
    elif top_rerank >= float(active["min_rerank_score"]):
        confidence += 0.08
    confidence += min(max(score_gap, 0.0), 1.0) * 0.12
    confidence += min(source_diversity, 3) * 0.06
    if hybrid_overlap:
        confidence += 0.12
    confidence = round(max(0.0, min(1.0, confidence)), 4)

    reasons: list[str] = []
    if len(sources) < int(active["min_source_count"]):
        reasons.append("too_few_sources")
    if top_rerank < float(active["min_rerank_score"]):
        reasons.append("low_rerank_score")
    if top_vector is not None and top_vector < float(active["min_vector_score"]) and top_rerank < float(active["min_rerank_score"]):
        reasons.append("low_vector_score")
    if not hybrid_overlap and len(sources) > 1 and confidence < 0.55 and top_rerank < float(active["min_rerank_score"]):
        reasons.append("low_vector_bm25_agreement")
    if confidence < float(active["min_confidence"]):
        reasons.append("low_confidence")

    blocking_reasons = {"no_sources", "too_few_sources", "low_confidence"}
    no_answer = any(reason in blocking_reasons for reason in reasons)
    return {
        "confidence": confidence,
        "uncertainty": "high" if confidence < 0.45 else "medium" if confidence < 0.7 else "low",
        "no_answer": no_answer,
        "reason": "Weak retrieval evidence." if no_answer else "Retrieved sources are usable.",
        "reasons": reasons,
        "thresholds": active,
        "agreement": {"hybrid_overlap": hybrid_overlap, "source_diversity": source_diversity},
        "top_scores": {
            "rerank": round(float(top_rerank), 6),
            "vector": round(float(top_vector), 6) if top_vector is not None else None,
            "bm25": round(float(top_bm25), 6) if top_bm25 is not None else None,
            "score_gap": round(float(score_gap), 6),
        },
    }


def index_health(app_state) -> dict[str, Any]:
    documents = storage.fetchone(app_state.sqlite, """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN substr(status, 1, 6) = 'failed' THEN 1 ELSE 0 END) AS failed,
               SUM(CASE WHEN stale_embedding THEN 1 ELSE 0 END) AS stale,
               SUM(CASE WHEN parse_warnings IS NOT NULL AND parse_warnings != '' THEN 1 ELSE 0 END) AS parse_warnings,
               SUM(CASE WHEN last_retrieved_at IS NULL OR last_retrieved_at = 0 THEN 1 ELSE 0 END) AS never_retrieved
        FROM documents WHERE type = 'file'
    """)
    chunk_summary = storage.fetchone(app_state.sqlite, """
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN embedding_model_id IS NOT NULL AND embedding_model_id != '' THEN 1 ELSE 0 END) AS embedded
        FROM chunks
    """)
    chunk_lengths = [row[0] for row in storage.fetchall(app_state.sqlite, """
        SELECT COALESCE(NULLIF(chunk_length, 0), NULLIF(char_count, 0), LENGTH(text), 0)
        FROM chunks
    """)]
    duplicate_chunks = storage.fetchone(app_state.sqlite, """
        SELECT COALESCE(SUM(copies - 1), 0) AS duplicate_count FROM (
            SELECT COUNT(*) AS copies FROM chunks
            WHERE text_hash IS NOT NULL AND text_hash != ''
            GROUP BY text_hash HAVING COUNT(*) > 1
        )
    """)["duplicate_count"]
    model_counts = {
        row["name"]: row["count"] for row in storage.fetchall(app_state.sqlite, """
            SELECT COALESCE(NULLIF(embedding_model_id, ''), 'unknown') AS name, COUNT(*) AS count
            FROM chunks GROUP BY name
        """)
    }
    profile_counts = {
        row["name"]: row["count"] for row in storage.fetchall(app_state.sqlite, """
            SELECT COALESCE(NULLIF(chunking_profile, ''), 'unknown') AS name, COUNT(*) AS count
            FROM chunks GROUP BY name
        """)
    }
    top_documents = storage.fetchall(app_state.sqlite, """
        SELECT id, display_name, path, retrieval_count FROM documents
        WHERE type = 'file' ORDER BY COALESCE(retrieval_count, 0) DESC, rowid LIMIT 10
    """)
    index_path = os.path.join(app_state.settings.data_dir, "lancedb")
    index_size = 0
    if os.path.exists(index_path):
        for root, _, files in os.walk(index_path):
            for name in files:
                try:
                    index_size += os.path.getsize(os.path.join(root, name))
                except OSError:
                    pass

    return {
        "document_count": documents["total"],
        "chunk_count": chunk_summary["total"],
        "embedded_chunk_count": chunk_summary["embedded"] or 0,
        "stale_document_count": documents["stale"] or 0,
        "failed_ingestion_count": documents["failed"] or 0,
        "parse_warning_count": documents["parse_warnings"] or 0,
        "duplicate_chunk_count": duplicate_chunks,
        "duplicate_chunk_rate": round(duplicate_chunks / chunk_summary["total"], 6) if chunk_summary["total"] else 0.0,
        "average_chunk_length": round(sum(chunk_lengths) / len(chunk_lengths), 2) if chunk_lengths else 0,
        "median_chunk_length": statistics.median(chunk_lengths) if chunk_lengths else 0,
        "min_chunk_length": min(chunk_lengths) if chunk_lengths else 0,
        "max_chunk_length": max(chunk_lengths) if chunk_lengths else 0,
        "documents_never_retrieved": documents["never_retrieved"] or 0,
        "index_size_bytes": index_size,
        "embedding_model_counts": dict(model_counts),
        "chunking_profile_counts": dict(profile_counts),
        "top_retrieved_documents": [
            {"id": row["id"], "name": row["display_name"] or os.path.basename(row["path"]), "retrieval_count": row["retrieval_count"] or 0}
            for row in top_documents
        ],
    }
