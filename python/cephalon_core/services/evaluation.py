from __future__ import annotations

import time
import uuid
from typing import Any

from .. import storage


def retrieval_metrics(
    *,
    expected_doc_ids: list[str],
    expected_chunk_ids: list[str],
    retrieved: list[dict[str, Any]],
    k: int,
) -> dict[str, float]:
    window = retrieved[:k]
    expected_docs = set(expected_doc_ids)
    expected_chunks = set(expected_chunk_ids)
    retrieved_docs = [str(item.get("doc_id", "")) for item in window]
    retrieved_chunks = [str(item.get("chunk_id") or item.get("id") or "") for item in window]

    doc_hits = sum(1 for doc_id in expected_docs if doc_id in retrieved_docs)
    chunk_hits = sum(1 for chunk_id in expected_chunks if chunk_id in retrieved_chunks)
    expected_total = len(expected_chunks) if expected_chunks else len(expected_docs)
    hit_total = chunk_hits if expected_chunks else doc_hits

    reciprocal_rank = 0.0
    for idx, item in enumerate(window, start=1):
        doc_match = item.get("doc_id") in expected_docs
        chunk_match = (item.get("chunk_id") or item.get("id")) in expected_chunks
        if chunk_match or (not expected_chunks and doc_match):
            reciprocal_rank = 1.0 / idx
            break

    return {
        "recall_at_k": _ratio(hit_total, expected_total),
        "mrr": round(reciprocal_rank, 6),
        "expected_doc_hit_rate": _ratio(doc_hits, len(expected_docs)),
        "expected_chunk_hit_rate": _ratio(chunk_hits, len(expected_chunks)),
    }


def run_eval_set(conn, eval_items: list[dict[str, Any]], pipeline: str, retrieved_by_id: dict[str, list[dict[str, Any]]], k: int) -> dict[str, Any]:
    run_id = str(uuid.uuid4())
    started = int(time.time())
    rows = []
    for item in eval_items:
        item_id = str(item["id"])
        metrics = retrieval_metrics(
            expected_doc_ids=[str(value) for value in item.get("expected_doc_ids", [])],
            expected_chunk_ids=[str(value) for value in item.get("expected_chunk_ids", [])],
            retrieved=retrieved_by_id.get(item_id, []),
            k=k,
        )
        rows.append({"eval_id": item_id, "question": item.get("question", ""), "metrics": metrics})

    aggregate = {
        "recall_at_k": _avg(row["metrics"]["recall_at_k"] for row in rows),
        "mrr": _avg(row["metrics"]["mrr"] for row in rows),
    }
    storage.save_eval_run(conn, {
        "id": run_id,
        "pipeline": pipeline,
        "top_k": k,
        "created_at": started,
        "aggregate": aggregate,
        "results": rows,
    })
    return storage.get_eval_run(conn, run_id)


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(max(0.0, min(1.0, numerator / denominator)), 6)


def _avg(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return round(sum(values) / len(values), 6)
