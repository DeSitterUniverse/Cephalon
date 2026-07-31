"""Deterministic retrieval and answer metrics for Cephalon RAG evaluations.

The benchmark deliberately avoids using the answer model as its sole judge.
Human-authored evidence targets, lexical requirements, numeric assertions, and
citation tags produce repeatable primary scores. Model-based semantic judging
can be recorded separately by the benchmark runner, but it never replaces these
metrics or controls whether a stacked PR passes its regression gate.
"""

from __future__ import annotations

import math
import re
import time
import uuid
from collections import defaultdict
from typing import Any, Iterable

from .. import storage

CITATION_PATTERN = re.compile(r"\[\[src:([A-Za-z0-9_-]+)\]\]", re.IGNORECASE)
NUMBER_PATTERN = re.compile(
    r"(?<![\w.])[-+]?(?:\d+(?:,\d{3})*(?:\.\d+)?|\.\d+)(?:[eE][-+]?\d+)?"
)
REFUSAL_MARKERS = (
    "insufficient evidence",
    "not enough evidence",
    "cannot determine",
    "can't determine",
    "not provided in the sources",
)


def retrieval_metrics(
    *,
    expected_doc_ids: list[str],
    expected_chunk_ids: list[str],
    retrieved: list[dict[str, Any]],
    k: int,
) -> dict[str, float]:
    """Score a ranked retrieval window against document or chunk targets.

    Chunk targets take precedence when present because they are the more
    specific evidence label. nDCG uses binary relevance and therefore remains
    interpretable across cases with different numbers of gold targets.
    """

    window = retrieved[:k]
    expected_docs = {str(value) for value in expected_doc_ids if str(value)}
    expected_chunks = {str(value) for value in expected_chunk_ids if str(value)}
    retrieved_docs = [str(item.get("doc_id", "")) for item in window]
    retrieved_chunks = [str(item.get("chunk_id") or item.get("id") or "") for item in window]

    doc_hits = len(expected_docs.intersection(retrieved_docs))
    chunk_hits = len(expected_chunks.intersection(retrieved_chunks))
    target_ids = expected_chunks or expected_docs
    ranked_ids = retrieved_chunks if expected_chunks else retrieved_docs
    # A gold document can contribute several retrieved child chunks. Counting
    # every child as a separate hit lets recall and nDCG exceed 1.0 and rewards
    # redundant context. Credit each labelled target once, at its best rank;
    # later occurrences remain in the precision denominator as redundancy.
    credited_targets: set[str] = set()
    relevance = []
    for value in ranked_ids:
        is_new_target = value in target_ids and value not in credited_targets
        relevance.append(1 if is_new_target else 0)
        if is_new_target:
            credited_targets.add(value)

    reciprocal_rank = 0.0
    for idx, relevant in enumerate(relevance, start=1):
        if relevant:
            reciprocal_rank = 1.0 / idx
            break

    relevant_result_count = sum(relevance)
    return {
        "recall_at_k": _ratio(relevant_result_count, len(target_ids)),
        "precision_at_k": _ratio(relevant_result_count, len(window)),
        "mrr": round(reciprocal_rank, 6),
        "ndcg_at_k": _ndcg(relevance, min(len(target_ids), k)),
        "expected_doc_hit_rate": _ratio(doc_hits, len(expected_docs)),
        "expected_chunk_hit_rate": _ratio(chunk_hits, len(expected_chunks)),
    }


def answer_metrics(
    *,
    item: dict[str, Any],
    answer: str,
    sources: list[dict[str, Any]],
) -> dict[str, float]:
    """Compute deterministic answer, requirement, citation, and numeric scores."""

    clean_answer = str(answer or "").strip()
    answer_lower = clean_answer.casefold()
    requirements = item.get("requirements", [])
    matched_requirements = sum(1 for requirement in requirements if _requirement_matches(requirement, answer_lower))
    accepted_answers = [str(value).casefold() for value in item.get("accepted_answers", []) if str(value).strip()]
    accepted_match = not accepted_answers or any(value in answer_lower for value in accepted_answers)
    expected_refusal = bool(item.get("expected_refusal"))
    refused = any(marker in answer_lower for marker in REFUSAL_MARKERS)

    numeric_assertions = item.get("numeric_assertions", [])
    numeric_hits = sum(1 for assertion in numeric_assertions if _numeric_assertion_matches(assertion, clean_answer))

    cited_source_ids = {match.group(1).upper() for match in CITATION_PATTERN.finditer(clean_answer)}
    available_source_ids = {
        str(source.get("source_id", "")).upper()
        for source in sources
        if str(source.get("source_id", "")).strip()
    }
    valid_citations = cited_source_ids.intersection(available_source_ids)
    invalid_citations = cited_source_ids.difference(available_source_ids)
    source_doc_ids = {
        str(source.get("doc_id", ""))
        for source in sources
        if str(source.get("doc_id", "")).strip()
    }
    expected_doc_ids = {str(value) for value in item.get("expected_doc_ids", []) if str(value)}
    expected_sources = sum(1 for doc_id in expected_doc_ids if doc_id in source_doc_ids)

    # Metrics with no applicable denominator are omitted instead of emitted as
    # zero. Otherwise, for example, 104 non-numeric cases would dilute the 16
    # numeric cases and make their accuracy impossible to interpret.
    metrics = {
        "correct_refusal": 1.0 if refused == expected_refusal else 0.0,
        "over_refusal": 1.0 if refused and not expected_refusal else 0.0,
    }
    if requirements:
        metrics["requirement_coverage"] = _ratio(matched_requirements, len(requirements))
    if accepted_answers:
        metrics["accepted_answer_match"] = 1.0 if accepted_match else 0.0
    if numeric_assertions:
        metrics["numeric_accuracy"] = _ratio(numeric_hits, len(numeric_assertions))
    if cited_source_ids:
        metrics["citation_precision"] = _ratio(len(valid_citations), len(cited_source_ids))
        metrics["invalid_citation_rate"] = _ratio(len(invalid_citations), len(cited_source_ids))
    if expected_doc_ids:
        metrics["citation_source_recall"] = _ratio(expected_sources, len(expected_doc_ids))
    return metrics


def run_eval_set(
    conn,
    eval_items: list[dict[str, Any]],
    pipeline: str,
    retrieved_by_id: dict[str, list[dict[str, Any]]],
    k: int,
    *,
    answers_by_id: dict[str, str] | None = None,
    sources_by_id: dict[str, list[dict[str, Any]]] | None = None,
    run_meta: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Persist an evaluation while preserving the legacy retrieval-only API."""

    run_id = str(uuid.uuid4())
    started = int(time.time())
    answers_by_id = answers_by_id or {}
    sources_by_id = sources_by_id or retrieved_by_id
    rows = []
    for item in eval_items:
        item_id = str(item["id"])
        metrics = retrieval_metrics(
            expected_doc_ids=[str(value) for value in item.get("expected_doc_ids", [])],
            expected_chunk_ids=[str(value) for value in item.get("expected_chunk_ids", [])],
            retrieved=retrieved_by_id.get(item_id, []),
            k=k,
        )
        if item_id in answers_by_id:
            metrics.update(
                answer_metrics(
                    item=item,
                    answer=answers_by_id[item_id],
                    sources=sources_by_id.get(item_id, []),
                )
            )
        rows.append(
            {
                "eval_id": item_id,
                "question": item.get("question", ""),
                "case": item,
                "answer": answers_by_id.get(item_id),
                "metrics": metrics,
            }
        )

    metric_names = sorted({name for row in rows for name in row["metrics"]})
    aggregate = {name: _avg(row["metrics"][name] for row in rows if name in row["metrics"]) for name in metric_names}
    aggregate["case_count"] = len(rows)
    aggregate["domain_breakdown"] = _grouped_metrics(rows, "domain")
    aggregate["category_breakdown"] = _grouped_metrics(rows, "category")
    storage.save_eval_run(
        conn,
        {
            "id": run_id,
            "pipeline": pipeline,
            "top_k": k,
            "created_at": started,
            "aggregate": aggregate,
            "meta": run_meta or {},
            "results": rows,
        },
    )
    return storage.get_eval_run(conn, run_id)


def _requirement_matches(requirement: dict[str, Any], answer_lower: str) -> bool:
    terms = [str(value).casefold() for value in requirement.get("required_terms", []) if str(value).strip()]
    if not terms:
        return False
    matches = [term in answer_lower for term in terms]
    return any(matches) if requirement.get("match_mode") == "any" else all(matches)


def _numeric_assertion_matches(assertion: dict[str, Any], answer: str) -> bool:
    expected = float(assertion["expected_value"])
    tolerance = max(0.0, float(assertion.get("tolerance", 0.0)))
    unit = str(assertion.get("unit") or "").strip().casefold()
    if unit and unit not in answer.casefold():
        return False
    for match in NUMBER_PATTERN.finditer(answer):
        try:
            observed = float(match.group(0).replace(",", ""))
        except ValueError:
            continue
        if abs(observed - expected) <= tolerance:
            return True
    return False


def _ndcg(relevance: list[int], ideal_relevant_count: int) -> float:
    if ideal_relevant_count <= 0:
        return 0.0
    dcg = sum(value / math.log2(index + 2) for index, value in enumerate(relevance))
    ideal = sum(1.0 / math.log2(index + 2) for index in range(ideal_relevant_count))
    return round(dcg / ideal, 6) if ideal else 0.0


def _grouped_metrics(rows: list[dict[str, Any]], field: str) -> dict[str, dict[str, float]]:
    groups: dict[str, list[dict[str, float]]] = defaultdict(list)
    for row in rows:
        key = str(row.get("case", {}).get(field) or "unspecified")
        groups[key].append(row["metrics"])
    result: dict[str, dict[str, float]] = {}
    for key, metrics_rows in sorted(groups.items()):
        names = sorted({name for metrics in metrics_rows for name in metrics})
        result[key] = {
            name: _avg(metrics[name] for metrics in metrics_rows if name in metrics)
            for name in names
        }
        result[key]["case_count"] = len(metrics_rows)
    return result


def _ratio(numerator: int, denominator: int) -> float:
    if denominator <= 0:
        return 0.0
    return round(max(0.0, min(1.0, numerator / denominator)), 6)


def _avg(values: Iterable[float]) -> float:
    materialized = list(values)
    if not materialized:
        return 0.0
    return round(sum(materialized) / len(materialized), 6)
