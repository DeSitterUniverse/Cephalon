"""Create a bounded request-scoped ledger from selected retrieval evidence.

The ledger is the control-plane contract shared by context selection, gap
retrieval, and answer verification.  This first implementation is deliberately
observability-only: it reads deterministic subqueries and final sources but
does not change ranking, context, prompts, or generation.

At most eight requirements and twenty sources are examined. Requirement/source
assignment is O(r*s), and potential-conflict comparison is hard-capped at 64
pairs. Evidence excerpts are capped at 500 characters so a normal trace stays
well below the 32 KiB operational target.
"""

from __future__ import annotations

import re
from typing import Any

from ..schemas import SourceChunk


MAX_REQUIREMENTS = 8
MAX_LEDGER_SOURCES = 20
MAX_EVIDENCE_PER_REQUIREMENT = 4
MAX_CONFLICT_COMPARISONS = 64
MAX_EVIDENCE_CHARS = 500
SUFFICIENT_TERM_COVERAGE = 0.34
PARTIAL_TERM_COVERAGE = 0.12

LEDGER_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "the", "to", "was", "were", "what", "when", "which", "who",
    "with", "versus", "vs", "compare", "explain", "describe", "paper", "study",
}
NUMBER_WITH_UNIT_RE = re.compile(r"(?<!\w)(-?\d+(?:\.\d+)?)\s*(%|[a-zA-Zµμ°]+)?")
NEGATION_RE = re.compile(r"\b(?:no|not|never|neither|without|failed to|did not|does not)\b", re.IGNORECASE)


def build_evidence_ledger(
    query_id: str,
    raw_query: str,
    subqueries: list[dict[str, str]],
    sources: list[SourceChunk],
    *,
    retrieval_round: int = 0,
) -> dict[str, Any]:
    """Build and return a compact JSON-compatible evidence ledger.

    A direct ``subquery_id`` match is treated as an intentional assignment;
    otherwise material-term coverage supplies a deterministic fallback.
    Conflict records are conservative diagnostics, not factual judgments: they
    require high lexical similarity plus either opposite negation or differing
    numeric values with the same explicit unit.
    """

    requirements = plan_requirements(raw_query, subqueries)
    evidence: list[dict[str, Any]] = []
    assignments: dict[str, list[tuple[str, float]]] = {item["id"]: [] for item in requirements}
    for index, source in enumerate(sources[:MAX_LEDGER_SOURCES], start=1):
        evidence_id = f"E{index}"
        text = (source.evidence_text or source.snippet or "").strip()
        assigned_ids: list[str] = []
        for requirement in requirements:
            coverage = _coverage(requirement["text"], text)
            direct = requirement["subquery_id"] in _source_subquery_ids(source)
            if direct or coverage >= PARTIAL_TERM_COVERAGE:
                score = max(coverage, 0.5 if direct else 0.0)
                assignments[requirement["id"]].append((evidence_id, round(score, 6)))
                assigned_ids.append(requirement["id"])
        source.evidence_ids = [evidence_id]
        source.requirement_ids = assigned_ids
        source.retrieval_round = retrieval_round
        evidence.append({
            "id": evidence_id,
            "source_id": source.source_id,
            "chunk_id": source.chunk_id,
            "doc_id": source.doc_id,
            "source_kind": source.source_kind or "text",
            "requirement_ids": assigned_ids,
            "retrieval_round": retrieval_round,
            "span": text[:MAX_EVIDENCE_CHARS],
            "page_number": source.page_number,
            "parent_id": source.parent_id,
            "status": "supporting" if assigned_ids else "unassigned",
        })

    conflicts = _potential_conflicts(evidence)
    conflicting_requirements = {
        requirement_id
        for conflict in conflicts
        for requirement_id in conflict["requirement_ids"]
    }
    requirement_records: list[dict[str, Any]] = []
    for requirement in requirements:
        ranked = sorted(assignments[requirement["id"]], key=lambda item: item[1], reverse=True)[:MAX_EVIDENCE_PER_REQUIREMENT]
        best = ranked[0][1] if ranked else 0.0
        if requirement["id"] in conflicting_requirements:
            status = "conflicting"
        elif best >= SUFFICIENT_TERM_COVERAGE:
            status = "sufficient"
        elif ranked:
            status = "partial"
        else:
            status = "missing"
        requirement_records.append({
            **requirement,
            "status": status,
            "evidence_ids": [evidence_id for evidence_id, _ in ranked],
            "best_coverage": round(best, 6),
        })

    counts = {
        status: sum(item["status"] == status for item in requirement_records)
        for status in ("sufficient", "partial", "missing", "conflicting")
    }
    return {
        "ledger_id": query_id,
        "query": raw_query,
        "state": "assessed",
        "retrieval_round": retrieval_round,
        "requirements": requirement_records,
        "evidence": evidence,
        "conflicts": conflicts,
        "summary": {**counts, "requirement_count": len(requirement_records), "evidence_count": len(evidence)},
        "limits": {
            "max_requirements": MAX_REQUIREMENTS,
            "max_sources": MAX_LEDGER_SOURCES,
            "max_evidence_per_requirement": MAX_EVIDENCE_PER_REQUIREMENT,
            "max_conflict_comparisons": MAX_CONFLICT_COMPARISONS,
            "max_evidence_chars": MAX_EVIDENCE_CHARS,
        },
    }


def plan_requirements(raw_query: str, subqueries: list[dict[str, str]]) -> list[dict[str, str]]:
    """Return the stable deterministic requirement plan used by control stages."""

    components = [item for item in subqueries if item.get("id") != "q0"]
    if not components:
        components = subqueries[:1] or [{"id": "q1", "text": raw_query}]
    return [
        {"id": f"R{index}", "subquery_id": str(item.get("id") or f"q{index}"), "text": str(item.get("text") or raw_query)}
        for index, item in enumerate(components[:MAX_REQUIREMENTS], start=1)
    ]


def _source_subquery_ids(source: SourceChunk) -> set[str]:
    return {value.strip() for value in (source.subquery_id or "").split(",") if value.strip()}


def _terms(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[\wµμ]+", text.lower(), flags=re.UNICODE)
        if len(token) >= 3 and token not in LEDGER_STOPWORDS
    }


def _coverage(requirement: str, evidence: str) -> float:
    required = _terms(requirement)
    if not required:
        return 0.0
    return len(required & _terms(evidence)) / len(required)


def _potential_conflicts(evidence: list[dict[str, Any]]) -> list[dict[str, Any]]:
    conflicts: list[dict[str, Any]] = []
    comparisons = 0
    for left_index, left in enumerate(evidence):
        for right in evidence[left_index + 1:]:
            if comparisons >= MAX_CONFLICT_COMPARISONS:
                return conflicts
            comparisons += 1
            shared_requirements = sorted(set(left["requirement_ids"]) & set(right["requirement_ids"]))
            if not shared_requirements or left["doc_id"] == right["doc_id"]:
                continue
            left_terms, right_terms = _terms(left["span"]), _terms(right["span"])
            union = left_terms | right_terms
            similarity = len(left_terms & right_terms) / len(union) if union else 0.0
            if similarity < 0.45:
                continue
            reason = _conflict_reason(left["span"], right["span"])
            if reason:
                conflicts.append({
                    "id": f"C{len(conflicts) + 1}",
                    "evidence_ids": [left["id"], right["id"]],
                    "requirement_ids": shared_requirements,
                    "status": "potential",
                    "reason": reason,
                })
    return conflicts


def _conflict_reason(left: str, right: str) -> str | None:
    if bool(NEGATION_RE.search(left)) != bool(NEGATION_RE.search(right)):
        return "highly similar evidence has opposite negation polarity"
    left_values = {(unit.lower(), float(value)) for value, unit in NUMBER_WITH_UNIT_RE.findall(left) if unit}
    right_values = {(unit.lower(), float(value)) for value, unit in NUMBER_WITH_UNIT_RE.findall(right) if unit}
    for unit, left_value in left_values:
        same_unit = [value for candidate_unit, value in right_values if candidate_unit == unit]
        if same_unit and all(abs(left_value - value) > max(1e-9, abs(left_value) * 0.001) for value in same_unit):
            return f"highly similar evidence reports different values with unit {unit}"
    return None
