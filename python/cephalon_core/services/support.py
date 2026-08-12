from __future__ import annotations

import re
from typing import Any

from ..schemas import SourceChunk
from .claim_verification import strip_hidden_reasoning, verify_answer_claims


SOURCE_TAG_PATTERN = re.compile(r"\[\[\s*src\s*:\s*([A-Za-z0-9_-]+)\s*\]\]", re.IGNORECASE)
SOURCE_LIKE_PATTERN = re.compile(r"\[\[[^\]\n]*src[^\]\n]*(?:\]\]|$)", re.IGNORECASE | re.MULTILINE)


def extract_cited_source_ids(answer_text: str) -> list[str]:
    """Return cited source IDs in first-use order, without duplicates."""
    cited: list[str] = []
    seen: set[str] = set()
    for match in SOURCE_TAG_PATTERN.finditer(answer_text or ""):
        source_id = match.group(1).upper()
        if source_id not in seen:
            cited.append(source_id)
            seen.add(source_id)
    return cited


def classify_citation_support(
    chunk_id: str,
    final_context: list[SourceChunk],
    *,
    supported_rerank: float = 0.45,
    weak_rerank: float = 0.1,
    supported_score: float = 0.55,
) -> dict[str, Any]:
    source = next((item for item in final_context if item.chunk_id == chunk_id), None)
    if source is None:
        return {"chunk_id": chunk_id, "status": "unsupported", "reason": "Citation is not in the final context."}

    rerank = source.rerank_score if source.rerank_score is not None else source.score
    if rerank >= supported_rerank or source.score >= supported_score:
        status = "supported"
        reason = "Citation is present in final context with strong retrieval score."
    elif rerank >= weak_rerank:
        status = "weak"
        reason = "Citation is present in final context but retrieval score is weak."
    else:
        status = "unsupported"
        reason = "Citation score is below support threshold."
    return {
        "chunk_id": chunk_id,
        "source_id": source.source_id,
        "doc_id": source.doc_id,
        "doc_name": source.doc_name,
        "status": status,
        "reason": reason,
        "score": source.score,
        "rerank_score": source.rerank_score,
    }


def classify_answer_support(answer_text: str, sources: list[SourceChunk]) -> dict[str, Any]:
    visible_answer = strip_hidden_reasoning(answer_text)
    raw_tags = [match.group(0) for match in SOURCE_TAG_PATTERN.finditer(visible_answer)]
    cited_source_ids = extract_cited_source_ids(visible_answer)
    malformed_citations = [
        match.group(0)
        for match in SOURCE_LIKE_PATTERN.finditer(visible_answer)
        if SOURCE_TAG_PATTERN.fullmatch(match.group(0)) is None
    ]
    source_by_id = {
        source.source_id.upper(): source
        for source in sources
        if source.source_id
    }
    citations = []
    for source_id in cited_source_ids:
        source = source_by_id.get(source_id)
        if source is None:
            citations.append({
                "chunk_id": f"unknown:{source_id}",
                "source_id": source_id,
                "status": "unsupported",
                "reason": "Citation tag does not identify a source in the final context.",
            })
        else:
            citations.append(classify_citation_support(source.chunk_id, sources))

    if malformed_citations:
        status = "unsupported"
    elif not citations and not sources:
        status = "not_applicable"
    elif not citations:
        status = "unsupported"
    elif any(item["status"] == "unsupported" for item in citations):
        status = "unsupported"
    elif any(item["status"] == "weak" for item in citations):
        status = "weak"
    else:
        status = "supported"

    valid_source_ids = [
        citation["source_id"]
        for citation in citations
        if not str(citation["chunk_id"]).startswith("unknown:")
    ]
    invalid_source_ids = [
        citation["source_id"]
        for citation in citations
        if str(citation["chunk_id"]).startswith("unknown:")
    ]
    available_source_ids = [source_id for source_id in source_by_id]
    duplicate_source_ids = sorted({
        source_id
        for source_id in cited_source_ids
        if sum(1 for raw in raw_tags if SOURCE_TAG_PATTERN.fullmatch(raw).group(1).upper() == source_id) > 1
    })
    uncited_source_ids = sorted(set(available_source_ids) - set(cited_source_ids))
    claim_validation = validate_answer_claims(visible_answer, sources)
    claim_ids_by_source: dict[str, list[str]] = {}
    claim_text_by_source: dict[str, list[str]] = {}
    for claim in claim_validation["claims"]:
        for source_id in claim["source_ids"]:
            claim_ids_by_source.setdefault(source_id, []).append(claim["claim_id"])
            claim_text_by_source.setdefault(source_id, []).append(claim["text"])
    for citation in citations:
        source_id = citation.get("source_id")
        source = source_by_id.get(str(source_id).upper()) if source_id else None
        citation["claim_ids"] = claim_ids_by_source.get(str(source_id), [])
        citation["claims"] = claim_text_by_source.get(str(source_id), [])
        citation["evidence"] = (
            (source.evidence_text if source.evidence_text is not None else source.snippet)
            if source else None
        )
    unused_citation_source_ids = [
        source_id
        for source_id in cited_source_ids
        if not claim_ids_by_source.get(source_id)
    ]
    accounting = {
        "citation_count": len(raw_tags),
        "unique_citation_count": len(cited_source_ids),
        "cited_source_ids": cited_source_ids,
        "valid_source_ids": valid_source_ids,
        "invalid_source_ids": invalid_source_ids,
        "available_source_count": len(available_source_ids),
        "duplicate_source_ids": duplicate_source_ids,
        "malformed_citations": malformed_citations,
        "unused_citation_source_ids": unused_citation_source_ids,
        "uncited_source_ids": uncited_source_ids,
        "uncited_source_count": len(uncited_source_ids),
        "citation_precision": round(len(valid_source_ids) / len(cited_source_ids), 6) if cited_source_ids else 0.0,
    }
    if claim_validation["unsupported_claim_count"] > 0:
        status = "unsupported"
    elif claim_validation["weak_claim_count"] > 0 and status == "supported":
        status = "weak"
    return {
        "status": status,
        "answer_sanitized": visible_answer != str(answer_text or ""),
        "citations": citations,
        "accounting": accounting,
        "claim_validation": claim_validation,
    }


def validate_answer_claims(answer_text: str, sources: list[SourceChunk]) -> dict[str, Any]:
    """Return deterministic entailment plus backward-compatible support aliases."""

    return verify_answer_claims(answer_text, sources)


def sources_from_context(context: str) -> list[SourceChunk]:
    """Reconstruct only the evidence text actually present in a model prompt."""
    matches = list(SOURCE_TAG_PATTERN.finditer(context or ""))
    evidence_by_id: dict[str, list[str]] = {}
    order: list[str] = []
    for index, match in enumerate(matches):
        source_id = match.group(1).upper()
        if source_id not in evidence_by_id:
            evidence_by_id[source_id] = []
            order.append(source_id)
        end = matches[index + 1].start() if index + 1 < len(matches) else len(context)
        evidence = context[match.end():end].strip()
        if evidence:
            evidence_by_id[source_id].append(evidence)
    return [
        SourceChunk(
            rank=index,
            source_id=source_id,
            doc_id=f"context:{source_id}",
            doc_name="Model-visible evidence",
            chunk_id=f"context:{source_id}",
            score=1.0,
            snippet="\n".join(evidence_by_id[source_id]),
            evidence_text="\n".join(evidence_by_id[source_id]),
        )
        for index, source_id in enumerate(order, start=1)
    ]
