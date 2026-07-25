from __future__ import annotations

import re
from typing import Any

from ..schemas import SourceChunk


SOURCE_TAG_PATTERN = re.compile(r"\[\[\s*src\s*:\s*([A-Za-z0-9_-]+)\s*\]\]", re.IGNORECASE)


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
    cited_source_ids = extract_cited_source_ids(answer_text)
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

    if not citations:
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
    accounting = {
        "citation_count": len(SOURCE_TAG_PATTERN.findall(answer_text or "")),
        "unique_citation_count": len(cited_source_ids),
        "cited_source_ids": cited_source_ids,
        "valid_source_ids": valid_source_ids,
        "invalid_source_ids": invalid_source_ids,
        "available_source_count": len(available_source_ids),
        "uncited_source_count": len(set(available_source_ids) - set(cited_source_ids)),
        "citation_precision": round(len(valid_source_ids) / len(cited_source_ids), 6) if cited_source_ids else 0.0,
    }
    return {"status": status, "citations": citations, "accounting": accounting}
