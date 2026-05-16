from __future__ import annotations

from typing import Any

from ..schemas import SourceChunk


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


def classify_answer_support(sources: list[SourceChunk]) -> dict[str, Any]:
    citations = [classify_citation_support(source.chunk_id, sources) for source in sources]
    if not citations:
        status = "unsupported"
    elif any(item["status"] == "supported" for item in citations):
        status = "supported"
    elif any(item["status"] == "weak" for item in citations):
        status = "weak"
    else:
        status = "unsupported"
    return {"status": status, "citations": citations}
