from __future__ import annotations

import re
from typing import Any

from ..schemas import SourceChunk


SOURCE_TAG_PATTERN = re.compile(r"\[\[\s*src\s*:\s*([A-Za-z0-9_-]+)\s*\]\]", re.IGNORECASE)
CLAIM_SPLIT_PATTERN = re.compile(r"(?<=[.!?])\s+|\n+")
CLAIM_STOPWORDS = {
    "about", "after", "again", "also", "because", "before", "being", "between",
    "could", "does", "from", "have", "into", "more", "most", "other", "should",
    "than", "that", "their", "there", "these", "they", "this", "those", "through",
    "using", "very", "were", "what", "when", "where", "which", "while", "with",
    "would", "your",
}


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
    claim_validation = validate_answer_claims(answer_text, sources)
    if claim_validation["unsupported_claim_count"] > 0:
        status = "unsupported"
    elif claim_validation["weak_claim_count"] > 0 and status == "supported":
        status = "weak"
    return {
        "status": status,
        "citations": citations,
        "accounting": accounting,
        "claim_validation": claim_validation,
    }


def validate_answer_claims(answer_text: str, sources: list[SourceChunk]) -> dict[str, Any]:
    source_by_id = {
        source.source_id.upper(): source
        for source in sources
        if source.source_id
    }
    claims: list[dict[str, Any]] = []
    for index, statement in enumerate(_claim_statements(answer_text), start=1):
        clean = statement.strip()
        if not clean or clean.startswith("<think>") or clean.startswith("</think>"):
            continue
        cited_ids = extract_cited_source_ids(clean)
        claim_text = SOURCE_TAG_PATTERN.sub("", clean).strip(" -*#\t")
        claim_terms = _claim_terms(claim_text)
        if len(claim_terms) < 2:
            continue
        known_sources = [source_by_id[source_id] for source_id in cited_ids if source_id in source_by_id]
        unknown_ids = [source_id for source_id in cited_ids if source_id not in source_by_id]
        coverage_by_source = {
            source.source_id or "unknown": _term_coverage(claim_terms, _claim_terms(source.snippet))
            for source in known_sources
        }
        best_coverage = max(coverage_by_source.values(), default=0.0)
        if unknown_ids:
            status = "unsupported"
            reason = "One or more citation tags do not identify supplied evidence."
        elif not cited_ids:
            status = "uncited"
            reason = "The claim has no source tag."
        elif best_coverage >= 0.55:
            status = "supported"
            reason = "The cited evidence contains most of the claim's material terms."
        elif best_coverage >= 0.25:
            status = "weak"
            reason = "The cited evidence has partial lexical support for the claim."
        else:
            status = "unsupported"
            reason = "The cited evidence does not contain enough of the claim's material terms."
        claims.append({
            "claim_id": f"C{index}",
            "text": claim_text,
            "source_ids": cited_ids,
            "status": status,
            "reason": reason,
            "coverage": round(best_coverage, 6),
            "coverage_by_source": {
                source_id: round(coverage, 6)
                for source_id, coverage in coverage_by_source.items()
            },
        })

    return {
        "method": "deterministic_claim_coverage_v1",
        "claim_count": len(claims),
        "supported_claim_count": sum(claim["status"] == "supported" for claim in claims),
        "weak_claim_count": sum(claim["status"] == "weak" for claim in claims),
        "unsupported_claim_count": sum(claim["status"] == "unsupported" for claim in claims),
        "uncited_claim_count": sum(claim["status"] == "uncited" for claim in claims),
        "claims": claims,
    }


def _claim_statements(answer_text: str) -> list[str]:
    tag = r"\[\[\s*src\s*:\s*[A-Za-z0-9_-]+\s*\]\]"
    normalized = re.sub(
        rf"([.!?])\s+((?:{tag}\s*)+)",
        lambda match: f"{match.group(1)} {match.group(2).strip()}\n",
        answer_text or "",
        flags=re.IGNORECASE,
    )
    return [
        statement
        for statement in CLAIM_SPLIT_PATTERN.split(normalized)
        if statement.strip()
    ]


def _claim_terms(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[\w.-]+", text.lower(), flags=re.UNICODE)
        if len(term) >= 3 and term not in CLAIM_STOPWORDS
    }


def _term_coverage(claim_terms: set[str], evidence_terms: set[str]) -> float:
    if not claim_terms:
        return 0.0
    return len(claim_terms & evidence_terms) / len(claim_terms)
