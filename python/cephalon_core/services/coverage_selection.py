"""Select reranked evidence by coverage and marginal context value.

The selector operates on Jina's already-reranked candidates; it does not
replace dense, lexical, RRF, or listwise ranking. A bounded greedy objective
balances normalized relevance, uncovered requirement terms, document
diversity, parent coherence, redundancy, and estimated token cost. Dense-rank
anchors remain a recall safety net.

With at most 100 candidates and 20 output slots, complexity is bounded by
O(k*n*(r+s)), where r <= 8 requirements and s <= 20 selected items.
"""

from __future__ import annotations

import re
from typing import Any

from .evidence_ledger import document_identity_matches, evidence_need_coverage, is_qualifying_evidence


RELEVANCE_WEIGHT = 0.42
UNCOVERED_REQUIREMENT_WEIGHT = 0.38
DIVERSITY_BONUS = 0.07
STRUCTURAL_COHERENCE_BONUS = 0.05
DENSE_ANCHOR_BONUS = 0.12
REDUNDANCY_PENALTY = 0.16
TOKEN_COST_PENALTY = 0.04
MIN_REQUIREMENT_COVERAGE = 0.34

SELECTION_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "the", "to", "was", "were", "what", "when", "which", "who",
    "with", "versus", "vs", "compare", "explain", "describe", "paper", "study",
}


def select_coverage_aware_results(
    ranked: list[dict[str, Any]],
    requirements: list[dict[str, str]],
    limit: int,
    *,
    anchor_ids: set[str] | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Greedily select results and attach transparent objective components."""

    if not ranked or limit <= 0:
        return [], {"selected": [], "covered_requirement_ids": [], "candidate_count": len(ranked)}
    anchor_ids = anchor_ids or set()
    candidates = ranked[:100]
    scores = [float(item.get("score", 0.0)) for item in candidates]
    minimum, maximum = min(scores), max(scores)
    score_range = max(maximum - minimum, 1e-9)
    named_requirements = [item for item in requirements if item.get("named_document")]
    requirements_by_id = {str(item["id"]): item for item in requirements}
    requirement_terms = {item["id"]: _requirement_terms(item) for item in requirements}
    eligible = [
        item for item in candidates
        if (
            float(item.get("score", 0.0)) >= max(0.05, maximum * 0.20)
            or item.get("lexical_rank") is not None
            or str(item.get("id")) in anchor_ids
        )
    ] or candidates[:limit]
    # A low-scoring but qualifying chunk from an explicitly named paper is a
    # higher-value recall target than a generic high-scoring distractor. Keep
    # it in the bounded selection window so the reservation pass can see it.
    for item in candidates:
        if any(
            document_identity_matches(requirement, item)
            and is_qualifying_evidence(str(item.get("text", "")))
            and evidence_need_coverage(requirement, str(item.get("text", ""))) >= MIN_REQUIREMENT_COVERAGE
            for requirement in named_requirements
        ) and item not in eligible:
            eligible.append(item)

    selected: list[dict[str, Any]] = []
    selected_terms: list[set[str]] = []
    selected_docs: set[str] = set()
    selected_parents: set[str] = set()
    covered_requirements: set[str] = set()
    remaining = list(eligible)
    decisions: list[dict[str, Any]] = []
    reserved_named_ids: list[str] = []
    reserved_named_requirement_ids: list[str] = []

    # Reserve one qualifying source per named document before greedy
    # complementarity.  Identity matching is based on document metadata, not
    # chunk text, so a bibliography mention cannot satisfy another paper.
    for requirement in named_requirements:
        if len(selected) >= limit:
            break
        matching = [
            item for item in remaining
            if document_identity_matches(requirement, item)
            and is_qualifying_evidence(str(item.get("text", "")))
            and evidence_need_coverage(requirement, str(item.get("text", ""))) >= MIN_REQUIREMENT_COVERAGE
        ]
        if not matching:
            continue
        reserved = max(
            matching,
            # A high reranker score for a title/heading fragment must not beat
            # a lower-ranked substantive span that actually answers the
            # requested contribution/result/method/limitation need.
            key=lambda item: (
                evidence_need_coverage(requirement, str(item.get("text", ""))),
                float(item.get("score", 0.0)),
                -int(item.get("rerank_rank", 0) or 0),
            ),
        )
        selected_item = dict(reserved)
        coverage = {
            key: round(
                evidence_need_coverage(requirements_by_id[key], str(reserved.get("text", "")))
                if requirements_by_id[key].get("named_document")
                else _coverage(terms, _terms(str(reserved.get("text", "")))),
                6,
            )
            for key, terms in requirement_terms.items()
        }
        payload = {
            "objective": round(float(reserved.get("score", 0.0)), 6),
            "normalized_relevance": 1.0,
            "requirement_coverage": coverage,
            "new_requirement_coverage": 1.0,
            "diversity_bonus": True,
            "structural_coherence": False,
            "dense_anchor": str(reserved.get("id")) in anchor_ids,
            "redundancy": 0.0,
            "estimated_tokens": max(1, (len(str(reserved.get("text", ""))) + 3) // 4),
            "decision": "reserved qualifying source for named document",
        }
        selected_item["context_selection"] = payload
        selected.append(selected_item)
        decisions.append({"chunk_id": selected_item.get("id"), **payload})
        reserved_named_ids.append(str(selected_item.get("id")))
        reserved_named_requirement_ids.append(str(requirement["id"]))
        covered_requirements.add(str(requirement["id"]))
        selected_terms.append(_terms(str(selected_item.get("text", ""))))
        selected_docs.add(str(selected_item.get("doc_id", "")))
        if selected_item.get("parent_id"):
            selected_parents.add(str(selected_item["parent_id"]))
        remaining = [item for item in remaining if item.get("id") != reserved.get("id")]

    while remaining and len(selected) < limit:
        best_item = None
        best_payload = None
        for item in remaining:
            item_terms = _terms(str(item.get("text", "")))
            normalized_relevance = (float(item.get("score", 0.0)) - minimum) / score_range if maximum != minimum else 1.0
            coverage = {
                key: (
                    evidence_need_coverage(requirements_by_id[key], str(item.get("text", "")))
                    if requirements_by_id[key].get("named_document")
                    and document_identity_matches(requirements_by_id[key], item)
                    else _coverage(terms, item_terms)
                    if not requirements_by_id[key].get("named_document")
                    else 0.0
                )
                for key, terms in requirement_terms.items()
            }
            new_coverage = max(
                (value for key, value in coverage.items() if key not in covered_requirements),
                default=0.0,
            )
            redundancy = max((_jaccard(item_terms, terms) for terms in selected_terms), default=0.0)
            doc_id = str(item.get("doc_id", ""))
            parent_id = str(item.get("parent_id", ""))
            diversity = 1.0 if doc_id and doc_id not in selected_docs else 0.0
            coherence = 1.0 if parent_id and parent_id in selected_parents else 0.0
            estimated_tokens = max(1, (len(str(item.get("text", ""))) + 3) // 4)
            token_cost = min(1.0, estimated_tokens / 650.0)
            anchor = 1.0 if str(item.get("id")) in anchor_ids else 0.0
            objective = (
                RELEVANCE_WEIGHT * normalized_relevance
                + UNCOVERED_REQUIREMENT_WEIGHT * new_coverage
                + DIVERSITY_BONUS * diversity
                + STRUCTURAL_COHERENCE_BONUS * coherence
                + DENSE_ANCHOR_BONUS * anchor
                - REDUNDANCY_PENALTY * redundancy
                - TOKEN_COST_PENALTY * token_cost
            )
            payload = {
                "objective": round(objective, 6),
                "normalized_relevance": round(normalized_relevance, 6),
                "requirement_coverage": {key: round(value, 6) for key, value in coverage.items()},
                "new_requirement_coverage": round(new_coverage, 6),
                "diversity_bonus": bool(diversity),
                "structural_coherence": bool(coherence),
                "dense_anchor": bool(anchor),
                "redundancy": round(redundancy, 6),
                "estimated_tokens": estimated_tokens,
            }
            if best_payload is None or (payload["objective"], float(item.get("score", 0.0))) > (
                best_payload["objective"], float(best_item.get("score", 0.0))
            ):
                best_item, best_payload = item, payload
        assert best_item is not None and best_payload is not None
        selected_item = dict(best_item)
        selected_item["context_selection"] = best_payload
        selected.append(selected_item)
        decisions.append({"chunk_id": selected_item.get("id"), **best_payload})
        terms = _terms(str(selected_item.get("text", "")))
        selected_terms.append(terms)
        selected_docs.add(str(selected_item.get("doc_id", "")))
        if selected_item.get("parent_id"):
            selected_parents.add(str(selected_item["parent_id"]))
        covered_requirements.update(
            key for key, value in best_payload["requirement_coverage"].items()
            if float(value) >= MIN_REQUIREMENT_COVERAGE
            and (
                not requirements_by_id[key].get("named_document")
                or (
                    document_identity_matches(requirements_by_id[key], selected_item)
                    and is_qualifying_evidence(str(selected_item.get("text", "")))
                    and evidence_need_coverage(requirements_by_id[key], str(selected_item.get("text", ""))) >= MIN_REQUIREMENT_COVERAGE
                )
            )
        )
        remaining = [item for item in remaining if item.get("id") != best_item.get("id")]

    return selected, {
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "selected": decisions,
        "covered_requirement_ids": sorted(covered_requirements),
        "reserved_named_requirement_ids": reserved_named_requirement_ids,
        "reserved_named_chunk_ids": reserved_named_ids,
        "uncovered_named_requirement_ids": [
            item["id"] for item in named_requirements if item["id"] not in covered_requirements
        ],
        "requirement_count": len(requirements),
        "weights": {
            "relevance": RELEVANCE_WEIGHT,
            "uncovered_requirement": UNCOVERED_REQUIREMENT_WEIGHT,
            "diversity_bonus": DIVERSITY_BONUS,
            "structural_coherence_bonus": STRUCTURAL_COHERENCE_BONUS,
            "dense_anchor_bonus": DENSE_ANCHOR_BONUS,
            "redundancy_penalty": REDUNDANCY_PENALTY,
            "token_cost_penalty": TOKEN_COST_PENALTY,
        },
    }


def _terms(text: str) -> set[str]:
    return {
        token for token in re.findall(r"[\wµμ]+", text.lower(), flags=re.UNICODE)
        if len(token) >= 3 and token not in SELECTION_STOPWORDS
    }


def _coverage(required: set[str], evidence: set[str]) -> float:
    return len(required & evidence) / len(required) if required else 0.0


def _requirement_terms(requirement: dict[str, Any]) -> set[str]:
    if requirement.get("named_document"):
        return {
            str(value).casefold()
            for value in requirement.get("evidence_need", [])
            if len(str(value)) >= 3
        }
    return _terms(str(requirement.get("text", "")))


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0
