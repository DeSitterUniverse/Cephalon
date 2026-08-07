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
    eligible = [
        item for item in candidates
        if (
            float(item.get("score", 0.0)) >= max(0.05, maximum * 0.20)
            or item.get("lexical_rank") is not None
            or str(item.get("id")) in anchor_ids
        )
    ] or candidates[:limit]

    requirement_terms = {item["id"]: _terms(item["text"]) for item in requirements}
    selected: list[dict[str, Any]] = []
    selected_terms: list[set[str]] = []
    selected_docs: set[str] = set()
    selected_parents: set[str] = set()
    covered_requirements: set[str] = set()
    remaining = list(eligible)
    decisions: list[dict[str, Any]] = []

    while remaining and len(selected) < limit:
        best_item = None
        best_payload = None
        for item in remaining:
            item_terms = _terms(str(item.get("text", "")))
            normalized_relevance = (float(item.get("score", 0.0)) - minimum) / score_range if maximum != minimum else 1.0
            coverage = {key: _coverage(terms, item_terms) for key, terms in requirement_terms.items()}
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
        )
        remaining = [item for item in remaining if item.get("id") != best_item.get("id")]

    return selected, {
        "candidate_count": len(candidates),
        "eligible_count": len(eligible),
        "selected": decisions,
        "covered_requirement_ids": sorted(covered_requirements),
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


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return len(left & right) / len(union) if union else 0.0
