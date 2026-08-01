"""Run the bounded evidence-acquisition state machine for a RAG request.

The normal path delegates once to the existing retriever. Thorough mode may
perform one additional deterministic gap query when the request ledger still
contains missing, partial, or conflicting requirements. This controller never
calls the chat model: reformulation is lexical, and the second pass reuses the
existing embedder, hybrid search, Jina reranker, assembly, and compression.

The gap round has one query, a 20-second timeout, at most three novel sources,
and at most 50 percent of the initial context's estimated tokens. Duplicate
queries and chunks already present through hierarchical/layout expansion stop
the round without changing context.
"""

from __future__ import annotations

import asyncio
import re
import time
from typing import Any

from ..schemas import RagSettings, SourceChunk
from . import retrieval
from .evidence_ledger import build_evidence_ledger, document_identity_matches, is_qualifying_evidence
from .prompt_budget import estimate_tokens


MAX_GAP_QUERIES = 1
MAX_GAP_SOURCES = 3
GAP_CONTEXT_SHARE = 0.50
GAP_TIMEOUT_SECONDS = 20.0
MIN_GAP_CONTEXT_TOKENS = 128
MAX_GAP_TOP_K = 12

GAP_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "did", "do", "does", "for", "from", "how",
    "in", "is", "it", "of", "on", "or", "the", "to", "was", "were", "what", "when", "which", "who",
    "with", "compare", "describe", "explain", "paper", "study",
}


async def retrieve_with_gap_control(
    app_state,
    prompt: str,
    query_vector: list[float],
    settings: RagSettings,
    *,
    enable_gap_round: bool,
) -> tuple[str, list[SourceChunk], dict[str, Any]]:
    """Retrieve initial evidence and optionally execute one Thorough gap round."""

    context, sources, meta = await retrieval.retrieve_context(app_state, prompt, query_vector, settings)
    if not enable_gap_round:
        return context, sources, meta

    ledger = meta.get("evidence_ledger") or {}
    gaps = [
        item for item in ledger.get("requirements", [])
        if item.get("status") in {"missing", "partial", "conflicting"}
    ]
    control_trace = {
        "enabled": True,
        "round": 1,
        "attempted": False,
        "status": "not_needed" if not gaps else "pending",
        "triggering_requirement_ids": [item.get("id") for item in gaps],
        "limits": {
            "max_queries": MAX_GAP_QUERIES,
            "max_sources": MAX_GAP_SOURCES,
            "context_share": GAP_CONTEXT_SHARE,
            "timeout_seconds": GAP_TIMEOUT_SECONDS,
            "max_top_k": MAX_GAP_TOP_K,
        },
    }
    if not gaps:
        _record_control_trace(meta, control_trace)
        return context, sources, meta

    gap_query = _build_gap_query(prompt, gaps)
    control_trace["query"] = gap_query
    control_trace["target_titles"] = [
        str(item.get("requested_title"))
        for item in gaps
        if item.get("named_document") and item.get("requested_title")
    ]
    if _normalize_query(gap_query) == _normalize_query(prompt):
        control_trace["status"] = "duplicate_query"
        _record_control_trace(meta, control_trace)
        return context, sources, meta

    control_trace["attempted"] = True
    gap_settings = settings.model_copy(update={
        # The targeted query gets the full bounded retrieval width even when
        # ordinary answer settings use a smaller final context. Selection and
        # admission below still enforce the three-source gap budget.
        "top_k": MAX_GAP_TOP_K,
        # Search the full bounded candidate pool before document-aware
        # admission. The context round still adds at most MAX_GAP_SOURCES
        # qualifying chunks, but limiting reranking to that same count can
        # hide an exact-title source behind generic evidence from other papers.
        "rerank_top_n": MAX_GAP_TOP_K,
        "conversation_memory": False,
        "trace_persistence": False,
    })
    started = time.perf_counter()
    try:
        gap_vector = await retrieval.get_embedding(app_state, gap_query)
        gap_context, gap_sources, gap_meta = await asyncio.wait_for(
            retrieval.retrieve_context(app_state, gap_query, gap_vector, gap_settings),
            timeout=GAP_TIMEOUT_SECONDS,
        )
    except asyncio.CancelledError:
        raise
    except TimeoutError:
        control_trace["status"] = "timeout"
        control_trace["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        _record_control_trace(meta, control_trace)
        return context, sources, meta
    except Exception as error:
        # A refinement round is optional; a transient embedder/reranker error
        # must not discard a valid initial retrieval in Thorough mode.
        control_trace["status"] = "error"
        control_trace["error_type"] = type(error).__name__
        control_trace["error"] = str(error)[:300]
        control_trace["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        _record_control_trace(meta, control_trace)
        return context, sources, meta

    seen_chunk_ids = _represented_chunk_ids(sources)
    token_budget = min(
        settings.parent_max_tokens,
        max(MIN_GAP_CONTEXT_TOKENS, int(estimate_tokens(context) * GAP_CONTEXT_SHARE)),
    )
    added: list[SourceChunk] = []
    added_blocks: list[str] = []
    spent_tokens = 0
    triggering_ids = [str(item.get("id")) for item in gaps if item.get("id")]
    named_gaps = [item for item in gaps if item.get("named_document")]
    ordered_gap_sources = sorted(
        gap_sources,
        key=lambda candidate: (
            any(document_identity_matches(gap, candidate) for gap in named_gaps),
            any(
                document_identity_matches(gap, candidate)
                and is_qualifying_evidence(
                    candidate.evidence_text if candidate.evidence_text is not None else candidate.snippet
                )
                for gap in named_gaps
            ),
            float(candidate.rerank_score if candidate.rerank_score is not None else candidate.score),
        ),
        reverse=True,
    )
    for candidate in ordered_gap_sources:
        if len(added) >= MAX_GAP_SOURCES or candidate.chunk_id in seen_chunk_ids:
            continue
        evidence = (
            candidate.evidence_text
            if candidate.evidence_text is not None
            else candidate.snippet or ""
        ).strip()
        if named_gaps:
            matching_gaps = [gap for gap in named_gaps if document_identity_matches(gap, candidate)]
            # A targeted named-paper round must not spend its bounded slots on
            # an unrelated document or a bibliography-only fragment.
            if not matching_gaps or not any(is_qualifying_evidence(evidence) for _ in matching_gaps):
                continue
        cost = estimate_tokens(evidence)
        if not evidence or spent_tokens + cost > token_budget:
            continue
        candidate.source_id = f"S{len(sources) + len(added) + 1}"
        candidate.rank = len(sources) + len(added) + 1
        candidate.retrieval_round = 1
        candidate.triggering_gap = ",".join(triggering_ids) or None
        added.append(candidate)
        added_blocks.append(retrieval.format_source_context(
            candidate.source_id,
            candidate.doc_name,
            candidate.chunk_id,
            evidence,
        ))
        spent_tokens += cost
        seen_chunk_ids.update(_represented_chunk_ids([candidate]))

    control_trace.update({
        "latency_ms": round((time.perf_counter() - started) * 1000, 2),
        "candidate_count": len(gap_sources),
        "added_source_count": len(added),
        "added_context_tokens": spent_tokens,
        "token_budget": token_budget,
        "status": "completed" if added else "no_novel_evidence",
        "gap_retrieval_latency_ms": gap_meta.get("retrieval_latency_ms"),
        "gap_context_available": bool(gap_context),
    })
    if not added:
        _record_control_trace(meta, control_trace)
        return context, sources, meta

    combined_sources = [*sources, *added]
    combined_context = "\n\n".join([context, *added_blocks])
    combined_ledger = build_evidence_ledger(
        str(meta.get("query_id") or ""),
        prompt,
        meta.get("subqueries") or [],
        combined_sources,
        retrieval_round=1,
    )
    combined_ledger["state"] = "gap_assessed"
    combined_ledger["gap_retrieval"] = control_trace
    confidence = retrieval.confidence_from_sources(combined_sources, settings)
    meta.update(confidence)
    meta["evidence_ledger"] = combined_ledger
    meta["retrieval_latency_ms"] = round(
        float(meta.get("retrieval_latency_ms") or 0.0) + float(control_trace["latency_ms"]),
        2,
    )
    meta.setdefault("search_modes", []).append("gap_round")
    trace = meta.get("trace") or {}
    trace["evidence_ledger"] = combined_ledger
    trace["final_context"] = [source.model_dump() for source in combined_sources]
    trace["gap_retrieval"] = control_trace
    trace["retrieval_mode"] = "+".join(filter(None, [trace.get("retrieval_mode"), "gap_round"]))
    trace["no_answer"] = confidence
    trace.setdefault("latency", {})["gap_ms"] = control_trace["latency_ms"]
    trace["latency"]["total_ms"] = meta["retrieval_latency_ms"]
    meta["trace"] = trace
    return combined_context, combined_sources, meta


def _build_gap_query(prompt: str, gaps: list[dict[str, Any]]) -> str:
    titles: list[str] = []
    needs: list[str] = []
    material: list[str] = []
    for gap in gaps:
        title = str(gap.get("requested_title") or "").strip()
        if title and title.casefold() not in {item.casefold() for item in titles}:
            titles.append(title)
        for need in gap.get("evidence_need", []):
            if str(need) not in needs:
                needs.append(str(need))
        for term in re.findall(r"[\wµμ]+", str(gap.get("text", "")).lower(), flags=re.UNICODE):
            if len(term) >= 3 and term not in GAP_STOPWORDS and term not in material:
                material.append(term)
    # Evidence hints make a single-question retry distinct from the initial
    # natural-language question while favoring result-bearing scientific text.
    hints = ["results", "measured", "evidence", "values", "units", "limitations"]
    exact_titles = [f'"{title}"' for title in titles[:3]]
    query = " ".join([
        *exact_titles,
        *needs[:4],
        *material[:18],
        *[hint for hint in hints if hint not in material and hint not in needs],
    ])
    return query or f"{prompt.strip()} supporting evidence results"


def _normalize_query(query: str) -> str:
    return " ".join(re.findall(r"[\wµμ]+", query.lower(), flags=re.UNICODE))


def _represented_chunk_ids(sources: list[SourceChunk]) -> set[str]:
    represented: set[str] = set()
    for source in sources:
        represented.add(source.chunk_id)
        represented.update(str(value) for value in source.context_assembly.get("expanded_chunk_ids", []))
        represented.update(str(value) for value in source.context_assembly.get("layout_chunk_ids", []))
    return represented


def _record_control_trace(meta: dict[str, Any], control_trace: dict[str, Any]) -> None:
    ledger = meta.get("evidence_ledger") or {}
    ledger["gap_retrieval"] = control_trace
    meta["evidence_ledger"] = ledger
    trace = meta.get("trace") or {}
    trace["gap_retrieval"] = control_trace
    trace["evidence_ledger"] = ledger
    meta["trace"] = trace
