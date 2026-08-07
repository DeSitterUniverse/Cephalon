"""Build bounded hierarchical context from reranked child chunks.

This module sits between reranking and lexical context compression.  It adapts
HiChunk-style auto-merging to Cephalon's existing parent/child SQLite index:
several independently retrieved siblings may become a contiguous span, and a
well-covered parent may replace that span.  The returned anchor is always an
actual retrieved child, so citations keep exact document provenance even when
the model receives additional surrounding text.

The implementation deliberately performs no embedding or model calls.  For a
request with ``n`` selected results it loads each distinct parent once and
examines at most ``MAX_SPAN_CHILDREN`` rows around a match.  The hard request
budget is no larger than the former worst case of ``n * parent_max_tokens``.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .. import storage


MIN_PARENT_MATCHES = 2
"""Independent selected children required before parent promotion."""

MIN_PARENT_COVERAGE = 0.40
"""Minimum fraction of parent child-token mass matched by retrieval."""

MAX_SPAN_CHILDREN = 5
"""Maximum contiguous children in a sibling span, including neighbours."""


@dataclass(frozen=True)
class ContextAssembly:
    """One context unit anchored to a selected retrieval result.

    ``matched_chunk_ids`` are independently selected children.  In contrast,
    ``expanded_chunk_ids`` also includes adjacent children added only for
    coherence.  Consumers must cite ``anchor_id`` rather than an expanded ID.
    """

    anchor_id: str
    result: dict[str, Any]
    text: str
    context_kind: str
    parent_id: str | None
    matched_chunk_ids: tuple[str, ...]
    expanded_chunk_ids: tuple[str, ...]
    coverage: float
    token_count: int
    decision: str

    def trace_payload(self) -> dict[str, Any]:
        """Return the backward-compatible JSON payload stored with a source."""

        return {
            "context_kind": self.context_kind,
            "anchor_chunk_id": self.anchor_id,
            "parent_id": self.parent_id,
            "matched_chunk_ids": list(self.matched_chunk_ids),
            "expanded_chunk_ids": list(self.expanded_chunk_ids),
            "parent_coverage": round(self.coverage, 6),
            "context_tokens": self.token_count,
            "decision": self.decision,
        }


def assemble_hierarchical_context(
    conn,
    selected: list[dict[str, Any]],
    *,
    parent_max_tokens: int,
    enable_merge: bool = True,
) -> list[ContextAssembly]:
    """Return ordered, non-overlapping context units for reranked results.

    Parent promotion requires two unique child matches, at least 40 percent
    token coverage, and enough remaining request budget.  Otherwise nearby
    matches may form a bounded sibling span.  Wide match ranges are never
    bridged because the intervening text is likely unrelated.  Parentless,
    summary, and memory results pass through unchanged.
    """

    if not selected:
        return []

    request_budget = max(1, len(selected)) * max(1, int(parent_max_tokens))
    remaining_budget = request_budget
    by_parent: dict[str, list[dict[str, Any]]] = {}
    for result in selected:
        parent_id = result.get("parent_id")
        if enable_merge and parent_id and result.get("source_kind", "child") == "child":
            by_parent.setdefault(str(parent_id), []).append(result)

    assemblies_by_anchor: dict[str, ContextAssembly] = {}
    covered_ids: set[str] = set()
    for parent_id, matches in by_parent.items():
        unique_matches = list({str(item["id"]): item for item in matches}.values())
        if len(unique_matches) < MIN_PARENT_MATCHES:
            continue
        child_rows = storage.fetchall(
            conn,
            """
            SELECT id, chunk_index, text, token_count
            FROM chunks
            WHERE parent_id = ?
            ORDER BY chunk_index, id
            """,
            (parent_id,),
        )
        if not child_rows:
            continue
        row_by_id = {str(row["id"]): row for row in child_rows}
        matched_rows = [row_by_id[str(item["id"])] for item in unique_matches if str(item["id"]) in row_by_id]
        if len(matched_rows) < MIN_PARENT_MATCHES:
            continue

        parent = storage.fetchone(
            conn,
            "SELECT text, token_count FROM parent_chunks WHERE id = ?",
            (parent_id,),
        )
        parent_tokens = max(1, int(parent["token_count"] or 0)) if parent else 0
        matched_tokens = sum(max(1, int(row["token_count"] or 0)) for row in matched_rows)
        coverage = min(1.0, matched_tokens / parent_tokens) if parent_tokens else 0.0
        # Selected order is rerank order, so the first group member remains the
        # exact high-confidence child to which the expanded context is tied.
        anchor_result = next(item for item in selected if item in unique_matches)
        anchor_id = str(anchor_result["id"])
        matched_ids = tuple(str(row["id"]) for row in sorted(matched_rows, key=lambda row: (row["chunk_index"], row["id"])))

        if (
            parent
            and coverage >= MIN_PARENT_COVERAGE
            and parent_tokens <= parent_max_tokens
            and parent_tokens <= remaining_budget
        ):
            assembly = ContextAssembly(
                anchor_id=anchor_id,
                result=anchor_result,
                text=str(parent["text"]),
                context_kind="parent",
                parent_id=parent_id,
                matched_chunk_ids=matched_ids,
                expanded_chunk_ids=tuple(str(row["id"]) for row in child_rows),
                coverage=coverage,
                token_count=parent_tokens,
                decision="promoted: independent sibling matches cover at least 40% of parent",
            )
        else:
            positions = sorted(child_rows.index(row) for row in matched_rows)
            first, last = positions[0], positions[-1]
            if last - first + 1 > MAX_SPAN_CHILDREN:
                continue
            # Add one neighbour on either side when the fixed five-child and
            # parent-sized token bounds still hold.  This captures definitions
            # and conclusions without allowing an open-ended graph walk.
            span_first = max(0, first - 1)
            span_last = min(len(child_rows) - 1, last + 1)
            while span_last - span_first + 1 > MAX_SPAN_CHILDREN:
                if first - span_first > span_last - last:
                    span_first += 1
                else:
                    span_last -= 1
            span_rows = child_rows[span_first:span_last + 1]
            span_tokens = sum(max(1, int(row["token_count"] or 0)) for row in span_rows)
            if span_tokens > parent_max_tokens or span_tokens > remaining_budget:
                span_rows = child_rows[first:last + 1]
                span_tokens = sum(max(1, int(row["token_count"] or 0)) for row in span_rows)
            if span_tokens > parent_max_tokens or span_tokens > remaining_budget:
                continue
            rejected_reason = "parent rejected: insufficient coverage"
            if parent_tokens > parent_max_tokens:
                rejected_reason = "parent rejected: exceeds per-unit token budget"
            elif parent_tokens > remaining_budget:
                rejected_reason = "parent rejected: exceeds remaining request budget"
            assembly = ContextAssembly(
                anchor_id=anchor_id,
                result=anchor_result,
                text="\n\n".join(str(row["text"]) for row in span_rows),
                context_kind="sibling_span",
                parent_id=parent_id,
                matched_chunk_ids=matched_ids,
                expanded_chunk_ids=tuple(str(row["id"]) for row in span_rows),
                coverage=coverage,
                token_count=span_tokens,
                decision=rejected_reason,
            )

        assemblies_by_anchor[anchor_id] = assembly
        covered_ids.update(matched_ids)
        remaining_budget -= assembly.token_count

    output: list[ContextAssembly] = []
    for result in selected:
        result_id = str(result["id"])
        if result_id in covered_ids and result_id not in assemblies_by_anchor:
            continue
        assembly = assemblies_by_anchor.get(result_id)
        if assembly is None:
            token_count = max(1, int(result.get("token_count") or ((len(str(result.get("text", ""))) + 3) // 4)))
            assembly = ContextAssembly(
                anchor_id=result_id,
                result=result,
                text=str(result.get("text", "")),
                context_kind="child",
                parent_id=str(result["parent_id"]) if result.get("parent_id") else None,
                matched_chunk_ids=(result_id,),
                expanded_chunk_ids=(result_id,),
                coverage=0.0,
                token_count=token_count,
                decision="exact selected result; sibling promotion criteria not met",
            )
        output.append(assembly)
        remaining_budget = max(0, remaining_budget - (assembly.token_count if result_id not in assemblies_by_anchor else 0))
    return output
