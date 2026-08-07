"""Conditionally add bounded document-layout evidence to selected context.

Cephalon already stores page, block, heading, asset, and parent provenance in
SQLite.  This module derives a small request-local evidence graph from that
data instead of maintaining a graph database.  It runs after hierarchical
assembly and before compression, adding context only when the query or anchor
indicates that layout relationships matter.

Expansion is breadth-first, at most two hops, six nodes, four neighbours per
node, and 25 percent of the pre-expansion context tokens.  These limits make
runtime and prompt growth independent of document size while preserving the
retrieved child as the public citation anchor.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any

from .. import storage
from .context_assembly import ContextAssembly


MAX_LAYOUT_HOPS = 2
MAX_LAYOUT_NODES = 6
MAX_NEIGHBORS_PER_NODE = 4
LAYOUT_TOKEN_SHARE = 0.25
NEIGHBOR_WINDOW = 3

LAYOUT_QUERY_RE = re.compile(
    r"\b(?:figure|fig\.?|panel|image|diagram|caption|table|row|column|header|page|section|above|below|preceding|following)\b",
    re.IGNORECASE,
)
ANAPHORA_RE = re.compile(r"^(?:this|these|those|such|it|they|the former|the latter)\b", re.IGNORECASE)


@dataclass(frozen=True)
class LayoutEdge:
    """A single structural relationship traversed during expansion."""

    edge_type: str
    from_chunk_id: str
    to_chunk_id: str
    hop: int
    reason: str


@dataclass(frozen=True)
class LayoutExpansion:
    """Additional evidence and trace data assigned to one citation anchor."""

    anchor_id: str
    text_blocks: tuple[str, ...]
    chunk_ids: tuple[str, ...]
    token_count: int
    edges: tuple[LayoutEdge, ...]

    def trace_payload(self) -> dict[str, Any]:
        """Serialize expansion data without exposing internal SQLite rows."""

        return {
            "layout_tokens": self.token_count,
            "layout_chunk_ids": list(self.chunk_ids),
            "structural_relationships": [
                {
                    "edge_type": edge.edge_type,
                    "from_chunk_id": edge.from_chunk_id,
                    "to_chunk_id": edge.to_chunk_id,
                    "hop": edge.hop,
                    "reason": edge.reason,
                }
                for edge in self.edges
            ],
        }


def expand_layout_evidence(
    conn,
    query: str,
    assemblies: list[ContextAssembly],
) -> dict[str, LayoutExpansion]:
    """Return request-bounded structural context keyed by citation anchor.

    The budget is shared across anchors in rerank order. Exact text already
    present in a parent or sibling assembly is excluded. A candidate must fit
    in the remaining token budget in full; partial chunks would obscure their
    original page and block provenance.
    """

    base_tokens = sum(max(1, item.token_count) for item in assemblies)
    remaining_tokens = int(base_tokens * LAYOUT_TOKEN_SHARE)
    if remaining_tokens <= 0:
        return {}

    layout_query = bool(LAYOUT_QUERY_RE.search(query))
    output: dict[str, LayoutExpansion] = {}
    for assembly in assemblies:
        if remaining_tokens <= 0 or assembly.result.get("source_kind", "child") != "child":
            continue
        anchor = _chunk_row(conn, assembly.anchor_id)
        if anchor is None or not _should_expand(anchor, layout_query):
            continue
        excluded = set(assembly.expanded_chunk_ids)
        visited = {assembly.anchor_id, *excluded}
        frontier = [(anchor, 0)]
        selected_rows: list[Any] = []
        selected_edges: list[LayoutEdge] = []
        while frontier and len(selected_rows) < MAX_LAYOUT_NODES and remaining_tokens > 0:
            current, current_hop = frontier.pop(0)
            if current_hop >= MAX_LAYOUT_HOPS:
                continue
            neighbours = _neighbours(conn, current, layout_query)
            for candidate, edge_type, reason in neighbours[:MAX_NEIGHBORS_PER_NODE]:
                candidate_id = str(candidate["id"])
                if candidate_id in visited:
                    continue
                visited.add(candidate_id)
                tokens = max(1, int(candidate["token_count"] or ((len(str(candidate["text"])) + 3) // 4)))
                if tokens > remaining_tokens:
                    continue
                hop = current_hop + 1
                selected_rows.append(candidate)
                selected_edges.append(LayoutEdge(edge_type, str(current["id"]), candidate_id, hop, reason))
                remaining_tokens -= tokens
                frontier.append((candidate, hop))
                if len(selected_rows) >= MAX_LAYOUT_NODES or remaining_tokens <= 0:
                    break
        if selected_rows:
            output[assembly.anchor_id] = LayoutExpansion(
                anchor_id=assembly.anchor_id,
                text_blocks=tuple(
                    f"[Structural context: {edge.edge_type}]\n{row['text']}"
                    for row, edge in zip(selected_rows, selected_edges, strict=True)
                ),
                chunk_ids=tuple(str(row["id"]) for row in selected_rows),
                token_count=sum(max(1, int(row["token_count"] or 0)) for row in selected_rows),
                edges=tuple(selected_edges),
            )
    return output


def _chunk_row(conn, chunk_id: str):
    return storage.fetchone(
        conn,
        """
        SELECT id, doc_id, parent_id, chunk_index, text, token_count,
               block_type, heading_path, page_number, page_end, block_index,
               provenance_json
        FROM chunks WHERE id = ?
        """,
        (chunk_id,),
    )


def _should_expand(row, layout_query: bool) -> bool:
    if layout_query or row["block_type"] in {"table", "caption", "footnote"}:
        return True
    if _asset_ids(row):
        return True
    return bool(ANAPHORA_RE.search(str(row["text"]).lstrip()))


def _neighbours(conn, row, layout_query: bool) -> list[tuple[Any, str, str]]:
    candidates = storage.fetchall(
        conn,
        """
        SELECT id, doc_id, parent_id, chunk_index, text, token_count,
               block_type, heading_path, page_number, page_end, block_index,
               provenance_json
        FROM chunks
        WHERE doc_id = ? AND chunk_index BETWEEN ? AND ? AND id != ?
        ORDER BY ABS(chunk_index - ?), chunk_index
        LIMIT 12
        """,
        (
            row["doc_id"],
            int(row["chunk_index"]) - NEIGHBOR_WINDOW,
            int(row["chunk_index"]) + NEIGHBOR_WINDOW,
            row["id"],
            int(row["chunk_index"]),
        ),
    )
    related: list[tuple[int, Any, str, str]] = []
    for candidate in candidates:
        relationship = _relationship(row, candidate, layout_query)
        if relationship is None:
            continue
        priority, edge_type, reason = relationship
        related.append((priority, candidate, edge_type, reason))
    related.sort(key=lambda item: (item[0], abs(int(item[1]["chunk_index"]) - int(row["chunk_index"])), int(item[1]["chunk_index"])))
    return [(candidate, edge_type, reason) for _, candidate, edge_type, reason in related]


def _relationship(left, right, layout_query: bool) -> tuple[int, str, str] | None:
    left_assets = _asset_ids(left)
    right_assets = _asset_ids(right)
    shared_assets = left_assets & right_assets
    if shared_assets:
        edge = "caption_of" if "caption" in {left["block_type"], right["block_type"]} else "asset_context"
        return 0, edge, "chunks reference the same extracted document asset"

    same_heading = bool(left["heading_path"] and left["heading_path"] == right["heading_path"])
    same_parent = bool(left["parent_id"] and left["parent_id"] == right["parent_id"])
    page_delta = _page_delta(left, right)
    index_delta = int(right["chunk_index"]) - int(left["chunk_index"])
    if page_delta == 1 and (same_heading or same_parent):
        return 1, "continues_on_next_page", "adjacent pages retain the same section or parent"

    if left["block_type"] == "table" and right["block_type"] == "table" and same_heading:
        if index_delta < 0:
            return 1, "table_header_for", "earlier table segment in the same section supplies schema context"
        return 2, "table_continuation", "adjacent table segment retains row and column context"

    if layout_query and "caption" in {left["block_type"], right["block_type"]} and left["page_number"] == right["page_number"]:
        return 2, "caption_of", "caption and referenced content occur on the same page"

    if abs(index_delta) == 1 and (same_heading or same_parent or layout_query):
        edge = "next_block" if index_delta > 0 else "previous_block"
        return 3, edge, "immediately adjacent document block"

    if same_heading:
        return 4, "same_section", "nearby chunk has the same heading path"
    return None


def _asset_ids(row) -> set[str]:
    try:
        provenance = json.loads(row["provenance_json"] or "{}")
    except (TypeError, json.JSONDecodeError):
        return set()
    values = provenance.get("asset_ids", []) if isinstance(provenance, dict) else []
    return {str(value) for value in values if value}


def _page_delta(left, right) -> int | None:
    if left["page_number"] is None or right["page_number"] is None:
        return None
    return abs(int(right["page_number"]) - int(left["page_number"]))
