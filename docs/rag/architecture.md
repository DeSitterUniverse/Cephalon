# RAG architecture

Cephalon indexes extractive parent summaries and child chunks in LanceDB while
retaining document structure in SQLite. A request follows this bounded path:

```text
question -> dense + FTS5 retrieval -> RRF -> Jina v3.5 rerank
         -> hierarchical context assembly -> compression -> generation
         -> citation and claim diagnostics
```

The Jina reranker is an external llama.cpp Vulkan service. Retrieval and
context assembly do not load a chat model; the selected chat model is required
only for answer generation and later model-assisted verification.

## Hierarchical context assembly

`services/context_assembly.py` converts reranked child hits into non-overlapping
context units. A unit always retains a retrieved child as its citation anchor.
It may contain:

- `child`: the exact retrieved result.
- `sibling_span`: at most five contiguous children, including at most one
  coherence neighbour on each side.
- `parent`: the complete parent when at least two independently retrieved
  children cover 40% of its child-token mass.

Parent and span text must fit `parent_max_tokens` and the request-level budget.
The request budget is capped at `rerank_top_n * parent_max_tokens`, matching the
previous worst case in which every selected child injected a full parent. Wide
sibling matches are not joined because unrelated interior material would make
the apparent evidence less precise.

Each source exposes `context_assembly` with the anchor, matched and expanded
chunk IDs, coverage, token count, and decision. Existing clients may ignore
this optional field. No reindex is needed for context-assembly changes; setting
the feature aside is equivalent to sending exact child chunks.

## Conditional layout evidence

`services/layout_expansion.py` derives request-local edges from chunk order,
heading paths, pages, block types, and shared asset IDs. It recognizes adjacent
blocks, same-section context, captions/assets, table headers/continuations, and
cross-page continuations. Expansion activates for structural block types,
layout-language questions, asset-bearing chunks, or anaphoric text.

The traversal is breadth-first and limited to two hops, six added chunks, four
neighbours per node, and 25% of pre-expansion context tokens. Migration 016
adds only `(parent_id, chunk_index)` and `(doc_id, chunk_index)` indexes; it
does not change document content and requires no reindex. The indexes can be
dropped to roll back with only a query-performance cost.

## Request-scoped evidence ledger

`services/evidence_ledger.py` maps deterministic query requirements to final
sources after compression. Requirements have `sufficient`, `partial`,
`missing`, or `conflicting` states; evidence records retain source/chunk/page,
retrieval round, and a bounded excerpt. Potential conflicts require high text
similarity plus opposite negation or incompatible values with the same unit.

The A5 ledger is observability-only: it cannot alter retrieval, context,
prompts, or answers. Migration 017 adds a JSON trace column with an empty-object
default. Traces remain readable if the column is absent, and disabling trace
persistence removes its durable cost. Hard caps are eight requirements, twenty
sources, four evidence assignments per requirement, 64 conflict comparisons,
and 500 excerpt characters.

## Coverage-aware selection and compression

`services/coverage_selection.py` greedily chooses from Jina's reranked list by
marginal value: normalized relevance and uncovered requirement coverage receive
the largest weights, with smaller source-diversity, parent-coherence, and dense
anchor bonuses; redundancy and estimated token cost are penalties. Every
source exposes the objective components in `context_selection`.

Compression uses the same requirement plan. It gives each selected source an
initial representation opportunity, rewards uncovered requirements, and adds
explicit preservation bonuses for tables/code/lists, numbers and units,
negation, and definitions. The existing `rerank_top_n * 3` output-block ceiling
is unchanged, and candidate examination is capped at 320 blocks. Setting all
coverage weights to zero and using relevance order provides a rollback without
reindexing or schema changes.

## Thorough gap retrieval

`services/retrieval_control.py` wraps the normal retriever. Balanced and Quick
return after the initial pass. Thorough inspects the ledger and may issue one
deterministic evidence query for missing, partial, or conflicting requirements.
The gap query uses the existing embedder, hybrid retrieval, Jina reranker, and
context path; it does not make a chat-model planning call.

The round is limited to one query, 12 initial candidates, three novel sources,
20 seconds, and 50% of initial context tokens (also capped by
`parent_max_tokens`). Duplicate queries and chunks represented by existing
parent/span/layout context stop expansion. Gap sources retain round `1` and the
triggering requirement IDs. Disabling the Thorough effort returns to the exact
single-pass path without reindexing or migration.

## Provenance invariant

Expansion can add context but cannot invent a citation target. The public
`chunk_id`, page, bounding box, assets, and source ID always belong to the
retrieved anchor. Added sibling IDs are diagnostic provenance only. This keeps
old conversations and text citations valid.
