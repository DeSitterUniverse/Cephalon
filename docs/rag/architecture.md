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

## Provenance invariant

Expansion can add context but cannot invent a citation target. The public
`chunk_id`, page, bounding box, assets, and source ID always belong to the
retrieved anchor. Added sibling IDs are diagnostic provenance only. This keeps
old conversations and text citations valid.
