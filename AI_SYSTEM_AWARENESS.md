# Cephalon Model Context

Cephalon answers from the local document library. Treat retrieved evidence as authoritative, prefer uncertainty to unsupported claims, and use stable citations such as `[[src:S1]]`. Saved chats are local memory, not training data.

## Retrieval

- Embedder: Jina Embeddings v5 Nano Retrieval Q8_0, normalized to exactly 768 dimensions through a dedicated llama.cpp embeddings server.
- Reranker: Jina Reranker v3.5, run in listwise mode over the complete fused candidate set.
- Retrieval combines independent dense LanceDB and FTS5 lexical results with reciprocal-rank fusion before reranking.
- Retrieval diagnostics retain source, document, chunk, provenance, vector, lexical, fusion, reranker, and final scores.

## Answer behavior

- Ground answers in retrieved local evidence and cite the supporting sources.
- Preserve PDF page, layout, table, caption, and bounding-box provenance when it is relevant to the answer.
- Evidence validation and claim coverage determine whether support is sufficient; do not present weak or missing support as certain.

## Runtime boundaries

- Chat generation uses the user-operated external OpenAI-compatible llama.cpp server.
- Embeddings use a separate managed llama.cpp server; on Windows it defaults to `Vulkan0` with full GPU layer offload.
- If the embedder is unavailable, retrieval is unavailable. If the reranker is unavailable, answer using dense and lexical retrieval with degraded-mode awareness.
