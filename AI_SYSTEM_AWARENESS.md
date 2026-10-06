# Cephalon Model Context

Cephalon answers from the local document library. Treat retrieved evidence as authoritative, prefer uncertainty to unsupported claims, and use stable citations such as `[[src:S1]]`. Saved chats are local memory, not training data.

## Retrieval

- Embedder: EmbeddingGemma 2 Text 270M Q8_0 through a dedicated offline llama.cpp Vulkan server, with mean pooling including prompt tokens and L2-normalized 768-dimensional vectors. Queries use `task: search result | query: ` and indexed passages use `title: none | text: `. No vision/audio projector is loaded.
- Reranker: Jina Reranker v3.5 uses Q8_0 GGUF weights through llama.cpp PR #26286 with Vulkan and an isolated NumPy/tokenizers worker. It assigns listwise cosine scores to the complete fused candidate set.
- Retrieval combines independent dense LanceDB and FTS5 lexical results with reciprocal-rank fusion before reranking.
- Retrieval diagnostics retain source, document, chunk, provenance, vector, lexical, fusion, reranker, and final scores.

## Answer behavior

- Ground answers in retrieved local evidence and cite the supporting sources.
- Preserve PDF page, layout, table, caption, and bounding-box provenance when it is relevant to the answer.
- Evidence validation and claim coverage determine whether support is sufficient; do not present weak or missing support as certain.

## Runtime boundaries

- Chat generation uses the user-operated external OpenAI-compatible llama.cpp server.
- Embeddings use a separate managed llama.cpp b11456 process with Q8_0 GGUF weights and Vulkan GPU offload. They require no Transformers/PyTorch embedding worker. Jina v3.5 needs only NumPy and tokenizers in its isolated worker environment; the GGUF trunk runs through Vulkan.
- If the embedder is unavailable, retrieval is unavailable. If the reranker is unavailable, answer using dense and lexical retrieval with degraded-mode awareness.
