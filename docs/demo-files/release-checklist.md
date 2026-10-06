# Launch Notes

- Keep the library organized around source documents.
- Confirm the separate chat llama.cpp server is healthy before asking questions.
- In Settings, confirm the EmbeddingGemma 2 Text 270M Q8_0 768-dimension embedder and Jina v3.5 reranker are installed, verified, and running.
- Check the embedder and reranker runtime state; a reranker warning means degraded dense + lexical retrieval, while an embedder error blocks retrieval.
- Reindex all documents after installing the retrieval stack or when Settings reports a stale index.
- Open Sources and Retrieval Trace to inspect provenance, vector/lexical/fusion scores, raw reranker score, listwise rank, and final score.

## Demo Query

Ask: "How should I use the retrieval scope selector?"
