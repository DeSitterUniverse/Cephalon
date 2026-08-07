# Research-to-implementation mapping

Cephalon adopts bounded mechanisms from RAG research rather than copying whole
reference systems or their model dependencies.

| Change | Research basis | Cephalon adaptation | Intentional deviation |
|---|---|---|---|
| Deterministic evaluation | RAGChecker, OpenScholar | Requirement, evidence, citation, refusal, numeric, and runtime metrics use backward-compatible case schemas. | Deterministic grading is authoritative; model grading is supplementary. Corpus files and generated reports remain outside Git. |
| Parent summary v2 | HiChunk, FreeChunker | Extractive summaries emphasize headings, definitions, entities, results, numbers, limitations, and conclusions. | No learned chunker or additional ingestion model. Reindex is explicit. |
| Sibling-span auto-merge | HiChunk, FreeChunker | Selected siblings form a bounded span; sufficiently covered parents may be promoted. Exact child provenance remains the citation anchor. | No model/runtime replacement. Five-child, 40% coverage, per-unit, and request token bounds prevent open-ended expansion. |

Implementation thresholds are named and documented in
`services/context_assembly.py`. Benchmark comparisons and large scientific
corpora are operational artifacts and are intentionally not committed.
