# Research-to-implementation mapping

Cephalon adopts bounded mechanisms from RAG research rather than copying whole
reference systems or their model dependencies.

| Change | Research basis | Cephalon adaptation | Intentional deviation |
|---|---|---|---|
| Deterministic evaluation | RAGChecker, OpenScholar | Requirement, evidence, citation, refusal, numeric, and runtime metrics use backward-compatible case schemas. | Deterministic grading is authoritative; model grading is supplementary. Corpus files and generated reports remain outside Git. |
| Parent summary v2 | HiChunk, FreeChunker | Extractive summaries emphasize headings, definitions, entities, results, numbers, limitations, and conclusions. | No learned chunker or additional ingestion model. Reindex is explicit. |
| Sibling-span auto-merge | HiChunk, FreeChunker | Selected siblings form a bounded span; sufficiently covered parents may be promoted. Exact child provenance remains the citation anchor. | No model/runtime replacement. Five-child, 40% coverage, per-unit, and request token bounds prevent open-ended expansion. |
| Conditional layout evidence | LAD-RAG, SciRAG | Request-local SQLite relationships connect adjacent blocks, sections, captions/assets, table segments, and page continuations. | No graph database or corpus-wide graph. Traversal is limited to two hops, six nodes, and 25% of context tokens. |
| Request-scoped evidence ledger | S2G-RAG, SciRAG | Deterministic requirements map to bounded source evidence with sufficiency, round, and potential-conflict state. | A5 is observability-only and adds no planner-model call or retrieval loop. |
| Coverage-aware context | S2G-RAG, SciRAG, OpenScholar | Greedy selection balances requirement coverage, relevance, diversity, coherence, redundancy, and token cost; compression preserves fragile scientific evidence. | Fixed weights and hard block limits avoid an additional selector model or unbounded context growth. |
| One-round gap retrieval | S2G-RAG | Thorough mode turns unresolved deterministic requirements into one bounded targeted retrieval pass and reassesses the ledger. | No planner/judge LLM call, recursive loop, or more than one reformulated query. |
| Claim verification and repair | OpenScholar, RAGChecker | Deterministic entailment, contradiction, unit/arithmetic checks, supplementary semantic audit, and one conditional repair. | Existing generation model is reused; hard deterministic failures cannot be overruled and repair never loops. |
| Named-document sufficiency | S2G-RAG, SciRAG, OpenScholar | Quoted study targets bind to document identity and require qualifying substantive evidence before ledger coverage or gap completion. | No cross-document graph or planner model; one matching source per named study and one bounded gap query. |
| Typed table ingestion | T-RAG, T2-RAGBench | PDF, CSV, and XLSX tables receive stable normalized table/column/row/cell records while their text remains in hybrid retrieval. | Deterministic conservative typing only; no learned parser, generated SQL, formula recalculation, or replacement of text RAG. |
| Typed table execution | T-RAG, T2-RAGBench, OpenScholar | A validated application plan performs bounded lookup, filtering, sorting, grouping, aggregation, comparison, difference, and percentage arithmetic over normalized cells. Ambiguous expressed-unit questions return all cited candidates deterministically. | No model-generated SQL, planner-model call, semantic selection among ambiguous values, cross-document arithmetic, or recursive execution loop. |
| Cell citations and numerical verification | T-RAG, T2-RAGBench, RAGChecker | Exact result cells, headers, values, units, and locations survive the public source contract; deterministic support checks recompute the validated table operation and the evaluator scores only cells attached to cited sources. | No generated expressions, workbook/formula execution, semantic override of hard numeric failures, or citation credit from uncited retrieval results. |

Implementation thresholds are named and documented in
`services/context_assembly.py`. Benchmark comparisons and large scientific
corpora are operational artifacts and are intentionally not committed.
