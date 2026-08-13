# Scientific RAG evaluation

The behavioral benchmark is intentionally private and outside Git. This keeps
72 PDFs, 120 curated cases, generated answers, database copies, and large
reports out of the product repository while production evaluator schemas and
metrics remain versioned in Cephalon.

## Frozen inputs

The default Windows private workspace is `C:\tmp\cephalon-private-rag`:

- Corpus PDFs: `C:\tmp\cephalon-scientific-corpus-frozen`
- Manifest and cases: `a1-full-private\benchmarks\scientific_rag`
- Versioned execution profiles: `a1-full-private\benchmarks\scientific_rag\profiles.json`
- Runner: `a1-full-private\scripts\scientific_rag_benchmark.py`
- Reusable 72-paper index: `a2-live-data`
- Per-PR reports and database copies: sibling directories under the private
  workspace

VDocRAG, OCR adapters, and image-caption adapters are excluded. Missing or
changed PDFs are reported; a replacement may be selected for app testing, but
it must receive a new private manifest entry and baseline rather than silently
reusing the old hash.

## Commands

Run from the Cephalon repository in PowerShell with Python 3.14:

```powershell
$private = 'C:\tmp\cephalon-private-rag\a1-full-private'
$runner = "$private\scripts\scientific_rag_benchmark.py"
$manifest = "$private\benchmarks\scientific_rag\corpus_manifest.json"
$cases = "$private\benchmarks\scientific_rag\cases.json"
$profiles = "$private\benchmarks\scientific_rag\profiles.json"
$env:PYTHONIOENCODING = 'utf-8'

# Validate frozen metadata without network or models.
py -3.14 $runner validate --manifest $manifest --cases $cases --profiles $profiles

# Full end-to-end run against an isolated backend.
py -3.14 $runner run --manifest $manifest --cases $cases `
  --cache-dir 'C:\tmp\cephalon-scientific-corpus-frozen' `
  --data-dir 'C:\tmp\cephalon-private-rag\a8-live-data' `
  --base-url 'http://127.0.0.1:8767' `
  --model 'ling-3.0-tiny' `
  --output 'C:\tmp\cephalon-private-rag\a8-e2e.json' --skip-download

# Cross-path smoke profile: 12 logical cases and 15 generated answers.
py -3.14 $runner run --manifest $manifest --cases $cases `
  --cache-dir 'C:\tmp\cephalon-scientific-corpus-frozen' `
  --data-dir 'C:\tmp\cephalon-private-rag\a8-live-data' `
  --base-url 'http://127.0.0.1:8767' `
  --model 'ling-3.0-tiny' `
  --output 'C:\tmp\cephalon-private-rag\a8-smoke.json' `
  --skip-download --skip-ingest --profile smoke-v1 --profiles $profiles

# Routine paired PR gate: 36 logical cases and 42 generated answers.
py -3.14 $runner run --manifest $manifest --cases $cases `
  --cache-dir 'C:\tmp\cephalon-scientific-corpus-frozen' `
  --data-dir 'C:\tmp\cephalon-private-rag\a8-live-data' `
  --base-url 'http://127.0.0.1:8767' `
  --model 'ling-3.0-tiny' `
  --output 'C:\tmp\cephalon-private-rag\a8-pr-core.json' `
  --skip-download --skip-ingest --profile pr-core-v1 --profiles $profiles

# Full retrieval-only release run (one cold and four warm repetitions).
py -3.14 $runner run --manifest $manifest --cases $cases `
  --cache-dir 'C:\tmp\cephalon-scientific-corpus-frozen' `
  --data-dir 'C:\tmp\cephalon-private-rag\a8-live-data' `
  --base-url 'http://127.0.0.1:8767' `
  --model 'ling-3.0-tiny' `
  --output 'C:\tmp\cephalon-private-rag\a8-retrieval.json' `
  --skip-download --skip-ingest --retrieval-only

# Compact base/head comparison.
py -3.14 $runner compare BASE.json HEAD.json `
  --output 'C:\tmp\cephalon-private-rag\base-head-comparison.json'
```

Profiled retrieval runs use the repetition count pinned by their definition.
`pr-core-v1`, `a8-critical-v1`, and `tables-v1` perform one cold and one warm
pass. Use the unprofiled full command for cache work or release measurement.

The comparator uses aggregate metrics only when both reports contain the same
evaluated case IDs. When profiles differ, it recomputes metrics over the exact
case-ID intersection, labels the result `matched_cases`, and omits the
whole-run latency delta. This keeps legacy full reports usable without
presenting a reduced profile as equivalent to the 120-case baseline.

For the focused named-paper gate, limit the private runner without changing
the frozen cases file:

```powershell
py -3.14 $runner run --manifest $manifest --cases $cases `
  --cache-dir 'C:\tmp\cephalon-scientific-corpus-frozen' `
  --data-dir 'C:\tmp\cephalon-private-rag\a2-live-data' `
  --base-url 'http://127.0.0.1:8767' `
  --model 'ling-3.0-tiny' `
  --output 'C:\tmp\cephalon-private-rag\a8-synthesis-focused.json' `
  --skip-download --skip-ingest --category synthesis_list
```

Run one frozen domain without changing the validated cases file:

```powershell
py -3.14 $runner run --manifest $manifest --cases $cases `
  --cache-dir 'C:\tmp\cephalon-scientific-corpus-frozen' `
  --data-dir 'C:\tmp\cephalon-private-rag\a8-live-data' `
  --base-url 'http://127.0.0.1:8767' `
  --model 'ling-3.0-tiny' `
  --output 'C:\tmp\cephalon-private-rag\a8-climate.json' `
  --skip-download --skip-ingest --domain climate
```

## Interpretation and gates

The 120-case file is frozen gold data; profiles change only which cases and
repetitions execute. Do not delete cases or edit `run_in_both_modes` and
`latency_sentinel` merely to shorten a run. Every new report records the
profile name, definition hash, selected-case hash, counts, and repetition
policy. The unprofiled full schedule preserves the legacy checkpoint
signature.

Use the gates in increasing cost order:

1. `smoke-v1` while developing.
2. `pr-core-v1` for ordinary behavioral PRs, plus the relevant feature slice.
3. Full 120-case retrieval for retrieval-affecting changes.
4. Full 136-answer generation for the final stack PR, releases, and changes to
   the model, prompts, or reranker contract.

Use `a8-critical-v1` for evidence-control and verification work,
`tables-v1` for every Stack B behavioral PR, and `performance-v1` when repeated
latency—not answer quality—is the target. Reuse a validated immutable index for
request-time changes. Reingest all 72 papers after parsing, chunking, summary,
embedding, table-schema, or index-version changes.

Deterministic grading is authoritative; chat-model semantic grading is
supplementary. Compare retrieval evidence/requirement coverage, answer and
numerical correctness, refusal behavior, citation precision/completeness,
latency, prompt/completion calls, peak RSS, and SQLite/Lance/assets/trace size.
Report every domain separately. For synthesis, report candidate retrieval
recall, final selected-document recall, distinct named-document coverage,
component coverage and citation attachment, gap rounds, validator parse and
fallback counts, reasoning leakage, calls, tokens, latency, and storage. A
changed manifest, cases file, prompt, model, llama.cpp build, seed, or server
argument invalidates a paired baseline.

Generated JSON and PDFs must remain untracked. Only compact PR descriptions and
durable production code/docs belong in Git.

## B2 table-execution result

The 2026-08-12 B1/B2 pair reused one immutable 72-paper index and ran
`tables-v1` with Ling 3.0 Tiny Q6_K and the adopted Jina v3.5 Q8_0 reranker.
B1 scored 1/18 numeric assertions (5.56%); B2 scored 17/18 (94.44%), a 94.12%
relative reduction in numeric error and above the 90% exact-value gate. There
were no request failures. Mean answer latency changed from 21.47 s to 5.95 s;
the bounded unit-answer route made zero chat-completion calls. Exact-string
accepted-answer matching fell to zero because ambiguity-preserving candidate
lists intentionally do not copy the frozen prose answer, so numeric grading,
valid citation accounting, and the explicit safety behavior are the relevant
B2 gates.
