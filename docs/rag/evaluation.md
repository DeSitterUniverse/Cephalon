# Scientific RAG evaluation

The behavioral benchmark is intentionally private and outside Git. This keeps
72 PDFs, 120 curated cases, generated answers, database copies, and large
reports out of the product repository while production evaluator schemas and
metrics remain versioned in Cephalon.

## Frozen inputs

The default Windows private workspace is `C:\tmp\cephalon-private-rag`:

- Corpus PDFs: `C:\tmp\cephalon-scientific-corpus-frozen`
- Manifest and cases: `a1-full-private\benchmarks\scientific_rag`
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
$env:PYTHONIOENCODING = 'utf-8'

# Validate frozen metadata without network or models.
py -3.14 $runner validate --manifest $manifest --cases $cases

# Full end-to-end run against an isolated backend.
py -3.14 $runner run --manifest $manifest --cases $cases `
  --cache-dir 'C:\tmp\cephalon-scientific-corpus-frozen' `
  --data-dir 'C:\tmp\cephalon-private-rag\a8-live-data' `
  --base-url 'http://127.0.0.1:8767' `
  --model 'gemma-4-E4B-it-UD-Q5_K_XL' `
  --output 'C:\tmp\cephalon-private-rag\a8-e2e.json' --skip-download

# One-case smoke (the first frozen case).
py -3.14 $runner run --manifest $manifest --cases $cases `
  --cache-dir 'C:\tmp\cephalon-scientific-corpus-frozen' `
  --data-dir 'C:\tmp\cephalon-private-rag\a8-live-data' `
  --base-url 'http://127.0.0.1:8767' `
  --model 'gemma-4-E4B-it-UD-Q5_K_XL' `
  --output 'C:\tmp\cephalon-private-rag\a8-one-case.json' `
  --skip-download --skip-ingest --limit 1

# Retrieval-only run (the runner performs five complete repetitions).
py -3.14 $runner run --manifest $manifest --cases $cases `
  --cache-dir 'C:\tmp\cephalon-scientific-corpus-frozen' `
  --data-dir 'C:\tmp\cephalon-private-rag\a8-live-data' `
  --base-url 'http://127.0.0.1:8767' `
  --model 'gemma-4-E4B-it-UD-Q5_K_XL' `
  --output 'C:\tmp\cephalon-private-rag\a8-retrieval.json' `
  --skip-download --skip-ingest --retrieval-only

# Compact base/head comparison.
py -3.14 $runner compare BASE.json HEAD.json `
  --output 'C:\tmp\cephalon-private-rag\base-head-comparison.json'
```

For the focused named-paper gate, limit the private runner without changing
the frozen cases file:

```powershell
py -3.14 $runner run --manifest $manifest --cases $cases `
  --cache-dir 'C:\tmp\cephalon-scientific-corpus-frozen' `
  --data-dir 'C:\tmp\cephalon-private-rag\a2-live-data' `
  --base-url 'http://127.0.0.1:8767' `
  --model 'gemma-4-E4B-it-UD-Q5_K_XL' `
  --output 'C:\tmp\cephalon-private-rag\a8-synthesis-focused.json' `
  --skip-download --skip-ingest --category synthesis_list
```

Run one frozen domain without changing the validated cases file:

```powershell
py -3.14 $runner run --manifest $manifest --cases $cases `
  --cache-dir 'C:\tmp\cephalon-scientific-corpus-frozen' `
  --data-dir 'C:\tmp\cephalon-private-rag\a8-live-data' `
  --base-url 'http://127.0.0.1:8767' `
  --model 'gemma-4-E4B-it-UD-Q5_K_XL' `
  --output 'C:\tmp\cephalon-private-rag\a8-climate.json' `
  --skip-download --skip-ingest --domain climate
```

## Interpretation and gates

Deterministic grading is authoritative; Gemma semantic grading is
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
