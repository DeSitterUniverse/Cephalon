# Evidence control

The evidence ledger is the request-local state shared by later coverage,
gap-retrieval, and verification stages.

```json
{
  "state": "assessed",
  "retrieval_round": 0,
  "requirements": [
    {"id": "R1", "text": "measured latency of system B", "status": "missing", "evidence_ids": []}
  ],
  "evidence": [
    {"id": "E1", "source_id": "S1", "chunk_id": "doc_4_18", "requirement_ids": [], "retrieval_round": 0}
  ],
  "conflicts": []
}
```

Requirement states move from `missing` or `partial` to `sufficient` only when
new evidence crosses the documented assignment threshold. A potential conflict
uses `conflicting` until verification resolves it. A request starts at round
zero; later gap retrieval may add one bounded round. Duplicate queries, token
limits, time limits, and final stopping conditions are enforced by the control
stage that owns the transition.

The ledger never replaces source provenance. Evidence IDs point to exact
`SourceChunk` records; source IDs remain the citation contract used by existing
conversations and clients.

Quoted paper titles and explicit named targets create document-aware
requirements. Identity matching uses document metadata and aliases rather than
similar words inside a chunk. A source must also contain a substantive,
evidence-marked excerpt of at least the qualifying length; bibliography-only,
reference-number-only, metadata-only, and extremely short fragments do not
count. Each distinct named paper needs its own matching document, preventing a
generic or unrelated chunk from satisfying multiple study requirements.

Coverage selection uses this requirement list before the final ledger is
materialized. Selection decisions record normalized relevance, per-requirement
coverage, diversity/coherence flags, redundancy, estimated tokens, and the
complete named weight set. Compression reports requirement coverage and source
representation so benchmark traces can distinguish retrieval misses from
context-pruning misses.

Thorough mode permits one transition from `assessed` to `gap_assessed`. The
transition occurs only when a requirement is `missing`, `partial`, or
`conflicting`; `not_needed`, `duplicate_query`, `timeout`, and
`no_novel_evidence` are terminal states. The trace records the triggering
requirements, generated query, candidate/source/token counts, latency, bounds,
and stop reason. It cannot recurse into another gap round.

After synthesis, deterministic verification is authoritative for citation
presence, negation, and arithmetic. The semantic audit may recognize a valid
paraphrase that lexical coverage marked partial, but it cannot override a
deterministic contradiction or numerical failure. Thorough repair is a single
terminal transition; repaired text is verified again by the ordinary final
support classifier before storage. Reasoning-channel tokens and explicit
`<think>` blocks are removed before this final classification, and validator
JSON-schema failures are recorded with a deterministic fallback rather than a
second repair loop.
