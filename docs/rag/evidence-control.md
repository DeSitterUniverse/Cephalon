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

Coverage selection uses this requirement list before the final ledger is
materialized. Selection decisions record normalized relevance, per-requirement
coverage, diversity/coherence flags, redundancy, estimated tokens, and the
complete named weight set. Compression reports requirement coverage and source
representation so benchmark traces can distinguish retrieval misses from
context-pruning misses.
