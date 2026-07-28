"""Isolated stdio worker for Jina Reranker v3.5 custom Transformers code."""

from __future__ import annotations

import json
import sys


def main(model_dir: str) -> int:
    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, dtype="auto", trust_remote_code=True)
    model.eval()
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            results = model.rerank(payload["query"], payload["documents"], top_n=None)
            # Jina v3.5 returns listwise results sorted by relevance and keeps
            # each original input index, which is the identity mapping main
            # retrieval uses to preserve chunk/provenance records.
            output = {
                "id": payload["id"],
                "results": [
                    {"index": int(item["index"]), "relevance_score": float(item["relevance_score"])}
                    for item in results
                ],
            }
        except Exception as exc:
            output = {"id": payload.get("id") if "payload" in locals() else None, "error": str(exc)}
        print(json.dumps(output), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
