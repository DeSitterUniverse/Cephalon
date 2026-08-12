"""Compatibility worker for the legacy Transformers Jina v3.5 install.

The primary reranker is the official Q8_0 GGUF/Vulkan runtime. This isolated
worker preserves existing installations and provides an automatic rollback
path when the GGUF assets or required llama.cpp features are unavailable.
Custom Hugging Face code remains out of the API process by design.
"""

from __future__ import annotations

import json
import sys


def main(model_dir: str) -> int:
    """Serve newline-delimited listwise rerank requests over stdio."""

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_dir, trust_remote_code=True)
    model = AutoModel.from_pretrained(model_dir, dtype="auto", trust_remote_code=True)
    model.eval()
    for line in sys.stdin:
        try:
            payload = json.loads(line)
            results = model.rerank(payload["query"], payload["documents"], top_n=None)
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
