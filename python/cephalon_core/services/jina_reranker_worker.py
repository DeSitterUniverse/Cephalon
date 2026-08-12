"""Official Jina v3.5 Q8_0 GGUF reranking through llama.cpp/Vulkan.

This worker implements Jina's listwise LBNL inference contract around three
pinned artifacts: the Qwen3 GGUF trunk, a two-layer MLP projector, and the
tokenizer. ``llama-embedding`` executes each bounded block with irregular
sliding-window attention and returns hidden states only at Jina's document and
query marker tokens. The worker projects those states, fuses block-level query
vectors, and returns cosine-ranked input indexes to the API process.

Why a dedicated worker is necessary:

* The projector is intentionally not part of the GGUF file.
* Jina v3.5 requires the ``--output-token-ids`` llama.cpp feature so token
  positions remain exact without serialising every hidden state.
* Ranking is listwise; independently scoring query/document pairs changes the
  model and its quality.
* Every block has explicit document, token, context, process-time, and output
  bounds. A malformed model response fails closed rather than silently falling
  back to a different scoring equation.

The prompt, block fusion, and projector follow Jina's v3.5 GGUF reference and
the llama.cpp Qwen3 SWA work in PR 26286. Cephalon keeps the previous isolated
Transformers worker as a compatibility fallback.
"""

from __future__ import annotations

import json
import os
import struct
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import numpy as np


DOC_EMBED_TOKEN_ID = 151670
QUERY_EMBED_TOKEN_ID = 151671
DOC_EMBED_TOKEN = "<|embed_token|>"
QUERY_EMBED_TOKEN = "<|rerank_token|>"
SCORE_TOKEN = "<|score_token|>"
SPECIAL_TOKEN_STRINGS = (DOC_EMBED_TOKEN, QUERY_EMBED_TOKEN, SCORE_TOKEN)

HIDDEN_SIZE = 1024
PROJECTED_SIZE = 512
MAX_BLOCK_DOCUMENTS = 125
MAX_QUERY_TOKENS = 2048
MAX_DOCUMENT_TOKENS = 8192
CONTEXT_ROUNDING_TOKENS = 256
LLAMA_UBATCH_TOKENS = 512
LLAMA_TIMEOUT_SECONDS = 180

SYSTEM_PROMPT = (
    "<|im_start|>system\n"
    "You are a search relevance expert who can determine a ranking of the passages based on "
    "how relevant they are to the query. If the query is a question, how relevant a passage is "
    "depends on how well it answers the question. If not, try to analyze the intent of the query "
    "and assess how well each passage satisfies the intent. If an instruction is provided, you "
    "should follow the instruction when determining the ranking."
    "<|im_end|>\n<|im_start|>user\n"
)
NO_THINK_SUFFIX = "<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n"
RANKING_SUFFIX = (
    "\nPlease provide the ranking of all passages based on their relevance "
    "to the search query, in descending order of relevance, with each label "
    "enclosed in square brackets (e.g., [2] > [1] > [3] > [0])."
)


def _sanitize(text: str) -> str:
    """Remove control tokens so corpus text cannot inject ranking markers."""

    for token in SPECIAL_TOKEN_STRINGS:
        text = text.replace(token, "")
    return text


def format_prompt(query: str, documents: list[str]) -> str:
    """Build Jina's dual-marker listwise prompt.

    Marker order is an invariant consumed by :meth:`GgufReranker._score_block`:
    early query, one document marker per input document, then late query.
    Only the late query state participates in final scoring.
    """

    clean_query = _sanitize(query)
    clean_documents = [_sanitize(document) for document in documents]
    body = (
        f"I will provide you with {len(clean_documents)} passages, each indicated by a numerical identifier. "
        f"Rank the passages based on their relevance to query: {clean_query}{QUERY_EMBED_TOKEN}\n"
    )
    body += "\n".join(
        f'<passage id="{index}">\n{document}{DOC_EMBED_TOKEN}\n</passage>'
        for index, document in enumerate(clean_documents)
    )
    body += f"\n<query>\n{clean_query}{QUERY_EMBED_TOKEN}\n</query>"
    return SYSTEM_PROMPT + body + RANKING_SUFFIX + NO_THINK_SUFFIX


class MlpProjector:
    """Float32 inference for Jina's bias-free 1024→512→512 projection."""

    def __init__(self, first: np.ndarray, second: np.ndarray) -> None:
        if first.shape != (PROJECTED_SIZE, HIDDEN_SIZE):
            raise ValueError(f"Unexpected first projector shape: {first.shape}.")
        if second.shape != (PROJECTED_SIZE, PROJECTED_SIZE):
            raise ValueError(f"Unexpected second projector shape: {second.shape}.")
        self.first = first
        self.second = second

    def __call__(self, values: np.ndarray) -> np.ndarray:
        hidden = np.maximum(0.0, values @ self.first.T)
        return hidden @ self.second.T


def load_projector(path: Path) -> MlpProjector:
    """Load the two BF16 tensors without requiring PyTorch in the app bundle.

    Safetensors stores BF16 as little-endian 16-bit words. Promoting those
    words into the high half of IEEE float32 is exact; no quantization or
    arithmetic occurs during conversion. The parser intentionally accepts
    only the two documented Jina projector layouts and rejects aliases,
    overlaps, malformed offsets, and all other dtypes.
    """

    with path.open("rb") as stream:
        header_size_bytes = stream.read(8)
        if len(header_size_bytes) != 8:
            raise ValueError("Projector safetensors header is truncated.")
        header_size = struct.unpack("<Q", header_size_bytes)[0]
        if header_size <= 0 or header_size > 1024 * 1024:
            raise ValueError(f"Projector safetensors header size is invalid: {header_size}.")
        try:
            header = json.loads(stream.read(header_size).decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise ValueError("Projector safetensors header is malformed.") from exc
        data_start = 8 + header_size

        names = set(header) - {"__metadata__"}
        if {"projector.0.weight", "projector.2.weight"}.issubset(names):
            first_name, second_name = "projector.0.weight", "projector.2.weight"
        elif {"linear1.weight", "linear2.weight"}.issubset(names):
            first_name, second_name = "linear1.weight", "linear2.weight"
        else:
            raise ValueError(f"Unsupported projector tensor names: {sorted(names)}.")

        def read_bf16(name: str) -> np.ndarray:
            metadata = header.get(name)
            if not isinstance(metadata, dict) or metadata.get("dtype") != "BF16":
                raise ValueError(f"Projector tensor {name!r} must use BF16.")
            shape = metadata.get("shape")
            offsets = metadata.get("data_offsets")
            if (
                not isinstance(shape, list)
                or not shape
                or not all(isinstance(value, int) and value > 0 for value in shape)
                or not isinstance(offsets, list)
                or len(offsets) != 2
                or not all(isinstance(value, int) and value >= 0 for value in offsets)
                or offsets[1] < offsets[0]
            ):
                raise ValueError(f"Projector tensor {name!r} has invalid metadata.")
            element_count = int(np.prod(shape, dtype=np.int64))
            byte_count = offsets[1] - offsets[0]
            if byte_count != element_count * 2:
                raise ValueError(f"Projector tensor {name!r} has an invalid byte range.")
            stream.seek(data_start + offsets[0])
            raw = stream.read(byte_count)
            if len(raw) != byte_count:
                raise ValueError(f"Projector tensor {name!r} is truncated.")
            bf16_words = np.frombuffer(raw, dtype="<u2").astype(np.uint32)
            return (bf16_words << 16).view("<f4").reshape(shape).copy()

        first = read_bf16(first_name)
        second = read_bf16(second_name)
    return MlpProjector(first, second)


class GgufReranker:
    """Bounded listwise reranking using Jina's official Q8_0 GGUF artifacts."""

    def __init__(
        self,
        model_dir: Path,
        llama_embedding_bin: Path,
        device: str,
        gpu_layers: int,
        max_context_tokens: int,
    ) -> None:
        from tokenizers import Tokenizer

        from ..config import RERANKER_GGUF_FILE, RERANKER_PROJECTOR_FILE, RERANKER_TOKENIZER_FILE

        self.model_path = model_dir / RERANKER_GGUF_FILE
        self.llama_embedding_bin = llama_embedding_bin
        if not self.model_path.is_file():
            raise FileNotFoundError(f"Jina GGUF is missing: {self.model_path}.")
        if not self.llama_embedding_bin.is_file():
            raise FileNotFoundError(f"llama-embedding is missing: {self.llama_embedding_bin}.")
        if not 4096 <= max_context_tokens <= 131072:
            raise ValueError("max_context_tokens must be between 4,096 and 131,072.")
        self.device = device
        self.gpu_layers = gpu_layers
        self.max_context_tokens = max_context_tokens
        self.tokenizer = Tokenizer.from_file(str(model_dir / RERANKER_TOKENIZER_FILE))
        self.tokenizer.no_padding()
        self.tokenizer.no_truncation()
        self.projector = load_projector(model_dir / RERANKER_PROJECTOR_FILE)

    def _truncate(self, text: str, maximum: int) -> tuple[str, int]:
        """Return tokenizer-exact text no longer than ``maximum`` tokens."""

        token_ids = self.tokenizer.encode(text, add_special_tokens=False).ids
        if len(token_ids) >= maximum:
            token_ids = token_ids[:maximum]
            text = self.tokenizer.decode(token_ids)
        return text, len(token_ids)

    def _make_blocks(self, query: str, documents: list[str]) -> tuple[str, list[list[str]]]:
        """Truncate and pack documents under hard list and context bounds.

        Complexity is O(total input tokens). The conservative capacity reserve
        leaves room for one maximum-sized document and prompt scaffolding; the
        exact formatted length is checked again before execution.
        """

        clean_documents: list[str] = []
        lengths: list[int] = []
        for document in documents:
            clean, length = self._truncate(document, MAX_DOCUMENT_TOKENS - 1)
            clean_documents.append(clean)
            lengths.append(length)
        clean_query, query_length = self._truncate(query, MAX_QUERY_TOKENS - 64)

        capacity = self.max_context_tokens - query_length
        current_tokens = query_length
        current: list[str] = []
        blocks: list[list[str]] = []
        for length, document in zip(lengths, clean_documents):
            current.append(document)
            capacity -= length
            current_tokens += length
            if (
                len(current) >= MAX_BLOCK_DOCUMENTS
                or capacity < MAX_DOCUMENT_TOKENS
                or current_tokens >= self.max_context_tokens - MAX_DOCUMENT_TOKENS
            ):
                blocks.append(current)
                current = []
                capacity = self.max_context_tokens - query_length
                current_tokens = query_length
        if current:
            blocks.append(current)
        return clean_query, blocks

    def _score_block(self, query: str, documents: list[str]) -> tuple[np.ndarray, np.ndarray, float]:
        """Execute one block and return projected docs, late query, and weight."""

        prompt = format_prompt(query, documents)
        prompt_tokens = len(self.tokenizer.encode(prompt).ids)
        if prompt_tokens > self.max_context_tokens:
            raise ValueError(
                f"Jina reranker block has {prompt_tokens} tokens; "
                f"configured limit is {self.max_context_tokens}."
            )
        context_tokens = min(
            self.max_context_tokens,
            ((prompt_tokens + CONTEXT_ROUNDING_TOKENS - 1) // CONTEXT_ROUNDING_TOKENS)
            * CONTEXT_ROUNDING_TOKENS,
        )

        prompt_path = ""
        try:
            with tempfile.NamedTemporaryFile(
                mode="w",
                suffix=".txt",
                delete=False,
                encoding="utf-8",
            ) as prompt_file:
                prompt_file.write(prompt)
                prompt_path = prompt_file.name
            command = [
                str(self.llama_embedding_bin),
                "-m",
                str(self.model_path),
                "-f",
                prompt_path,
                "--no-escape",
                "--pooling",
                "none",
                "--embd-separator",
                "<#CEPHALON_JINA_SEPARATOR#>",
                "--embd-normalize",
                "-1",
                "--embd-output-format",
                "json",
                "--output-token-ids",
                f"{DOC_EMBED_TOKEN_ID},{QUERY_EMBED_TOKEN_ID}",
                "--ubatch-size",
                str(LLAMA_UBATCH_TOKENS),
                "--ctx-size",
                str(context_tokens),
                "--flash-attn",
                "on",
                "--fit",
                "off",
                "--device",
                self.device,
                "--gpu-layers",
                str(self.gpu_layers),
            ]
            completed = subprocess.run(
                command,
                capture_output=True,
                text=True,
                timeout=LLAMA_TIMEOUT_SECONDS,
                check=True,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError(f"Vulkan reranking exceeded {LLAMA_TIMEOUT_SECONDS} seconds.") from exc
        except subprocess.CalledProcessError as exc:
            detail = (exc.stderr or exc.stdout or "llama-embedding failed").strip()[-2000:]
            raise RuntimeError(f"Vulkan reranking failed: {detail}") from exc
        finally:
            if prompt_path:
                try:
                    os.unlink(prompt_path)
                except OSError:
                    pass

        try:
            payload = json.loads(completed.stdout)
            compact = np.asarray(
                [item["embedding"] for item in payload["data"]],
                dtype=np.float32,
            )
        except (KeyError, TypeError, ValueError, json.JSONDecodeError) as exc:
            raise RuntimeError("llama-embedding returned malformed selected-token vectors.") from exc

        expected_rows = len(documents) + 2
        if compact.shape != (expected_rows, HIDDEN_SIZE):
            raise RuntimeError(
                f"Expected selected-token shape {(expected_rows, HIDDEN_SIZE)}, got {compact.shape}."
            )
        # Encounter order is early query, every document, then late query.
        document_vectors = self.projector(compact[1:-1])
        query_vector = self.projector(compact[-1:])
        scores = _cosine_scores(document_vectors, query_vector[0])
        block_weight = float(np.max((1.0 + scores) / 2.0))
        return document_vectors, query_vector[0], block_weight

    def rerank(self, query: str, documents: list[str]) -> list[dict[str, Any]]:
        """Return every input index sorted by descending listwise relevance."""

        if not documents:
            return []
        clean_query, blocks = self._make_blocks(query, documents)
        document_vectors: list[np.ndarray] = []
        query_vectors: list[np.ndarray] = []
        weights: list[float] = []
        for block in blocks:
            docs, block_query, block_weight = self._score_block(clean_query, block)
            document_vectors.append(docs)
            query_vectors.append(block_query)
            weights.append(block_weight)

        all_documents = np.concatenate(document_vectors, axis=0)
        all_queries = np.stack(query_vectors, axis=0)
        safe_weights = np.asarray(weights, dtype=np.float32)
        if float(np.sum(safe_weights)) <= 1e-8:
            fused_query = np.mean(all_queries, axis=0)
        else:
            fused_query = np.average(all_queries, axis=0, weights=safe_weights)
        scores = _cosine_scores(all_documents, fused_query)
        order = np.argsort(scores)[::-1]
        return [
            {"index": int(index), "relevance_score": float(scores[index])}
            for index in order
        ]


def _cosine_scores(documents: np.ndarray, query: np.ndarray) -> np.ndarray:
    """Compute Jina's cosine equation with its exact numerical epsilon."""

    denominator = np.linalg.norm(documents, axis=1) * np.linalg.norm(query)
    return (documents @ query) / (denominator + 1e-8)


def main(
    model_dir: str,
    llama_embedding_bin: str,
    device: str,
    gpu_layers: str,
    max_context_tokens: str,
) -> int:
    """Serve newline-delimited rerank requests over stdio."""

    reranker = GgufReranker(
        Path(model_dir),
        Path(llama_embedding_bin),
        device,
        int(gpu_layers),
        int(max_context_tokens),
    )
    for line in sys.stdin:
        request: dict[str, Any] = {}
        try:
            request = json.loads(line)
            if not isinstance(request, dict):
                raise TypeError("Reranker request must be a JSON object.")
            results = reranker.rerank(
                str(request["query"]),
                [str(document) for document in request["documents"]],
            )
            response = {"id": request["id"], "results": results}
        except Exception as exc:
            response = {
                "id": request.get("id"),
                "error": str(exc),
            }
        print(json.dumps(response), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(*sys.argv[1:6]))
