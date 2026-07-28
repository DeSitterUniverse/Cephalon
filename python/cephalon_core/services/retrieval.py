import asyncio
import hashlib
import json
import os
import re
import time
import uuid
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import storage
from ..config import EMBEDDING_DIMENSION
from ..schemas import RagSettings, SourceChunk
from . import metrics
from . import observability
from . import jina_runtime

RRF_K = 60
CORE_MEMORY_DOC_ID = "core_memory"
EMBEDDING_CACHE_LIMIT = 128
EMBEDDING_INFERENCE_BATCH_SIZE = 16
RERANK_CACHE_LIMIT = 96
RERANK_TEXT_LIMIT = 700
MEMORY_ONLY_REQUEST = re.compile(r"\b(?:past|previous|earlier)\s+(?:conversation|chat)\s+only\b", re.IGNORECASE)
MEMORY_PREFERRED_REQUEST = re.compile(
    r"\b(?:past|previous|earlier)\s+(?:conversation|chat)\b|\b(?:we\s+(?:just\s+)?discuss(?:ed)?|remember)\b",
    re.IGNORECASE,
)
QUERY_STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "best", "for", "from", "how", "i", "in", "is", "it",
    "amount", "day", "me", "my", "of", "on", "or", "show", "the", "to", "what", "when", "which", "with",
}


@dataclass
class CompressionSource:
    source_id: str
    text: str
    rank: int
    score: float


def vector_table_name(app_state=None) -> str:
    return storage.active_vector_table_name(app_state)


def ensure_retrieval_index(app_state) -> dict[str, Any]:
    table_name = vector_table_name(app_state)
    try:
        storage.ensure_chunks_fts(app_state.sqlite)
        lexical_available = True
        lexical_error = None
    except Exception as exc:
        lexical_available = False
        lexical_error = str(exc)

    dense_available = table_name in app_state.lance.table_names()
    app_state.retrieval_index = {
        "mode": "sqlite_fts5_rrf",
        "dense_available": dense_available,
        "lexical_available": lexical_available,
        "table": table_name,
        "error": lexical_error,
    }
    return app_state.retrieval_index


def ensure_vector_table(app_state, rows: list[dict[str, Any]]):
    table_name = vector_table_name(app_state)
    if table_name in app_state.lance.table_names():
        table = app_state.lance.open_table(table_name)
        if not _vector_table_has_current_schema(table):
            app_state.lance.drop_table(table_name)
            table = app_state.lance.create_table(
                table_name,
                data=rows,
                schema=storage.vector_schema(getattr(app_state, "embedding_dim", EMBEDDING_DIMENSION)),
            )
            ensure_retrieval_index(app_state)
            return table
        if rows:
            table.add(rows)
    else:
        table = app_state.lance.create_table(table_name, data=rows, schema=storage.vector_schema(getattr(app_state, "embedding_dim", EMBEDDING_DIMENSION)))
    ensure_retrieval_index(app_state)
    return table


def _vector_table_has_current_schema(table) -> bool:
    try:
        schema = table.schema
        names = set(schema.names if hasattr(schema, "names") else schema.to_arrow_schema().names)
    except Exception:
        return True
    return {"parent_id", "source_kind", "embedding_model_id", "embedding_dim", "chunk_length"} <= names


async def get_embedding(app_state, text: str) -> list[float]:
    return await asyncio.to_thread(_get_embedding_sync, app_state, text)


async def get_embeddings(app_state, texts: list[str]) -> list[list[float]]:
    return await asyncio.to_thread(_get_embeddings_sync, app_state, texts)


def _get_embedding_sync(app_state, text: str) -> list[float]:
    return _get_embeddings_sync(app_state, [text])[0]


def _get_embeddings_sync(app_state, texts: list[str]) -> list[list[float]]:
    if getattr(app_state, "retrieval_error", None):
        raise RuntimeError(app_state.retrieval_error)
    if not texts:
        return []

    cache: OrderedDict[str, list[float]] = getattr(app_state, "embedding_cache", None)
    if cache is None:
        cache = OrderedDict()
        app_state.embedding_cache = cache

    normalized_texts = [" ".join(text.strip().split()) for text in texts]
    cache_keys = [hashlib.sha256(text.encode("utf-8")).hexdigest() for text in normalized_texts]
    vectors: list[list[float] | None] = [None] * len(texts)
    missing_indices: list[int] = []
    missing_texts: list[str] = []
    for index, cache_key in enumerate(cache_keys):
        if cache_key in cache:
            cache.move_to_end(cache_key)
            vectors[index] = list(cache[cache_key])
        else:
            missing_indices.append(index)
            missing_texts.append(normalized_texts[index])

    if missing_texts:
        configured_batch_size = getattr(app_state, "embedding_batch_size", EMBEDDING_INFERENCE_BATCH_SIZE)
        try:
            batch_size = max(1, int(configured_batch_size))
        except (TypeError, ValueError):
            batch_size = EMBEDDING_INFERENCE_BATCH_SIZE
        embedded = []
        for start in range(0, len(missing_texts), batch_size):
            batch = missing_texts[start:start + batch_size]
            try:
                embedded.extend(_run_embedding_batch(app_state, batch))
            except Exception:
                if len(batch) == 1:
                    raise
                embedded.extend(_run_embedding_batch(app_state, [text])[0] for text in batch)
        for index, vector in zip(missing_indices, embedded, strict=True):
            cache_key = cache_keys[index]
            cache[cache_key] = vector
            cache.move_to_end(cache_key)
            vectors[index] = list(vector)
        while len(cache) > EMBEDDING_CACHE_LIMIT:
            cache.popitem(last=False)

    return [list(vector) for vector in vectors if vector is not None]


def _run_embedding_batch(app_state, texts: list[str]) -> list[list[float]]:
    # The dedicated llama.cpp service performs Nano's required last-token
    # pooling.  Request batches are OpenAI-compatible and server-normalized;
    # normalize once more here to keep persisted vectors unit length.
    vectors = jina_runtime.embed(app_state, texts)
    normalized = []
    for vector in vectors:
        array = np.asarray(vector, dtype=np.float32)
        norm = np.linalg.norm(array)
        if not norm:
            raise RuntimeError("Jina Nano returned a zero embedding.")
        array = array / norm
        if array.size != EMBEDDING_DIMENSION:
            raise RuntimeError(f"Jina Nano returned {array.size} dimensions; expected fixed {EMBEDDING_DIMENSION}.")
        normalized.append(array.tolist())
    return normalized


async def save_permanent_memory(app_state, conversation_id: str, message_id: str, user_prompt: str, answer_text: str) -> None:
    """Persist a compact, searchable local memory for a completed conversation turn."""
    if not getattr(storage.get_rag_settings(app_state.sqlite), "conversation_memory", True):
        return
    memory_id = f"mem_{uuid.uuid4()}"
    # Reasoning traces are useful for the live response, but they are neither a
    # user-facing answer nor useful semantic-memory content.  Keeping them out
    # prevents a long trace from drowning out the actual answer on retrieval.
    clean_answer = re.sub(r"<think>.*?</think>", "", answer_text, flags=re.IGNORECASE | re.DOTALL).strip()
    memory_text = (
        f"[Past Conversation Context]\nUser: {user_prompt.strip()[:1600]}\n"
        f"Assistant: {(clean_answer or answer_text.strip())[:3200]}"
    )
    try:
        vector = await get_embedding(app_state, memory_text)
        lance_data = [{
            "vector": vector,
            "id": memory_id,
            "doc_id": "core_memory",
            "text": memory_text,
            "chunk_index": -1,
            "parent_id": None,
            "source_kind": "memory",
            **storage.active_embedding_metadata(app_state),
            "chunk_length": len(memory_text),
        }]
        await asyncio.to_thread(ensure_vector_table, app_state, lance_data)
        storage.execute(
            app_state.sqlite,
            "INSERT INTO conversation_memory (id, conversation_id, message_id, created_at) VALUES (?, ?, ?, ?)",
            (memory_id, conversation_id, message_id, int(time.time())),
        )
    except Exception:
        pass


def delete_conversation_memory(app_state, conversation_id: str) -> None:
    rows = storage.fetchall(app_state.sqlite, "SELECT id FROM conversation_memory WHERE conversation_id = ?", (conversation_id,))
    table_name = vector_table_name(app_state)
    if rows and table_name in app_state.lance.table_names():
        table = app_state.lance.open_table(table_name)
        for row in rows:
            try:
                table.delete(f"id = {_quote_lance_string(row['id'])}")
            except Exception:
                continue
    storage.execute(app_state.sqlite, "DELETE FROM conversation_memory WHERE conversation_id = ?", (conversation_id,))


def _quote_lance_string(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def rerank(app_state, prompt: str, results: list[dict]) -> list[dict]:
    if not results:
        return []
    cache_key = _rerank_cache_key(prompt, results)
    cache: OrderedDict[str, list[dict]] = getattr(app_state, "rerank_cache", None)
    if cache is None:
        cache = OrderedDict()
        app_state.rerank_cache = cache
    cached_scores = cache.get(cache_key)
    if cached_scores is not None and len(cached_scores) == len(results):
        cache.move_to_end(cache_key)
        listwise_results = cached_scores
    else:
        documents = [_rerank_text(res.get("text", "")) for res in results]
        try:
            listwise_results = jina_runtime.rerank(app_state, prompt, documents)
        except RuntimeError as exc:
            # Retrieval remains useful through its separately preserved dense
            # and FTS5/RRF stages when the isolated worker is unavailable.
            runtime_status = getattr(app_state, "reranker_runtime_status", None)
            if runtime_status is None:
                runtime_status = {}
                app_state.reranker_runtime_status = runtime_status
            runtime_status["last_failure"] = str(exc)
            runtime_status["status"] = "error"
            for res in results:
                res["rerank_score"] = None
                res["reranker_raw_score"] = None
                res["listwise_rank"] = None
                res["score"] = round(_retrieval_prior_score(prompt, res), 6)
                res["final_score"] = res["score"]
            return sorted(results, key=lambda item: item["score"], reverse=True)
        cache[cache_key] = listwise_results
        cache.move_to_end(cache_key)
        while len(cache) > RERANK_CACHE_LIMIT:
            cache.popitem(last=False)
    by_index = {int(item["index"]): item for item in listwise_results}
    for listwise_rank, item in enumerate(listwise_results, start=1):
        input_index = int(item["index"])
        if input_index not in by_index or not 0 <= input_index < len(results):
            raise RuntimeError("Jina v3.5 returned an invalid listwise candidate index.")
        res = results[input_index]
        raw_score = float(item["relevance_score"])
        res["reranker_raw_score"] = raw_score
        res["listwise_rank"] = listwise_rank
        res["rerank_score"] = raw_score
        res["retrieval_prior_score"] = _retrieval_prior_score(prompt, res)
        # v3.5 scores are cosine relevance values, not v3 logits.  Keep the
        # raw value separately and use a deterministic bounded fusion.
        res["score"] = round(0.75 * raw_score + 0.25 * res["retrieval_prior_score"], 6)
        res["final_score"] = res["score"]
    return sorted(results, key=lambda x: (x["score"], -int(x.get("listwise_rank") or 10**9)), reverse=True)


def _rerank_text(text: str) -> str:
    cleaned = " ".join(str(text).split())
    if len(cleaned) <= RERANK_TEXT_LIMIT:
        return cleaned
    return cleaned[:RERANK_TEXT_LIMIT]


def _rerank_cache_key(prompt: str, results: list[dict]) -> str:
    candidate_key = "|".join(f"{res.get('id')}:{hashlib.sha256(str(res.get('text', '')).encode('utf-8')).hexdigest()[:16]}" for res in results)
    return hashlib.sha256(f"{prompt.strip().lower()}::{candidate_key}".encode("utf-8")).hexdigest()


def _select_relevant_results(ranked: list[dict], limit: int) -> list[dict]:
    if not ranked:
        return []
    window = ranked[:limit]
    trusted_doc_ids = {result["doc_id"] for result in window if result.get("lexical_rank") is not None}
    if not trusted_doc_ids:
        return window

    threshold = max(0.75, float(window[0].get("score", 0.0)) * 0.2)
    selected: list[dict] = []
    for result in ranked:
        if len(selected) >= limit:
            break
        score = float(result.get("score", 0.0))
        if (
            result.get("lexical_rank") is not None
            or result.get("doc_id") in trusted_doc_ids
            or score >= threshold
        ):
            selected.append(result)
    return selected


def _memory_request_mode(prompt: str) -> tuple[bool, bool]:
    """Return (memory_only, memory_preferred) for explicit conversation references."""
    memory_only = bool(MEMORY_ONLY_REQUEST.search(prompt))
    return memory_only, memory_only or bool(MEMORY_PREFERRED_REQUEST.search(prompt))


def _retrieval_prior_score(prompt: str, result: dict) -> float:
    prior = 0.0
    if result.get("lexical_rank") is not None:
        prior += max(0.5, 3.0 - 0.12 * (int(result["lexical_rank"]) - 1))
    if result.get("dense_rank") is not None:
        prior += max(0.05, 0.8 - 0.03 * (int(result["dense_rank"]) - 1))

    important_terms = [
        term for term in re.findall(r"[\w]+", prompt.lower(), flags=re.UNICODE)
        if len(term) >= 3 and term not in QUERY_STOPWORDS
    ]
    if important_terms:
        text = result.get("text", "").lower()
        overlap = sum(1 for term in set(important_terms) if term in text)
        prior += min(1.0, overlap * 0.25)
    return round(prior, 6)


def plan_subqueries(prompt: str) -> list[dict[str, str]]:
    clean = " ".join(prompt.strip().split())
    parts = [p.strip(" ,;") for p in re.split(r"\b(?:and|also|versus|vs\.?|compare)\b|[?;]", clean, flags=re.I) if p.strip(" ,;")]
    # A short fragment such as "2027" is not an independent retrieval
    # question.  Likewise, output/citation instructions describe how to
    # answer, not what evidence to retrieve.  Searching for either dilutes
    # the candidate pool for multi-part factual questions.
    useful_parts = []
    for part in parts:
        lowered = part.lower()
        if re.match(r"^(?:cite|include|use|answer|respond|format)\b", lowered):
            continue
        terms = [
            term for term in re.findall(r"[\w]+", lowered, flags=re.UNICODE)
            if len(term) >= 3 and term not in QUERY_STOPWORDS
        ]
        if len(terms) >= 2:
            useful_parts.append(part)
    if len(parts) <= 1:
        return [{"id": "q1", "text": clean}]
    if len(useful_parts) <= 1:
        return [{"id": "q0", "text": clean}]
    # Keep the complete question as the primary retrieval intent. Decomposed
    # parts improve recall but must not erase comparison/relationship context.
    return [{"id": "q0", "text": clean}] + [{"id": f"q{idx}", "text": part} for idx, part in enumerate(useful_parts[:5], start=1)]


def hydrate_sources(app_state, results: list[dict], subquery_id: str | None = None, start_rank: int = 1) -> list[SourceChunk]:
    if not results:
        return []
    doc_ids = list({res["doc_id"] for res in results if res["doc_id"] != "core_memory"})
    path_map = {}
    if doc_ids:
        placeholders = ",".join("?" * len(doc_ids))
        rows = storage.fetchall(app_state.sqlite, f"SELECT id, path, display_name FROM documents WHERE id IN ({placeholders})", tuple(doc_ids))
        path_map = {row["id"]: row["display_name"] or os.path.basename(row["path"]) for row in rows}
    chunk_ids = [res["id"] for res in results if res.get("doc_id") != "core_memory"]
    provenance_map: dict[str, dict[str, Any]] = {}
    asset_map: dict[tuple[str, str], dict[str, Any]] = {}
    if doc_ids:
        placeholders = ",".join("?" * len(doc_ids))
        asset_rows = storage.fetchall(
            app_state.sqlite,
            f"""
            SELECT id, doc_id, page_number, bounding_box, mime_type, caption,
                   width, height
            FROM document_assets
            WHERE doc_id IN ({placeholders})
            """,
            tuple(doc_ids),
        )
        for row in asset_rows:
            bbox = _json_metadata(row["bounding_box"], None)
            asset_map[(row["doc_id"], row["id"])] = {
                "asset_id": row["id"],
                "page_number": row["page_number"],
                "bounding_box": tuple(bbox) if isinstance(bbox, list) and len(bbox) == 4 else None,
                "mime_type": row["mime_type"],
                "caption": row["caption"],
                "width": row["width"],
                "height": row["height"],
                "url": f"/documents/{row['doc_id']}/assets/{row['id']}",
            }
    if chunk_ids:
        placeholders = ",".join("?" * len(chunk_ids))
        rows = storage.fetchall(
            app_state.sqlite,
            f"""
            SELECT id, block_type, section_heading, heading_path, page_number,
                   page_end, block_index, bounding_box, provenance_json
            FROM chunks
            WHERE id IN ({placeholders})
            """,
            tuple(chunk_ids),
        )
        for row in rows:
            heading_path = _json_metadata(row["heading_path"], [])
            bounding_box = _json_metadata(row["bounding_box"], None)
            provenance = _json_metadata(row["provenance_json"], {})
            provenance_map[row["id"]] = {
                "block_type": row["block_type"],
                "section_heading": row["section_heading"],
                "heading_path": heading_path if isinstance(heading_path, list) else [],
                "page_number": row["page_number"],
                "page_end": row["page_end"],
                "block_index": row["block_index"],
                "bounding_box": tuple(bounding_box) if isinstance(bounding_box, list) and len(bounding_box) == 4 else None,
                "element_ids": provenance.get("element_ids", []) if isinstance(provenance, dict) else [],
                "provenance": provenance if isinstance(provenance, dict) else {},
            }

    sources: list[SourceChunk] = []
    for rank, res in enumerate(results, start=start_rank):
        doc_id = res["doc_id"]
        doc_name = "Core Memory" if doc_id == "core_memory" else path_map.get(doc_id, "Unknown")
        text = res["text"].strip()
        source_id = f"S{rank}"
        provenance = provenance_map.get(res["id"], {})
        source_asset_ids = provenance.get("provenance", {}).get("asset_ids", [])
        source_assets = [
            asset_map[(doc_id, asset_id)]
            for asset_id in source_asset_ids
            if (doc_id, asset_id) in asset_map
        ]
        sources.append(SourceChunk(
            rank=rank,
            source_id=source_id,
            doc_id=doc_id,
            doc_name=doc_name,
            chunk_id=res["id"],
            parent_id=res.get("parent_id"),
            score=float(res.get("score", 0)),
            final_score=float(res["final_score"]) if res.get("final_score") is not None else float(res.get("score", 0)),
            vector_score=float(res["_distance"]) if "_distance" in res and res["_distance"] is not None else None,
            rerank_score=float(res["rerank_score"]) if res.get("rerank_score") is not None else None,
            reranker_raw_score=float(res["reranker_raw_score"]) if res.get("reranker_raw_score") is not None else None,
            listwise_rank=int(res["listwise_rank"]) if res.get("listwise_rank") is not None else None,
            lexical_score=float(res["lexical_score"]) if res.get("lexical_score") is not None else None,
            fusion_score=float(res["fusion_score"]) if res.get("fusion_score") is not None else None,
            snippet=text[:500],
            subquery_id=subquery_id or ",".join(res.get("subquery_ids", [])) or None,
            assets=source_assets,
            **provenance,
        ))
    return sources


def _json_metadata(value: str | None, default: Any) -> Any:
    if not value:
        return default
    try:
        return json.loads(value)
    except (TypeError, json.JSONDecodeError):
        return default


def source_location_label(source: SourceChunk | None) -> str:
    if source is None:
        return ""
    details: list[str] = []
    if source.page_number is not None:
        page_label = str(source.page_number)
        if source.page_end is not None and source.page_end != source.page_number:
            page_label = f"{page_label}-{source.page_end}"
        details.append(f"page {page_label}")
    if source.section_heading:
        details.append(source.section_heading)
    if source.block_type and source.block_type != "paragraph":
        details.append(source.block_type)
    return " | ".join(details)


def _merge_candidates(target: dict[str, dict], results: list[dict], subquery_id: str) -> None:
    for result in results:
        chunk_id = result["id"]
        existing = target.get(chunk_id)
        if existing is None:
            merged = dict(result)
            merged["subquery_ids"] = [subquery_id]
            target[chunk_id] = merged
            continue
        if subquery_id not in existing["subquery_ids"]:
            existing["subquery_ids"].append(subquery_id)
        # A candidate can arrive through different subqueries with different
        # RRF scores.  Preserve its best original retrieval ranks even when a
        # later result replaces the score-bearing row, so that a top dense hit
        # remains eligible for the context safety net below.
        best_ranks = {
            field: min(
                value for value in (existing.get(field), result.get(field))
                if value is not None
            )
            for field in ("dense_rank", "lexical_rank", "summary_rank")
            if any(value is not None for value in (existing.get(field), result.get(field)))
        }
        # RRF scores are comparable across these equally configured searches.
        if float(result.get("score", 0)) > float(existing.get("score", 0)):
            subqueries = existing["subquery_ids"]
            existing.clear()
            existing.update(dict(result))
            existing["subquery_ids"] = subqueries
        existing.update(best_ranks)


async def _search_once(app_state, prompt: str, query_vector: list[float], settings: RagSettings) -> tuple[list[dict], str, dict[str, Any]]:
    table_name = vector_table_name(app_state)
    index = ensure_retrieval_index(app_state)
    latency: dict[str, float] = {}
    stage_started = time.perf_counter()
    summary_results = _summary_dense_search(app_state, table_name, query_vector, max(4, settings.top_k // 3))
    latency["summary_ms"] = round((time.perf_counter() - stage_started) * 1000, 2)
    summary_parent_rank = {
        result["parent_id"]: rank
        for rank, result in enumerate(summary_results, start=1)
        if result.get("parent_id")
    }
    stage_started = time.perf_counter()
    dense_results = _dense_search(app_state, table_name, query_vector, settings.top_k)
    latency["vector_ms"] = round((time.perf_counter() - stage_started) * 1000, 2)
    stage_started = time.perf_counter()
    lexical_results = _lexical_search(app_state, prompt, settings.top_k)
    latency["bm25_ms"] = round((time.perf_counter() - stage_started) * 1000, 2)
    _apply_summary_parent_boost(dense_results, summary_parent_rank)
    _apply_summary_parent_boost(lexical_results, summary_parent_rank)
    stage_started = time.perf_counter()
    fused = _fuse_rrf(dense_results, lexical_results, settings.top_k)
    latency["fusion_ms"] = round((time.perf_counter() - stage_started) * 1000, 2)
    modes = []
    if summary_results:
        modes.append("summary_dense")
    if dense_results:
        modes.append("dense")
    if lexical_results:
        modes.append("sqlite_fts5")
    if not index.get("lexical_available"):
        modes.append("lexical_unavailable")
    trace = {
        "summary_candidates": [_trace_candidate(row, rank) for rank, row in enumerate(summary_results, start=1)],
        "vector_candidates": [_trace_candidate(row, rank) for rank, row in enumerate(dense_results, start=1)],
        "bm25_candidates": [_trace_candidate(row, rank) for rank, row in enumerate(lexical_results, start=1)],
        "fused_candidates": [_trace_candidate(row, rank) for rank, row in enumerate(fused, start=1)],
        "latency": latency,
    }
    return fused, "+".join(modes) if modes else "empty", trace


def _summary_dense_search(app_state, table_name: str, query_vector: list[float], limit: int) -> list[dict]:
    if table_name not in app_state.lance.table_names():
        return []
    table = app_state.lance.open_table(table_name)
    try:
        query = table.search(query_vector, vector_column_name="vector")
    except TypeError:
        query = table.search(query_vector)
    rows = query.limit(max(limit * 4, limit + 20)).to_list()
    summaries = [
        dict(row) for row in rows
        if row.get("doc_id") != CORE_MEMORY_DOC_ID and row.get("source_kind") == "summary" and row.get("parent_id")
    ]
    for rank, row in enumerate(summaries[:limit], start=1):
        row["summary_rank"] = rank
    return summaries[:limit]


def _apply_summary_parent_boost(results: list[dict], summary_parent_rank: dict[str, int]) -> None:
    for result in results:
        parent_id = result.get("parent_id")
        if parent_id in summary_parent_rank:
            result["summary_rank"] = summary_parent_rank[parent_id]


def _dense_search(app_state, table_name: str, query_vector: list[float], limit: int) -> list[dict]:
    if table_name not in app_state.lance.table_names():
        return []
    table = app_state.lance.open_table(table_name)
    search_limit = max(limit + 50, limit * 5)
    try:
        query = table.search(query_vector, vector_column_name="vector")
    except TypeError:
        query = table.search(query_vector)
    try:
        rows = query.where(f"doc_id != '{CORE_MEMORY_DOC_ID}'").limit(search_limit).to_list()
    except Exception:
        rows = query.limit(search_limit).to_list()
    rows = [
        row for row in rows
        if row.get("doc_id") != CORE_MEMORY_DOC_ID and row.get("source_kind", "child") != "summary"
    ][:limit]
    results = []
    for rank, row in enumerate(rows, start=1):
        item = dict(row)
        item.setdefault("parent_id", row.get("parent_id") if hasattr(row, "get") else None)
        item.setdefault("source_kind", row.get("source_kind") if hasattr(row, "get") else "child")
        item["dense_rank"] = rank
        item["vector_score"] = _distance_to_score(item.get("_distance"))
        results.append(item)
    return results


def _memory_dense_search(app_state, table_name: str, query_vector: list[float], limit: int = 2) -> list[dict]:
    if table_name not in app_state.lance.table_names():
        return []
    table = app_state.lance.open_table(table_name)
    try:
        query = table.search(query_vector, vector_column_name="vector")
    except TypeError:
        query = table.search(query_vector)
    try:
        rows = query.where("source_kind = 'memory'").limit(limit).to_list()
    except Exception:
        rows = query.limit(max(limit * 10, 20)).to_list()
    results = []
    for rank, row in enumerate((item for item in rows if item.get("source_kind") == "memory") , start=1):
        if rank > limit:
            break
        item = dict(row)
        item["dense_rank"] = rank
        item["vector_score"] = _distance_to_score(item.get("_distance"))
        item["score"] = float(item["vector_score"] or 0.0)
        results.append(item)
    return results


def _lexical_search(app_state, prompt: str, limit: int) -> list[dict]:
    storage.ensure_chunks_fts(app_state.sqlite)
    rows = storage.fetchall(
        app_state.sqlite,
        """
        SELECT
            chunks.id,
            chunks.doc_id,
            chunks.chunk_index,
            chunks.text,
            chunks.parent_id,
            chunks.chunk_length,
            chunks.embedding_model_id,
            chunks.embedding_dim,
            bm25(chunks_fts) AS bm25_score
        FROM chunks_fts
        JOIN chunks ON chunks.id = chunks_fts.chunk_id
        WHERE chunks_fts MATCH ?
        ORDER BY bm25_score
        LIMIT ?
        """,
        (_fts_query(prompt), limit),
    )
    results = []
    for rank, row in enumerate(rows, start=1):
        item = {key: row[key] for key in row.keys()}
        item["lexical_rank"] = rank
        item["lexical_score"] = float(item["bm25_score"])
        item["score"] = -float(item["bm25_score"])
        results.append(item)
    return results


def _fts_query(prompt: str) -> str:
    raw_terms = [term.lower() for term in re.findall(r"[\w]+", prompt, flags=re.UNICODE)]
    terms: list[str] = []
    for term in raw_terms:
        if len(term) < 3 or term in QUERY_STOPWORDS:
            continue
        terms.append(term)
        if term.endswith("s") and len(term) > 4:
            terms.append(term[:-1])
    if not terms:
        return '""'
    unique_terms = list(dict.fromkeys(terms))
    return " OR ".join(f"{term}*" for term in unique_terms[:24])


def _fuse_rrf(dense_results: list[dict], lexical_results: list[dict], limit: int) -> list[dict]:
    fused: dict[str, dict] = {}
    for result in dense_results:
        chunk_id = result["id"]
        entry = fused.setdefault(chunk_id, dict(result))
        entry["dense_rank"] = result["dense_rank"]
        entry["vector_score"] = result.get("vector_score")
        entry["fusion_score"] = entry.get("fusion_score", 0.0) + 1.0 / (RRF_K + result["dense_rank"])
    for result in lexical_results:
        chunk_id = result["id"]
        entry = fused.setdefault(chunk_id, dict(result))
        entry["lexical_rank"] = result["lexical_rank"]
        entry["lexical_score"] = result.get("lexical_score")
        entry["fusion_score"] = entry.get("fusion_score", 0.0) + 1.0 / (RRF_K + result["lexical_rank"])
    for entry in fused.values():
        if entry.get("summary_rank") is not None:
            entry["fusion_score"] = entry.get("fusion_score", 0.0) + 1.0 / (RRF_K + int(entry["summary_rank"]))
        entry["score"] = float(entry.get("fusion_score", 0.0))
    return sorted(fused.values(), key=lambda item: item["score"], reverse=True)[:limit]


def _distance_to_score(distance: Any) -> float | None:
    if distance is None:
        return None
    try:
        return 1.0 / (1.0 + float(distance))
    except Exception:
        return None


def split_sentences(text: str) -> list[str]:
    return [
        sentence.strip()
        for sentence in re.split(r"(?<=[.!?])\s+|\n+", text)
        if sentence.strip()
    ]


def split_context_blocks(text: str) -> list[tuple[str, str]]:
    """Keep tables, code, and lists intact instead of slicing them as prose."""
    blocks: list[tuple[str, str]] = []
    for block in re.split(r"\n\s*\n", text.strip()):
        clean = block.strip()
        if not clean:
            continue
        lines = [line.rstrip() for line in clean.splitlines() if line.strip()]
        structured = clean.startswith("```") or any("\t" in line or "|" in line for line in lines) or all(
            re.match(r"\s*(?:[-*•]|\d+[.)])\s+", line) for line in lines
        )
        if structured:
            blocks.append(("structured", clean))
        else:
            blocks.extend(("prose", sentence) for sentence in split_sentences(clean))
    return blocks


def format_source_context(source_id: str, doc_name: str, chunk_id: str, text: str) -> str:
    return f"[[src:{source_id}]] Source: {doc_name} | Chunk: {chunk_id}\n{text.strip()}"


def _term_set(text: str) -> set[str]:
    return {
        term
        for term in re.findall(r"[\w]+", text.lower(), flags=re.UNICODE)
        if len(term) >= 3 and term not in QUERY_STOPWORDS
    }


def _sentence_relevance(query_terms: set[str], sentence: str) -> float:
    sentence_terms = _term_set(sentence)
    if not query_terms or not sentence_terms:
        return 0.0
    return len(query_terms & sentence_terms) / len(query_terms | sentence_terms)


def compress_context(
    query: str,
    sources: list[CompressionSource],
    max_sentences: int = 10,
) -> tuple[str, dict[str, Any], dict[str, str]]:
    query_terms = _term_set(query)
    candidates: list[dict[str, Any]] = []
    for source in sources:
        for block_type, text in split_context_blocks(source.text):
            relevance = _sentence_relevance(query_terms, text)
            candidates.append({
                "source_id": source.source_id,
                "sentence": text,
                "block_type": block_type,
                "rank": source.rank,
                "score": relevance + (source.score * 0.08) + (1.0 / max(source.rank, 1) * 0.04),
                "relevant": relevance > 0,
            })
    candidates.sort(key=lambda item: item["score"], reverse=True)

    kept: list[dict[str, Any]] = []
    seen_terms: set[str] = set()
    for candidate in candidates:
        terms = _term_set(candidate["sentence"])
        overlap = len(terms & seen_terms) / max(len(terms), 1)
        if overlap > 0.75 and len(kept) >= 1:
            continue
        kept.append(candidate)
        seen_terms.update(terms)
        if len(kept) >= max_sentences:
            break

    if not kept and candidates:
        kept = candidates[:max_sentences]

    total_sentences = len(candidates)
    relevant_total = sum(1 for item in candidates if item["relevant"])
    compressed = "\n".join(f"[[src:{item['source_id']}]] {item['sentence']}" for item in kept)
    evidence_by_source: dict[str, list[str]] = {}
    for item in kept:
        evidence_by_source.setdefault(item["source_id"], []).append(item["sentence"])
    stats = {
        "input_sentences": total_sentences,
        "kept_sentences": len(kept),
        "relevant_sentence_count": relevant_total,
        "context_relevance": round(relevant_total / total_sentences, 6) if total_sentences else 0.0,
    }
    return compressed, stats, {
        source_id: "\n".join(evidence)
        for source_id, evidence in evidence_by_source.items()
    }


def confidence_from_sources(sources: list[SourceChunk], settings: RagSettings | None = None) -> dict[str, Any]:
    thresholds = None
    if settings is not None:
        thresholds = {
            "min_confidence": settings.no_answer_min_confidence,
            "min_rerank_score": settings.no_answer_min_rerank_score,
            "min_vector_score": settings.no_answer_min_vector_score,
            "min_source_count": settings.no_answer_min_source_count,
        }
    return observability.no_answer_diagnostics(sources, thresholds)


async def retrieve_context(app_state, prompt: str, query_vector: list[float], settings: RagSettings) -> tuple[str, list[SourceChunk], dict[str, Any]]:
    started = time.perf_counter()
    query_id = str(uuid.uuid4())
    context_chunks: list[str] = []
    all_sources: list[SourceChunk] = []
    compression_inputs: list[CompressionSource] = []
    search_modes: list[str] = []
    trace: dict[str, Any] = {
        "query_id": query_id,
        "raw_query": prompt,
        "normalized_query": " ".join(prompt.strip().split()),
        "rewritten_query": None,
        "timestamp": int(time.time()),
        "retrieval_mode": "",
        "subqueries": [],
        "vector_candidates": [],
        "memory_candidates": [],
        "bm25_candidates": [],
        "fused_candidates": [],
        "reranked_candidates": [],
        "final_context": [],
        "unused_candidates": [],
        "latency": {"preprocessing_ms": 0, "rewrite_ms": 0, "vector_ms": 0, "bm25_ms": 0, "fusion_ms": 0, "rerank_ms": 0, "context_ms": 0},
        "no_answer": {},
    }
    subqueries = plan_subqueries(prompt)
    memory_only, memory_preferred = _memory_request_mode(prompt)
    trace["subqueries"] = subqueries
    numeric_context, numeric_sources = _structured_numeric_analysis_for_query(app_state, prompt)
    if numeric_context:
        context_chunks.extend(numeric_context)
        all_sources.extend(numeric_sources)
        compression_inputs.extend([
            CompressionSource(source_id=source.source_id or f"S{source.rank}", text=numeric_context[0], rank=source.rank, score=source.score)
            for source in numeric_sources
        ])
        search_modes.append("numeric_scan")
        trace["reranked_candidates"] = [source.model_dump() for source in numeric_sources]
    else:
        merged_candidates: dict[str, dict] = {}
        table_name = vector_table_name(app_state)
        # Respect the user's per-query memory setting.  Otherwise unrelated
        # remembered conversations can consume a context slot for a document
        # question even when the caller explicitly turned memory off.
        memory_results = _memory_dense_search(app_state, table_name, query_vector) if settings.conversation_memory else []
        if memory_results:
            _merge_candidates(merged_candidates, memory_results, "memory")
            trace["memory_candidates"] = [_trace_candidate(row, rank) for rank, row in enumerate(memory_results, start=1)]
            search_modes.append("conversation_memory")

        if not memory_only:
            for subquery in subqueries:
                vector = query_vector if subquery["text"] == prompt else await get_embedding(app_state, subquery["text"])
                search_output = await _search_once(app_state, subquery["text"], vector, settings)
                if len(search_output) == 2:
                    results, mode = search_output
                    stage_trace = {"latency": {}}
                else:
                    results, mode, stage_trace = search_output
                search_modes.append(mode)
                _merge_candidates(merged_candidates, results, subquery["id"])
                trace["vector_candidates"].extend(stage_trace.get("vector_candidates", []))
                trace["bm25_candidates"].extend(stage_trace.get("bm25_candidates", []))
                trace["fused_candidates"].extend(stage_trace.get("fused_candidates", []))
                for key in ("vector_ms", "bm25_ms", "fusion_ms"):
                    trace["latency"][key] = round(trace["latency"].get(key, 0) + stage_trace.get("latency", {}).get(key, 0), 2)

        candidates = sorted(merged_candidates.values(), key=lambda row: float(row.get("score", 0)), reverse=True)
        # Memory requests must not be crowded out before reranking by a larger
        # document corpus.  For an explicit "past conversation only" request,
        # documents are intentionally excluded above; otherwise reserve the
        # leading rerank slot for the best semantic-memory candidate.
        if memory_preferred:
            memory_candidates = [row for row in candidates if row.get("doc_id") == CORE_MEMORY_DOC_ID]
            non_memory_candidates = [row for row in candidates if row.get("doc_id") != CORE_MEMORY_DOC_ID]
            candidates = memory_candidates + non_memory_candidates
        # RRF is excellent for consensus, but a precise dense hit can score
        # below many weaker lexical matches when a question is decomposed.
        # Keep each subquery's top dense result in the rerank window and in the
        # final context as a small recall safety net.
        anchor_ids = {
            row["id"] for row in candidates
            if row.get("doc_id") != CORE_MEMORY_DOC_ID and row.get("dense_rank") == 1
        }
        anchors = [row for row in candidates if row["id"] in anchor_ids]
        non_anchors = [row for row in candidates if row["id"] not in anchor_ids]
        # Jina v3.5 is listwise: its relevance scores depend on the complete
        # candidate set, so never pre-cut the fused set before one rerank call.
        rerank_candidates = anchors + non_anchors
        rerank_started = time.perf_counter()
        all_ranked = await asyncio.to_thread(rerank, app_state, prompt, rerank_candidates)
        trace["latency"]["rerank_ms"] = round((time.perf_counter() - rerank_started) * 1000, 2)
        if memory_only:
            reranked = all_ranked[:settings.rerank_top_n]
        else:
            reranked = _select_relevant_results(all_ranked, settings.rerank_top_n)
            anchor_ranked = [row for row in all_ranked if row["id"] in anchor_ids]
            if anchor_ranked:
                reranked = (anchor_ranked + [
                    row for row in reranked if row["id"] not in anchor_ids
                ])[:settings.rerank_top_n]
            if memory_preferred:
                memory_ranked = [row for row in all_ranked if row.get("doc_id") == CORE_MEMORY_DOC_ID]
                if memory_ranked:
                    # An explicit reference to prior chat is stronger evidence
                    # of intent than a generic document tie-breaker.
                    reranked = (memory_ranked[:1] + [
                        row for row in reranked if row.get("id") != memory_ranked[0].get("id")
                    ])[:settings.rerank_top_n]
        selected_ids = {item["id"] for item in reranked}
        trace["reranked_candidates"] = [_trace_candidate(row, rank) for rank, row in enumerate(all_ranked, start=1)]
        trace["unused_candidates"].extend([
            {**_trace_candidate(row, rank), "reason": "below final context cutoff"}
            for rank, row in enumerate(all_ranked, start=1)
            if row["id"] not in selected_ids
        ][: max(settings.top_k, 10)])
        all_sources = hydrate_sources(app_state, reranked)
        source_by_chunk = {source.chunk_id: source for source in all_sources}
        for res in reranked:
            source = source_by_chunk.get(res["id"])
            if res["doc_id"] == CORE_MEMORY_DOC_ID:
                context_chunks.append(format_source_context(source.source_id or "S1", "Past conversation", res["id"], res["text"]))
                if source:
                    source.evidence_text = res["text"].strip()
                continue
            label = source.doc_name if source else "Unknown"
            if location := source_location_label(source):
                label = f"{label} | {location}"
            source_id = source.source_id if source and source.source_id else f"S{len(all_sources) + 1}"
            context_text = _parent_context(app_state, res) or res["text"]
            context_chunks.append(format_source_context(source_id, label, res["id"], context_text))
            # Parent text provides useful surrounding context for ordinary
            # results.  For a top dense anchor, however, retain the exact
            # child hit during compression so a large parent section cannot
            # crowd out the sentence that earned the top semantic rank.
            compression_text = res["text"] if res["id"] in anchor_ids else context_text
            compression_inputs.append(CompressionSource(source_id=source_id, text=compression_text, rank=source.rank if source else 99, score=source.score if source else 0.0))

    if compression_inputs:
        compressed_context, compression_stats, evidence_by_source = compress_context(
            prompt,
            compression_inputs,
            max_sentences=max(6, settings.rerank_top_n * 3),
        )
        if compressed_context:
            memory_context = [chunk for chunk in context_chunks if "Past conversation" in chunk]
            context_chunks = memory_context + [compressed_context]
            for source in all_sources:
                source.evidence_text = evidence_by_source.get(source.source_id or "")
    else:
        compression_stats = {"input_sentences": 0, "kept_sentences": 0, "context_relevance": 0.0}

    _mark_sources_retrieved(app_state, all_sources)
    confidence = confidence_from_sources(all_sources, settings)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
    trace["retrieval_mode"] = "+".join(search_modes) if search_modes else "empty"
    trace["final_context"] = [source.model_dump() for source in all_sources]
    trace["no_answer"] = confidence
    trace["latency"]["total_ms"] = elapsed_ms
    try:
        metrics_path = metrics.append_retrieval_event(app_state, {
            "query_hash": hashlib.sha256(prompt.encode("utf-8")).hexdigest(),
            "query_length": len(prompt),
            "subquery_count": len(subqueries),
            "retrieval_latency_ms": elapsed_ms,
            "search_modes": search_modes,
            "source_count": len(all_sources),
            "scores": [source.score for source in all_sources],
            "vector_scores": [source.vector_score for source in all_sources if source.vector_score is not None],
            "lexical_scores": [source.lexical_score for source in all_sources if source.lexical_score is not None],
            "fusion_scores": [source.fusion_score for source in all_sources if source.fusion_score is not None],
            "rerank_scores": [source.rerank_score for source in all_sources if source.rerank_score is not None],
            "confidence": confidence["confidence"],
            "no_answer": confidence["no_answer"],
            "context_relevance": compression_stats.get("context_relevance", 0.0),
        })
    except OSError as error:
        metrics_path = None
        app_state.last_metrics_error = str(error)
    meta = {
        **confidence,
        "query_id": query_id,
        "subqueries": subqueries,
        "retrieval_latency_ms": elapsed_ms,
        "search_modes": search_modes,
        "metrics_path": metrics_path,
        "compression": compression_stats,
        "trace": trace,
    }
    return "\n\n".join(context_chunks) if context_chunks else "No relevant memories or documents found.", all_sources, meta


def _trace_candidate(row: dict[str, Any], rank: int) -> dict[str, Any]:
    return {
        "rank": rank,
        "chunk_id": _trace_value(row.get("id") or row.get("chunk_id")),
        "doc_id": _trace_value(row.get("doc_id")),
        "parent_id": _trace_value(row.get("parent_id")),
        "source_kind": _trace_value(row.get("source_kind")),
        "chunk_index": _trace_value(row.get("chunk_index")),
        "score": _trace_value(row.get("score")),
        "final_score": _trace_value(row.get("final_score", row.get("score"))),
        "vector_score": _trace_value(row.get("vector_score")),
        "lexical_score": _trace_value(row.get("lexical_score")),
        "fusion_score": _trace_value(row.get("fusion_score")),
        "rerank_score": _trace_value(row.get("rerank_score")),
        "reranker_raw_score": _trace_value(row.get("reranker_raw_score")),
        "listwise_rank": _trace_value(row.get("listwise_rank")),
        "dense_rank": _trace_value(row.get("dense_rank")),
        "lexical_rank": _trace_value(row.get("lexical_rank")),
        "summary_rank": _trace_value(row.get("summary_rank")),
        "snippet": str(row.get("text", ""))[:500],
    }


def _trace_value(value: Any) -> Any:
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            return str(value)
    return value


def _parent_context(app_state, result: dict) -> str | None:
    parent_id = result.get("parent_id")
    if not parent_id:
        return None
    row = storage.fetchone(app_state.sqlite, "SELECT text FROM parent_chunks WHERE id = ?", (parent_id,))
    return row["text"] if row else None


def _structured_numeric_analysis_for_query(app_state, prompt: str) -> tuple[list[str], list[SourceChunk]]:
    lowered = prompt.lower()
    if not _looks_like_numeric_record_question(lowered):
        return [], []
    if not any(term in lowered for term in ("heaviest", "highest", "max", "maximum", "most", "largest")):
        return [], []

    rows = storage.fetchall(
        app_state.sqlite,
        """
        SELECT documents.id AS doc_id, documents.path, documents.display_name, chunks.id AS chunk_id, chunks.text
        FROM documents
        JOIN chunks ON chunks.doc_id = documents.id
        WHERE documents.type = 'file'
        ORDER BY documents.ingested_at DESC, chunks.chunk_index
        """,
    )
    best: dict[str, Any] | None = None
    metric = _numeric_record_metric(lowered)
    for row in rows:
        for match in re.finditer(r"(\d{4}[/-]\d{2}[/-]\d{2})\s+(\d+)(?:/(\d+))?", row["text"]):
            first_value = int(match.group(2))
            second_value = int(match.group(3)) if match.group(3) is not None else None
            total_value = first_value + second_value if second_value is not None else first_value
            value = second_value if metric == "second" and second_value is not None else first_value if metric == "first" else total_value
            if best is None or value > best["value"]:
                best = {
                    "date": match.group(1),
                    "first_value": first_value,
                    "second_value": second_value,
                    "total_value": total_value,
                    "value": value,
                    "metric": metric,
                    "doc_id": row["doc_id"],
                    "doc_name": row["display_name"] or os.path.basename(row["path"]),
                    "chunk_id": row["chunk_id"],
                }
    if best is None:
        return [], []

    text = (
        f"[Computed Source: {best['doc_name']} | Chunk: {best['chunk_id']}]\n"
        f"Structured numeric analysis over indexed rows: highest {best['metric']} value is on {best['date']} "
        f"with total={best['total_value']}, first={best['first_value']}"
        f"{'' if best['second_value'] is None else f', second={best['second_value']}'}."
    )
    source = SourceChunk(
        rank=1,
        source_id="S1",
        doc_id=best["doc_id"],
        doc_name=best["doc_name"],
        chunk_id=best["chunk_id"],
        score=1.0,
        snippet=(
            f"Highest {best['metric']}: {best['date']} total={best['total_value']} first={best['first_value']}"
            f"{'' if best['second_value'] is None else f' second={best['second_value']}'}"
        ),
        rerank_score=1.0,
        fusion_score=1.0,
        subquery_id="computed",
    )
    return [text], [source]


def _looks_like_numeric_record_question(lowered_prompt: str) -> bool:
    return bool(re.search(r"\b(?:record|row|date|day|amount|value|data|total|first|second|download(?:ed|s)?|upload(?:ed|s)?)\b", lowered_prompt))


def _numeric_record_metric(lowered_prompt: str) -> str:
    if re.search(r"\b(?:most|highest|max(?:imum)?|largest)\s+(?:\w+\s+){0,2}(?:download(?:ed|s)?|second)\b", lowered_prompt):
        return "second"
    if re.search(r"\b(?:most|highest|max(?:imum)?|largest)\s+(?:\w+\s+){0,2}(?:upload(?:ed|s)?|first)\b", lowered_prompt):
        return "first"
    if re.search(r"\bheaviest\s+(?:download(?:ed|s)?|second)\b", lowered_prompt):
        return "second"
    if re.search(r"\bheaviest\s+(?:upload(?:ed|s)?|first)\b", lowered_prompt):
        return "first"
    return "total"


def _mark_sources_retrieved(app_state, sources: list[SourceChunk]) -> None:
    doc_ids = {source.doc_id for source in sources if source.doc_id != "core_memory"}
    for doc_id in doc_ids:
        storage.execute(
            app_state.sqlite,
            "UPDATE documents SET last_retrieved_at = ?, retrieval_count = COALESCE(retrieval_count, 0) + 1 WHERE id = ?",
            (int(time.time()), doc_id),
        )
