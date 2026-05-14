import hashlib
import os
import re
import time
import uuid
from dataclasses import dataclass
from typing import Any

import numpy as np

from .. import storage
from ..config import EMBEDDING_DIMENSION
from ..schemas import RagSettings, SourceChunk
from . import metrics

RRF_K = 60
CORE_MEMORY_DOC_ID = "core_memory"
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
        if rows:
            table.add(rows)
    else:
        table = app_state.lance.create_table(table_name, data=rows, schema=storage.vector_schema(getattr(app_state, "embedding_dim", EMBEDDING_DIMENSION)))
    ensure_retrieval_index(app_state)
    return table


async def get_embedding(app_state, text: str) -> list[float]:
    if getattr(app_state, "embedder", None) is None:
        raise RuntimeError("Embedding engine is not ready.")

    fixed_length = getattr(app_state, "embedding_fixed_sequence_length", None)
    tokenizer_kwargs = {"truncation": True, "return_tensors": "np"}
    if fixed_length:
        tokenizer_kwargs.update({"padding": "max_length", "max_length": int(fixed_length)})
    else:
        tokenizer_kwargs["padding"] = True
    inputs = app_state.embed_tokenizer(text, **tokenizer_kwargs)
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in inputs:
        ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)

    outs = app_state.embedder.run(None, ort_inputs)
    output = np.asarray(outs[0])
    if output.ndim == 2:
        vec = output[0]
    elif output.ndim == 3:
        pooling = getattr(app_state, "embedding_pooling", "cls")
        hidden = output[0]
        if pooling == "last_token":
            seq_len = int(ort_inputs["attention_mask"][0].sum()) - 1
            vec = hidden[max(seq_len, 0)]
        else:
            vec = hidden[0]
        norm = np.linalg.norm(vec)
        vec = vec / norm if norm else vec
    else:
        raise RuntimeError(f"Unsupported embedding output rank: {output.ndim}")
    expected_dim = getattr(app_state, "embedding_dim", EMBEDDING_DIMENSION)
    if len(vec) != expected_dim:
        raise RuntimeError(f"Embedding dimension mismatch: got {len(vec)}, expected {expected_dim}. Re-export ONNX models and rebuild the index.")
    return vec.tolist()


async def save_permanent_memory(app_state, user_prompt: str, vector: list[float]) -> None:
    memory_id = f"mem_{uuid.uuid4()}"
    memory_text = f"[Past Conversation Context]: The user stated/asked: '{user_prompt}'"
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
    try:
        ensure_vector_table(app_state, lance_data)
    except Exception:
        pass


def rerank(app_state, prompt: str, results: list[dict]) -> list[dict]:
    if not results:
        return []
    pairs = [[prompt, res["text"]] for res in results]
    inputs = app_state.tokenizer(pairs, padding=True, truncation=True, return_tensors="np")
    ort_inputs = {
        "input_ids": inputs["input_ids"].astype(np.int64),
        "attention_mask": inputs["attention_mask"].astype(np.int64),
    }
    if "token_type_ids" in inputs:
        ort_inputs["token_type_ids"] = inputs["token_type_ids"].astype(np.int64)
    raw_scores = np.asarray(app_state.reranker.run(None, ort_inputs)[0])
    scores = _reranker_scores(app_state, raw_scores)
    for idx, res in enumerate(results):
        res["rerank_score"] = float(scores[idx])
        res["retrieval_prior_score"] = _retrieval_prior_score(prompt, res)
        res["score"] = _final_retrieval_score(prompt, res, float(scores[idx]))
    return sorted(results, key=lambda x: x["score"], reverse=True)


def _reranker_scores(app_state, raw_scores: np.ndarray) -> np.ndarray:
    mode = getattr(app_state, "reranker_score_mode", "auto")
    if raw_scores.ndim == 2 and raw_scores.shape[1] == 2:
        if mode == "logit_margin_0_minus_1":
            return raw_scores[:, 0] - raw_scores[:, 1]
        if mode == "logit_margin_1_minus_0":
            return raw_scores[:, 1] - raw_scores[:, 0]
        if mode == "class_0":
            return raw_scores[:, 0]
        if mode == "class_1":
            return raw_scores[:, 1]
        return raw_scores[:, 0] - raw_scores[:, 1]
    if raw_scores.ndim == 2 and raw_scores.shape[1] > 1:
        return raw_scores[:, -1]
    return raw_scores.reshape(-1)


def _bounded_rerank_score(raw_score: float) -> float:
    return float(np.tanh(raw_score / 2.0))


def _final_retrieval_score(prompt: str, result: dict, raw_rerank_score: float) -> float:
    return round(_retrieval_prior_score(prompt, result) + _bounded_rerank_score(raw_rerank_score), 6)


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
    if len(parts) <= 1:
        return [{"id": "q1", "text": clean}]
    return [{"id": f"q{idx}", "text": part} for idx, part in enumerate(parts[:5], start=1)]


def hydrate_sources(app_state, results: list[dict], subquery_id: str | None = None) -> list[SourceChunk]:
    if not results:
        return []
    doc_ids = list({res["doc_id"] for res in results if res["doc_id"] != "core_memory"})
    path_map = {}
    if doc_ids:
        placeholders = ",".join("?" * len(doc_ids))
        rows = storage.fetchall(app_state.sqlite, f"SELECT id, path, display_name FROM documents WHERE id IN ({placeholders})", tuple(doc_ids))
        path_map = {row["id"]: row["display_name"] or os.path.basename(row["path"]) for row in rows}

    sources: list[SourceChunk] = []
    for rank, res in enumerate(results, start=1):
        doc_id = res["doc_id"]
        doc_name = "Core Memory" if doc_id == "core_memory" else path_map.get(doc_id, "Unknown")
        text = res["text"].strip()
        source_id = f"S{rank}"
        sources.append(SourceChunk(
            rank=rank,
            source_id=source_id,
            doc_id=doc_id,
            doc_name=doc_name,
            chunk_id=res["id"],
            parent_id=res.get("parent_id"),
            score=float(res.get("score", 0)),
            vector_score=float(res["_distance"]) if "_distance" in res and res["_distance"] is not None else None,
            rerank_score=float(res.get("rerank_score", res.get("score", 0))),
            lexical_score=float(res["lexical_score"]) if res.get("lexical_score") is not None else None,
            fusion_score=float(res["fusion_score"]) if res.get("fusion_score") is not None else None,
            snippet=text[:500],
            subquery_id=subquery_id,
        ))
    return sources


async def _search_once(app_state, prompt: str, query_vector: list[float], settings: RagSettings) -> tuple[list[dict], str]:
    table_name = vector_table_name(app_state)
    index = ensure_retrieval_index(app_state)
    summary_results = _summary_dense_search(app_state, table_name, query_vector, max(4, settings.top_k // 3))
    summary_parent_rank = {
        result["parent_id"]: rank
        for rank, result in enumerate(summary_results, start=1)
        if result.get("parent_id")
    }
    dense_results = _dense_search(app_state, table_name, query_vector, settings.top_k)
    lexical_results = _lexical_search(app_state, prompt, settings.top_k)
    _apply_summary_parent_boost(dense_results, summary_parent_rank)
    _apply_summary_parent_boost(lexical_results, summary_parent_rank)
    fused = _fuse_rrf(dense_results, lexical_results, settings.top_k)
    modes = []
    if summary_results:
        modes.append("summary_dense")
    if dense_results:
        modes.append("dense")
    if lexical_results:
        modes.append("sqlite_fts5")
    if not index.get("lexical_available"):
        modes.append("lexical_unavailable")
    return fused, "+".join(modes) if modes else "empty"


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


def compress_context(query: str, sources: list[CompressionSource], max_sentences: int = 10) -> tuple[str, dict[str, Any]]:
    query_terms = _term_set(query)
    candidates: list[dict[str, Any]] = []
    for source in sources:
        for sentence in split_sentences(source.text):
            relevance = _sentence_relevance(query_terms, sentence)
            candidates.append({
                "source_id": source.source_id,
                "sentence": sentence,
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
    stats = {
        "input_sentences": total_sentences,
        "kept_sentences": len(kept),
        "relevant_sentence_count": relevant_total,
        "context_relevance": round(relevant_total / total_sentences, 6) if total_sentences else 0.0,
    }
    return compressed, stats


def confidence_from_sources(sources: list[SourceChunk]) -> dict[str, Any]:
    if not sources:
        return {"confidence": 0.0, "uncertainty": "high", "no_answer": True, "reason": "No retrieved sources."}
    scores = [source.rerank_score if source.rerank_score is not None else source.score for source in sources]
    top = max(scores)
    second = sorted(scores, reverse=True)[1] if len(scores) > 1 else top
    spread = abs(top - second)
    confidence = max(0.0, min(1.0, 0.45 + min(top, 1.0) * 0.35 + min(spread, 1.0) * 0.2))
    no_answer = len(sources) < 1 or confidence < 0.35
    return {
        "confidence": round(confidence, 4),
        "uncertainty": "high" if confidence < 0.45 else "medium" if confidence < 0.7 else "low",
        "no_answer": no_answer,
        "reason": "Closest matches are weak." if no_answer else "Retrieved sources are usable.",
    }


async def retrieve_context(app_state, prompt: str, query_vector: list[float], settings: RagSettings) -> tuple[str, list[SourceChunk], dict[str, Any]]:
    started = time.perf_counter()
    context_chunks: list[str] = []
    all_sources: list[SourceChunk] = []
    compression_inputs: list[CompressionSource] = []
    search_modes: list[str] = []
    subqueries = plan_subqueries(prompt)
    numeric_context, numeric_sources = _structured_numeric_analysis_for_query(app_state, prompt)
    if numeric_context:
        context_chunks.extend(numeric_context)
        all_sources.extend(numeric_sources)
        compression_inputs.extend([
            CompressionSource(source_id=source.source_id or f"S{source.rank}", text=numeric_context[0], rank=source.rank, score=source.score)
            for source in numeric_sources
        ])
        search_modes.append("numeric_scan")
    else:
        for subquery in subqueries:
            vector = query_vector if subquery["text"] == prompt else await get_embedding(app_state, subquery["text"])
            results, mode = await _search_once(app_state, subquery["text"], vector, settings)
            search_modes.append(mode)
            reranked = _select_relevant_results(rerank(app_state, subquery["text"], results), settings.rerank_top_n)
            sources = hydrate_sources(app_state, reranked, subquery["id"])
            all_sources.extend(sources)
            source_by_chunk = {source.chunk_id: source for source in sources}

            for res in reranked:
                source = source_by_chunk.get(res["id"])
                if res["doc_id"] == CORE_MEMORY_DOC_ID:
                    context_chunks.append(res["text"])
                else:
                    label = source.doc_name if source else "Unknown"
                    source_id = source.source_id if source and source.source_id else f"S{len(all_sources) + 1}"
                    context_text = _parent_context(app_state, res) or res["text"]
                    context_chunks.append(format_source_context(source_id, label, res["id"], context_text))
                    compression_inputs.append(CompressionSource(source_id=source_id, text=context_text, rank=source.rank if source else 99, score=source.score if source else 0.0))

    if compression_inputs:
        compressed_context, compression_stats = compress_context(prompt, compression_inputs, max_sentences=max(6, settings.rerank_top_n * 3))
        if compressed_context:
            context_chunks = [compressed_context]
    else:
        compression_stats = {"input_sentences": 0, "kept_sentences": 0, "context_relevance": 0.0}

    _mark_sources_retrieved(app_state, all_sources)
    confidence = confidence_from_sources(all_sources)
    elapsed_ms = round((time.perf_counter() - started) * 1000, 2)
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
        "subqueries": subqueries,
        "retrieval_latency_ms": elapsed_ms,
        "search_modes": search_modes,
        "metrics_path": metrics_path,
        "compression": compression_stats,
    }
    return "\n\n".join(context_chunks) if context_chunks else "No relevant memories or documents found.", all_sources, meta


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
