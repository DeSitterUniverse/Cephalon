import sqlite3
import time

from cephalon_core import storage
from cephalon_core.config import Settings
from cephalon_core.schemas import SourceChunk
from cephalon_core.services import evaluation, observability, support


def build_conn():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    storage.run_migrations(conn, Settings())
    return conn


def test_retrieval_trace_persistence_roundtrip():
    conn = build_conn()
    trace = {
        "query_id": "query-1",
        "raw_query": "stress supplements",
        "normalized_query": "stress supplements",
        "retrieval_mode": "dense+sqlite_fts5",
        "subqueries": [{"id": "q1", "text": "stress supplements"}],
        "vector_candidates": [{"rank": 1, "chunk_id": "c1", "doc_id": "d1", "score": 0.8, "vector_score": 0.8}],
        "bm25_candidates": [{"rank": 1, "chunk_id": "c2", "doc_id": "d2", "score": -4.2, "lexical_score": -4.2}],
        "fused_candidates": [{"rank": 1, "chunk_id": "c1", "doc_id": "d1", "fusion_score": 0.032}],
        "reranked_candidates": [{"rank": 1, "chunk_id": "c1", "doc_id": "d1", "rerank_score": 1.7}],
        "final_context": [{"rank": 1, "chunk_id": "c1", "doc_id": "d1", "source_id": "S1", "text": "Ashwagandha helps stress."}],
        "unused_candidates": [{"rank": 2, "chunk_id": "c2", "doc_id": "d2", "reason": "below final context cutoff"}],
        "latency": {"vector_ms": 2.0, "bm25_ms": 1.0, "fusion_ms": 0.5, "rerank_ms": 3.0, "total_ms": 8.5},
        "no_answer": {"confidence": 0.82, "no_answer": False},
    }

    storage.save_retrieval_trace(conn, trace)
    loaded = storage.get_retrieval_trace(conn, "query-1")

    assert loaded["query_id"] == "query-1"
    assert loaded["latency"]["rerank_ms"] == 3.0
    assert loaded["candidates"]["vector"][0]["chunk_id"] == "c1"
    assert loaded["candidates"]["bm25"][0]["lexical_score"] == -4.2
    assert loaded["candidates"]["fused"][0]["fusion_score"] == 0.032
    assert loaded["candidates"]["reranked"][0]["rerank_score"] == 1.7
    assert loaded["final_context"][0]["source_id"] == "S1"


def test_stale_embedding_detection_uses_hashes_versions_and_models():
    baseline = {
        "content_hash": "file-a",
        "chunking_config_hash": "chunk-a",
        "parser_version": "parser-a",
        "embedding_model_id": "embed-a",
    }

    assert observability.detect_stale_state(baseline, baseline)["stale"] is False
    assert observability.detect_stale_state(baseline, {**baseline, "content_hash": "file-b"})["reasons"] == ["file_hash_changed"]
    assert observability.detect_stale_state(baseline, {**baseline, "chunking_config_hash": "chunk-b"})["reasons"] == ["chunking_config_changed"]
    assert observability.detect_stale_state(baseline, {**baseline, "parser_version": "parser-b"})["reasons"] == ["parser_version_changed"]
    assert observability.detect_stale_state(baseline, {**baseline, "embedding_model_id": "embed-b"})["reasons"] == ["embedding_model_changed"]


def test_eval_metrics_recall_and_mrr_are_deterministic():
    metrics = evaluation.retrieval_metrics(
        expected_doc_ids=["doc-b"],
        expected_chunk_ids=["chunk-b"],
        retrieved=[
            {"doc_id": "doc-a", "chunk_id": "chunk-a"},
            {"doc_id": "doc-b", "chunk_id": "chunk-b"},
            {"doc_id": "doc-c", "chunk_id": "chunk-c"},
        ],
        k=3,
    )

    assert metrics["recall_at_k"] == 1.0
    assert metrics["mrr"] == 0.5
    assert metrics["expected_doc_hit_rate"] == 1.0
    assert metrics["expected_chunk_hit_rate"] == 1.0


def test_no_answer_gating_uses_strength_and_agreement():
    weak = observability.no_answer_diagnostics(
        [
            SourceChunk(rank=1, doc_id="doc-a", doc_name="a", chunk_id="a1", score=0.12, snippet="weak", vector_score=0.08, lexical_score=None, fusion_score=0.01, rerank_score=0.05),
        ],
        thresholds={"min_confidence": 0.35, "min_rerank_score": 0.3, "min_vector_score": 0.2, "min_source_count": 1},
    )
    strong = observability.no_answer_diagnostics(
        [
            SourceChunk(rank=1, doc_id="doc-a", doc_name="a", chunk_id="a1", score=0.92, snippet="strong", vector_score=0.7, lexical_score=-1.0, fusion_score=0.03, rerank_score=1.2),
            SourceChunk(rank=2, doc_id="doc-b", doc_name="b", chunk_id="b1", score=0.81, snippet="also", vector_score=0.6, lexical_score=-1.2, fusion_score=0.02, rerank_score=0.9),
        ],
        thresholds={"min_confidence": 0.35, "min_rerank_score": 0.3, "min_vector_score": 0.2, "min_source_count": 1},
    )

    assert weak["no_answer"] is True
    assert "low_rerank_score" in weak["reasons"]
    assert strong["no_answer"] is False
    assert strong["agreement"]["hybrid_overlap"] is True


def test_citation_support_classification_is_score_based():
    final_context = [
        SourceChunk(rank=1, doc_id="doc-a", doc_name="a", chunk_id="a1", score=0.8, snippet="supported", rerank_score=0.9),
        SourceChunk(rank=2, doc_id="doc-b", doc_name="b", chunk_id="b1", score=0.3, snippet="weak", rerank_score=0.2),
    ]

    assert support.classify_citation_support("a1", final_context)["status"] == "supported"
    assert support.classify_citation_support("b1", final_context)["status"] == "weak"
    assert support.classify_citation_support("missing", final_context)["status"] == "unsupported"


def test_schema_initialization_is_idempotent_and_observability_tables_exist():
    conn = build_conn()
    storage.run_migrations(conn, Settings())

    tables = {row["name"] for row in storage.fetchall(conn, "SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"retrieval_queries", "retrieval_candidates", "retrieval_context", "retrieval_latency", "answer_records", "answer_citations", "eval_runs", "eval_results", "user_feedback"} <= tables

    chunk_columns = storage.table_columns(conn, "chunks")
    document_columns = storage.table_columns(conn, "documents")
    assert {"text_hash", "chunking_config_hash", "parser_version", "embedding_status"} <= chunk_columns
    assert {"text_hash", "chunking_config_hash", "parser_version", "embedding_config_hash", "parse_warnings"} <= document_columns
