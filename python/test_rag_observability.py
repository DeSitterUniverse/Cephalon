import hashlib
import sqlite3
import time
from types import SimpleNamespace

from cephalon_core import storage
from cephalon_core.config import Settings
from cephalon_core.schemas import RagSettings, SourceChunk
from cephalon_core.services import evaluation, ingestion, observability, support


def build_conn():
    conn = sqlite3.connect(":memory:", check_same_thread=False)
    conn.row_factory = sqlite3.Row
    storage.run_migrations(conn, Settings())
    return conn


def test_answer_metrics_recognize_current_and_legacy_canonical_refusals():
    item = {"expected_refusal": True, "accepted_answers": ["insufficient evidence"]}
    current = evaluation.answer_metrics(
        item=item,
        answer="I found insufficient evidence in your local documents to answer that reliably.",
        sources=[],
    )
    legacy = evaluation.answer_metrics(
        item={"expected_refusal": True},
        answer="I could not find sufficient supporting evidence in your local documents to answer that reliably.",
        sources=[],
    )
    natural = evaluation.answer_metrics(
        item={"expected_refusal": True},
        answer="The retrieved documents do not contain information about that measurement.",
        sources=[],
    )
    natural_first_person = evaluation.answer_metrics(
        item={"expected_refusal": True, "accepted_answers": ["insufficient evidence"]},
        answer="I was unable to find any mention of that value in the retrieved documents.",
        sources=[],
    )
    partial = evaluation.answer_metrics(
        item={"expected_refusal": False},
        answer="The first source supports the result, but the second does not contain that comparison.",
        sources=[],
    )
    assert current["correct_refusal"] == 1.0
    assert current["accepted_answer_match"] == 1.0
    assert legacy["correct_refusal"] == 1.0
    assert natural["correct_refusal"] == 1.0
    assert natural_first_person["correct_refusal"] == 1.0
    assert natural_first_person["accepted_answer_match"] == 1.0
    assert partial["correct_refusal"] == 1.0
    assert partial["over_refusal"] == 0.0


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


def test_stale_document_refresh_tracks_parser_chunk_model_and_file_changes(tmp_path):
    conn = build_conn()
    path = tmp_path / "paper.pdf"
    path.write_bytes(b"original machine PDF")
    rag_settings = RagSettings()
    content_hash = hashlib.sha256(path.read_bytes()).hexdigest()
    size_bytes, modified_at = path.stat().st_size, int(path.stat().st_mtime)
    embedding_model_id = "test/embedder"
    embedding_dim = 1024
    chunking_hash = observability.chunking_config_hash(
        ingestion.CHUNKING_PROFILE,
        ingestion._chunking_config(rag_settings),
    )
    storage.execute(
        conn,
        """
        INSERT INTO documents (
            id, path, display_name, content_hash, ingested_at, chunk_count, status,
            type, size_bytes, modified_at, embedding_model_id, embedding_dim,
            text_hash, parser_version, chunking_profile, chunking_config_hash,
            embedding_config_hash, stale_embedding
        )
        VALUES (?, ?, ?, ?, ?, 1, 'ready', 'file', ?, ?, ?, ?, ?, ?, ?, ?, ?, 0)
        """,
        (
            "doc-stale",
            str(path),
            path.name,
            content_hash,
            int(time.time()),
            size_bytes,
            modified_at,
            embedding_model_id,
            embedding_dim,
            "text-hash",
            "old-pdf-parser",
            ingestion.CHUNKING_PROFILE,
            chunking_hash,
            f"{embedding_model_id}:{embedding_dim}",
        ),
    )
    app_state = SimpleNamespace(
        sqlite=conn,
        embedding_model_id=embedding_model_id,
        embedding_dim=embedding_dim,
    )

    first = ingestion.refresh_document_staleness(app_state, rag_settings)
    payload = storage.get_document_payload(conn, "doc-stale")
    assert first["stale_document_count"] == 1
    assert payload["stale_reasons"] == ["parser_version_changed"]

    storage.execute(
        conn,
        "UPDATE documents SET parser_version = ? WHERE id = ?",
        (ingestion.parser_version_for_path(str(path)), "doc-stale"),
    )
    second = ingestion.refresh_document_staleness(app_state, rag_settings)
    assert second["stale_document_count"] == 0
    assert storage.get_document_payload(conn, "doc-stale")["stale_embedding"] is False

    path.write_bytes(b"changed machine PDF with a different size")
    third = ingestion.refresh_document_staleness(app_state, rag_settings)
    assert third["reasons_by_document"]["doc-stale"] == ["file_hash_changed"]


def test_legacy_saved_rag_fields_are_ignored_and_not_resaved():
    conn = build_conn()
    storage.execute(
        conn,
        "UPDATE app_settings SET value = ? WHERE key = 'rag'",
        ('{"top_k":12,"chunk_size":900,"chunk_overlap":90,"full_context":true}',),
    )

    loaded = storage.get_rag_settings(conn)
    saved = storage.save_rag_settings(conn, loaded)

    assert loaded.top_k == 12
    assert {"chunk_size", "chunk_overlap", "full_context"}.isdisjoint(saved.model_dump())


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


def test_answer_support_accounts_only_for_citations_used_in_the_answer():
    final_context = [
        SourceChunk(rank=1, source_id="S1", doc_id="doc-a", doc_name="a", chunk_id="a1", score=0.8, snippet="supported", rerank_score=0.9),
        SourceChunk(rank=2, source_id="S2", doc_id="doc-b", doc_name="b", chunk_id="b1", score=0.3, snippet="unused", rerank_score=0.2),
    ]

    result = support.classify_answer_support(
        "The supported fact appears here. [[src:S1]] Repeated. [[SRC:S1]] Unknown. [[src:S9]]",
        final_context,
    )

    assert result["status"] == "unsupported"
    assert [item["source_id"] for item in result["citations"]] == ["S1", "S9"]
    assert result["accounting"] == {
        "citation_count": 3,
        "unique_citation_count": 2,
        "cited_source_ids": ["S1", "S9"],
        "valid_source_ids": ["S1"],
        "invalid_source_ids": ["S9"],
        "duplicate_source_ids": ["S1"],
        "malformed_citations": [],
        "unused_citation_source_ids": ["S9"],
        "uncited_source_ids": ["S2"],
        "available_source_count": 2,
        "uncited_source_count": 1,
        "citation_precision": 0.5,
    }


def test_answer_without_citations_is_not_marked_supported_by_unused_context():
    source = SourceChunk(
        rank=1,
        source_id="S1",
        doc_id="doc-a",
        doc_name="a",
        chunk_id="a1",
        score=0.9,
        snippet="strong but uncited",
        rerank_score=1.2,
    )

    result = support.classify_answer_support("An uncited answer.", [source])

    assert result["status"] == "unsupported"
    assert result["citations"] == []
    assert result["accounting"]["uncited_source_count"] == 1


def test_claim_validation_checks_each_claim_against_its_cited_source():
    sources = [
        SourceChunk(
            rank=1,
            source_id="S1",
            doc_id="doc-a",
            doc_name="a",
            chunk_id="a1",
            score=0.9,
            snippet="The RATE method improved retrieval recall to 81.7 percent.",
            rerank_score=1.2,
        ),
    ]

    result = support.validate_answer_claims(
        "RATE improved retrieval recall to 81.7 percent. [[src:S1]] "
        "It also reduced GPU use by half. [[src:S1]]",
        sources,
    )

    assert result["claim_count"] == 2
    assert result["supported_claim_count"] == 1
    assert result["unsupported_claim_count"] == 1
    assert [claim["status"] for claim in result["claims"]] == ["supported", "unsupported"]


def test_answer_support_exposes_claim_evidence_and_malformed_tags():
    source = SourceChunk(
        rank=1,
        source_id="S1",
        doc_id="doc-a",
        doc_name="a",
        chunk_id="a1",
        score=0.9,
        snippet="A longer raw parent passage.",
        evidence_text="RATE improved retrieval recall to 81.7 percent.",
        rerank_score=1.2,
    )

    result = support.classify_answer_support(
        "RATE improved retrieval recall to 81.7 percent. [[src:S1]] Broken [[src S2]]",
        [source],
    )

    assert result["status"] == "unsupported"
    assert result["accounting"]["malformed_citations"] == ["[[src S2]]"]
    assert result["citations"][0]["claim_ids"] == ["C1"]
    assert result["citations"][0]["evidence"] == source.evidence_text
    unattached = support.classify_answer_support("[[src:S1]]", [source])
    assert unattached["accounting"]["unused_citation_source_ids"] == ["S1"]


def test_schema_initialization_is_idempotent_and_observability_tables_exist():
    conn = build_conn()
    storage.run_migrations(conn, Settings())

    tables = {row["name"] for row in storage.fetchall(conn, "SELECT name FROM sqlite_master WHERE type = 'table'")}
    assert {"retrieval_queries", "retrieval_candidates", "retrieval_context", "retrieval_latency", "answer_records", "answer_citations", "eval_runs", "eval_results", "user_feedback"} <= tables

    chunk_columns = storage.table_columns(conn, "chunks")
    document_columns = storage.table_columns(conn, "documents")
    assert {
        "text_hash",
        "chunking_config_hash",
        "parser_version",
        "embedding_status",
        "page_end",
        "block_index",
        "bounding_box",
        "provenance_json",
    } <= chunk_columns
    assert {"text_hash", "chunking_config_hash", "parser_version", "embedding_config_hash", "parse_warnings", "stale_reasons"} <= document_columns
