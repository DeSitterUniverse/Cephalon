import asyncio
import os
import sqlite3
from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from cephalon_core.config import Settings
from cephalon_core.events import EventBus
from cephalon_core.schemas import RagSettings
from cephalon_core.app_factory import _validate_embedder_meta, _validate_reranker_meta
from cephalon_core.app_factory import _read_model_meta
from cephalon_core import routes
from cephalon_core.services import metrics, retrieval
from cephalon_core.services import models
from cephalon_core.services.documents import collect_obsidian_files
from cephalon_core.services.ingestion import delete_document_rows, delete_document_vectors, process_single_file
from cephalon_core.services.jobs import JobManager
from cephalon_core.services.retrieval import vector_table_name
from cephalon_core import storage
from cephalon_core.validators import validate_document_id, validate_model_filename


class FakeTable:
    def __init__(self) -> None:
        self.rows = []
        self.deleted_filters = []

    def add(self, rows):
        self.rows.extend(rows)

    def delete(self, filter_expr: str) -> None:
        self.deleted_filters.append(filter_expr)

    def search(self, *_args, **_kwargs):
        return FakeSearch(self.rows)


class FakeSearch:
    def __init__(self, rows):
        self.rows = rows
        self.count = len(rows)

    def limit(self, count: int):
        self.count = count
        return self

    def to_list(self):
        return self.rows[:self.count]


class FakeTokenizer:
    def __call__(self, pairs, **_kwargs):
        import numpy as np

        return {
            "input_ids": np.ones((len(pairs), 2), dtype="int64"),
            "attention_mask": np.ones((len(pairs), 2), dtype="int64"),
        }


class FakeLance:
    def __init__(self) -> None:
        self.table = None

    def table_names(self):
        return [vector_table_name()] if self.table else []

    def open_table(self, _name: str):
        return self.table

    def create_table(self, _name: str, data, schema):
        assert schema.equals(storage.VECTOR_SCHEMA)
        self.table = FakeTable()
        self.table.add(data)
        return self.table


def build_memory_state(conn=None):
    sqlite_conn = conn or sqlite3.connect(":memory:", check_same_thread=False)
    sqlite_conn.row_factory = sqlite3.Row
    settings = Settings()
    storage.run_migrations(sqlite_conn, settings)
    return SimpleNamespace(sqlite=sqlite_conn, lance=FakeLance(), settings=settings)


def test_settings_reads_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("CEPHALON_DATA_DIR", str(tmp_path / "data"))
    monkeypatch.setenv("CEPHALON_MODEL_DIR", str(tmp_path / "models"))
    monkeypatch.setenv("CEPHALON_PORT", "9999")
    monkeypatch.setenv("CEPHALON_MAX_TOKENS", "64")
    monkeypatch.setenv("CEPHALON_CORS_ORIGINS", "http://localhost:1420,http://tauri.localhost")

    settings = Settings()

    assert settings.data_dir.endswith("data")
    assert settings.model_dir.endswith("models")
    assert settings.port == 9999
    assert settings.max_tokens == 64
    assert settings.cors_origins == ["http://localhost:1420", "http://tauri.localhost"]


def test_validate_model_filename_blocks_paths(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    (model_dir / "good.gguf").write_text("model", encoding="utf-8")

    assert validate_model_filename("good.gguf", str(model_dir)).endswith("good.gguf")

    with pytest.raises(HTTPException):
        validate_model_filename("../bad.gguf", str(model_dir))
    with pytest.raises(HTTPException):
        validate_model_filename("bad.txt", str(model_dir))
    with pytest.raises(HTTPException):
        validate_model_filename("missing.gguf", str(model_dir))


def test_validate_document_id_rejects_unsafe_values():
    validate_document_id("11111111-1111-4111-8111-111111111111")

    with pytest.raises(HTTPException):
        validate_document_id("core_memory")
    with pytest.raises(HTTPException):
        validate_document_id("abc' OR '1'='1")


def test_migrations_create_workbench_tables():
    state = build_memory_state()

    tables = {row["name"] for row in storage.fetchall(state.sqlite, "SELECT name FROM sqlite_master WHERE type = 'table'")}

    assert {
        "documents",
        "chunks",
        "schema_migrations",
        "jobs",
        "job_events",
        "document_tags",
        "app_settings",
        "parent_chunks",
        "summary_nodes",
        "conversations",
        "messages",
        "message_sources",
    } <= tables
    assert storage.fetchone(state.sqlite, "SELECT id FROM documents WHERE id = 'core_memory'")
    assert storage.get_rag_settings(state.sqlite).top_k == 20
    assert storage.get_rag_settings(state.sqlite).context_tokens == 32768


def test_model_inventory_separates_chat_and_auxiliary_ggufs(tmp_path):
    model_dir = tmp_path / "models"
    model_dir.mkdir()
    for filename in [
        "granite-4.1-8b-Q4_K_S.gguf",
        "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf",
        "jina-reranker-v3-Q8_0.gguf",
        "v5-small-retrieval-Q8_0.gguf",
    ]:
        (model_dir / filename).write_text("model", encoding="utf-8")
    settings = Settings()
    settings.model_dir = str(model_dir)

    inventory = models.model_inventory(settings)

    assert inventory["chat_models"] == [
        "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf",
        "granite-4.1-8b-Q4_K_S.gguf",
    ]
    assert inventory["auxiliary_gguf"] == [
        "jina-reranker-v3-Q8_0.gguf",
        "v5-small-retrieval-Q8_0.gguf",
    ]


def test_packaged_llama_dll_discovery_uses_sidecar_path_when_present():
    discovered = models._discover_packaged_llama_dll_dir()

    if discovered is not None:
        assert discovered.endswith("llama_cpp\\lib") or discovered.endswith("llama_cpp/lib")
        assert any(name.startswith("ggml") for name in os.listdir(discovered))


def test_quiet_llama_stderr_preserves_loader_exceptions():
    with pytest.raises(RuntimeError, match="load failed"):
        with models._quiet_llama_stderr():
            raise RuntimeError("load failed")


def test_conversation_persistence_roundtrip():
    state = build_memory_state()
    conversation = storage.create_conversation(state.sqlite, "Stress supplements")
    storage.append_message(
        state.sqlite,
        conversation["id"],
        "user",
        "What helps stress?",
        model="granite.gguf",
        settings={"reasoning_mode": "balanced"},
    )
    assistant = storage.append_message(
        state.sqlite,
        conversation["id"],
        "assistant",
        "Use the cited source. [[src:S1]]",
        model="granite.gguf",
        meta={"confidence": 0.8},
    )
    storage.save_message_sources(state.sqlite, assistant["id"], [
        {"source_id": "S1", "doc_name": "stress.docx", "chunk_id": "chunk-1", "score": 0.9}
    ])

    listed = storage.list_conversations(state.sqlite)
    loaded = storage.get_conversation(state.sqlite, conversation["id"])

    assert listed[0]["title"] == "Stress supplements"
    assert [message["role"] for message in loaded["messages"]] == ["user", "assistant"]
    assert loaded["messages"][1]["sources"][0]["source_id"] == "S1"


def test_process_single_file_skips_duplicate_hash(monkeypatch, tmp_path):
    state = build_memory_state()
    file_path = tmp_path / "note.md"
    file_path.write_text("The 4-7-8 method is a breathing exercise.", encoding="utf-8")

    async def fake_embedding(_app_state, _text: str):
        return [0.0] * storage.active_embedding_metadata()["embedding_dim"]

    monkeypatch.setattr("cephalon_core.services.ingestion.get_embedding", fake_embedding)

    asyncio.run(process_single_file(state, str(file_path), RagSettings()))
    asyncio.run(process_single_file(state, str(file_path), RagSettings()))

    rows = storage.fetchall(state.sqlite, "SELECT path, content_hash, status FROM documents WHERE type = 'file'")
    chunks = storage.fetchall(state.sqlite, "SELECT id FROM chunks")

    assert len(rows) == 1
    assert rows[0]["status"] == "ready"
    assert len(chunks) == 1
    fts_rows = storage.fetchall(state.sqlite, "SELECT chunk_id FROM chunks_fts")
    assert len(fts_rows) == 1
    assert state.lance.table is not None
    assert len(state.lance.table.rows) == 2
    assert {row["source_kind"] for row in state.lance.table.rows} == {"summary", "child"}


def test_force_text_import_allows_unknown_extension(monkeypatch, tmp_path):
    state = build_memory_state()
    file_path = tmp_path / "notes.custom"
    file_path.write_text("Custom extension should still import as text.", encoding="utf-8")

    async def fake_embedding(_app_state, _text: str):
        return [0.0] * storage.active_embedding_metadata()["embedding_dim"]

    monkeypatch.setattr("cephalon_core.services.ingestion.get_embedding", fake_embedding)

    result = asyncio.run(process_single_file(state, str(file_path), RagSettings(), force_text=True))
    row = storage.fetchone(state.sqlite, "SELECT extraction_mode, embedding_dim FROM documents WHERE id = ?", (result["doc_id"],))

    assert result["status"] == "ready"
    assert row["extraction_mode"] == "text"
    assert row["embedding_dim"] == storage.active_embedding_metadata()["embedding_dim"]


def test_process_single_file_creates_parent_child_summary_metadata(monkeypatch, tmp_path):
    state = build_memory_state()
    file_path = tmp_path / "semantic.md"
    file_path.write_text(
        "Stress support includes ashwagandha and rhodiola. Magnesium can help sleep.\n\n"
        "Deployment pipelines should run tests before packaging. Release artifacts need names.\n\n"
        "Traffic records are date and number rows that should keep each row intact.",
        encoding="utf-8",
    )

    async def fake_embedding(_app_state, text: str):
        base = float(len(text) % 7) / 10.0
        return [base] * storage.active_embedding_metadata()["embedding_dim"]

    monkeypatch.setattr("cephalon_core.services.ingestion.get_embedding", fake_embedding)

    result = asyncio.run(process_single_file(state, str(file_path), RagSettings()))
    parents = storage.fetchall(state.sqlite, "SELECT id, summary FROM parent_chunks WHERE doc_id = ?", (result["doc_id"],))
    summaries = storage.fetchall(state.sqlite, "SELECT id, parent_id, summary FROM summary_nodes WHERE doc_id = ?", (result["doc_id"],))
    chunks = storage.fetchall(state.sqlite, "SELECT parent_id, semantic_role FROM chunks WHERE doc_id = ?", (result["doc_id"],))

    assert result["status"] == "ready"
    assert parents
    assert summaries
    assert all(row["parent_id"] for row in chunks)
    assert {row["semantic_role"] for row in chunks} == {"child"}


def test_unknown_text_file_imports_without_force_text(monkeypatch, tmp_path):
    state = build_memory_state()
    file_path = tmp_path / "numbers.dataset"
    file_path.write_text("quarter,revenue\nQ1,120\nQ2,143\n", encoding="utf-8")

    async def fake_embedding(_app_state, _text: str):
        return [0.0] * storage.active_embedding_metadata()["embedding_dim"]

    monkeypatch.setattr("cephalon_core.services.ingestion.get_embedding", fake_embedding)

    result = asyncio.run(process_single_file(state, str(file_path), RagSettings()))
    row = storage.fetchone(state.sqlite, "SELECT status, extraction_mode FROM documents WHERE id = ?", (result["doc_id"],))

    assert result["status"] == "ready"
    assert row["status"] == "ready"
    assert row["extraction_mode"] == "text"


def test_obsidian_collection_skips_internal_config(tmp_path):
    vault = tmp_path / "Obsidian Vault"
    vault.mkdir()
    (vault / "Daily.md").write_text("# Daily\nA note about local RAG.", encoding="utf-8")
    (vault / "Board.canvas").write_text('{"nodes":[]}', encoding="utf-8")
    internal = vault / ".obsidian"
    internal.mkdir()
    (internal / "app.json").write_text('{"theme":"obsidian"}', encoding="utf-8")

    collected = [path.replace("\\", "/") for path in collect_obsidian_files(str(vault))]

    assert any(path.endswith("/Daily.md") for path in collected)
    assert any(path.endswith("/Board.canvas") for path in collected)
    assert not any("/.obsidian/" in path for path in collected)


def test_unknown_binary_file_fails_with_clear_reason(tmp_path):
    state = build_memory_state()
    file_path = tmp_path / "image.unknown"
    file_path.write_bytes(b"\x00\x01\x02\x03\x00\xff")

    result = asyncio.run(process_single_file(state, str(file_path), RagSettings()))

    assert result["status"] == "failed"
    assert "binary" in result["error"].lower()


def test_jina_model_metadata_is_strict():
    assert _validate_embedder_meta({
        "model_id": "jinaai/jina-embeddings-v5-text-small",
        "dimension": 1024,
        "validated": True,
    }) is None
    assert _validate_reranker_meta({
        "model_id": "jinaai/jina-reranker-v3",
        "validated": True,
        "score_mode": "logit_margin_0_minus_1",
    }) is None

    assert "Embedder model mismatch" in _validate_embedder_meta({"model_id": "other", "dimension": 1024, "validated": True})
    assert "score_mode" in _validate_reranker_meta({"model_id": "jinaai/jina-reranker-v3", "validated": True})


def test_query_requires_explicit_loaded_model():
    app_state = SimpleNamespace(startup_error=None, active_model_name="loaded.gguf")
    with pytest.raises(HTTPException) as exc:
        routes._ensure_query_model_loaded(app_state, "other.gguf")

    assert exc.value.status_code == 409
    assert "Load the selected GGUF model" in exc.value.detail


def test_query_accepts_loaded_model():
    app_state = SimpleNamespace(startup_error=None, active_model_name="loaded.gguf")

    routes._ensure_query_model_loaded(app_state, "loaded.gguf")


def test_model_metadata_reader_rejects_non_object_json(tmp_path):
    model_dir = tmp_path / "model"
    model_dir.mkdir()
    (model_dir / "cephalon_onnx_meta.json").write_text("null", encoding="utf-8")

    assert _read_model_meta(str(model_dir)) == {}


def test_retrieval_uses_sqlite_fts_dense_and_rrf(monkeypatch, tmp_path):
    state = build_memory_state()
    file_path = tmp_path / "fixture.md"
    file_path.write_text("Cephalon retrieval fixture mentions sqlite lexical search.", encoding="utf-8")

    async def fake_embedding(_app_state, _text: str):
        return [0.0] * storage.active_embedding_metadata()["embedding_dim"]

    monkeypatch.setattr("cephalon_core.services.ingestion.get_embedding", fake_embedding)
    result = asyncio.run(process_single_file(state, str(file_path), RagSettings()))
    state.tokenizer = FakeTokenizer()
    state.reranker = SimpleNamespace(run=lambda *_args, **_kwargs: [__import__("numpy").array([[1.0, 0.0]])])
    state.reranker_score_mode = "logit_margin_0_minus_1"

    context, sources, meta = asyncio.run(
        retrieval.retrieve_context(state, "sqlite lexical search", [0.0] * storage.active_embedding_metadata()["embedding_dim"], RagSettings())
    )

    assert result["status"] == "ready"
    assert sources
    assert sources[0].chunk_id == f"{result['doc_id']}_0"
    assert sources[0].fusion_score is not None
    assert "sqlite_fts5" in meta["search_modes"][0]
    assert "Cephalon retrieval fixture" in context


def test_context_compressor_keeps_relevant_cited_sentences():
    sources = [
        retrieval.CompressionSource(
            source_id="S1",
            text="Ashwagandha lowers stress. Distributed systems use quorum writes. Rhodiola can reduce fatigue.",
            rank=1,
            score=0.9,
        )
    ]

    compressed, stats = retrieval.compress_context("stress supplements", sources, max_sentences=2)

    assert "[[src:S1]]" in compressed
    assert "Ashwagandha" in compressed
    assert stats["kept_sentences"] == 2
    assert 0 < stats["context_relevance"] <= 1


def test_source_tags_are_stable_and_separate_from_user_filenames():
    source = retrieval.format_source_context("S3", "Stress advice.docx", "chunk-1", "Ashwagandha helps stress.")

    assert source.startswith("[[src:S3]] Source: Stress advice.docx | Chunk: chunk-1")
    assert "[[src:Stress advice.docx]]" not in source


def test_fts_query_ignores_question_stopwords():
    query = retrieval._fts_query("what are the best supplements for stress advice")

    assert "what" not in query
    assert "supplement*" in query
    assert "stress*" in query
    assert "advice*" in query


def test_retrieval_prior_keeps_exact_lexical_matches_above_bad_raw_rerank():
    lexical = {"text": "domain specific exact terms alpha beta gamma", "lexical_rank": 1}
    unrelated = {"text": "distributed systems cap theorem consistency availability", "dense_rank": 1}

    lexical_score = retrieval._final_retrieval_score("domain exact terms", lexical, -0.6)
    unrelated_score = retrieval._final_retrieval_score("domain exact terms", unrelated, 1.9)

    assert lexical_score > unrelated_score


def test_relevant_selection_keeps_same_document_and_drops_weak_dense_strays():
    ranked = [
        {"id": "domain_0", "doc_id": "domain", "score": 3.7, "lexical_rank": 1},
        {"id": "domain_1", "doc_id": "domain", "score": 0.84, "dense_rank": 2},
        {"id": "unrelated_0", "doc_id": "unrelated", "score": 0.64, "dense_rank": 3},
    ]

    selected = retrieval._select_relevant_results(ranked, 3)

    assert [result["id"] for result in selected] == ["domain_0", "domain_1"]


def test_dense_search_excludes_core_memory_rows():
    state = build_memory_state()
    state.lance.table = FakeTable()
    state.lance.table.rows = [
        {"id": "mem_1", "doc_id": "core_memory", "text": "Repeated user prompt", "_distance": 0.0},
        {"id": "doc_1_0", "doc_id": "11111111-1111-4111-8111-111111111111", "text": "Document chunk", "_distance": 0.5},
    ]

    results = retrieval._dense_search(state, vector_table_name(state), [0.0], 1)

    assert [row["id"] for row in results] == ["doc_1_0"]


def test_structured_numeric_analysis_scans_all_chunks(tmp_path):
    state = build_memory_state()
    doc_id = "11111111-1111-4111-8111-111111111111"
    path = str(tmp_path / "numeric_records.dat")
    storage.execute(
        state.sqlite,
        """
        INSERT INTO documents (id, path, display_name, content_hash, chunk_count, status, type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (doc_id, path, "numeric_records.dat", "hash", 2, "ready", "file"),
    )
    chunks = [
        ("11111111-1111-4111-8111-111111111111_0", "2026/04/12 24403931/1000\n"),
        ("11111111-1111-4111-8111-111111111111_1", "2025/08/14 19747776/177090636\n"),
    ]
    for index, (chunk_id, text) in enumerate(chunks):
        storage.execute(
            state.sqlite,
            "INSERT INTO chunks (id, doc_id, chunk_index, text, chunk_length) VALUES (?, ?, ?, ?, ?)",
            (chunk_id, doc_id, index, text, len(text)),
        )
        storage.upsert_chunk_fts(state.sqlite, chunk_id, doc_id, text)

    context, sources = retrieval._structured_numeric_analysis_for_query(
        state,
        "what is my heaviest data day, show amount from the second value",
    )

    assert "highest total value is on 2025/08/14" in context[0]
    assert "second=177090636" in context[0]
    assert sources[0].chunk_id.endswith("_1")


def test_structured_numeric_analysis_can_rank_by_named_column_when_explicit(tmp_path):
    assert retrieval._numeric_record_metric("highest second value day") == "second"
    assert retrieval._numeric_record_metric("heaviest data day show amount from the second value") == "total"


def test_structured_numeric_query_uses_computed_context_only(monkeypatch, tmp_path):
    state = build_memory_state()
    doc_id = "11111111-1111-4111-8111-111111111111"
    storage.execute(
        state.sqlite,
        """
        INSERT INTO documents (id, path, display_name, content_hash, chunk_count, status, type)
        VALUES (?, ?, ?, ?, ?, ?, ?)
        """,
        (doc_id, str(tmp_path / "numeric_records.dat"), "numeric_records.dat", "hash", 1, "ready", "file"),
    )
    storage.execute(
        state.sqlite,
        "INSERT INTO chunks (id, doc_id, chunk_index, text, chunk_length) VALUES (?, ?, ?, ?, ?)",
        (f"{doc_id}_0", doc_id, 0, "2025/08/14 19747776/177090636", 31),
    )

    async def fail_search(*_args, **_kwargs):
        raise AssertionError("structured numeric max queries should not run generic retrieval")

    monkeypatch.setattr(retrieval, "_search_once", fail_search)

    context, sources, meta = asyncio.run(retrieval.retrieve_context(
        state,
        "what is my heaviest data day, show amount from the second value",
        [0.0],
        RagSettings(),
    ))

    assert "2025/08/14" in context
    assert sources[0].subquery_id == "computed"
    assert meta["search_modes"] == ["numeric_scan"]


def test_delete_vectors_uses_active_table_and_safe_filter(tmp_path):
    state = build_memory_state()
    state.lance.table = FakeTable()

    delete_document_vectors(state, "11111111-1111-4111-8111-111111111111")

    assert state.lance.table.deleted_filters == ["doc_id = '11111111-1111-4111-8111-111111111111'"]


def test_delete_document_rows_cleans_sqlite_fts(monkeypatch, tmp_path):
    state = build_memory_state()
    file_path = tmp_path / "delete.md"
    file_path.write_text("Delete should clean full text search rows.", encoding="utf-8")

    async def fake_embedding(_app_state, _text: str):
        return [0.0] * storage.active_embedding_metadata()["embedding_dim"]

    monkeypatch.setattr("cephalon_core.services.ingestion.get_embedding", fake_embedding)
    result = asyncio.run(process_single_file(state, str(file_path), RagSettings()))
    delete_document_vectors(state, result["doc_id"])
    delete_document_rows(state, result["doc_id"])

    assert storage.fetchall(state.sqlite, "SELECT chunk_id FROM chunks_fts WHERE doc_id = ?", (result["doc_id"],)) == []


def test_job_manager_lifecycle_and_events(monkeypatch, tmp_path):
    state = build_memory_state()
    event_bus = EventBus(state.sqlite)
    manager = JobManager(state, event_bus)
    file_path = tmp_path / "fixture.md"
    file_path.write_text("Cephalon job queue fixture.", encoding="utf-8")

    async def fake_embedding(_app_state, _text: str):
        return [0.0] * storage.active_embedding_metadata()["embedding_dim"]

    monkeypatch.setattr("cephalon_core.services.ingestion.get_embedding", fake_embedding)

    async def run_job():
        job = await manager.enqueue_ingest(str(file_path))
        await manager._run_job(job["id"])
        return manager.get_job(job["id"])

    finished = asyncio.run(run_job())
    events = storage.fetchall(state.sqlite, "SELECT event_type FROM job_events WHERE job_id = ?", (finished["id"],))

    assert finished["status"] == "succeeded"
    assert finished["processed_files"] == 1
    assert any(row["event_type"] == "job" for row in events)


def test_reindex_preserves_display_name_and_tags(monkeypatch, tmp_path):
    state = build_memory_state()
    event_bus = EventBus(state.sqlite)
    manager = JobManager(state, event_bus)
    file_path = tmp_path / "fixture.md"
    file_path.write_text("Cephalon reindex fixture.", encoding="utf-8")

    async def fake_embedding(_app_state, _text: str):
        return [0.0] * storage.active_embedding_metadata()["embedding_dim"]

    monkeypatch.setattr("cephalon_core.services.ingestion.get_embedding", fake_embedding)
    first = asyncio.run(process_single_file(state, str(file_path), RagSettings()))
    storage.execute(state.sqlite, "UPDATE documents SET display_name = ? WHERE id = ?", ("Renamed Fixture", first["doc_id"]))
    storage.execute(state.sqlite, "INSERT INTO document_tags (doc_id, tag) VALUES (?, ?)", (first["doc_id"], "rag"))
    file_path.write_text("Cephalon reindex fixture changed.", encoding="utf-8")

    async def run_job():
        job = await manager.enqueue_ingest(str(file_path), kind="reindex", target_doc_id=first["doc_id"])
        await manager._run_job(job["id"])
        return manager.get_job(job["id"])

    finished = asyncio.run(run_job())
    row = storage.fetchone(state.sqlite, "SELECT display_name, status FROM documents WHERE id = ?", (first["doc_id"],))
    tags = storage.get_document_tags(state.sqlite, first["doc_id"])

    assert finished["status"] == "succeeded"
    assert row["display_name"] == "Renamed Fixture"
    assert row["status"] == "ready"
    assert tags == ["rag"]


def test_metrics_export_writes_numeric_snapshot(tmp_path):
    settings = Settings()
    settings.metrics_dir = str(tmp_path / "metrics")
    state = build_memory_state()
    state.settings = settings

    path = metrics.export_corpus_snapshot(state)

    assert os.path.exists(path)
    with open(path, encoding="utf-8") as f:
        header = f.readline()
        row = f.readline()
    assert "document_count" in header
    assert row


def test_quality_metrics_are_numeric_and_bounded():
    values = metrics.quality_metrics(
        query="what supplements help stress",
        answer="Ashwagandha helps stress. Space elevators are unrelated.",
        context="Ashwagandha helps stress. Rhodiola helps fatigue.",
        relevant_sentence_count=1,
        total_sentence_count=2,
        supported_statement_count=1,
        total_statement_count=2,
        answer_query_similarity=0.42,
    )

    assert values == {
        "context_relevance": 0.5,
        "groundedness": 0.5,
        "answer_relevance": 0.42,
    }


def test_retrieval_metrics_write_failure_is_nonfatal(monkeypatch):
    state = build_memory_state()

    async def fake_search(_app_state, _prompt, _query_vector, _settings):
        return [], "hybrid"

    def fail_metrics(_app_state, _payload):
        raise OSError("metrics directory unavailable")

    monkeypatch.setattr(retrieval, "_search_once", fake_search)
    monkeypatch.setattr(metrics, "append_retrieval_event", fail_metrics)

    context, sources, meta = asyncio.run(retrieval.retrieve_context(state, "missing answer", [0.0], RagSettings()))

    assert context == "No relevant memories or documents found."
    assert sources == []
    assert meta["metrics_path"] is None
    assert meta["no_answer"] is True
    assert state.last_metrics_error == "metrics directory unavailable"
