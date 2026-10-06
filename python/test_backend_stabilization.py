import asyncio
from datetime import date
import os
import re
import sqlite3
import threading
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.request import urlopen
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import numpy as np
from fastapi import HTTPException

from cephalon_core.config import Settings
from cephalon_core.events import EventBus
from cephalon_core.schemas import Message, QueryRequest, RagSettings
from cephalon_core.routes import _settings_for_retrieval_scope
from cephalon_core.routes import documents as document_routes
from cephalon_core.app_factory import create_app
from cephalon_core import routes
from cephalon_core.services import document_assets, embedding_runtime, generation, ingestion, metrics, observability, pdf_parser, reranker_runtime, retrieval, table_ingestion, table_models
from cephalon_core.services.memory_jobs import MemoryJobManager
from cephalon_core.services import models
from cephalon_core.services import documents
from cephalon_core.services.prompt_budget import budget_prompt
from cephalon_core.runtime import ModelRuntime
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
        if filter_expr.startswith("doc_id = "):
            doc_id = filter_expr.split("=", 1)[1].strip().strip("'")
            self.rows = [row for row in self.rows if row.get("doc_id") != doc_id]

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
    monkeypatch.setenv("CEPHALON_EMBEDDER_DEVICE", "Vulkan1")
    monkeypatch.setenv("CEPHALON_EMBEDDER_LLAMA_SERVER_BIN", str(tmp_path / "llama-server.exe"))
    monkeypatch.setenv("CEPHALON_EMBEDDER_GPU_LAYERS", "0")
    monkeypatch.setenv("CEPHALON_CORS_ORIGINS", "http://localhost:9999,https://example.test")

    settings = Settings()

    assert settings.data_dir.endswith("data")
    assert settings.model_dir.endswith("models")
    assert settings.port == 9999
    assert settings.max_tokens == 64
    assert settings.embedder_device == "Vulkan1"
    assert settings.embedder_llama_server_bin == str(tmp_path / "llama-server.exe")
    assert settings.embedder_gpu_layers == 0
    assert settings.cors_origins == ["http://localhost:9999", "https://example.test"]


def test_cors_defaults_to_no_browser_origins():
    assert Settings._parse_cors_origins(None) == []


def test_create_app_defers_runtime_directory_writes_until_lifespan(monkeypatch, tmp_path):
    data_dir = tmp_path / "read-only-host-data"
    model_dir = tmp_path / "read-only-host-models"
    monkeypatch.setenv("CEPHALON_DATA_DIR", str(data_dir))
    monkeypatch.setenv("CEPHALON_MODEL_DIR", str(model_dir))

    create_app(Settings())

    assert not data_dir.exists()
    assert not model_dir.exists()


def test_scope_modes_change_retrieval_not_model_style():
    base = RagSettings(top_k=20, rerank_top_n=4, max_tokens=777, temperature=0.72)

    low = _settings_for_retrieval_scope(base, "low")
    medium = _settings_for_retrieval_scope(base, "medium")
    high = _settings_for_retrieval_scope(base, "high")

    assert (low.top_k, low.rerank_top_n) == (12, 3)
    assert (medium.top_k, medium.rerank_top_n) == (20, 4)
    assert (high.top_k, high.rerank_top_n) == (28, 6)
    assert low.temperature == medium.temperature == high.temperature == 0.72
    assert low.max_tokens == medium.max_tokens == high.max_tokens == 777


def test_auto_retrieval_router_skips_clear_conversation_but_defaults_to_safe_recall():
    assert routes.plan_retrieval_route("Hello!", "auto")["resolved"] == "off"
    assert routes.plan_retrieval_route("Brainstorm names for a space game", "auto")["resolved"] == "off"
    assert routes.plan_retrieval_route("What does the RATE paper report?", "auto")["resolved"] == "medium"
    assert routes.plan_retrieval_route("Compare the projected growth rates for 2026 and 2027?", "auto")["resolved"] == "medium"
    assert routes.plan_retrieval_route("Anything", "off")["retrieve"] is False
    assert routes.plan_retrieval_route("Hello", "auto", evidence_required=True)["retrieve"] is True


def test_query_request_keeps_retrieval_scope_and_response_effort_independent():
    default = QueryRequest(prompt="Hello")
    thorough = QueryRequest(
        prompt="Compare the documents carefully.",
        retrieval_scope="high",
        response_effort="thorough",
    )

    assert (default.retrieval_scope, default.response_effort) == ("medium", "balanced")
    assert (thorough.retrieval_scope, thorough.response_effort) == ("high", "thorough")


def test_query_decomposition_keeps_the_original_question_and_deduplicates_candidates():
    subqueries = retrieval.plan_subqueries("Compare alpha versus beta")
    assert subqueries[0] == {"id": "q0", "text": "Compare alpha versus beta"}

    merged = {}
    retrieval._merge_candidates(merged, [{"id": "chunk-1", "score": 0.2}], "q0")
    retrieval._merge_candidates(merged, [{"id": "chunk-1", "score": 0.4}], "q1")

    assert list(merged) == ["chunk-1"]
    assert merged["chunk-1"]["subquery_ids"] == ["q0", "q1"]
    assert merged["chunk-1"]["score"] == 0.4


def test_query_decomposition_drops_format_instructions_and_orphaned_years():
    subqueries = retrieval.plan_subqueries(
        "What global growth rates are projected for 2026 and 2027? "
        "Explain the reasons for the change; cite only the local source."
    )
    assert subqueries[0]["id"] == "q0"
    texts = [item["text"] for item in subqueries]
    assert "2027" not in texts
    assert all(not text.lower().startswith("cite") for text in texts)


def test_generation_instruction_requires_checking_supplied_evidence_before_refusal():
    state = SimpleNamespace(architecture_context="")
    instruction = generation.build_system_instruction(state, "What does the report say?", "[[src:S1]] growth is 2.6 percent")
    assert "Before saying that the local documents do not contain an answer" in instruction


def test_candidate_merge_preserves_a_top_dense_rank_across_subqueries():
    merged = {}
    retrieval._merge_candidates(merged, [{"id": "chunk-1", "score": 0.01, "dense_rank": 1}], "q0")
    retrieval._merge_candidates(merged, [{"id": "chunk-1", "score": 0.03, "lexical_rank": 1}], "q1")

    assert merged["chunk-1"]["dense_rank"] == 1
    assert merged["chunk-1"]["lexical_rank"] == 1


def test_embedding_runtime_uses_bounded_batches(monkeypatch):
    state = SimpleNamespace(embedder=object(), embedding_batch_size=2)
    batch_sizes = []

    def fake_run(_app_state, role: str, texts: list[str]):
        assert role == "document"
        batch_sizes.append(len(texts))
        return [[float(index)] * 3 for index, _text in enumerate(texts)]

    monkeypatch.setattr(retrieval, "_run_embedding_batch", fake_run)

    vectors = retrieval._get_embeddings_sync(state, "document", ["one", "two", "three", "four", "five"])

    assert batch_sizes == [2, 2, 1]
    assert len(vectors) == 5


def test_thorough_response_effort_drafts_then_refines(monkeypatch):
    calls = []

    class FakeResponse:
        def __init__(self, stream: bool):
            self.stream = stream

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return b'{"choices":[{"message":{"content":"Draft answer with one gap."}}]}'

        def __iter__(self):
            if self.stream:
                return iter([
                    b'data: {"choices":[{"delta":{"content":"Final "}}]}\n',
                    b'data: {"choices":[{"delta":{"content":"answer"}}]}\n',
                    b"data: [DONE]\n",
                ])
            return iter([])

    def fake_completion(_app_state, messages, settings, *, stream, response_format=None):
        calls.append({
            "messages": messages,
            "settings": settings,
            "stream": stream,
            "response_format": response_format,
        })
        return FakeResponse(stream)

    monkeypatch.setattr(generation, "_server_completion", fake_completion)

    state = SimpleNamespace(
        architecture_context="",
    )

    tokens = list(generation.stream_response(
        state,
        "What changed?",
        "[[src:S1]] The release changed.",
        [Message(role="user", content="Earlier context")],
        RagSettings(max_tokens=512),
        {"confidence": 0.9},
        response_effort="thorough",
    ))

    assert tokens == ["Final ", "answer"]
    assert len(calls) == 3
    assert calls[0]["stream"] is False
    assert calls[1]["stream"] is False
    assert calls[2]["stream"] is True
    assert calls[0]["settings"].max_tokens == 6144
    assert calls[1]["settings"].max_tokens == 4096
    assert calls[1]["response_format"]["type"] == "json_schema"
    assert calls[2]["settings"].max_tokens == 6144
    assert "Draft answer with one gap." in calls[2]["messages"][0]["content"]
    assert "--- CLAIM AUDIT ---" in calls[2]["messages"][0]["content"]
    assert '"claims":' in calls[2]["messages"][0]["content"]


def test_response_effort_reserves_thinking_capacity_separately_from_final_output():
    settings = RagSettings(max_tokens=16)

    quick = generation._settings_for_response_effort(settings, "quick")
    balanced = generation._settings_for_response_effort(settings, "balanced")
    thorough = generation._settings_for_response_effort(settings, "thorough")

    assert quick.max_tokens == 2048 + generation.THINKING_TOKEN_ALLOCATION
    assert balanced.max_tokens == 4096 + generation.THINKING_TOKEN_ALLOCATION
    assert thorough.max_tokens == 4096 + generation.THINKING_TOKEN_ALLOCATION


def test_prompt_budget_preserves_first_intent_and_recent_messages():
    history = [Message(role="user", content="Original intent: compare release behavior.")]
    history.extend(
        Message(role="assistant" if index % 2 else "user", content=(f"message {index} " * 80))
        for index in range(12)
    )
    bounded_history, bounded_context = budget_prompt(
        history,
        "evidence " * 5000,
        context_window=1200,
        output_tokens=200,
    )

    assert bounded_history[0].content.startswith("Original intent")
    assert bounded_history[-1].content.startswith("message 11")
    assert len(bounded_history) < len(history)
    assert len(bounded_context) < len("evidence " * 5000)


def test_model_runtime_serializes_model_access():
    runtime = ModelRuntime()
    entered = []

    def worker(name):
        with runtime.exclusive():
            entered.append(f"{name}-start")
            import time as _time
            _time.sleep(0.02)
            entered.append(f"{name}-end")

    import threading
    first = threading.Thread(target=worker, args=("first",))
    second = threading.Thread(target=worker, args=("second",))
    first.start()
    second.start()
    first.join()
    second.join()

    assert entered in [
        ["first-start", "first-end", "second-start", "second-end"],
        ["second-start", "second-end", "first-start", "first-end"],
    ]


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


def test_model_inventory_reports_the_configured_external_server(monkeypatch):
    monkeypatch.setenv("CEPHALON_LLAMA_SERVER_MODEL", "My external model")
    inventory = models.model_inventory(SimpleNamespace(settings=Settings()))
    assert inventory["chat_models"] == ["My external model"]
    assert inventory["chat_model_details"] == []
    assert inventory["auxiliary_gguf"] == []


def test_load_llm_connects_to_configured_external_server(monkeypatch):
    monkeypatch.setenv("CEPHALON_LLAMA_SERVER_MODEL", "My external model")
    monkeypatch.setenv("CEPHALON_LLAMA_SERVER_CONTEXT_TOKENS", "32768")
    state = build_memory_state()
    state.llm = None
    monkeypatch.setattr(models, "_server_request", lambda _state, _path: {"status": "ok"})

    models.load_llm(state, "My external model")

    assert state.active_model_name == "My external model"
    assert state.active_context_tokens == 32768


def test_model_backend_reports_offline_connecting_and_connected(monkeypatch):
    offline_state = SimpleNamespace(settings=Settings(), active_model_name=None)
    monkeypatch.setattr(
        models,
        "_server_request",
        lambda _state, _path: (_ for _ in ()).throw(RuntimeError("server offline")),
    )
    offline = models.llama_backend_info(offline_state, probe=True)
    assert offline["server_available"] is False
    assert offline["connection_status"] == "offline"

    monkeypatch.setattr(models, "_server_request", lambda _state, _path: {"status": "ok"})
    connecting_state = SimpleNamespace(settings=Settings(), active_model_name=None)
    connecting = models.llama_backend_info(connecting_state, probe=True)
    assert connecting["server_available"] is True
    assert connecting["connection_status"] == "connecting"

    connected_state = SimpleNamespace(settings=Settings(), active_model_name="Ling-3.0-tiny-Q6_K")
    connected = models.llama_backend_info(connected_state, probe=True)
    assert connected["connection_status"] == "connected"


def test_server_failure_clears_stale_model_state(monkeypatch):
    state = SimpleNamespace(
        settings=Settings(),
        active_model_name="stale-model",
        active_context_tokens=8192,
        active_model_context_tokens=8192,
    )
    monkeypatch.setattr(
        models,
        "_server_request",
        lambda _state, _path: (_ for _ in ()).throw(RuntimeError("server offline")),
    )

    with pytest.raises(HTTPException) as exc:
        models.ensure_server_connected(state)

    assert exc.value.status_code == 503
    assert state.active_model_name is None
    assert state.active_context_tokens is None
    assert state.active_model_context_tokens is None
    assert state.last_model_load_error == "server offline"
    assert state.last_model_error == "server offline"


def test_streaming_generation_rejects_empty_and_truncated_output(monkeypatch):
    class FakeResponse:
        def __init__(self, lines):
            self.lines = lines

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def __iter__(self):
            return iter(self.lines)

    state = SimpleNamespace(architecture_context="")
    settings = RagSettings(max_tokens=64)

    monkeypatch.setattr(
        generation,
        "_server_completion",
        lambda *_args, **_kwargs: FakeResponse([
            b'data: {"choices":[]}\n',
            b"data: [DONE]\n",
        ]),
    )
    with pytest.raises(RuntimeError, match="no visible answer"):
        list(generation._stream_server_completion(state, [], settings))
    assert "no visible answer" in state.last_model_error

    monkeypatch.setattr(
        generation,
        "_server_completion",
        lambda *_args, **_kwargs: FakeResponse([
            b'data: {"choices":[{"delta":{"content":"partial"}}]}\n',
        ]),
    )
    with pytest.raises(RuntimeError, match="completion terminator"):
        list(generation._stream_server_completion(state, [], settings))


def test_non_streaming_generation_rejects_missing_visible_content():
    with pytest.raises(RuntimeError, match="no completion choices"):
        generation._response_content({"choices": []})

    with pytest.raises(RuntimeError, match="no visible answer"):
        generation._response_content({"choices": [{"message": {"content": "<think>private</think>"}}]})


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


def test_non_retrieval_answer_persistence_does_not_reference_missing_query():
    state = build_memory_state()
    route = routes.plan_retrieval_route("Write a short draft", "off")
    meta = routes._empty_retrieval_meta(route, evidence_required=False)
    conversation = storage.create_conversation(state.sqlite, "Non-retrieval answer")
    assistant = storage.append_message(
        state.sqlite,
        conversation["id"],
        "assistant",
        "A generated answer.",
        model="external-model",
    )

    assert meta["query_id"] is None
    storage.save_answer_record(
        state.sqlite,
        {
            "id": assistant["id"],
            "query_id": "orphan-query-id",
            "conversation_id": conversation["id"],
            "message_id": assistant["id"],
            "answer_text": assistant["content"],
            "support_status": "unsupported",
            "meta": meta,
            "citations": [],
        },
    )

    row = storage.fetchone(
        state.sqlite,
        "SELECT query_id FROM answer_records WHERE id = ?",
        (assistant["id"],),
    )
    assert row["query_id"] is None


def test_completed_answer_commits_evidence_and_memory_job_together(monkeypatch):
    state = build_memory_state()
    conversation = storage.create_conversation(state.sqlite, "Atomic answer")
    record = {"answer_text": "Answer", "support_status": "supported", "citations": []}
    message = storage.save_completed_answer(
        state.sqlite, conversation["id"], "Answer", model="test", settings={}, meta={},
        sources=[{"source_id": "S1"}], record=record, memory_prompt="Question",
    )
    assert storage.get_conversation(state.sqlite, conversation["id"])["messages"][0]["sources"][0]["source_id"] == "S1"
    assert storage.fetchone(state.sqlite, "SELECT 1 FROM answer_records WHERE message_id = ?", (message["id"],))
    assert storage.fetchone(state.sqlite, "SELECT prompt FROM memory_jobs WHERE message_id = ?", (message["id"],))["prompt"] == "Question"

    def fail_record(*_args, **_kwargs):
        raise RuntimeError("record write failed")

    monkeypatch.setattr(storage, "save_answer_record", fail_record)
    with pytest.raises(RuntimeError, match="record write failed"):
        storage.save_completed_answer(
            state.sqlite, conversation["id"], "Second answer", model="test", settings={}, meta={},
            sources=[{"source_id": "S2"}], record=record, memory_prompt="Second question",
        )
    assert [item["content"] for item in storage.get_conversation(state.sqlite, conversation["id"])["messages"]] == ["Answer"]
    assert storage.fetchone(state.sqlite, "SELECT COUNT(*) AS count FROM memory_jobs")["count"] == 1


def test_memory_job_recovers_and_finishes_without_blocking_answer(monkeypatch):
    state = build_memory_state()
    conversation = storage.create_conversation(state.sqlite, "Deferred memory")
    message = storage.save_completed_answer(
        state.sqlite, conversation["id"], "Answer", model="test", settings={}, meta={},
        sources=[], record={"answer_text": "Answer", "citations": []}, memory_prompt="Question",
    )
    finished = threading.Event()

    async def fake_save(_state, _conversation_id, message_id, prompt, answer):
        assert (message_id, prompt, answer) == (message["id"], "Question", "Answer")
        finished.set()

    monkeypatch.setattr(retrieval, "save_permanent_memory", fake_save)

    async def run():
        manager = MemoryJobManager(state)
        await manager.start()
        try:
            assert await asyncio.to_thread(finished.wait, 1)
            for _ in range(20):
                if storage.fetchone(state.sqlite, "SELECT 1 FROM memory_jobs WHERE message_id = ?", (message["id"],)) is None:
                    break
                await asyncio.sleep(0.01)
        finally:
            await manager.stop()

    asyncio.run(run())
    assert storage.fetchone(state.sqlite, "SELECT 1 FROM memory_jobs WHERE message_id = ?", (message["id"],)) is None


def test_document_payloads_batch_tags_and_limit_previews():
    state = build_memory_state()
    for index in range(3):
        doc_id = f"doc-{index}"
        storage.execute(
            state.sqlite,
            "INSERT INTO documents (id, path, display_name, content_hash, chunk_count, status, type) VALUES (?, ?, ?, ?, ?, 'ready', 'file')",
            (doc_id, f"/tmp/{doc_id}.md", doc_id, f"hash-{index}", 30),
        )
        storage.execute(state.sqlite, "INSERT INTO document_tags (doc_id, tag) VALUES (?, ?)", (doc_id, "test"))
        for chunk_index in range(30):
            storage.execute(
                state.sqlite,
                "INSERT INTO chunks (id, doc_id, chunk_index, text) VALUES (?, ?, ?, ?)",
                (f"{doc_id}-{chunk_index}", doc_id, chunk_index, f"chunk {chunk_index}"),
            )

    statements = []
    state.sqlite.set_trace_callback(statements.append)
    documents = storage.list_document_payloads(state.sqlite)
    detail = storage.get_document_payload(state.sqlite, "doc-0", preview_limit=20)
    state.sqlite.set_trace_callback(None)

    assert len(documents) == 3
    assert all(document["tags"] == ["test"] for document in documents)
    assert len(detail["chunk_preview"]) == 20
    assert sum("FROM document_tags" in statement for statement in statements) == 2
    assert any("LIMIT 20" in statement for statement in statements)


def test_index_health_aggregates_without_loading_chunk_text(tmp_path):
    state = build_memory_state()
    state.settings = SimpleNamespace(data_dir=str(tmp_path))
    storage.execute(
        state.sqlite,
        """INSERT INTO documents
           (id, path, display_name, content_hash, status, type, stale_embedding, retrieval_count)
           VALUES ('health-doc', 'health.md', 'Health', 'hash', 'failed: parsing', 'file', 1, 3)""",
    )
    for index, length in enumerate((4, 6, 8)):
        storage.execute(
            state.sqlite,
            """INSERT INTO chunks
               (id, doc_id, chunk_index, text, chunk_length, text_hash, embedding_model_id, chunking_profile)
               VALUES (?, 'health-doc', ?, ?, ?, ?, ?, ?)""",
            (f"health-{index}", index, "x" * length, length, "same" if index < 2 else "other", "embed" if index < 2 else None, "profile"),
        )
    statements = []
    state.sqlite.set_trace_callback(statements.append)
    health = observability.index_health(state)
    state.sqlite.set_trace_callback(None)

    assert health["document_count"] == 1
    assert health["chunk_count"] == 3
    assert health["embedded_chunk_count"] == 2
    assert health["duplicate_chunk_count"] == 1
    assert health["median_chunk_length"] == 6
    assert health["failed_ingestion_count"] == health["stale_document_count"] == 1
    assert health["embedding_model_counts"] == {"embed": 2, "unknown": 1}
    assert all("SELECT * FROM CHUNKS" not in statement.upper() for statement in statements)


def test_conversation_sources_are_loaded_in_one_query():
    state = build_memory_state()
    conversation = storage.create_conversation(state.sqlite, "Batch sources")
    for index in range(4):
        message = storage.append_message(state.sqlite, conversation["id"], "assistant", f"answer {index}")
        storage.save_message_sources(state.sqlite, message["id"], [{"source_id": f"S{index + 1}"}])

    statements = []
    state.sqlite.set_trace_callback(statements.append)
    loaded = storage.get_conversation(state.sqlite, conversation["id"])
    state.sqlite.set_trace_callback(None)

    assert len(loaded["messages"]) == 4
    assert sum("FROM message_sources" in statement for statement in statements) == 1


def test_conversation_messages_are_paginated_from_the_newest():
    state = build_memory_state()
    conversation = storage.create_conversation(state.sqlite, "Long chat")
    for index in range(5):
        storage.append_message(state.sqlite, conversation["id"], "user", f"message {index}")

    newest = storage.get_conversation(state.sqlite, conversation["id"], message_limit=2)
    older = storage.get_conversation(
        state.sqlite,
        conversation["id"],
        message_limit=2,
        before=newest["next_before"],
    )

    assert [message["content"] for message in newest["messages"]] == ["message 3", "message 4"]
    assert newest["has_more"] is True
    assert [message["content"] for message in older["messages"]] == ["message 1", "message 2"]


def test_process_single_file_skips_duplicate_hash(monkeypatch, tmp_path):
    state = build_memory_state()
    file_path = tmp_path / "note.md"
    file_path.write_text("The 4-7-8 method is a breathing exercise.", encoding="utf-8")

    async def fake_embedding(_app_state, _text: str):
        return [0.0] * storage.active_embedding_metadata()["embedding_dim"]

    monkeypatch.setattr("cephalon_core.services.ingestion.get_document_embedding", fake_embedding)

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


def test_document_readers_stream_hash_and_avoid_duplicate_extraction(monkeypatch, tmp_path):
    file_path = tmp_path / "large.bin"
    file_path.write_bytes(b"x" * (1024 * 1024 + 17))
    read_sizes = []
    original_open = open

    class TrackingFile:
        def __init__(self, file_obj):
            self.file_obj = file_obj

        def __enter__(self):
            return self

        def __exit__(self, *args):
            self.file_obj.close()

        def read(self, size=-1):
            read_sizes.append(size)
            return self.file_obj.read(size)

    monkeypatch.setattr("builtins.open", lambda path, mode="r", *args, **kwargs: TrackingFile(original_open(path, mode, *args, **kwargs)))
    documents.get_file_hash(str(file_path))
    assert read_sizes
    assert -1 not in read_sizes

    calls = []

    monkeypatch.setattr(
        documents,
        "parse_pdf",
        lambda _path: (calls.append("extract"), SimpleNamespace(text="page text\npage text"))[1],
    )
    text, _mode = documents.extract_text("fixture.pdf")
    assert text == "page text\npage text"
    assert calls == ["extract"]

    workbook = SimpleNamespace(worksheets=[])
    load_workbook = lambda *_args, **kwargs: (calls.append(kwargs), workbook)[1]
    monkeypatch.setattr(documents.openpyxl, "load_workbook", load_workbook)
    documents.extract_text("fixture.xlsx")
    assert calls[-1]["read_only"] is True


def test_typed_table_values_are_conservative_and_exact():
    cases = {
        "-12": ("-12", "integer", None),
        "1.20e-7": ("0.00000012", "decimal", None),
        "12.50%": ("12.5", "percentage", "%"),
        "-4.25 kg": ("-4.25", "decimal", "kg"),
        "12345678901234567890.123400": ("12345678901234567890.1234", "decimal", None),
        "1,23": ("1,23", "text", None),
        "": (None, "missing", None),
    }
    for raw, expected in cases.items():
        typed = table_models.type_value(raw)
        assert (typed.normalized_value, typed.value_type, typed.unit) == expected
    assert table_models.type_value(date(2026, 8, 12)).normalized_value == "2026-08-12"
    xlsx_percent = table_models.type_value(0.125, number_format="0.0%")
    assert (xlsx_percent.raw_value, xlsx_percent.normalized_value, xlsx_percent.unit) == ("0.125", "12.5", "%")
    assert table_models.type_value(0, number_format="0%").normalized_value == "0"
    assert table_models.type_value(0.125, number_format='0.0"%"').value_type == "decimal"


def test_typed_table_limits_are_bounded_and_warn(monkeypatch):
    monkeypatch.setattr(table_models, "MAX_TABLE_ROWS", 2)
    monkeypatch.setattr(table_models, "MAX_CELL_CHARACTERS", 4)
    table = table_models.build_table(
        [["Head"], ["12345"], ["discarded"]], source_type="csv", table_index=0
    )
    assert table.row_count == 2
    assert "row_limit_reached" in table.warnings
    assert table.cells[1].raw_value == "12345"
    assert table.cells[1].parse_warnings == ["cell_length_limit_exceeded"]


def test_csv_typed_tables_round_trip_stable_ids_and_cascade(monkeypatch, tmp_path):
    state = build_memory_state()
    file_path = tmp_path / "measurements.csv"
    file_path.write_text(
        "Name,Percent,Mass,Scientific,Missing\nalpha,12.50%,-4.25 kg,1.20e-7,\n",
        encoding="utf-8",
    )

    async def fake_embedding(_app_state, _text: str):
        return [0.0] * storage.active_embedding_metadata()["embedding_dim"]

    monkeypatch.setattr("cephalon_core.services.ingestion.get_document_embedding", fake_embedding)
    first = asyncio.run(process_single_file(state, str(file_path), RagSettings()))
    assert first["status"] == "ready"
    first_tables = table_ingestion.load_document_tables(state.sqlite, first["doc_id"])
    first_ids = [cell["id"] for cell in first_tables[0]["cells"]]
    values = {cell["cell_ref"]: cell for cell in first_tables[0]["cells"]}
    assert values["csv:r2c2"]["normalized_value"] == "12.5"
    assert values["csv:r2c2"]["unit"] == "%"
    assert values["csv:r2c3"]["normalized_value"] == "-4.25"
    assert values["csv:r2c3"]["unit"] == "kg"
    assert values["csv:r2c4"]["normalized_value"] == "0.00000012"
    assert values["csv:r2c5"]["value_type"] == "missing"

    repeated = asyncio.run(process_single_file(
        state, str(file_path), RagSettings(), existing_doc_id=first["doc_id"]
    ))
    assert repeated["status"] == "ready", repeated
    second_ids = [cell["id"] for cell in table_ingestion.load_document_tables(state.sqlite, first["doc_id"])[0]["cells"]]
    assert second_ids == first_ids

    delete_document_rows(state, first["doc_id"])
    assert storage.fetchone(state.sqlite, "SELECT COUNT(*) AS count FROM tables")["count"] == 0
    assert storage.fetchone(state.sqlite, "SELECT COUNT(*) AS count FROM table_cells")["count"] == 0


def test_csv_encoding_validation_covers_bytes_after_the_dialect_sample(tmp_path):
    path = tmp_path / "late-latin1.csv"
    path.write_bytes(b"Name,Value\nrow," + (b"a" * 8200) + b"\xe9\n")

    extracted = documents.extract_document(str(path))

    assert "csv_encoding_fallback:latin-1" in extracted.warnings
    assert extracted.tables[0].provenance["encoding"] == "latin-1"
    assert extracted.tables[0].raw_rows[1][1].endswith("\xe9")


def test_typed_table_reindex_failure_rolls_back_previous_rows(monkeypatch, tmp_path):
    state = build_memory_state()
    file_path = tmp_path / "atomic.csv"
    file_path.write_text("Name,Value\nold,1\n", encoding="utf-8")

    async def fake_embedding(_app_state, _text: str):
        return [0.0] * storage.active_embedding_metadata()["embedding_dim"]

    monkeypatch.setattr("cephalon_core.services.ingestion.get_document_embedding", fake_embedding)
    first = asyncio.run(process_single_file(state, str(file_path), RagSettings()))
    original = table_ingestion.persistence_rows
    file_path.write_text("Name,Value\nnew,2\n", encoding="utf-8")

    def duplicate_cell(*args, **kwargs):
        rows = original(*args, **kwargs)
        rows["cells"].append(rows["cells"][0])
        return rows

    monkeypatch.setattr(table_ingestion, "persistence_rows", duplicate_cell)
    failed = asyncio.run(process_single_file(
        state, str(file_path), RagSettings(), existing_doc_id=first["doc_id"]
    ))
    assert failed["status"] == "failed"
    restored = table_ingestion.load_document_tables(state.sqlite, first["doc_id"])
    assert [cell["raw_value"] for cell in restored[0]["cells"]] == ["Name", "Value", "old", "1"]
    assert state.sqlite.execute(
        "SELECT status FROM documents WHERE id = ?", (first["doc_id"],)
    ).fetchone()[0] == "ready"


def test_typed_table_migration_preserves_existing_stack_a_rows():
    state = build_memory_state()
    state.sqlite.execute(
        "INSERT INTO documents (id, path, content_hash, status) VALUES ('old-doc', 'old.txt', 'hash', 'ready')"
    )
    state.sqlite.commit()
    for table in ("table_cells", "table_rows", "table_columns", "tables"):
        state.sqlite.execute(f"DROP TABLE {table}")
    state.sqlite.execute("DELETE FROM schema_migrations WHERE version = '018_typed_tables'")
    state.sqlite.commit()

    storage.run_migrations(state.sqlite, state.settings)

    assert state.sqlite.execute("SELECT path FROM documents WHERE id = 'old-doc'").fetchone()[0] == "old.txt"
    assert state.sqlite.execute(
        "SELECT COUNT(*) FROM schema_migrations WHERE version = '018_typed_tables'"
    ).fetchone()[0] == 1


def test_structured_table_parser_version_requires_explicit_reindex():
    from cephalon_core.services.ingestion import parser_version_for_path

    assert parser_version_for_path("measurements.csv") == table_models.TABLE_PARSER_VERSION
    assert parser_version_for_path("measurements.xlsx") == table_models.TABLE_PARSER_VERSION
    assert parser_version_for_path("paper.pdf") == documents.PDF_PARSER_VERSION


def test_xlsx_typed_tables_preserve_sheets_formulas_formats_and_merges(tmp_path):
    path = tmp_path / "workbook.xlsx"
    workbook = documents.openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Results"
    sheet.append(["Count", "Rate", "Formula", "Merged", None, "Date"])
    # Excel stores numeric cells with about 15 significant digits. Large exact
    # decimal handling is covered by the raw-text/CSV tests instead.
    sheet.append([123456789012345, 0.125, "=A2*2", "group", None, date(2026, 8, 12)])
    sheet["B2"].number_format = "0.0%"
    sheet.merge_cells("D1:E1")
    workbook.save(path)
    workbook.close()

    extracted = documents.extract_document(str(path))
    assert extracted.extraction_mode == "native_structured"
    assert len(extracted.tables) == 1
    table = extracted.tables[0]
    cells = {cell.cell_ref: cell for cell in table.cells}
    assert cells["Results!A2"].normalized_value == "123456789012345"
    assert cells["Results!B2"].value_type == "percentage"
    assert cells["Results!B2"].raw_value == "0.125"
    assert cells["Results!B2"].normalized_value == "12.5"
    assert cells["Results!C2"].formula == "=A2*2"
    assert cells["Results!C2"].effective_value is None
    assert cells["Results!F2"].value_type == "datetime"
    assert table.provenance["merged_ranges"] == ["D1:E1"]


def test_xlsx_formula_uses_cached_value_without_recalculation(tmp_path):
    path = tmp_path / "cached-formula.xlsx"
    workbook = documents.openpyxl.Workbook()
    sheet = workbook.active
    sheet.title = "Results"
    sheet.append(["Input", "Formula"])
    sheet.append([4, "=A2*2"])
    workbook.save(path)
    workbook.close()

    with zipfile.ZipFile(path, "r") as archive:
        members = {name: archive.read(name) for name in archive.namelist()}
    worksheet_name = "xl/worksheets/sheet1.xml"
    worksheet = members[worksheet_name].decode("utf-8")
    worksheet, replacements = re.subn(
        r'(<c r="B2"[^>]*><f>.*?</f><v>).*?(</v></c>)',
        r"\g<1>8\g<2>",
        worksheet,
        count=1,
    )
    assert replacements == 1
    members[worksheet_name] = worksheet.encode("utf-8")
    rewritten = tmp_path / "cached-formula-rewritten.xlsx"
    with zipfile.ZipFile(rewritten, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for name, payload in members.items():
            archive.writestr(name, payload)
    rewritten.replace(path)

    extracted = documents.extract_document(str(path))
    formula_cell = {cell.cell_ref: cell for cell in extracted.tables[0].cells}["Results!B2"]
    assert formula_cell.raw_value == "=A2*2"
    assert formula_cell.formula == "=A2*2"
    assert formula_cell.effective_value == "8"
    assert formula_cell.normalized_value == "8"
    assert formula_cell.value_type == "integer"
    assert "formula_cached_value_unavailable" not in formula_cell.parse_warnings


def test_corrupt_xlsx_fails_without_creating_partial_table(tmp_path):
    path = tmp_path / "corrupt.xlsx"
    path.write_bytes(b"not an office archive")

    with pytest.raises(Exception):
        documents.extract_document(str(path))


def test_typed_tables_flag_rolls_back_to_existing_text_index(monkeypatch, tmp_path):
    state = build_memory_state()
    state.settings.typed_tables = False
    path = tmp_path / "rollback.csv"
    path.write_text("Metric,Value\nRecall,0.93\n", encoding="utf-8")

    async def fake_embedding(_app_state, _text: str):
        return [0.0] * storage.active_embedding_metadata()["embedding_dim"]

    monkeypatch.setattr("cephalon_core.services.ingestion.get_document_embedding", fake_embedding)
    result = asyncio.run(process_single_file(state, str(path), RagSettings()))
    assert result["status"] == "ready"
    assert storage.fetchone(state.sqlite, "SELECT COUNT(*) AS count FROM tables")["count"] == 0
    assert storage.fetchone(state.sqlite, "SELECT COUNT(*) AS count FROM chunks WHERE doc_id = ?", (result["doc_id"],))["count"] > 0


def test_rich_pdf_parser_preserves_pages_headings_tables_and_removes_repeated_footer(monkeypatch):
    def word(text, x0, top, x1, bottom, size=10, fontname="Regular"):
        return {
            "text": text,
            "x0": x0,
            "top": top,
            "x1": x1,
            "bottom": bottom,
            "size": size,
            "fontname": fontname,
        }

    class FakeTable:
        bbox = (40, 180, 300, 240)
        rows = [
            SimpleNamespace(cells=[(40, 180, 160, 200), (160, 180, 300, 200)]),
            SimpleNamespace(cells=[(40, 200, 160, 220), (160, 200, 300, 220)]),
            SimpleNamespace(cells=[(40, 220, 160, 240), (160, 220, 300, 240)]),
        ]

        def extract(self):
            return [["Model", "Recall"], ["Baseline", "72.4"], ["RATE", "81.7"]]

    class FakePage:
        width = 600
        height = 800

        def __init__(self, page_number):
            self.page_number = page_number
            self.images = [{"x0": 40, "top": 300, "x1": 220, "bottom": 400}]

        def extract_words(self, **_kwargs):
            heading = [
                word("3", 40, 50, 50, 65, 16, "Bold"),
                word("Results", 55, 50, 130, 65, 16, "Bold"),
            ] if self.page_number == 1 else []
            return heading + [
                word("Retrieval", 40, 120, 95, 132),
                word("improved", 100, 120, 155, 132),
                word("substantially.", 160, 120, 235, 132),
                word("Figure", 40, 410, 75, 420, 9),
                word("1:", 80, 410, 90, 420, 9),
                word("Retrieval", 95, 410, 145, 420, 9),
                word("pipeline", 150, 410, 195, 420, 9),
                word("Proceedings", 250, 760, 320, 770, 8),
                word(str(self.page_number), 325, 760, 332, 770, 8),
            ]

        def find_tables(self, _settings):
            return [FakeTable()] if self.page_number == 1 else []

        def extract_text(self, **_kwargs):
            return "fallback"

        def close(self):
            return None

    class FakePdf:
        pages = [FakePage(1), FakePage(2)]

        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

    fake_image = SimpleNamespace(
        data=b"embedded-image",
        name="figure.png",
        image=SimpleNamespace(size=(180, 100)),
    )
    monkeypatch.setattr(pdf_parser.pdfplumber, "open", lambda *_args, **_kwargs: FakePdf())
    monkeypatch.setattr(
        pdf_parser,
        "PdfReader",
        lambda _path: SimpleNamespace(
            pages=[SimpleNamespace(images=[fake_image]), SimpleNamespace(images=[fake_image])]
        ),
    )

    parsed = pdf_parser.parse_pdf("fixture.pdf")

    assert parsed.page_count == 2
    assert "Proceedings" not in parsed.text
    assert any(block.block_type == "table" and "RATE | 81.7" in block.text for block in parsed.blocks)
    assert parsed.tables[0].cells[0].bounding_box == (40.0, 180.0, 160.0, 200.0)
    paragraph = next(block for block in parsed.blocks if "improved substantially" in block.text)
    assert paragraph.page_number == 1
    assert paragraph.heading_path == ["3 Results"]
    assert paragraph.element_id.startswith("el-")
    assert len(parsed.assets) == 2
    assert parsed.assets[0].caption == "Figure 1: Retrieval pipeline"
    assert parsed.assets[0].asset_id in next(
        block for block in parsed.blocks if block.block_type == "caption"
    ).asset_ids


def test_structured_pdf_ingestion_persists_source_provenance(monkeypatch, tmp_path):
    state = build_memory_state()
    state.settings.data_dir = str(tmp_path / "data")
    file_path = tmp_path / "paper.pdf"
    file_path.write_bytes(b"%PDF fixture")
    extracted = documents.ExtractedDocument(
        text="4 Findings\n\nThe measured result was 81.7.",
        extraction_mode="native_structured",
        page_count=1,
        parser_version=pdf_parser.PARSER_VERSION,
        assets=[
            pdf_parser.PdfAsset(
                asset_id="p1-img-0123456789abcdef",
                page_number=1,
                bounding_box=(40, 120, 240, 220),
                data=b"fixture-image",
                extension=".png",
                mime_type="image/png",
                sha256="0123456789abcdef",
                caption="Figure 1: Measured result",
                width=200,
                height=100,
            ),
        ],
        blocks=[
            pdf_parser.DocumentBlock(
                text="4 Findings",
                page_number=1,
                block_type="heading",
                heading_path=["4 Findings"],
                heading_level=1,
                bounding_box=(40, 40, 180, 60),
                block_index=0,
            ),
            pdf_parser.DocumentBlock(
                text="The measured result was 81.7.",
                page_number=1,
                block_type="paragraph",
                heading_path=["4 Findings"],
                bounding_box=(40, 80, 280, 105),
                block_index=1,
                element_id="el-measured-result",
                asset_ids=["p1-img-0123456789abcdef"],
            ),
        ],
    )
    monkeypatch.setattr(documents, "extract_document", lambda *_args, **_kwargs: extracted)

    async def fake_embedding(_app_state, _text):
        return [0.0] * storage.active_embedding_metadata()["embedding_dim"]

    monkeypatch.setattr("cephalon_core.services.ingestion.get_document_embedding", fake_embedding)

    result = asyncio.run(process_single_file(state, str(file_path), RagSettings()))
    chunk = storage.fetchone(
        state.sqlite,
        """
        SELECT block_type, section_heading, heading_path, page_number, page_end,
               block_index, bounding_box, parser_version, provenance_json
        FROM chunks WHERE doc_id = ?
        """,
        (result["doc_id"],),
    )

    assert result["status"] == "ready"
    assert chunk["block_type"] == "paragraph"
    assert chunk["section_heading"] == "4 Findings"
    assert chunk["heading_path"] == '["4 Findings"]'
    assert (chunk["page_number"], chunk["page_end"], chunk["block_index"]) == (1, 1, 1)
    assert chunk["parser_version"] == pdf_parser.PARSER_VERSION
    assert '"element_ids": ["el-measured-result"]' in chunk["provenance_json"]
    asset = storage.fetchone(
        state.sqlite,
        "SELECT id, filename, caption FROM document_assets WHERE doc_id = ?",
        (result["doc_id"],),
    )
    assert asset["id"] == "p1-img-0123456789abcdef"
    assert asset["caption"] == "Figure 1: Measured result"
    assert (tmp_path / "data" / "document-assets" / result["doc_id"] / asset["filename"]).read_bytes() == b"fixture-image"
    hydrated = retrieval.hydrate_sources(
        state,
        [{
            "id": f"{result['doc_id']}_0",
            "doc_id": result["doc_id"],
            "text": "The measured result was 81.7.",
            "score": 0.9,
        }],
    )
    assert hydrated[0].element_ids == ["el-measured-result"]
    assert hydrated[0].assets[0].url.endswith("/assets/p1-img-0123456789abcdef")
    response = document_routes.get_document_asset(
        SimpleNamespace(app=SimpleNamespace(state=state)),
        result["doc_id"],
        "p1-img-0123456789abcdef",
    )
    assert response.media_type == "image/png"
    assert response.path.endswith(asset["filename"])


def test_pdf_asset_reindex_rollback_restores_previous_files(tmp_path):
    doc_id = "11111111-1111-1111-1111-111111111111"
    old = pdf_parser.PdfAsset(
        asset_id="p1-img-old",
        page_number=1,
        bounding_box=None,
        data=b"old",
        extension=".png",
        mime_type="image/png",
        sha256="old",
    )
    first = document_assets.AssetTransaction.prepare(str(tmp_path), doc_id, [old])
    first.promote()
    first.finalize()

    new = pdf_parser.PdfAsset(
        asset_id="p1-img-new",
        page_number=1,
        bounding_box=None,
        data=b"new",
        extension=".png",
        mime_type="image/png",
        sha256="new",
    )
    replacement = document_assets.AssetTransaction.prepare(str(tmp_path), doc_id, [new])
    replacement.promote()
    replacement.rollback()

    active = tmp_path / "document-assets" / doc_id
    assert (active / "p1-img-old.png").read_bytes() == b"old"
    assert not (active / "p1-img-new.png").exists()
    document_assets.delete_document_assets(str(tmp_path), doc_id)
    assert not active.exists()


def test_force_text_import_allows_unknown_extension(monkeypatch, tmp_path):
    state = build_memory_state()
    file_path = tmp_path / "notes.custom"
    file_path.write_text("Custom extension should still import as text.", encoding="utf-8")

    async def fake_embedding(_app_state, _text: str):
        return [0.0] * storage.active_embedding_metadata()["embedding_dim"]

    monkeypatch.setattr("cephalon_core.services.ingestion.get_document_embedding", fake_embedding)

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

    monkeypatch.setattr("cephalon_core.services.ingestion.get_document_embedding", fake_embedding)

    result = asyncio.run(process_single_file(state, str(file_path), RagSettings()))
    parents = storage.fetchall(state.sqlite, "SELECT id, summary FROM parent_chunks WHERE doc_id = ?", (result["doc_id"],))
    summaries = storage.fetchall(state.sqlite, "SELECT id, parent_id, summary FROM summary_nodes WHERE doc_id = ?", (result["doc_id"],))
    chunks = storage.fetchall(state.sqlite, "SELECT parent_id, semantic_role FROM chunks WHERE doc_id = ?", (result["doc_id"],))

    assert result["status"] == "ready"
    assert parents
    assert summaries
    assert all(row["parent_id"] for row in chunks)
    assert {row["semantic_role"] for row in chunks} == {"child"}


def test_process_single_file_batches_final_embeddings(monkeypatch, tmp_path):
    state = build_memory_state()
    state.embedder = object()
    file_path = tmp_path / "batch.md"
    file_path.write_text("Batch final summary and child embeddings together.", encoding="utf-8")
    batch_sizes = []

    async def fake_embedding(_app_state, _text: str):
        return [0.0] * storage.active_embedding_metadata()["embedding_dim"]

    async def fake_embeddings(_app_state, texts: list[str]):
        batch_sizes.append(len(texts))
        return [
            [float(index)] * storage.active_embedding_metadata()["embedding_dim"]
            for index, _text in enumerate(texts)
        ]

    monkeypatch.setattr("cephalon_core.services.ingestion.get_document_embedding", fake_embedding)
    monkeypatch.setattr("cephalon_core.services.ingestion.get_document_embeddings", fake_embeddings)

    result = asyncio.run(process_single_file(state, str(file_path), RagSettings()))

    assert result["status"] == "ready"
    assert batch_sizes == [2]


def test_semantic_child_chunking_does_not_embed_each_sentence(monkeypatch):
    state = SimpleNamespace(embedder=object())

    async def unexpected_embedding(*_args, **_kwargs):
        raise AssertionError("Chunk boundary selection should not invoke the embedder.")

    monkeypatch.setattr("cephalon_core.services.ingestion.get_document_embeddings", unexpected_embedding)
    monkeypatch.setattr("cephalon_core.services.ingestion.get_document_embedding", unexpected_embedding)
    text = "First sentence has enough words to be a real unit. Second sentence is another separate unit. Third sentence completes the test."

    chunks = asyncio.run(ingestion.build_semantic_child_chunks(state, text, RagSettings()))

    assert chunks


def test_process_single_file_reports_stage_progress(monkeypatch, tmp_path):
    state = build_memory_state()
    file_path = tmp_path / "progress.md"
    file_path.write_text("Report extraction, chunking, embedding, and persistence.", encoding="utf-8")
    progress = []

    async def fake_embedding(_app_state, _text: str):
        return [0.0] * storage.active_embedding_metadata()["embedding_dim"]

    async def report(stage: str, percent: int):
        progress.append((stage, percent))

    monkeypatch.setattr("cephalon_core.services.ingestion.get_document_embedding", fake_embedding)
    result = asyncio.run(
        process_single_file(
            state,
            str(file_path),
            RagSettings(),
            progress=report,
        )
    )

    assert result["status"] == "ready"
    assert progress == [
        ("extracting", 10),
        ("chunking", 35),
        ("embedding", 65),
        ("persisting", 90),
        ("complete", 100),
    ]


def test_unknown_text_file_imports_without_force_text(monkeypatch, tmp_path):
    state = build_memory_state()
    file_path = tmp_path / "numbers.dataset"
    file_path.write_text("quarter,revenue\nQ1,120\nQ2,143\n", encoding="utf-8")

    async def fake_embedding(_app_state, _text: str):
        return [0.0] * storage.active_embedding_metadata()["embedding_dim"]

    monkeypatch.setattr("cephalon_core.services.ingestion.get_document_embedding", fake_embedding)

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


def test_query_requires_connected_server(monkeypatch):
    app_state = SimpleNamespace(startup_error=None, active_model_name="loaded.gguf")
    monkeypatch.setattr(models, "_server_request", lambda _state, _path: {"status": "ok"})
    with pytest.raises(HTTPException) as exc:
        routes._ensure_query_model_loaded(SimpleNamespace(startup_error=None, active_model_name=None), "other.gguf")

    assert exc.value.status_code == 409
    assert "Connect to the configured external llama.cpp server" in exc.value.detail


def test_query_accepts_server_reported_model_even_when_client_label_differs(monkeypatch):
    app_state = SimpleNamespace(startup_error=None, active_model_name="loaded.gguf")
    monkeypatch.setattr(models, "_server_request", lambda _state, _path: {"status": "ok"})

    routes._ensure_query_model_loaded(app_state, "External llama.cpp server")


def test_generation_event_stream_stops_after_client_disconnect():
    class FakeRequest:
        def __init__(self):
            self.calls = 0

        async def is_disconnected(self):
            self.calls += 1
            return self.calls > 1

    class ClosableEvents:
        def __init__(self):
            self.closed = False
            self.events = iter([("token", "first"), ("token", "second")])

        def __iter__(self):
            return self

        def __next__(self):
            return next(self.events)

        def close(self):
            self.closed = True

    async def collect():
        request = FakeRequest()
        events = ClosableEvents()
        received = [event async for event in routes._cancel_on_disconnect(request, events)]
        return received, events.closed

    received, closed = asyncio.run(collect())

    assert received == [("token", "first")]
    assert closed is True


def test_completed_generation_stream_keeps_cancellation_clear():
    class Request:
        async def is_disconnected(self):
            return False

    cancellation = generation.CompletionCancellation()

    async def collect():
        return [event async for event in routes._cancel_on_disconnect(Request(), iter([("token", "answer")]), cancellation)]

    assert asyncio.run(collect()) == [("token", "answer")]
    assert cancellation.is_cancelled() is False


def test_cancel_endpoint_marks_the_matching_query_only():
    active = generation.CompletionCancellation()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(active_queries={"query-1": active})))
    assert asyncio.run(routes.cancel_query(request, "other")) == {"cancelled": False}
    assert active.is_cancelled() is False
    assert asyncio.run(routes.cancel_query(request, "query-1")) == {"cancelled": True}
    assert active.is_cancelled() is True


def test_cancel_interrupts_a_stalled_urllib_response():
    release = threading.Event()
    reading = threading.Event()

    class Handler(BaseHTTPRequestHandler):
        def do_GET(self):
            self.send_response(200)
            self.end_headers()
            release.wait(3)

        def log_message(self, *_args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    cancellation = generation.CompletionCancellation()

    def read():
        with urlopen(f"http://127.0.0.1:{server.server_port}/", timeout=3) as response:
            cancellation.bind(response)
            reading.set()
            try:
                response.readline()
            except OSError:
                pass
            finally:
                cancellation.unbind(response)

    reader = threading.Thread(target=read, daemon=True)
    try:
        reader.start()
        assert reading.wait(1)
        cancellation.abort()
        reader.join(1)
        assert not reader.is_alive()
    finally:
        release.set()
        server.shutdown()
        server.server_close()


def test_query_stream_commits_answer_and_finishes_before_memory_embedding(monkeypatch):
    state = build_memory_state()
    state.startup_error = None
    state.active_queries = {}
    wake_calls = []
    state.memory_jobs = SimpleNamespace(wake=lambda: wake_calls.append(True))
    monkeypatch.setattr(routes, "_ensure_query_model_loaded", lambda *_args: None)
    monkeypatch.setattr(
        generation, "stream_response_events",
        lambda *_args, **_kwargs: iter([("phase", "answering"), ("token", "Hello"), ("token", " world")]),
    )
    monkeypatch.setattr(metrics, "append_retrieval_event", lambda *_args, **_kwargs: None)
    request = SimpleNamespace(app=SimpleNamespace(state=state))

    async def is_disconnected():
        return False

    request.is_disconnected = is_disconnected

    async def run():
        response = await routes.chat_and_remember(
            request,
            QueryRequest(
                prompt="Say hello", model="test", retrieval_scope="off", request_id="test-query",
                settings=RagSettings(conversation_memory=True),
            ),
        )
        return [chunk async for chunk in response.body_iterator]

    packets = asyncio.run(run())
    assert any("event: done" in packet for packet in packets)
    assert wake_calls == [True]
    assert state.active_queries == {}
    assistant = storage.fetchone(state.sqlite, "SELECT id, content FROM messages WHERE role = 'assistant'")
    assert assistant["content"] == "Hello world"
    assert storage.fetchone(state.sqlite, "SELECT 1 FROM answer_records WHERE message_id = ?", (assistant["id"],))
    assert storage.fetchone(state.sqlite, "SELECT 1 FROM memory_jobs WHERE message_id = ?", (assistant["id"],))


def test_generation_disconnect_interrupts_a_stalled_next_event():
    released = threading.Event()
    closed = threading.Event()

    class FakeRequest:
        calls = 0

        async def is_disconnected(self):
            self.calls += 1
            return self.calls > 1

    class StalledEvents:
        def __next__(self):
            released.wait(2)
            return "token", "late"

        def close(self):
            closed.set()

    class Cancellation:
        def is_cancelled(self):
            return released.is_set()

        def abort(self):
            released.set()

    async def collect():
        return [event async for event in routes._cancel_on_disconnect(FakeRequest(), StalledEvents(), Cancellation())]

    assert asyncio.run(asyncio.wait_for(collect(), timeout=1)) == []
    assert released.is_set()
    assert closed.wait(1)


def test_retrieval_uses_sqlite_fts_dense_and_rrf(monkeypatch, tmp_path):
    state = build_memory_state()
    file_path = tmp_path / "fixture.md"
    file_path.write_text("Cephalon retrieval fixture mentions sqlite lexical search.", encoding="utf-8")

    async def fake_embedding(_app_state, _text: str):
        return [0.0] * storage.active_embedding_metadata()["embedding_dim"]

    monkeypatch.setattr("cephalon_core.services.ingestion.get_document_embedding", fake_embedding)
    result = asyncio.run(process_single_file(state, str(file_path), RagSettings()))
    context, sources, meta = asyncio.run(
        retrieval.retrieve_context(state, "sqlite lexical search", [0.0] * storage.active_embedding_metadata()["embedding_dim"], RagSettings())
    )

    assert result["status"] == "ready"
    assert sources
    assert sources[0].chunk_id == f"{result['doc_id']}_0"
    assert sources[0].fusion_score is not None
    assert "sqlite_fts5" in meta["search_modes"][0]
    assert "Cephalon retrieval fixture" in context


def test_hydrate_sources_adds_structured_provenance():
    state = build_memory_state()
    doc_id = "11111111-1111-4111-8111-111111111111"
    storage.execute(
        state.sqlite,
        """
        INSERT INTO documents (id, path, display_name, content_hash, chunk_count, status, type)
        VALUES (?, ?, ?, ?, 1, 'ready', 'file')
        """,
        (doc_id, "paper.pdf", "Paper", "hash"),
    )
    storage.execute(
        state.sqlite,
        """
        INSERT INTO chunks (
            id, doc_id, chunk_index, text, block_type, section_heading,
            heading_path, page_number, page_end, block_index, bounding_box
        )
        VALUES (?, ?, 0, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            "chunk-1",
            doc_id,
            "Result table",
            "table",
            "Experimental Results",
            '["Results","Experimental Results"]',
            7,
            8,
            42,
            "[72.0,145.0,510.0,260.0]",
        ),
    )

    sources = retrieval.hydrate_sources(
        state,
        [{"id": "chunk-1", "doc_id": doc_id, "text": "Result table", "score": 0.9}],
    )

    assert sources[0].page_number == 7
    assert sources[0].page_end == 8
    assert sources[0].block_type == "table"
    assert sources[0].heading_path == ["Results", "Experimental Results"]
    assert sources[0].bounding_box == (72.0, 145.0, 510.0, 260.0)
    assert retrieval.source_location_label(sources[0]) == "page 7-8 | Experimental Results | table"


def test_reranker_uses_jina_candidate_indices(monkeypatch):
    state = SimpleNamespace(
        rerank_cache=OrderedDict(),
        reranker_runtime_status={},
    )
    calls = []

    def fake_listwise(_state, query, documents):
        calls.append((query, documents))
        return [
            {"index": 2, "score": 0.92},
            {"index": 0, "score": 0.61},
            {"index": 1, "score": 0.08},
        ]

    monkeypatch.setattr(reranker_runtime, "rerank", fake_listwise)
    results = [
        {"id": "a", "text": "first candidate", "score": 0.0},
        {"id": "b", "text": "second candidate", "score": 0.0},
        {"id": "c", "text": "third candidate", "score": 0.0},
    ]

    ranked = retrieval.rerank(state, "query", results)

    assert calls == [("query", ["first candidate", "second candidate", "third candidate"])]
    assert [item["id"] for item in ranked] == ["c", "a", "b"]
    assert ranked[0]["listwise_rank"] == 1
    assert ranked[0]["reranker_raw_score"] == 0.92


def test_unavailable_jina_is_explicitly_degraded_without_old_fallback(monkeypatch):
    state = SimpleNamespace(rerank_cache=OrderedDict(), reranker_runtime_status={})

    def unavailable(*_args, **_kwargs):
        raise RuntimeError("Jina worker unavailable")

    monkeypatch.setattr(reranker_runtime, "rerank", unavailable)
    rows = [{"id": "a", "text": "alpha", "score": 0.3}, {"id": "b", "text": "beta", "score": 0.2}]
    result = retrieval.rerank(state, "alpha", rows)

    assert len(result) == 2
    assert all(row["rerank_score"] is None and row["reranker_raw_score"] is None for row in result)
    assert state.reranker_runtime_status == {"status": "error", "last_failure": "Jina worker unavailable"}


def test_retrieval_submits_every_fused_candidate_to_listwise_reranker(monkeypatch):
    state = build_memory_state()
    state.rerank_cache = OrderedDict()
    candidates = [
        {"id": f"chunk-{index}", "doc_id": f"doc-{index}", "text": f"candidate {index}", "score": 1.0 - index / 100, "dense_rank": 1 if index == 0 else None}
        for index in range(7)
    ]
    rerank_calls = []

    async def fake_search(_state, _prompt, _vector, _settings):
        return candidates, "dense", {"latency": {}}

    def fake_rerank(_state, _query, rows):
        rerank_calls.append([row["id"] for row in rows])
        return [
            {**row, "reranker_raw_score": 1.0 - rank / 10, "rerank_score": 1.0 - rank / 10, "final_score": 1.0 - rank / 10, "listwise_rank": rank + 1}
            for rank, row in enumerate(rows)
        ]

    monkeypatch.setattr(retrieval, "plan_subqueries", lambda _prompt: [{"id": "q0", "text": "question"}])
    monkeypatch.setattr(retrieval, "_search_once", fake_search)
    monkeypatch.setattr(retrieval, "rerank", fake_rerank)
    monkeypatch.setattr(retrieval.metrics, "append_retrieval_event", lambda *_args, **_kwargs: None)

    _context, _sources, meta = asyncio.run(retrieval.retrieve_context(state, "question", [0.0] * storage.active_embedding_metadata()["embedding_dim"], RagSettings(top_k=2, rerank_top_n=2)))

    assert rerank_calls == [[f"chunk-{index}" for index in range(7)]]
    assert len(meta["trace"]["reranked_candidates"]) == 7
    assert meta["trace"]["reranked_candidates"][0]["reranker_raw_score"] == 1.0
    assert meta["trace"]["reranked_candidates"][0]["listwise_rank"] == 1


def test_reranker_verification_checks_pinned_snapshot(tmp_path, monkeypatch):
    import json
    from cephalon_core.config import RERANKER_MODEL_ID, RERANKER_REPO, RERANKER_REVISION, RERANKER_PACKAGES, RERANKER_PRECISION
    model_dir = tmp_path / "reranker"
    model_dir.mkdir()
    state = build_memory_state()
    state.settings.reranker_model_dir = str(model_dir)
    names = [*reranker_runtime.RERANKER_FILE_SHA256, reranker_runtime.WORKER_FILE]
    for name in names:
        (model_dir / name).write_bytes(name.encode())
    files = {name: reranker_runtime._hash(model_dir / name) for name in names}
    monkeypatch.setattr(reranker_runtime, "RERANKER_FILE_SHA256", {name: files[name] for name in names[:-1]})
    monkeypatch.setattr(reranker_runtime, "RERANKER_WORKER_SHA256", files[names[-1]])
    manifest = {
        "model_id": RERANKER_MODEL_ID, "repo": RERANKER_REPO, "revision": RERANKER_REVISION,
        "precision": RERANKER_PRECISION, "packages": RERANKER_PACKAGES, "files": files,
    }
    manifest_path = model_dir / reranker_runtime.MANIFEST_FILE
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert reranker_runtime.verify_model(state)["verified"] is True
    assert reranker_runtime.verify_model(state)["selected_backend"] == "llama_cpp_vulkan"
    (model_dir / names[0]).write_bytes(b"tampered")
    invalid = reranker_runtime.verify_model(state)
    assert invalid["verified"] is False
    assert "checksum mismatch" in invalid["error"]
    # Rewriting the manifest cannot bless substituted weights.
    manifest["files"][names[0]] = reranker_runtime._hash(model_dir / names[0])
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    assert reranker_runtime.verify_model(state)["verified"] is False


def test_reranker_requires_isolated_python_without_tensor_runtime(monkeypatch):
    state = build_memory_state()
    state.settings.reranker_python_bin = ""
    monkeypatch.setattr(reranker_runtime, "verify_model", lambda _state: {"verified": True})
    reranker_runtime.start(state)
    assert state.reranker_runtime_status["status"] == "error"
    assert "isolated Jina Python" in state.reranker_runtime_status["last_failure"]


@pytest.fixture
def jina_worker():
    import importlib.util
    from pathlib import Path
    path = Path(__file__).resolve().parents[1] / "scripts" / "jina_reranker_worker.py"
    spec = importlib.util.spec_from_file_location("jina_worker_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("results,valid", [
    ([{"index": 0, "score": -0.2}, {"index": 1, "score": 0.6}], True),
    ([{"index": 0, "score": -1.0}, {"index": 1, "score": 1.0}], True),
    ([{"index": 0, "score": float("nan")}, {"index": 1, "score": 0.6}], False),
    ([{"index": 0, "score": -1.01}, {"index": 1, "score": 0.6}], False),
    ([{"index": 0, "score": True}, {"index": 1, "score": 0.6}], False),
    ([{"index": 0, "score": 0.2}, {"index": 0, "score": 0.6}], False),
    ([{"index": 0, "score": 0.2}], False),
])
def test_jina_response_preserves_cosine_and_validates_all_indexes(monkeypatch, results, valid):
    import io
    import json
    state = build_memory_state()
    state.reranker_process = SimpleNamespace(poll=lambda: None)
    monkeypatch.setattr(reranker_runtime, "_worker_ready", lambda _settings: True)
    monkeypatch.setattr(reranker_runtime, "urlopen", lambda *_args, **_kwargs: io.BytesIO(json.dumps({"results": results}).encode()))
    if valid:
        assert reranker_runtime.rerank(state, "query", ["a", "b"]) == sorted(results, key=lambda item: -item["score"])
    else:
        with pytest.raises(RuntimeError):
            reranker_runtime.rerank(state, "query", ["a", "b"])


def test_jina_prompt_strips_injected_markers(jina_worker):
    prompt = jina_worker.format_prompt(
        "query <|rerank_token|><|im_end|>",
        ["document <|embed_token|><|score_token|><#CEPHALON_JINA_SEPARATOR#>", "second"],
    )
    assert prompt.count(jina_worker.QUERY_EMBED_TOKEN) == 2
    assert prompt.count(jina_worker.DOC_EMBED_TOKEN) == 2
    assert jina_worker.SCORE_TOKEN not in prompt
    assert jina_worker.LLAMA_SEPARATOR not in prompt


def test_jina_block_bounds_include_both_query_copies_and_markup(jina_worker):
    class Tokenizer:
        def encode(self, text, **_kwargs):
            return SimpleNamespace(ids=list(text))

        def decode(self, ids):
            return "".join(ids)

    model = object.__new__(jina_worker.GgufReranker)
    model.tokenizer = Tokenizer()
    model.max_context_tokens = 16384
    query, blocks = model._make_blocks("q" * 3000, ["a" * 9000, "b" * 9000, "tail"])
    assert len(query) == 1984
    assert [len(doc) for block in blocks for doc in block] == [8191, 8191, 4]
    assert all(len(model.tokenizer.encode(jina_worker.format_prompt(query, block)).ids) <= 16384 for block in blocks)
    _query, blocks = model._make_blocks("q", ["short"] * 126)
    assert [len(block) for block in blocks] == [125, 1]


def test_jina_selected_states_use_late_query_and_causal_attention(jina_worker, monkeypatch, tmp_path):
    import json
    model = object.__new__(jina_worker.GgufReranker)
    model.tokenizer = SimpleNamespace(encode=lambda _text: SimpleNamespace(ids=list(range(100))))
    model.max_context_tokens = 16384
    model.llama_embedding_bin = tmp_path / "llama-embedding"
    model.model_path = tmp_path / "model.gguf"
    model.device, model.gpu_layers = "Vulkan0", 99
    model.projector = lambda values: values[:, :512]
    compact = np.zeros((4, 1024), dtype=np.float32)
    compact[:, 0] = [-1, 1, -1, 1]  # early query, doc 0, doc 1, late query
    commands = []

    def helper(command, **_kwargs):
        commands.append(command)
        return SimpleNamespace(stdout=json.dumps({"data": [{"embedding": row.tolist()} for row in compact]}), stderr="")

    monkeypatch.setattr(jina_worker.subprocess, "run", helper)
    documents, query, weight = model._score_block("query", ["first", "second"])
    assert jina_worker._cosine_scores(documents, query).tolist() == [1.0, -1.0]
    assert weight == 1.0
    command = commands[0]
    assert command[command.index("--attention") + 1] == "causal"
    assert command[command.index("--output-token-ids") + 1] == "151670,151671"
    assert command[command.index("--pooling") + 1] == "none"
    assert command[command.index("--embd-normalize") + 1] == "-1"
    compact[1, 0] = float("nan")
    with pytest.raises(RuntimeError, match="non-finite"):
        model._score_block("query", ["first", "second"])


def test_jina_projector_loads_bf16_without_torch(jina_worker, tmp_path):
    import json
    import struct
    first = np.zeros((512, 1024), dtype=np.float32)
    second = np.zeros((512, 512), dtype=np.float32)
    first[0, 0], second[0, 0] = 1.5, 2.0
    payload, header = b"", {}
    for name, matrix in [("projector.0.weight", first), ("projector.2.weight", second)]:
        data = (matrix.view(np.uint32) >> 16).astype("<u2").tobytes()
        header[name] = {"dtype": "BF16", "shape": list(matrix.shape), "data_offsets": [len(payload), len(payload) + len(data)]}
        payload += data
    encoded = json.dumps(header).encode()
    path = tmp_path / "projector.safetensors"
    path.write_bytes(struct.pack("<Q", len(encoded)) + encoded + payload)
    projector = jina_worker.load_projector(path)
    values = np.zeros((2, 1024), dtype=np.float32)
    values[:, 0] = [2.0, -2.0]
    assert projector(values)[:, 0].tolist() == [6.0, 0.0]


def test_jina_block_query_fusion_preserves_global_indexes(jina_worker, monkeypatch):
    model = object.__new__(jina_worker.GgufReranker)
    monkeypatch.setattr(model, "_make_blocks", lambda *_args: ("query", [["a", "b"], ["c"]]))
    outputs = iter([
        (np.array([[1.0, 0.0], [0.0, 1.0]]), np.array([1.0, 0.0]), 0.75),
        (np.array([[1.0, 1.0]]), np.array([0.0, 1.0]), 0.25),
    ])
    monkeypatch.setattr(model, "_score_block", lambda *_args: next(outputs))
    results = model.rerank("query", ["a", "b", "c"])
    assert [item["index"] for item in results] == [0, 2, 1]
    expected = np.array([0.75, 1.0 / np.sqrt(2), 0.25]) / np.sqrt(0.75 ** 2 + 0.25 ** 2)
    assert np.allclose([item["score"] for item in results], expected)


def test_jina_runtime_rejects_wrong_pr_head_and_changed_binary(monkeypatch, tmp_path):
    import json
    executable = tmp_path / "llama-embedding.exe"
    executable.write_bytes(b"helper")
    manifest_path = tmp_path / "cephalon-runtime-manifest.json"
    manifest = {"source_commit": "old-head", "backend": "vulkan", "files": {executable.name: reranker_runtime._hash(executable)}}
    manifest_path.write_text(json.dumps(manifest))
    settings = SimpleNamespace(reranker_llama_embedding_bin=str(executable))
    with pytest.raises(RuntimeError, match="PR #26286"):
        reranker_runtime.verify_runtime(settings)
    manifest["source_commit"] = reranker_runtime.RERANKER_LLAMA_CPP_REVISION
    manifest_path.write_text(json.dumps(manifest))
    monkeypatch.setattr(reranker_runtime, "_runtime_version", lambda *_args: "version (build 1, commit 491219a)")
    reranker_runtime.verify_runtime(settings)
    executable.write_bytes(b"changed")
    with pytest.raises(RuntimeError, match="checksum"):
        reranker_runtime.verify_runtime(settings)


def test_jina_cache_distinguishes_query_case_and_block_context():
    rows = [{"id": "a", "text": "candidate"}]
    settings = Settings()
    original = retrieval._rerank_cache_key("Query", rows, settings)
    assert retrieval._rerank_cache_key("query", rows, settings) != original
    settings.reranker_max_context_tokens = 16384
    assert retrieval._rerank_cache_key("Query", rows, settings) != original


def test_pdf_asset_transaction_deduplicates_repeated_content_addressed_asset(tmp_path):
    asset = pdf_parser.PdfAsset(
        asset_id="p1-img-0123456789abcdef0123",
        page_number=1,
        bounding_box=None,
        data=b"same image bytes",
        extension=".png",
        mime_type="image/png",
        sha256="0123456789abcdef",
    )

    transaction = document_assets.AssetTransaction.prepare(str(tmp_path), "doc-1", [asset, asset])

    assert len(transaction.rows) == 1
    assert len(list(__import__("pathlib").Path(transaction.staging).iterdir())) == 1
    transaction.rollback()


def test_boundaryless_text_is_split_before_embedding():
    text = " ".join(f"token{index}" for index in range(401))
    settings = RagSettings(parent_target_tokens=100, parent_max_tokens=100, child_target_tokens=32, child_max_tokens=32)

    parents = ingestion.build_parent_chunks(text, settings)
    children = asyncio.run(ingestion.build_semantic_child_chunks(build_memory_state(), text, settings))

    assert max(ingestion.estimate_tokens(chunk) for chunk in parents) <= 100
    assert max(ingestion.estimate_tokens(chunk) for chunk in children) <= 32
    assert " ".join(parents) == text


def test_completed_reindex_progress_is_persisted():
    state = build_memory_state()
    run = storage.create_reindex_run(state.sqlite, "full", 2)
    now = 1778755000
    for job_id, status in (("complete", "succeeded"), ("failure", "failed")):
        storage.execute(
            state.sqlite,
            "INSERT INTO jobs (id, kind, path, status, created_at, updated_at, reindex_run_id) VALUES (?, 'reindex', ?, ?, ?, ?, ?)",
            (job_id, f"{job_id}.md", status, now, now, run["id"]),
        )

    completed = storage.refresh_reindex_run(state.sqlite, run["id"])

    assert completed["status"] == "completed_with_errors"
    assert completed["processed_documents"] == 2
    assert completed["succeeded_documents"] == 1
    assert completed["failed_documents"] == 1
    assert storage.latest_reindex_run(state.sqlite)["id"] == run["id"]


def test_embedder_calls_dedicated_server_in_one_batch(monkeypatch):
    state = build_memory_state()
    state.embedding_dim = storage.active_embedding_metadata()["embedding_dim"]
    calls = []
    state.retrieval_error = None

    def fake_embed(_state, role, texts):
        assert role == "document"
        calls.append(list(texts))
        return [[float(index + 1)] * state.embedding_dim for index, _ in enumerate(texts)]

    monkeypatch.setattr(embedding_runtime, "embed", fake_embed)

    vectors = asyncio.run(retrieval.get_document_embeddings(state, ["first", "second", "third"]))

    assert calls == [["first", "second", "third"]]
    assert len(vectors) == 3
    assert vectors[1][0] == pytest.approx(1 / np.sqrt(state.embedding_dim))


def test_query_and_passage_embeddings_have_separate_cache_entries(monkeypatch):
    state = build_memory_state()
    state.embedding_dim = storage.active_embedding_metadata()["embedding_dim"]
    state.retrieval_error = None
    calls = []

    def fake_embed(_state, role, texts):
        calls.append((role, list(texts)))
        return [[1.0 if role == "query" else -1.0] * state.embedding_dim for _ in texts]

    monkeypatch.setattr(embedding_runtime, "embed", fake_embed)
    query = asyncio.run(retrieval.get_query_embedding(state, "relativity"))
    passage = asyncio.run(retrieval.get_document_embedding(state, "relativity"))
    cached = asyncio.run(retrieval.get_query_embedding(state, "relativity"))

    assert calls == [("query", ["relativity"]), ("document", ["relativity"])]
    assert len(query) == len(passage) == state.embedding_dim
    assert query[0] > 0 and passage[0] < 0
    assert cached == query
    assert np.linalg.norm(query) == pytest.approx(1.0, abs=1e-5)


def test_embeddinggemma_prompts_are_role_specific():
    assert embedding_runtime.PROMPTS == {"query": "task: search result | query: ", "document": "title: none | text: "}


def _embeddinggemma_snapshot(tmp_path, monkeypatch):
    import hashlib
    import json
    from cephalon_core.config import embedding_vector_space
    files = {}
    for name in (embedding_runtime.EMBEDDER_MODEL_FILE,):
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(name.encode())
        files[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    monkeypatch.setattr(embedding_runtime, "EMBEDDING_GGUF_SHA256", files[embedding_runtime.EMBEDDER_MODEL_FILE])
    manifest = {"vector_space": embedding_vector_space(), "gguf_repo": embedding_runtime.EMBEDDING_GGUF_REPO, "gguf_revision": embedding_runtime.EMBEDDING_GGUF_REVISION, "files": files}
    (tmp_path / embedding_runtime.MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    state = build_memory_state()
    state.settings.embedder_model_dir = str(tmp_path)
    return state, manifest


def test_embeddinggemma_manifest_verifies_pinned_gguf_bytes(tmp_path, monkeypatch):
    state, _manifest = _embeddinggemma_snapshot(tmp_path, monkeypatch)
    assert embedding_runtime.verify_model(state)["verified"]
    (tmp_path / embedding_runtime.EMBEDDER_MODEL_FILE).write_text("changed", encoding="utf-8")
    result = embedding_runtime.verify_model(state)
    assert not result["verified"]
    assert embedding_runtime.EMBEDDER_MODEL_FILE in result["error"]


def test_embeddinggemma_rejects_same_dimension_legacy_vector_space(tmp_path, monkeypatch):
    import json
    state, manifest = _embeddinggemma_snapshot(tmp_path, monkeypatch)
    manifest["vector_space"]["model_id"] = "legacy-768d-model"
    (tmp_path / embedding_runtime.MANIFEST_FILE).write_text(json.dumps(manifest), encoding="utf-8")
    assert not embedding_runtime.verify_model(state)["verified"]


def test_embeddinggemma_ready_checks_gguf_identity_and_context(monkeypatch, tmp_path):
    state, _manifest = _embeddinggemma_snapshot(tmp_path, monkeypatch)
    identity = {
        "model_alias": embedding_runtime.MODEL_ALIAS, "model_ftype": "Q8_0",
        "modalities": {"vision": False, "video": False, "audio": False},
        "default_generation_settings": {"n_ctx": 8192}, "total_slots": 1,
        "model_path": str(tmp_path / embedding_runtime.EMBEDDER_MODEL_FILE),
    }
    monkeypatch.setattr(embedding_runtime, "_get", lambda _url, endpoint: {"status": "ok"} if endpoint == "/health" else identity)
    assert embedding_runtime._server_ready(state.settings)
    for field, wrong in [("model_alias", "other"), ("model_ftype", "BF16"), ("model_path", "other.gguf"), ("total_slots", 4), ("default_generation_settings", {"n_ctx": 2048}), ("modalities", {"vision": True})]:
        original = identity[field]
        identity[field] = wrong
        assert not embedding_runtime._server_ready(state.settings)
        identity[field] = original


@pytest.mark.parametrize("bad_data", [
    [{"index": 0, "embedding": [1.0] * 2048}],
    [{"index": 1, "embedding": [1.0] * 768}],
    [{"index": 0, "embedding": [float("nan")] * 768}],
    [{"index": 0, "embedding": [True] * 768}],
    [{"index": 0, "embedding": [0.0] * 768}],
])
def test_embeddinggemma_rejects_invalid_worker_vectors(monkeypatch, bad_data):
    state = build_memory_state()
    monkeypatch.setattr(embedding_runtime, "_server_ready", lambda _settings: True)
    monkeypatch.setattr(embedding_runtime, "_post", lambda _settings, endpoint, _payload: {"tokens": [1, 2]} if endpoint == "/tokenize" else {"data": bad_data})
    with pytest.raises(RuntimeError):
        embedding_runtime.embed(state, "query", ["test"])


def test_embeddinggemma_gguf_uses_role_prompts_and_preserves_batch_order(monkeypatch):
    state = build_memory_state()
    calls = []
    monkeypatch.setattr(embedding_runtime, "_server_ready", lambda _settings: True)
    def respond(_settings, endpoint, payload):
        calls.append((endpoint, payload))
        if endpoint == "/tokenize":
            return {"tokens": [1, 2, 3]}
        return {"data": [{"index": 1, "embedding": [-1.0] * 768}, {"index": 0, "embedding": [1.0] * 768}]}
    monkeypatch.setattr(embedding_runtime, "_post", respond)
    for role in ("query", "document"):
        calls.clear()
        vectors = embedding_runtime.embed(state, role, ["one", "two"])
        assert [vector[0] for vector in vectors] == [1.0, -1.0]
        assert [payload["content"] for endpoint, payload in calls if endpoint == "/tokenize"] == [embedding_runtime.PROMPTS[role] + "one", embedding_runtime.PROMPTS[role] + "two"]
        assert calls[-1][1]["input"] == [[1, 2, 3], [1, 2, 3]]


def test_embeddinggemma_truncates_token_arrays_before_inference(monkeypatch):
    state = build_memory_state()
    monkeypatch.setattr(embedding_runtime, "_server_ready", lambda _settings: True)
    def respond(_settings, endpoint, payload):
        if endpoint == "/tokenize":
            assert payload["add_special"] is True and payload["parse_special"] is False
            return {"tokens": list(range(9000))}
        assert payload["input"] == [list(range(8192))]
        return {"data": [{"index": 0, "embedding": [1.0] * 768}]}
    monkeypatch.setattr(embedding_runtime, "_post", respond)
    assert len(embedding_runtime.embed(state, "query", ["long"])[0]) == 768


def test_embeddinggemma_q8_command_retains_full_bidirectional_sequence():
    from cephalon_core.config import ACTIVE_VECTOR_TABLE, embedding_vector_space
    command = embedding_runtime.server_command(Settings())
    for option in ("--ctx-size", "--batch-size", "--ubatch-size"):
        assert command[command.index(option) + 1] == "8192"
    assert command[command.index("--parallel") + 1] == "1"
    assert command[command.index("--flash-attn") + 1] == "on"
    assert command[command.index("--pooling") + 1] == "mean"
    assert "--offline" in command and "--no-mmproj" in command
    assert "q8_0" in ACTIVE_VECTOR_TABLE
    assert embedding_vector_space()["quantization"] == "Q8_0"


def test_embedder_rejects_unexpected_service_on_its_port(monkeypatch):
    state = build_memory_state()
    monkeypatch.setattr(embedding_runtime, "verify_model", lambda _state: {"verified": True})
    monkeypatch.setattr(embedding_runtime, "_server_ready", lambda _settings: False)
    monkeypatch.setattr(embedding_runtime, "_port_occupied", lambda _port: True)

    embedding_runtime.start(state)

    assert "Another service occupies" in state.retrieval_error
    assert state.embedder_process is None


def test_embeddinggemma_start_isolates_chat_environment_and_stops_owned_process(monkeypatch, tmp_path):
    state = build_memory_state()
    binary = tmp_path / "llama-server.exe"
    binary.write_bytes(b"test binary")
    state.settings.embedder_llama_server_bin = str(binary)
    state.settings.data_dir = str(tmp_path)
    monkeypatch.setattr(embedding_runtime, "verify_model", lambda _state: {"verified": True})
    monkeypatch.setattr(embedding_runtime, "_port_occupied", lambda _port: False)
    monkeypatch.setattr(embedding_runtime, "_runtime_version", lambda *_args: "build 11456")
    monkeypatch.setenv("LLAMA_ARG_MODEL", "chat-model.gguf")
    monkeypatch.setenv("LLAMA_ARG_PORT", "8080")
    monkeypatch.setenv("LLAMA_API_KEY", "chat-key")
    process = SimpleNamespace(pid=123, terminated=False)
    process.poll = lambda: 0 if process.terminated else None
    process.terminate = lambda: setattr(process, "terminated", True)
    process.wait = lambda **_kwargs: 0
    def spawn(command, **kwargs):
        assert not any(name.startswith("LLAMA_ARG_") or name == "LLAMA_API_KEY" for name in kwargs["env"])
        assert command == embedding_runtime.server_command(state.settings)
        return process
    monkeypatch.setattr(embedding_runtime.subprocess, "Popen", spawn)
    embedding_runtime.start(state)
    assert state.retrieval_error is None and state.embedder_process_owned
    embedding_runtime.stop(state)
    assert process.terminated and not state.embedder_process_owned


def test_embeddinggemma_rejects_an_older_llama_build(monkeypatch, tmp_path):
    binary = str(tmp_path / "old-server.exe")
    monkeypatch.setattr(embedding_runtime.subprocess, "run", lambda *_args, **_kwargs: SimpleNamespace(returncode=0, stdout="version: build 10975", stderr=""))
    with pytest.raises(ValueError, match="b11456"):
        embedding_runtime._runtime_version(binary, 1, 1)


def test_upgrade_discards_legacy_memory_but_preserves_conversation():
    state = build_memory_state()
    storage.execute(state.sqlite, "INSERT INTO conversations (id, title, created_at, updated_at) VALUES ('c1', 'Test', 1, 2)")
    storage.execute(state.sqlite, "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES ('u1', 'c1', 'user', 'What is Aurora?', 1)")
    storage.execute(state.sqlite, "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES ('a1', 'c1', 'assistant', 'Aurora is a launch.', 2)")
    storage.execute(state.sqlite, "INSERT INTO conversation_memory (id, conversation_id, message_id, created_at) VALUES ('mem_a1', 'c1', 'a1', 2)")
    storage.execute(state.sqlite, "INSERT INTO memory_jobs (message_id, conversation_id, prompt) VALUES ('a1', 'c1', 'What is Aurora?')")
    storage.execute(state.sqlite, "DELETE FROM schema_migrations WHERE version = '022_conversation_memory_vector_space'")

    storage.run_migrations(state.sqlite, state.settings)

    assert storage.fetchone(state.sqlite, "SELECT COUNT(*) AS count FROM conversation_memory")["count"] == 0
    assert storage.fetchone(state.sqlite, "SELECT COUNT(*) AS count FROM memory_jobs")["count"] == 0
    assert storage.fetchone(state.sqlite, "SELECT content FROM messages WHERE id = 'a1'")["content"] == "Aurora is a launch."
    assert storage.fetchone(state.sqlite, "SELECT title FROM conversations WHERE id = 'c1'")["title"] == "Test"


def test_embeddinggemma_q8_upgrade_marks_bf16_documents_and_preserves_chat():
    import hashlib
    import json
    from cephalon_core.config import EMBEDDING_MODEL_ID, embedding_config_hash, embedding_vector_space
    state = build_memory_state()
    storage.execute(state.sqlite, "INSERT INTO conversations (id, title, created_at, updated_at) VALUES ('c1', 'Saved', 1, 2)")
    storage.execute(state.sqlite, "INSERT INTO messages (id, conversation_id, role, content, created_at) VALUES ('a1', 'c1', 'assistant', 'Saved answer', 2)")
    storage.execute(state.sqlite, "INSERT INTO conversation_memory (id, conversation_id, message_id, text, embedding_config_hash, created_at) VALUES ('old-memory', 'c1', 'a1', 'Old memory', 'old-space', 2)")
    storage.execute(state.sqlite, "INSERT INTO memory_jobs (message_id, conversation_id, prompt) VALUES ('a1', 'c1', 'Old query')")
    bf16_space = embedding_vector_space()
    bf16_space["quantization"] = "BF16"
    bf16_hash = hashlib.sha256(json.dumps(bf16_space, sort_keys=True).encode()).hexdigest()
    for doc_id, model, config_hash in [("old-768", EMBEDDING_MODEL_ID, bf16_hash), ("new-768", EMBEDDING_MODEL_ID, embedding_config_hash())]:
        storage.execute(state.sqlite, "INSERT INTO documents (id, path, content_hash, type, status, embedding_model_id, embedding_dim, embedding_config_hash, stale_embedding) VALUES (?, ?, 'content', 'file', 'ready', ?, 768, ?, 0)", (doc_id, doc_id + '.txt', model, config_hash))
    storage.execute(state.sqlite, "DELETE FROM schema_migrations WHERE version = '025_embeddinggemma2_q8_vector_space'")
    storage.run_migrations(state.sqlite, state.settings)
    assert storage.fetchone(state.sqlite, "SELECT stale_embedding FROM documents WHERE id = 'old-768'")["stale_embedding"] == 1
    assert storage.fetchone(state.sqlite, "SELECT stale_embedding FROM documents WHERE id = 'new-768'")["stale_embedding"] == 0
    assert storage.fetchone(state.sqlite, "SELECT COUNT(*) AS n FROM conversation_memory")["n"] == 0
    assert storage.fetchone(state.sqlite, "SELECT COUNT(*) AS n FROM memory_jobs")["n"] == 0
    assert storage.fetchone(state.sqlite, "SELECT content FROM messages WHERE id = 'a1'")["content"] == "Saved answer"
    assert storage.fetchone(state.sqlite, "SELECT 1 FROM schema_migrations WHERE version = '022_conversation_memory_vector_space'") is not None


def test_partial_reindex_queries_only_current_documents():
    state = build_memory_state()
    state.reindex_required = True
    for doc_id, stale in (("current-doc", 0), ("old-doc", 1)):
        storage.execute(
            state.sqlite,
            "INSERT INTO documents (id, path, display_name, content_hash, chunk_count, status, type, stale_embedding) VALUES (?, ?, ?, 'hash', 1, 'ready', 'file', ?)",
            (doc_id, f"{doc_id}.txt", doc_id, stale),
        )
        storage.execute(
            state.sqlite,
            "INSERT INTO chunks (id, doc_id, chunk_index, text) VALUES (?, ?, 0, 'aurora launch')",
            (f"{doc_id}-chunk", doc_id),
        )
        storage.upsert_chunk_fts(state.sqlite, f"{doc_id}-chunk", doc_id, "aurora launch")

    assert routes._retrieval_index_unavailable(state) is False
    assert [row["doc_id"] for row in retrieval._lexical_search(state, "aurora launch", 10)] == ["current-doc"]


def test_context_compressor_keeps_relevant_cited_sentences():
    sources = [
        retrieval.CompressionSource(
            source_id="S1",
            text="Ashwagandha lowers stress. Distributed systems use quorum writes. Rhodiola can reduce fatigue.",
            rank=1,
            score=0.9,
        )
    ]

    compressed, stats, evidence_by_source = retrieval.compress_context(
        "stress supplements",
        sources,
        max_sentences=2,
    )

    assert "[[src:S1]]" in compressed
    assert "Ashwagandha" in compressed
    assert evidence_by_source["S1"].startswith("Ashwagandha")
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


def test_relevant_selection_keeps_same_document_and_drops_weak_dense_strays():
    ranked = [
        {"id": "domain_0", "doc_id": "domain", "score": 3.7, "lexical_rank": 1},
        {"id": "domain_1", "doc_id": "domain", "score": 0.84, "dense_rank": 2},
        {"id": "unrelated_0", "doc_id": "unrelated", "score": 0.64, "dense_rank": 3},
    ]

    selected = retrieval._select_relevant_results(ranked, 3)

    assert [result["id"] for result in selected] == ["domain_0", "domain_1"]


def test_memory_request_mode_reserves_or_isolates_conversation_memory():
    assert retrieval._memory_request_mode("From our past conversation only, what did we decide?") == (True, True)
    assert retrieval._memory_request_mode("What did we just discuss about the rollback?") == (False, True)
    assert retrieval._memory_request_mode("What does the local document say about rollback?") == (False, False)


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

    monkeypatch.setattr("cephalon_core.services.ingestion.get_document_embedding", fake_embedding)
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

    monkeypatch.setattr("cephalon_core.services.ingestion.get_document_embedding", fake_embedding)

    async def run_job():
        job = await manager.enqueue_ingest(str(file_path))
        await manager._run_job(job["id"])
        return manager.get_job(job["id"])

    finished = asyncio.run(run_job())
    events = storage.fetchall(state.sqlite, "SELECT event_type FROM job_events WHERE job_id = ?", (finished["id"],))

    assert finished["status"] == "succeeded"
    assert finished["processed_files"] == 1
    assert any(row["event_type"] == "job" for row in events)


def test_job_manager_recovers_queued_and_marks_interrupted_running_jobs(tmp_path):
    state = build_memory_state()
    event_bus = EventBus(state.sqlite)
    manager = JobManager(state, event_bus)
    now = 1778755000
    for job_id, status in (("queued-job", "queued"), ("running-job", "running")):
        storage.execute(
            state.sqlite,
            """
            INSERT INTO jobs (id, kind, path, status, created_at, updated_at)
            VALUES (?, 'ingest', ?, ?, ?, ?)
            """,
            (job_id, str(tmp_path / f"{job_id}.md"), status, now, now),
        )

    queued_ids = manager.recover_interrupted_jobs()

    assert queued_ids == ["queued-job"]
    running = manager.get_job("running-job")
    assert running["status"] == "failed"
    assert "interrupted" in running["error"].lower()


def test_event_bus_drops_old_refresh_event_for_slow_subscriber():
    async def exercise():
        bus = EventBus()
        queue = asyncio.Queue(maxsize=1)
        await queue.put({"type": "document", "payload": {"old": True}})
        bus._subscribers.add(queue)
        await asyncio.wait_for(bus.publish("document", {"new": True}), timeout=0.1)
        return await queue.get()

    event = asyncio.run(exercise())
    assert event["payload"] == {"new": True}


def test_reindex_preserves_display_name_and_tags(monkeypatch, tmp_path):
    state = build_memory_state()
    event_bus = EventBus(state.sqlite)
    manager = JobManager(state, event_bus)
    file_path = tmp_path / "fixture.md"
    file_path.write_text("Cephalon reindex fixture.", encoding="utf-8")

    async def fake_embedding(_app_state, _text: str):
        return [0.0] * storage.active_embedding_metadata()["embedding_dim"]

    monkeypatch.setattr("cephalon_core.services.ingestion.get_document_embedding", fake_embedding)
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


def test_failed_reindex_keeps_previous_searchable_chunks(monkeypatch, tmp_path):
    state = build_memory_state()
    event_bus = EventBus(state.sqlite)
    manager = JobManager(state, event_bus)
    file_path = tmp_path / "fixture.md"
    file_path.write_text("Original searchable content.", encoding="utf-8")

    async def good_embedding(_app_state, _text: str):
        return [0.0] * storage.active_embedding_metadata()["embedding_dim"]

    monkeypatch.setattr("cephalon_core.services.ingestion.get_document_embedding", good_embedding)
    first = asyncio.run(process_single_file(state, str(file_path), RagSettings()))
    original_chunks = [row["text"] for row in storage.fetchall(state.sqlite, "SELECT text FROM chunks WHERE doc_id = ?", (first["doc_id"],))]
    file_path.write_text("Replacement content that fails.", encoding="utf-8")

    async def failed_embedding(_app_state, _text: str):
        raise RuntimeError("embedding failed")

    monkeypatch.setattr("cephalon_core.services.ingestion.get_document_embedding", failed_embedding)

    async def run_job():
        job = await manager.enqueue_ingest(str(file_path), kind="reindex", target_doc_id=first["doc_id"])
        await manager._run_job(job["id"])
        return manager.get_job(job["id"])

    finished = asyncio.run(run_job())
    remaining_chunks = [row["text"] for row in storage.fetchall(state.sqlite, "SELECT text FROM chunks WHERE doc_id = ?", (first["doc_id"],))]
    document = storage.fetchone(state.sqlite, "SELECT status FROM documents WHERE id = ?", (first["doc_id"],))

    assert finished["status"] == "failed"
    assert remaining_chunks == original_chunks
    assert document["status"] == "ready"


def test_failed_vector_replacement_rolls_back_reindex_rows(monkeypatch, tmp_path):
    state = build_memory_state()
    file_path = tmp_path / "fixture.md"
    file_path.write_text("Original searchable content.", encoding="utf-8")

    async def fake_embedding(_app_state, _text: str):
        return [0.0] * storage.active_embedding_metadata()["embedding_dim"]

    monkeypatch.setattr("cephalon_core.services.ingestion.get_document_embedding", fake_embedding)
    first = asyncio.run(process_single_file(state, str(file_path), RagSettings()))
    original_chunks = [
        row["text"]
        for row in storage.fetchall(state.sqlite, "SELECT text FROM chunks WHERE doc_id = ?", (first["doc_id"],))
    ]
    file_path.write_text("Replacement content reaches vector persistence.", encoding="utf-8")

    original_add = state.lance.table.add

    def fail_vector_write(rows):
        if any("Replacement content" in row["text"] for row in rows):
            raise RuntimeError("vector write failed")
        original_add(rows)

    monkeypatch.setattr(state.lance.table, "add", fail_vector_write)
    result = asyncio.run(
        process_single_file(
            state,
            str(file_path),
            RagSettings(),
            existing_doc_id=first["doc_id"],
        )
    )
    remaining_chunks = [
        row["text"]
        for row in storage.fetchall(state.sqlite, "SELECT text FROM chunks WHERE doc_id = ?", (first["doc_id"],))
    ]
    remaining_vectors = [
        row["text"]
        for row in state.lance.table.rows
        if row["doc_id"] == first["doc_id"]
    ]

    assert result["status"] == "failed"
    assert remaining_chunks == original_chunks
    assert set(remaining_vectors) == {"Original searchable content.", "Original searchable content."}


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
