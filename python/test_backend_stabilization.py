import asyncio
import os
import sqlite3
from collections import OrderedDict
from types import SimpleNamespace

import pytest
import numpy as np
from fastapi import HTTPException

from cephalon_core.config import Settings
from cephalon_core.events import EventBus
from cephalon_core.schemas import Message, QueryRequest, RagSettings
from cephalon_core.routes import _settings_for_retrieval_scope
from cephalon_core.app_factory import _validate_embedder_meta, _validate_reranker_meta
from cephalon_core.app_factory import _read_model_meta
from cephalon_core import routes
from cephalon_core.services import generation, ingestion, metrics, pdf_parser, retrieval
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


class FakeTokenizer:
    def __call__(self, pairs, **_kwargs):
        import numpy as np

        return {
            "input_ids": np.ones((len(pairs), 2), dtype="int64"),
            "attention_mask": np.ones((len(pairs), 2), dtype="int64"),
        }


class FakeReranker:
    def __init__(self):
        self.batch_sizes = []

    def get_inputs(self):
        return []

    def run(self, _outputs, inputs):
        batch_size = len(inputs["input_ids"])
        self.batch_sizes.append(batch_size)
        return [np.arange(batch_size, dtype=float).reshape(batch_size, 1)]


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


def test_query_request_defaults_to_auto_retrieval():
    assert QueryRequest(prompt="Hello").retrieval_scope == "auto"


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


def test_reranker_uses_safe_single_pair_batches_by_default():
    reranker = FakeReranker()
    state = SimpleNamespace(tokenizer=FakeTokenizer(), reranker=reranker)

    scores = retrieval._score_rerank_pairs(state, [["query", "one"], ["query", "two"], ["query", "three"]])

    assert reranker.batch_sizes == [1, 1, 1]
    assert scores.tolist() == [0.0, 0.0, 0.0]


def test_reranker_uses_declared_validated_batch_size():
    reranker = FakeReranker()
    state = SimpleNamespace(tokenizer=FakeTokenizer(), reranker=reranker, reranker_batch_size=2)

    scores = retrieval._score_rerank_pairs(state, [["query", "one"], ["query", "two"], ["query", "three"]])

    assert reranker.batch_sizes == [2, 1]
    assert scores.tolist() == [0.0, 1.0, 0.0]


def test_embedding_runtime_uses_bounded_batches(monkeypatch):
    state = SimpleNamespace(embedder=object(), embedding_batch_size=2)
    batch_sizes = []

    def fake_run(_app_state, texts: list[str]):
        batch_sizes.append(len(texts))
        return [[float(index)] * 3 for index, _text in enumerate(texts)]

    monkeypatch.setattr(retrieval, "_run_embedding_batch", fake_run)

    vectors = retrieval._get_embeddings_sync(state, ["one", "two", "three", "four", "five"])

    assert batch_sizes == [2, 2, 1]
    assert len(vectors) == 5


def test_query_request_separates_retrieval_scope_and_response_effort():
    request = QueryRequest(
        prompt="Compare the documents carefully.",
        retrieval_scope="high",
        response_effort="thorough",
    )

    assert request.retrieval_scope == "high"
    assert request.response_effort == "thorough"


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

    def fake_completion(_app_state, messages, settings, *, stream):
        calls.append({"messages": messages, "settings": settings, "stream": stream})
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
    assert calls[1]["settings"].max_tokens == 2048
    assert calls[2]["settings"].max_tokens == 6144
    assert "Draft answer with one gap." in calls[2]["messages"][0]["content"]
    assert "--- CLAIM AUDIT ---" in calls[2]["messages"][0]["content"]


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

    class FakePage:
        def extract_text(self):
            calls.append("extract")
            return "page text"

    monkeypatch.setattr(documents, "PdfReader", lambda _path: SimpleNamespace(pages=[FakePage(), FakePage()]))
    text, _mode = documents.extract_text("fixture.pdf")
    assert text == "page text\npage text"
    assert len(calls) == 2

    workbook = SimpleNamespace(worksheets=[])
    load_workbook = lambda *_args, **kwargs: (calls.append(kwargs), workbook)[1]
    monkeypatch.setattr(documents.openpyxl, "load_workbook", load_workbook)
    documents.extract_text("fixture.xlsx")
    assert calls[-1]["read_only"] is True


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

        def extract(self):
            return [["Model", "Recall"], ["Baseline", "72.4"], ["RATE", "81.7"]]

    class FakePage:
        width = 600
        height = 800

        def __init__(self, page_number):
            self.page_number = page_number

        def extract_words(self, **_kwargs):
            heading = [
                word("3", 40, 50, 50, 65, 16, "Bold"),
                word("Results", 55, 50, 130, 65, 16, "Bold"),
            ] if self.page_number == 1 else []
            return heading + [
                word("Retrieval", 40, 120, 95, 132),
                word("improved", 100, 120, 155, 132),
                word("substantially.", 160, 120, 235, 132),
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

    monkeypatch.setattr(pdf_parser.pdfplumber, "open", lambda *_args, **_kwargs: FakePdf())

    parsed = pdf_parser.parse_pdf("fixture.pdf")

    assert parsed.page_count == 2
    assert "Proceedings" not in parsed.text
    assert any(block.block_type == "table" and "RATE | 81.7" in block.text for block in parsed.blocks)
    paragraph = next(block for block in parsed.blocks if "improved substantially" in block.text)
    assert paragraph.page_number == 1
    assert paragraph.heading_path == ["3 Results"]


def test_structured_pdf_ingestion_persists_source_provenance(monkeypatch, tmp_path):
    state = build_memory_state()
    file_path = tmp_path / "paper.pdf"
    file_path.write_bytes(b"%PDF fixture")
    extracted = documents.ExtractedDocument(
        text="4 Findings\n\nThe measured result was 81.7.",
        extraction_mode="native_structured",
        page_count=1,
        parser_version=pdf_parser.PARSER_VERSION,
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
            ),
        ],
    )
    monkeypatch.setattr(documents, "extract_document", lambda *_args, **_kwargs: extracted)

    async def fake_embedding(_app_state, _text):
        return [0.0] * storage.active_embedding_metadata()["embedding_dim"]

    monkeypatch.setattr("cephalon_core.services.ingestion.get_embedding", fake_embedding)

    result = asyncio.run(process_single_file(state, str(file_path), RagSettings()))
    chunk = storage.fetchone(
        state.sqlite,
        """
        SELECT block_type, section_heading, heading_path, page_number, page_end,
               block_index, bounding_box, parser_version
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

    monkeypatch.setattr("cephalon_core.services.ingestion.get_embedding", fake_embedding)
    monkeypatch.setattr("cephalon_core.services.ingestion.get_embeddings", fake_embeddings)

    result = asyncio.run(process_single_file(state, str(file_path), RagSettings()))

    assert result["status"] == "ready"
    assert batch_sizes == [2]


def test_semantic_child_chunking_does_not_embed_each_sentence(monkeypatch):
    state = SimpleNamespace(embedder=object())

    async def unexpected_embedding(*_args, **_kwargs):
        raise AssertionError("Chunk boundary selection should not invoke the embedder.")

    monkeypatch.setattr("cephalon_core.services.ingestion.get_embeddings", unexpected_embedding)
    monkeypatch.setattr("cephalon_core.services.ingestion.get_embedding", unexpected_embedding)
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

    monkeypatch.setattr("cephalon_core.services.ingestion.get_embedding", fake_embedding)
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


def test_onnx_model_metadata_accepts_valid_non_jina_profiles():
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

    assert _validate_embedder_meta({"model_id": "other/embedder", "kind": "embedder", "dimension": 768, "validated": True}) is None
    assert _validate_reranker_meta({"model_id": "other/reranker", "kind": "reranker", "validated": True, "score_mode": "auto"}) is None
    assert "score_mode" in _validate_reranker_meta({"model_id": "other/reranker", "kind": "reranker", "validated": True})


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


def test_reranker_scores_candidates_in_one_onnx_batch():
    class BatchTokenizer:
        def __call__(self, pairs, **_kwargs):
            return {
                "input_ids": np.ones((len(pairs), 8), dtype=np.int64),
                "attention_mask": np.ones((len(pairs), 8), dtype=np.int64),
            }

    class BatchSession:
        def __init__(self):
            self.calls = 0

        def get_inputs(self):
            return [SimpleNamespace(name="input_ids"), SimpleNamespace(name="attention_mask")]

        def run(self, _output_names, ort_inputs):
            self.calls += 1
            batch_size = ort_inputs["input_ids"].shape[0]
            return [np.asarray([[2.0 + index, 0.5] for index in range(batch_size)], dtype=np.float32)]

    session = BatchSession()
    state = SimpleNamespace(
        tokenizer=BatchTokenizer(),
        reranker=session,
        reranker_score_mode="logit_margin_0_minus_1",
        reranker_batch_size=3,
        rerank_cache=OrderedDict(),
    )
    results = [
        {"id": "a", "text": "first candidate", "score": 0.0},
        {"id": "b", "text": "second candidate", "score": 0.0},
        {"id": "c", "text": "third candidate", "score": 0.0},
    ]

    ranked = retrieval.rerank(state, "query", results)

    assert session.calls == 1
    assert all("rerank_score" in item for item in ranked)
    assert ranked[0]["rerank_score"] > ranked[-1]["rerank_score"]


def test_embedder_scores_texts_in_one_onnx_batch():
    state = build_memory_state()
    state.embedding_dim = storage.active_embedding_metadata()["embedding_dim"]
    state.embedding_fixed_sequence_length = None
    state.embedding_pooling = "embedding"
    calls = []

    class BatchTokenizer:
        def __call__(self, texts, **_kwargs):
            count = len(texts)
            return {
                "input_ids": np.ones((count, 3), dtype=np.int64),
                "attention_mask": np.ones((count, 3), dtype=np.int64),
            }

    class BatchEmbedder:
        def run(self, _output_names, ort_inputs):
            batch_size = ort_inputs["input_ids"].shape[0]
            calls.append(batch_size)
            return [np.asarray([
                [float(index)] * state.embedding_dim
                for index in range(batch_size)
            ], dtype=np.float32)]

    state.embed_tokenizer = BatchTokenizer()
    state.embedder = BatchEmbedder()

    vectors = asyncio.run(retrieval.get_embeddings(state, ["first", "second", "third"]))

    assert calls == [3]
    assert len(vectors) == 3
    assert vectors[1][0] == 1.0


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


def test_failed_reindex_keeps_previous_searchable_chunks(monkeypatch, tmp_path):
    state = build_memory_state()
    event_bus = EventBus(state.sqlite)
    manager = JobManager(state, event_bus)
    file_path = tmp_path / "fixture.md"
    file_path.write_text("Original searchable content.", encoding="utf-8")

    async def good_embedding(_app_state, _text: str):
        return [0.0] * storage.active_embedding_metadata()["embedding_dim"]

    monkeypatch.setattr("cephalon_core.services.ingestion.get_embedding", good_embedding)
    first = asyncio.run(process_single_file(state, str(file_path), RagSettings()))
    original_chunks = [row["text"] for row in storage.fetchall(state.sqlite, "SELECT text FROM chunks WHERE doc_id = ?", (first["doc_id"],))]
    file_path.write_text("Replacement content that fails.", encoding="utf-8")

    async def failed_embedding(_app_state, _text: str):
        raise RuntimeError("embedding failed")

    monkeypatch.setattr("cephalon_core.services.ingestion.get_embedding", failed_embedding)

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

    monkeypatch.setattr("cephalon_core.services.ingestion.get_embedding", fake_embedding)
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
