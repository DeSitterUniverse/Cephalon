import json
import os
import time
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from . import storage
from .schemas import DocumentUpdateRequest, EvalRunRequest, IngestRequest, LoadModelRequest, QueryRequest, RagSettings, TagRequest
from .services import evaluation, generation, ingestion, metrics, models, observability, retrieval, support
from .validators import normalize_existing_path, validate_document_id, validate_tag


router = APIRouter()


def state(request: Request):
    return request.app.state


def _ensure_query_model_loaded(app_state, requested_model: str) -> None:
    if getattr(app_state, "active_model_name", None) != requested_model:
        raise HTTPException(status_code=409, detail="Load the selected GGUF model before querying.")


def _settings_for_reasoning_mode(settings: RagSettings, mode: str) -> RagSettings:
    clean = (mode or "balanced").lower()
    if clean == "fast":
        return settings.model_copy(update={
            "top_k": min(settings.top_k, 12),
            "rerank_top_n": min(settings.rerank_top_n, 3),
            "max_tokens": min(settings.max_tokens, 512),
            "temperature": min(settings.temperature, 0.35),
        })
    if clean == "deep":
        return settings.model_copy(update={
            "top_k": max(settings.top_k, 28),
            "rerank_top_n": max(settings.rerank_top_n, 6),
            "max_tokens": max(settings.max_tokens, 1024),
            "temperature": max(settings.temperature, 0.55),
        })
    return settings


@router.get("/health")
def health(request: Request):
    app_state = state(request)
    return {
        "status": "degraded" if app_state.startup_error else "ok",
        "startup_error": app_state.startup_error,
        "engines_ready": app_state.startup_error is None,
        "data_dir": app_state.settings.data_dir,
        "model_dir": app_state.settings.model_dir,
        "metrics_dir": app_state.settings.metrics_dir,
        "obsidian_vault_dir": app_state.settings.obsidian_vault_dir,
        "last_metrics_error": getattr(app_state, "last_metrics_error", None),
        "active_model": getattr(app_state, "active_model_name", None),
        "active_context_tokens": getattr(app_state, "active_context_tokens", None),
        "active_model_context_tokens": getattr(app_state, "active_model_context_tokens", None),
        "last_model_load_error": getattr(app_state, "last_model_load_error", None),
        "onnx_warmup": getattr(app_state, "onnx_warmup", None),
        "python_runtime": models.python_runtime_info(),
        "llama_backend": models.llama_backend_info(),
        "retrieval_index": getattr(app_state, "retrieval_index", None),
        "generated_index_backup": getattr(app_state, "generated_index_backup", None),
        "embedding": {
            "model_id": storage.active_embedding_metadata(app_state)["embedding_model_id"],
            "dimension": storage.active_embedding_metadata(app_state)["embedding_dim"],
            "table": retrieval.vector_table_name(app_state),
        },
    }


@router.get("/models")
def get_models(request: Request):
    app_state = state(request)
    inventory = models.model_inventory(app_state.settings)
    return {
        "models": inventory["chat_models"],
        "model_details": inventory.get("chat_model_details", []),
        "auxiliary_gguf": inventory["auxiliary_gguf"],
        "model_dir": app_state.settings.model_dir,
        "active_model": getattr(app_state, "active_model_name", None),
        "active_context_tokens": getattr(app_state, "active_context_tokens", None),
        "active_model_context_tokens": getattr(app_state, "active_model_context_tokens", None),
        "last_model_load_error": getattr(app_state, "last_model_load_error", None),
        "llama_backend": models.llama_backend_info(),
    }


@router.get("/vaults/obsidian")
def get_obsidian_vault(request: Request):
    app_state = state(request)
    vault_path = app_state.settings.obsidian_vault_dir
    return {
        "path": vault_path,
        "exists": os.path.isdir(vault_path),
    }


@router.post("/vaults/obsidian/ingest")
async def ingest_obsidian_vault(request: Request):
    app_state = state(request)
    vault_path = app_state.settings.obsidian_vault_dir
    if not os.path.isdir(vault_path):
        raise HTTPException(status_code=404, detail=f"Obsidian vault not found: {vault_path}")
    job = await app_state.job_manager.enqueue_ingest(vault_path, kind="obsidian", force_text=True)
    return {"job_id": job["id"], "status": job["status"], "message": "Obsidian vault ingestion queued.", "path": vault_path}


@router.post("/models/load")
def load_model(request: Request, req: LoadModelRequest):
    app_state = state(request)
    if app_state.startup_error:
        raise HTTPException(status_code=503, detail=app_state.startup_error)
    if not req.model.strip():
        raise HTTPException(status_code=400, detail="Select a local GGUF model before loading.")

    models.load_llm(app_state, req.model)
    return {
        "status": "loaded",
        "active_model": getattr(app_state, "active_model_name", None),
        "active_context_tokens": getattr(app_state, "active_context_tokens", None),
        "active_model_context_tokens": getattr(app_state, "active_model_context_tokens", None),
        "last_model_load_error": getattr(app_state, "last_model_load_error", None),
        "llama_backend": models.llama_backend_info(),
    }


@router.get("/settings")
def get_settings(request: Request):
    return storage.get_rag_settings(state(request).sqlite)


@router.put("/settings")
async def put_settings(request: Request, rag_settings: RagSettings):
    saved = storage.save_rag_settings(state(request).sqlite, rag_settings)
    await state(request).event_bus.publish("settings", saved.model_dump())
    return saved


@router.get("/documents")
def get_documents(request: Request):
    rows = storage.fetchall(
        state(request).sqlite,
        "SELECT * FROM documents WHERE type = 'file' ORDER BY ingested_at DESC",
    )
    return {"documents": [storage.document_payload(state(request).sqlite, row) for row in rows]}


@router.get("/documents/{doc_id}")
def get_document(request: Request, doc_id: str):
    doc_id = validate_document_id(doc_id)
    row = storage.fetchone(state(request).sqlite, "SELECT * FROM documents WHERE id = ? AND type = 'file'", (doc_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Document not found.")
    chunks = storage.fetchall(state(request).sqlite, "SELECT * FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (doc_id,))
    payload = storage.document_payload(state(request).sqlite, row)
    payload["chunk_preview"] = [
        {
            "id": c["id"],
            "index": c["chunk_index"],
            "text": c["text"][:500],
            "block_type": c["block_type"] if "block_type" in c.keys() else None,
            "token_count": c["token_count"] if "token_count" in c.keys() else None,
            "char_count": c["char_count"] if "char_count" in c.keys() else None,
            "chunking_profile": c["chunking_profile"] if "chunking_profile" in c.keys() else None,
            "embedding_status": c["embedding_status"] if "embedding_status" in c.keys() else None,
        }
        for c in chunks[:20]
    ]
    return payload


@router.patch("/documents/{doc_id}")
async def patch_document(request: Request, doc_id: str, body: DocumentUpdateRequest):
    doc_id = validate_document_id(doc_id)
    if body.display_name is None or not body.display_name.strip():
        raise HTTPException(status_code=400, detail="display_name is required.")
    storage.execute(state(request).sqlite, "UPDATE documents SET display_name = ? WHERE id = ? AND type = 'file'", (body.display_name.strip(), doc_id))
    await state(request).event_bus.publish("document", {"id": doc_id, "status": "updated"})
    return get_document(request, doc_id)


@router.post("/documents/{doc_id}/tags")
async def add_tag(request: Request, doc_id: str, body: TagRequest):
    doc_id = validate_document_id(doc_id)
    tag = validate_tag(body.tag)
    if not storage.fetchone(state(request).sqlite, "SELECT id FROM documents WHERE id = ? AND type = 'file'", (doc_id,)):
        raise HTTPException(status_code=404, detail="Document not found.")
    storage.execute(state(request).sqlite, "INSERT OR IGNORE INTO document_tags (doc_id, tag) VALUES (?, ?)", (doc_id, tag))
    await state(request).event_bus.publish("document", {"id": doc_id, "status": "tagged", "tag": tag})
    return {"status": "success", "tag": tag}


@router.delete("/documents/{doc_id}/tags/{tag}")
async def delete_tag(request: Request, doc_id: str, tag: str):
    doc_id = validate_document_id(doc_id)
    tag = validate_tag(tag)
    storage.execute(state(request).sqlite, "DELETE FROM document_tags WHERE doc_id = ? AND tag = ?", (doc_id, tag))
    await state(request).event_bus.publish("document", {"id": doc_id, "status": "untagged", "tag": tag})
    return {"status": "success"}


@router.post("/documents/{doc_id}/reindex")
async def reindex_document(request: Request, doc_id: str):
    doc_id = validate_document_id(doc_id)
    row = storage.fetchone(state(request).sqlite, "SELECT path, extraction_mode FROM documents WHERE id = ? AND type = 'file'", (doc_id,))
    if not row:
        raise HTTPException(status_code=404, detail="Document not found.")
    storage.execute(state(request).sqlite, "UPDATE documents SET status = 'queued', last_error = NULL WHERE id = ?", (doc_id,))
    job = await state(request).job_manager.enqueue_ingest(
        row["path"],
        kind="reindex",
        target_doc_id=doc_id,
        force_text=row["extraction_mode"] == "text",
    )
    return {"job_id": job["id"], "status": job["status"], "message": "Document queued for reindexing."}


@router.delete("/documents/{doc_id}")
async def delete_document(request: Request, doc_id: str):
    doc_id = validate_document_id(doc_id)
    if not storage.fetchone(state(request).sqlite, "SELECT id FROM documents WHERE id = ? AND type = 'file'", (doc_id,)):
        raise HTTPException(status_code=404, detail="Document not found.")
    ingestion.delete_document_vectors(state(request), doc_id)
    ingestion.delete_document_rows(state(request), doc_id)
    await state(request).event_bus.publish("document", {"id": doc_id, "status": "deleted"})
    return {"status": "success"}


@router.post("/ingest")
async def ingest_endpoint(request: Request, req: IngestRequest):
    target_path = normalize_existing_path(req.path)
    job = await state(request).job_manager.enqueue_ingest(target_path, force_text=req.force_text)
    return {"job_id": job["id"], "status": job["status"], "message": "Ingestion queued."}


@router.get("/jobs")
def list_jobs(request: Request):
    return {"jobs": state(request).job_manager.list_jobs()}


@router.get("/retrieval/traces")
def list_retrieval_traces(request: Request):
    return {"traces": storage.list_retrieval_traces(state(request).sqlite)}


@router.get("/retrieval/traces/{query_id}")
def get_retrieval_trace(request: Request, query_id: str):
    trace = storage.get_retrieval_trace(state(request).sqlite, query_id)
    if not trace:
        raise HTTPException(status_code=404, detail="Retrieval trace not found.")
    return trace


@router.get("/observability/index-health")
def get_index_health(request: Request):
    return observability.index_health(state(request))


@router.get("/eval/runs")
def list_eval_runs(request: Request):
    return {"runs": storage.list_eval_runs(state(request).sqlite)}


@router.get("/eval/runs/{run_id}")
def get_eval_run(request: Request, run_id: str):
    run = storage.get_eval_run(state(request).sqlite, run_id)
    if not run:
        raise HTTPException(status_code=404, detail="Eval run not found.")
    return run


@router.post("/eval/runs")
async def create_eval_run(request: Request, body: EvalRunRequest):
    app_state = state(request)
    if app_state.startup_error:
        raise HTTPException(status_code=503, detail=app_state.startup_error)
    settings = storage.get_rag_settings(app_state.sqlite).model_copy(update={"top_k": body.top_k, "rerank_top_n": min(body.top_k, 10)})
    retrieved_by_id = {}
    for item in body.evals:
        vector = await retrieval.get_embedding(app_state, item.question)
        _context, sources, _meta = await retrieval.retrieve_context(app_state, item.question, vector, settings)
        retrieved_by_id[item.id] = [
            {"doc_id": source.doc_id, "chunk_id": source.chunk_id, "score": source.score}
            for source in sources
        ]
    run = evaluation.run_eval_set(
        app_state.sqlite,
        [item.model_dump() for item in body.evals],
        body.pipeline,
        retrieved_by_id,
        body.top_k,
    )
    return run


@router.post("/feedback")
def save_feedback(request: Request, body: dict):
    storage.execute(
        state(request).sqlite,
        """
        INSERT INTO user_feedback (query_id, message_id, feedback_value, failure_reason, correction_text, expected_doc_id, created_at)
        VALUES (?, ?, ?, ?, ?, ?, strftime('%s','now'))
        """,
        (
            body.get("query_id"),
            body.get("message_id"),
            str(body.get("feedback_value", "")).strip()[:32] or "unknown",
            body.get("failure_reason"),
            body.get("correction_text"),
            body.get("expected_doc_id"),
        ),
    )
    return {"status": "success"}


@router.get("/conversations")
def list_conversations(request: Request):
    return {"conversations": storage.list_conversations(state(request).sqlite)}


@router.post("/conversations")
def create_conversation(request: Request):
    return storage.create_conversation(state(request).sqlite)


@router.get("/conversations/{conversation_id}")
def get_conversation(request: Request, conversation_id: str):
    payload = storage.get_conversation(state(request).sqlite, conversation_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return payload


@router.patch("/conversations/{conversation_id}")
async def rename_conversation(request: Request, conversation_id: str, body: dict):
    payload = storage.rename_conversation(state(request).sqlite, conversation_id, str(body.get("title", "")))
    if not payload:
        raise HTTPException(status_code=400, detail="Conversation title is required.")
    await state(request).event_bus.publish("conversation", {"id": conversation_id, "status": "renamed"})
    return payload


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(request: Request, conversation_id: str):
    storage.archive_conversation(state(request).sqlite, conversation_id)
    await state(request).event_bus.publish("conversation", {"id": conversation_id, "status": "deleted"})
    return {"status": "success"}


@router.get("/events")
async def events(request: Request):
    return StreamingResponse(state(request).event_bus.stream(), media_type="text/event-stream")


@router.post("/metrics/export")
async def export_metrics(request: Request):
    app_state = state(request)
    try:
        return {"status": "success", "path": metrics.export_corpus_snapshot(app_state), "error": None}
    except OSError as error:
        app_state.last_metrics_error = str(error)
        return {"status": "failed", "path": None, "error": str(error)}


@router.post("/query")
async def chat_and_remember(request: Request, req: QueryRequest):
    app_state = state(request)
    if app_state.startup_error:
        raise HTTPException(status_code=503, detail=app_state.startup_error)
    if not req.prompt.strip():
        raise HTTPException(status_code=400, detail="Prompt is required.")
    if not req.model.strip():
        raise HTTPException(status_code=400, detail="Select a local GGUF model before querying.")

    rag_settings = _settings_for_reasoning_mode(req.settings or storage.get_rag_settings(app_state.sqlite), req.reasoning_mode)
    _ensure_query_model_loaded(app_state, req.model)

    query_vector = await retrieval.get_embedding(app_state, req.prompt)
    context, sources, query_meta = await retrieval.retrieve_context(app_state, req.prompt, query_vector, rag_settings)
    query_meta["reasoning_mode"] = req.reasoning_mode
    if rag_settings.trace_persistence and query_meta.get("trace"):
        storage.save_retrieval_trace(app_state.sqlite, query_meta["trace"])
    conversation_id = req.conversation_id
    if not conversation_id:
        title = req.prompt.strip().replace("\n", " ")[:80]
        conversation_id = storage.create_conversation(app_state.sqlite, title)["id"]
    user_message = storage.append_message(
        app_state.sqlite,
        conversation_id,
        "user",
        req.prompt,
        model=req.model,
        settings={**rag_settings.model_dump(), "reasoning_mode": req.reasoning_mode},
    )

    def response_stream():
        answer_parts: list[str] = []
        for subquery in query_meta["subqueries"]:
            yield _sse("subquery", subquery)
        yield _sse("conversation", {"conversation_id": conversation_id, "user_message_id": user_message["id"]})
        for source in sources:
            yield _sse("source", source.model_dump())
        yield _sse("answer_meta", {key: value for key, value in query_meta.items() if key not in {"subqueries", "trace"}})
        try:
            generation_started = time.perf_counter()
            for token in generation.stream_llama(app_state, req.prompt, context, req.history, rag_settings, query_meta):
                answer_parts.append(token)
                yield _sse("token", {"text": token})
            answer_text = "".join(answer_parts)
            generation_ms = round((time.perf_counter() - generation_started) * 1000, 2)
            quality = metrics.estimate_answer_quality(req.prompt, answer_text, context)
            support_payload = support.classify_answer_support(sources)
            query_meta.update(quality)
            query_meta["support"] = support_payload
            query_meta["generation_latency_ms"] = generation_ms
            try:
                metrics.append_retrieval_event(app_state, {
                    "event_type": "answer_quality",
                    "conversation_id": conversation_id,
                    "model": req.model,
                    **quality,
                })
            except OSError as error:
                app_state.last_metrics_error = str(error)
            yield _sse("answer_meta", {**quality, "support": support_payload, "generation_latency_ms": generation_ms})
            assistant_message = storage.append_message(
                app_state.sqlite,
                conversation_id,
                "assistant",
                answer_text,
                model=req.model,
                settings={**rag_settings.model_dump(), "reasoning_mode": req.reasoning_mode},
                meta=query_meta,
            )
            storage.save_message_sources(app_state.sqlite, assistant_message["id"], [source.model_dump() for source in sources])
            storage.save_answer_record(app_state.sqlite, {
                "id": assistant_message["id"],
                "query_id": query_meta.get("query_id"),
                "conversation_id": conversation_id,
                "message_id": assistant_message["id"],
                "answer_text": answer_text,
                "confidence": query_meta.get("confidence"),
                "support_status": support_payload["status"],
                "meta": {key: value for key, value in query_meta.items() if key != "trace"},
                "citations": support_payload["citations"],
            })
            yield _sse("message", {"conversation_id": conversation_id, "assistant_message_id": assistant_message["id"]})
            yield _sse("done", {"ok": True})
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(response_stream(), media_type="text/event-stream")


def _sse(event_type: str, payload: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
