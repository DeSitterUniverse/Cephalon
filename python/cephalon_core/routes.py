import json
import os

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from . import storage
from .schemas import DocumentUpdateRequest, IngestRequest, QueryRequest, RagSettings, TagRequest
from .services import generation, ingestion, metrics, models, retrieval
from .validators import is_supported_file, normalize_existing_path, validate_document_id, validate_tag


router = APIRouter()


def state(request: Request):
    return request.app.state


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
        "last_metrics_error": getattr(app_state, "last_metrics_error", None),
        "active_model": getattr(app_state, "active_model_name", None),
        "active_context_tokens": getattr(app_state, "active_context_tokens", None),
        "active_model_context_tokens": getattr(app_state, "active_model_context_tokens", None),
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
    return {
        "models": models.list_models(app_state.settings),
        "model_dir": app_state.settings.model_dir,
        "active_model": getattr(app_state, "active_model_name", None),
        "active_context_tokens": getattr(app_state, "active_context_tokens", None),
        "active_model_context_tokens": getattr(app_state, "active_model_context_tokens", None),
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
    chunks = storage.fetchall(state(request).sqlite, "SELECT id, chunk_index, text FROM chunks WHERE doc_id = ? ORDER BY chunk_index", (doc_id,))
    payload = storage.document_payload(state(request).sqlite, row)
    payload["chunk_preview"] = [{"id": c["id"], "index": c["chunk_index"], "text": c["text"][:500]} for c in chunks[:10]]
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
    if os.path.isfile(target_path) and not is_supported_file(target_path) and not req.force_text:
        raise HTTPException(status_code=400, detail=f"Unsupported file type: {os.path.splitext(target_path)[1] or 'none'}")
    job = await state(request).job_manager.enqueue_ingest(target_path, force_text=req.force_text)
    return {"job_id": job["id"], "status": job["status"], "message": "Ingestion queued."}


@router.get("/jobs")
def list_jobs(request: Request):
    return {"jobs": state(request).job_manager.list_jobs()}


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

    rag_settings = req.settings or storage.get_rag_settings(app_state.sqlite)
    models.load_llm(app_state, req.model) if getattr(app_state, "active_model_name", None) != req.model else None

    query_vector = await retrieval.get_embedding(app_state, req.prompt)
    context, sources, query_meta = await retrieval.retrieve_context(app_state, req.prompt, query_vector, rag_settings)

    async def after_response():
        await retrieval.save_permanent_memory(app_state, req.prompt, query_vector)

    def response_stream():
        for subquery in query_meta["subqueries"]:
            yield _sse("subquery", subquery)
        for source in sources:
            yield _sse("source", source.model_dump())
        yield _sse("answer_meta", {key: value for key, value in query_meta.items() if key != "subqueries"})
        try:
            for token in generation.stream_llama(app_state, req.prompt, context, req.history, rag_settings, query_meta):
                yield _sse("token", {"text": token})
            yield _sse("done", {"ok": True})
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    # Save memory before returning; FastAPI BackgroundTasks cannot be attached to this manual stream cleanly here.
    await after_response()
    return StreamingResponse(response_stream(), media_type="text/event-stream")


def _sse(event_type: str, payload: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"
