import asyncio
import json
import os
import re
import time
import uuid
from collections.abc import Iterator
from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import StreamingResponse

from .. import storage
from ..schemas import EvalRunRequest, IngestRequest, LlamaServerSettings, LoadModelRequest, QueryRequest, RagSettings
from ..services import evaluation, generation, ingestion, jina_runtime, metrics, models, observability, retrieval, support
from ..validators import normalize_existing_path
from .conversations import router as conversations_router
from .documents import delete_document, get_documents, router as documents_router


router = APIRouter()
router.include_router(documents_router)
router.include_router(conversations_router)


def state(request: Request):
    return request.app.state


def _ensure_query_model_loaded(app_state, requested_model: str) -> None:
    models.ensure_server_connected(app_state, requested_model)


def _settings_for_retrieval_scope(settings: RagSettings, scope: str) -> RagSettings:
    clean = (scope or "medium").lower()
    if clean in {"low", "fast"}:
        return settings.model_copy(update={
            "top_k": min(settings.top_k, 12),
            "rerank_top_n": min(settings.rerank_top_n, 3),
        })
    if clean in {"high", "deep"}:
        return settings.model_copy(update={
            "top_k": max(settings.top_k, 28),
            "rerank_top_n": max(settings.rerank_top_n, 6),
        })
    return settings


def plan_retrieval_route(
    prompt: str,
    scope: str,
    *,
    evidence_required: bool = False,
) -> dict:
    requested = (scope or "auto").lower()
    if requested in {"low", "medium", "high"}:
        return {
            "requested": requested,
            "resolved": requested,
            "retrieve": True,
            "reason": "The user selected an explicit retrieval scope.",
        }
    if requested == "off":
        return {
            "requested": requested,
            "resolved": "off",
            "retrieve": False,
            "reason": "Document retrieval was explicitly disabled.",
        }
    if evidence_required:
        return {
            "requested": "auto",
            "resolved": "medium",
            "retrieve": True,
            "reason": "Evidence-required mode always searches local documents.",
        }

    clean = " ".join(prompt.lower().split())
    document_cues = (
        "according to", "citation", "cite ", "document", "file", "local source",
        "my notes", "paper", "pdf", "report", "research", "source", "spreadsheet",
        "table", "uploaded",
    )
    non_retrieval_cues = (
        "brainstorm", "draft ", "help me write", "make up", "proofread", "rewrite",
        "roleplay", "tell me a joke", "translate",
    )
    greeting = re.fullmatch(
        r"(?:hi|hello|hey|thanks|thank you|good (?:morning|afternoon|evening))[!. ]*",
        clean,
    )
    if any(cue in clean for cue in document_cues):
        resolved = "medium"
        reason = "The prompt refers to documents, sources, citations, or structured records."
    elif greeting or any(cue in clean for cue in non_retrieval_cues):
        resolved = "off"
        reason = "The prompt is clearly conversational or generative rather than document-seeking."
    elif len(clean.split()) <= 4 and not clean.endswith("?"):
        resolved = "off"
        reason = "The short prompt has no document-retrieval signal."
    else:
        resolved = "medium"
        reason = "Auto mode defaults to retrieval when document relevance is uncertain."
    return {
        "requested": "auto",
        "resolved": resolved,
        "retrieve": resolved != "off",
        "reason": reason,
    }


def _empty_retrieval_meta(route: dict, *, evidence_required: bool) -> dict:
    no_answer = bool(evidence_required)
    return {
        "query_id": str(uuid.uuid4()),
        "subqueries": [],
        "retrieval_latency_ms": 0.0,
        "search_modes": ["disabled"],
        "metrics_path": None,
        "confidence": 0.0,
        "uncertainty": "not_applicable" if not no_answer else "high",
        "no_answer": no_answer,
        "reason": (
            "Document retrieval is disabled."
            if not no_answer
            else "Evidence is required, but document retrieval is disabled."
        ),
        "reasons": ["retrieval_disabled"],
        "thresholds": {},
        "agreement": {"hybrid_overlap": False, "source_diversity": 0},
        "top_scores": {},
        "trace": None,
        "retrieval_route": route,
    }


@router.get("/health")
def health(request: Request):
    app_state = state(request)
    retrieval_error = getattr(app_state, "retrieval_error", None)
    return {
        "service": "cephalon",
        "api_version": 1,
        "status": "degraded" if app_state.startup_error or retrieval_error else "ok",
        "startup_error": app_state.startup_error,
        "engines_ready": app_state.startup_error is None and retrieval_error is None,
        "data_dir": app_state.settings.data_dir,
        "model_dir": app_state.settings.model_dir,
        "metrics_dir": app_state.settings.metrics_dir,
        "obsidian_vault_dir": app_state.settings.obsidian_vault_dir,
        "last_metrics_error": getattr(app_state, "last_metrics_error", None),
        "active_model": getattr(app_state, "active_model_name", None),
        "active_context_tokens": getattr(app_state, "active_context_tokens", None),
        "active_model_context_tokens": getattr(app_state, "active_model_context_tokens", None),
        "last_model_load_error": getattr(app_state, "last_model_load_error", None),
        "retrieval_error": retrieval_error,
        "python_runtime": models.python_runtime_info(),
        "llama_backend": models.llama_backend_info(app_state, probe=True),
        "retrieval_index": getattr(app_state, "retrieval_index", None),
        "generated_index_backup": getattr(app_state, "generated_index_backup", None),
        "embedding": {
            "model_id": storage.active_embedding_metadata(app_state)["embedding_model_id"],
            "dimension": storage.active_embedding_metadata(app_state)["embedding_dim"],
            "table": retrieval.vector_table_name(app_state),
        },
        "retrieval_stack": jina_runtime.model_status(app_state),
    }


@router.get("/models/status")
def get_model_status(request: Request):
    return jina_runtime.model_status(state(request))


@router.post("/models/download")
def download_model(request: Request, body: dict):
    app_state = state(request)
    try:
        payload = jina_runtime.download_model(app_state, str(body.get("kind", "")))
        # Downloads are intentionally not hot-loaded: a restart provides a
        # predictable model process boundary and avoids replacing active files.
        return {"status": "downloaded", "restart_required": True, "model": payload}
    except (ValueError, OSError, RuntimeError) as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@router.post("/models/verify")
def verify_model(request: Request, body: dict):
    try:
        return jina_runtime.verify_model(state(request), str(body.get("kind", "")))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/models/delete")
def delete_model(request: Request, body: dict):
    if body.get("confirmed") is not True:
        raise HTTPException(status_code=400, detail="Model deletion requires confirmed: true.")
    try:
        return jina_runtime.delete_model(state(request), str(body.get("kind", "")))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.post("/models/open")
def open_model_directory(request: Request, body: dict):
    try:
        return jina_runtime.open_model_directory(state(request), str(body.get("kind", "")))
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc


@router.get("/runtime/embedder/status")
def get_embedder_runtime(request: Request):
    return jina_runtime.embedder_status(state(request))


@router.get("/runtime/reranker/status")
def get_reranker_runtime(request: Request):
    return jina_runtime.reranker_status(state(request))


@router.get("/models")
def get_models(request: Request):
    app_state = state(request)
    inventory = models.model_inventory(app_state)
    return {
        "models": inventory["chat_models"],
        "model_details": inventory.get("chat_model_details", []),
        "auxiliary_gguf": inventory["auxiliary_gguf"],
        "model_dir": app_state.settings.model_dir,
        "active_model": getattr(app_state, "active_model_name", None),
        "active_context_tokens": getattr(app_state, "active_context_tokens", None),
        "active_model_context_tokens": getattr(app_state, "active_model_context_tokens", None),
        "last_model_load_error": getattr(app_state, "last_model_load_error", None),
        "llama_backend": models.llama_backend_info(app_state, probe=True),
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
    models.load_llm(app_state, req.model)
    return {
        "status": "loaded",
        "active_model": getattr(app_state, "active_model_name", None),
        "active_context_tokens": getattr(app_state, "active_context_tokens", None),
        "active_model_context_tokens": getattr(app_state, "active_model_context_tokens", None),
        "last_model_load_error": getattr(app_state, "last_model_load_error", None),
        "llama_backend": models.llama_backend_info(app_state, probe=True),
    }


@router.get("/models/server")
def get_llama_server_settings(request: Request):
    return state(request).llama_server_settings


@router.put("/models/server")
async def put_llama_server_settings(request: Request, server_settings: LlamaServerSettings):
    app_state = state(request)
    saved = storage.save_llama_server_settings(app_state.sqlite, server_settings)
    app_state.llama_server_settings = saved
    app_state.active_model_name = None
    app_state.active_context_tokens = None
    app_state.active_model_context_tokens = None
    app_state.last_model_load_error = None
    await app_state.event_bus.publish("llama_server", saved.model_dump())
    return saved


@router.get("/settings")
def get_settings(request: Request):
    return storage.get_rag_settings(state(request).sqlite)


@router.put("/settings")
async def put_settings(request: Request, rag_settings: RagSettings):
    app_state = state(request)
    previous = storage.get_rag_settings(app_state.sqlite)
    chunk_keys = ("parent_target_tokens", "parent_max_tokens", "child_target_tokens", "child_max_tokens", "child_overlap_tokens")
    reindex_required = any(getattr(previous, key) != getattr(rag_settings, key) for key in chunk_keys)
    saved = storage.save_rag_settings(app_state.sqlite, rag_settings)
    stale_summary = ingestion.refresh_document_staleness(app_state, saved)
    await app_state.event_bus.publish(
        "settings",
        {
            **saved.model_dump(),
            "reindex_required": reindex_required,
            "stale_document_count": stale_summary["stale_document_count"],
        },
    )
    return saved


@router.post("/ingest")
async def ingest_endpoint(request: Request, req: IngestRequest):
    if getattr(state(request), "retrieval_error", None):
        raise HTTPException(status_code=503, detail=state(request).retrieval_error)
    target_path = normalize_existing_path(req.path)
    job = await state(request).job_manager.enqueue_ingest(target_path, force_text=req.force_text)
    return {"job_id": job["id"], "status": job["status"], "message": "Ingestion queued."}


async def _queue_reindex(request: Request, *, stale_only: bool) -> dict:
    app_state = state(request)
    if getattr(app_state, "retrieval_error", None):
        raise HTTPException(status_code=503, detail=app_state.retrieval_error)
    ingestion.refresh_document_staleness(app_state)
    clause = "AND stale_embedding = 1" if stale_only else ""
    rows = storage.fetchall(
        app_state.sqlite,
        # A reindex must retain its previous searchable document on failure.
        # Include earlier interrupted runs with existing chunks, but never use
        # document status as a job-queue state.
        f"SELECT id, path, extraction_mode FROM documents WHERE type = 'file' AND (status = 'ready' OR chunk_count > 0) {clause} ORDER BY display_name, path",
    )
    run = storage.create_reindex_run(app_state.sqlite, "stale" if stale_only else "full", len(rows))
    job_ids = []
    for row in rows:
        job = await app_state.job_manager.enqueue_ingest(
            row["path"], kind="reindex", target_doc_id=row["id"], force_text=row["extraction_mode"] == "text", reindex_run_id=run["id"],
        )
        job_ids.append(job["id"])
    app_state.reindex_required = bool(rows)
    return {"status": run["status"], "run_id": run["id"], "mode": run["mode"], "total": len(rows), "job_ids": job_ids}


@router.post("/reindex/full")
async def reindex_full(request: Request):
    return await _queue_reindex(request, stale_only=False)


@router.post("/reindex/stale")
async def reindex_stale(request: Request):
    return await _queue_reindex(request, stale_only=True)


@router.get("/reindex/progress")
def reindex_progress(request: Request):
    app_state = state(request)
    stale = ingestion.refresh_document_staleness(app_state)
    run = storage.latest_reindex_run(app_state.sqlite)
    app_state.reindex_required = bool(stale["stale_document_count"])
    if run is None:
        return {"status": "idle", "processed": 0, "total": 0, "succeeded": 0, "failed": 0, "cancelled": 0, "stale_document_count": stale["stale_document_count"], "reindex_required": app_state.reindex_required}
    return {
        "run_id": run["id"],
        "status": run["status"],
        "processed": run["processed_documents"],
        "total": run["total_documents"],
        "succeeded": run["succeeded_documents"],
        "failed": run["failed_documents"],
        "cancelled": run["cancelled_documents"],
        "stale_document_count": stale["stale_document_count"],
        "reindex_required": app_state.reindex_required,
    }


@router.get("/jobs")
def list_jobs(request: Request):
    return {"jobs": state(request).job_manager.list_jobs()}


@router.post("/jobs/{job_id}/cancel")
async def cancel_job(request: Request, job_id: str):
    try:
        return await state(request).job_manager.cancel_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc


@router.post("/jobs/{job_id}/retry")
async def retry_job(request: Request, job_id: str):
    try:
        return await state(request).job_manager.retry_job(job_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="Job not found.") from exc
    except ValueError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc


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
    if getattr(app_state, "retrieval_error", None) or getattr(app_state, "reindex_required", False):
        raise HTTPException(status_code=503, detail=getattr(app_state, "retrieval_error", None) or "The 768-dimensional Jina index requires reindexing.")
    settings = storage.get_rag_settings(app_state.sqlite).model_copy(update={"top_k": body.top_k, "rerank_top_n": min(body.top_k, 10)})
    retrieved_by_id = {}
    for item in body.evals:
        if item.id in body.sources:
            # End-to-end evaluation must grade the evidence actually shown to
            # the generator and user. Re-running retrieval here can select a
            # different candidate set, doubles listwise-reranker cost, and
            # invalidates citation attachment metrics. Retrieval-only callers
            # omit this field and continue through the normal live path below.
            retrieved_by_id[item.id] = body.sources[item.id]
            continue
        vector = await retrieval.get_embedding(app_state, item.question)
        _context, sources, _meta = await retrieval.retrieve_context(app_state, item.question, vector, settings)
        # Preserve source identifiers and provenance so answer/citation metrics
        # use the same evidence contract as the chat UI.
        retrieved_by_id[item.id] = [source.model_dump() for source in sources]
    run = evaluation.run_eval_set(
        app_state.sqlite,
        [item.model_dump() for item in body.evals],
        body.pipeline,
        retrieved_by_id,
        body.top_k,
        answers_by_id=body.answers,
        sources_by_id=body.sources or retrieved_by_id,
        run_meta=body.run_meta,
    )
    return run


@router.post("/feedback")
def save_feedback(request: Request, body: dict):
    app_state = state(request)
    cursor = storage.execute(
        app_state.sqlite,
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
    feedback_id = cursor.lastrowid
    expected_doc_id = str(body.get("expected_doc_id") or "").strip()
    question = str(body.get("question") or body.get("correction_text") or "").strip()
    if expected_doc_id and question:
        storage.execute(
            app_state.sqlite,
            "INSERT INTO eval_cases (id, question, expected_doc_id, source_feedback_id, status, created_at) VALUES (?, ?, ?, ?, 'pending', ?)",
            (str(uuid.uuid4()), question[:4000], expected_doc_id, feedback_id, int(time.time())),
        )
    return {"status": "success", "feedback_id": feedback_id}


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
        raise HTTPException(status_code=400, detail="Connect to the configured external llama.cpp server before querying.")

    base_rag_settings = req.settings or storage.get_rag_settings(app_state.sqlite)
    retrieval_route = plan_retrieval_route(
        req.prompt,
        req.retrieval_scope,
        evidence_required=base_rag_settings.evidence_required,
    )
    rag_settings = _settings_for_retrieval_scope(base_rag_settings, retrieval_route["resolved"])
    if retrieval_route["retrieve"] and getattr(app_state, "retrieval_error", None):
        raise HTTPException(status_code=503, detail=app_state.retrieval_error)
    if retrieval_route["retrieve"] and getattr(app_state, "reindex_required", False):
        raise HTTPException(status_code=409, detail="The previous 1024-dimensional index is stale. Reindex documents before querying retrieval.")
    _ensure_query_model_loaded(app_state, req.model)

    async def response_stream():
        answer_parts: list[str] = []
        try:
            # Send an event immediately so slow embedding/retrieval cannot leave the UI
            # waiting for a response that has not begun streaming yet.
            yield _sse("phase", {"phase": "routing"})
            if retrieval_route["retrieve"]:
                yield _sse("phase", {"phase": "retrieving"})
                query_vector = await retrieval.get_embedding(app_state, req.prompt)
                context, sources, query_meta = await retrieval.retrieve_context(app_state, req.prompt, query_vector, rag_settings)
                query_meta["retrieval_route"] = retrieval_route
            else:
                context, sources = "", []
                query_meta = _empty_retrieval_meta(
                    retrieval_route,
                    evidence_required=rag_settings.evidence_required,
                )
            query_meta["retrieval_scope"] = req.retrieval_scope
            query_meta["response_effort"] = req.response_effort
            if rag_settings.trace_persistence and query_meta.get("trace"):
                storage.save_retrieval_trace(app_state.sqlite, query_meta["trace"])
            conversation_id = req.conversation_id
            if conversation_id:
                conversation = storage.get_conversation(app_state.sqlite, conversation_id)
                if conversation and conversation.get("title") == "New chat":
                    storage.rename_conversation(app_state.sqlite, conversation_id, _conversation_title(req.prompt))
            else:
                conversation_id = storage.create_conversation(app_state.sqlite, _conversation_title(req.prompt))["id"]
            user_message = storage.append_message(
                app_state.sqlite, conversation_id, "user", req.prompt, model=req.model,
                settings={**rag_settings.model_dump(), "retrieval_scope": req.retrieval_scope, "response_effort": req.response_effort},
            )
            for subquery in query_meta["subqueries"]:
                yield _sse("subquery", subquery)
            yield _sse("conversation", {"conversation_id": conversation_id, "user_message_id": user_message["id"]})
            for source in sources:
                yield _sse("source", source.model_dump())
            yield _sse("answer_meta", {key: value for key, value in query_meta.items() if key not in {"subqueries", "trace"}})
            generation_started = time.perf_counter()
            if rag_settings.evidence_required and query_meta.get("no_answer"):
                answer = "I could not find sufficient supporting evidence in your local documents to answer that reliably."
                answer_parts.append(answer)
                yield _sse("phase", {"phase": "evidence_required"})
                yield _sse("token", {"text": answer})
            else:
                generation_events = generation.stream_response_events(
                    app_state,
                    req.prompt,
                    context,
                    req.history,
                    rag_settings,
                    query_meta,
                    response_effort=req.response_effort,
                )
                async for event_type, value in _cancel_on_disconnect(request, generation_events):
                    if event_type == "phase":
                        yield _sse("phase", {"phase": value})
                    else:
                        answer_parts.append(value)
                        yield _sse("token", {"text": value})
            if await request.is_disconnected():
                return
            answer_text = "".join(answer_parts)
            generation_ms = round((time.perf_counter() - generation_started) * 1000, 2)
            quality = metrics.estimate_answer_quality(req.prompt, answer_text, context)
            support_payload = support.classify_answer_support(answer_text, sources)
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
                settings={
                    **rag_settings.model_dump(),
                    "retrieval_scope": req.retrieval_scope,
                    "response_effort": req.response_effort,
                },
                meta=query_meta,
            )
            used_source_ids = set(support_payload["accounting"]["valid_source_ids"])
            storage.save_message_sources(
                app_state.sqlite,
                assistant_message["id"],
                [
                    source.model_dump()
                    for source in sources
                    if source.source_id in used_source_ids
                ],
            )
            if rag_settings.conversation_memory and answer_text.strip():
                await retrieval.save_permanent_memory(
                    app_state,
                    conversation_id,
                    assistant_message["id"],
                    req.prompt,
                    answer_text,
                )
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
        except asyncio.CancelledError:
            raise
        except Exception as exc:
            yield _sse("error", {"message": str(exc)})

    return StreamingResponse(response_stream(), media_type="text/event-stream")


def _conversation_title(prompt: str) -> str:
    return " ".join(prompt.split())[:80] or "New chat"


def _sse(event_type: str, payload: dict) -> str:
    return f"event: {event_type}\ndata: {json.dumps(payload)}\n\n"


_END_OF_EVENTS = object()


async def _cancel_on_disconnect(request: Request, events: Iterator[tuple[str, str]]):
    try:
        while not await request.is_disconnected():
            event = await asyncio.to_thread(_next_event, events)
            if event is _END_OF_EVENTS:
                return
            yield event
    finally:
        close = getattr(events, "close", None)
        if close is not None:
            close()


def _next_event(events: Iterator[tuple[str, str]]):
    try:
        return next(events)
    except StopIteration:
        return _END_OF_EVENTS
