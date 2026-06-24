from fastapi import APIRouter, HTTPException, Request

from .. import storage
from ..schemas import DocumentUpdateRequest, TagRequest
from ..services import ingestion
from ..validators import validate_document_id, validate_tag


router = APIRouter()


@router.get("/documents")
def get_documents(request: Request):
    return {"documents": storage.list_document_payloads(request.app.state.sqlite)}


@router.get("/documents/{doc_id}")
def get_document(request: Request, doc_id: str):
    doc_id = validate_document_id(doc_id)
    payload = storage.get_document_payload(request.app.state.sqlite, doc_id)
    if not payload:
        raise HTTPException(status_code=404, detail="Document not found.")
    return payload


@router.patch("/documents/{doc_id}")
async def patch_document(request: Request, doc_id: str, body: DocumentUpdateRequest):
    doc_id = validate_document_id(doc_id)
    if body.display_name is None or not body.display_name.strip():
        raise HTTPException(status_code=400, detail="display_name is required.")
    storage.execute(
        request.app.state.sqlite,
        "UPDATE documents SET display_name = ? WHERE id = ? AND type = 'file'",
        (body.display_name.strip(), doc_id),
    )
    await request.app.state.event_bus.publish("document", {"id": doc_id, "status": "updated"})
    return get_document(request, doc_id)


@router.post("/documents/{doc_id}/tags")
async def add_tag(request: Request, doc_id: str, body: TagRequest):
    doc_id = validate_document_id(doc_id)
    tag = validate_tag(body.tag)
    if not storage.fetchone(
        request.app.state.sqlite,
        "SELECT id FROM documents WHERE id = ? AND type = 'file'",
        (doc_id,),
    ):
        raise HTTPException(status_code=404, detail="Document not found.")
    storage.execute(
        request.app.state.sqlite,
        "INSERT OR IGNORE INTO document_tags (doc_id, tag) VALUES (?, ?)",
        (doc_id, tag),
    )
    await request.app.state.event_bus.publish(
        "document",
        {"id": doc_id, "status": "tagged", "tag": tag},
    )
    return {"status": "success", "tag": tag}


@router.delete("/documents/{doc_id}/tags/{tag}")
async def delete_tag(request: Request, doc_id: str, tag: str):
    doc_id = validate_document_id(doc_id)
    tag = validate_tag(tag)
    storage.execute(
        request.app.state.sqlite,
        "DELETE FROM document_tags WHERE doc_id = ? AND tag = ?",
        (doc_id, tag),
    )
    await request.app.state.event_bus.publish(
        "document",
        {"id": doc_id, "status": "untagged", "tag": tag},
    )
    return {"status": "success"}


@router.post("/documents/{doc_id}/reindex")
async def reindex_document(request: Request, doc_id: str):
    doc_id = validate_document_id(doc_id)
    row = storage.fetchone(
        request.app.state.sqlite,
        "SELECT path, extraction_mode FROM documents WHERE id = ? AND type = 'file'",
        (doc_id,),
    )
    if not row:
        raise HTTPException(status_code=404, detail="Document not found.")
    storage.execute(
        request.app.state.sqlite,
        "UPDATE documents SET status = 'queued', last_error = NULL WHERE id = ?",
        (doc_id,),
    )
    job = await request.app.state.job_manager.enqueue_ingest(
        row["path"],
        kind="reindex",
        target_doc_id=doc_id,
        force_text=row["extraction_mode"] == "text",
    )
    return {
        "job_id": job["id"],
        "status": job["status"],
        "message": "Document queued for reindexing.",
    }


@router.delete("/documents/{doc_id}")
async def delete_document(request: Request, doc_id: str):
    doc_id = validate_document_id(doc_id)
    if not storage.fetchone(
        request.app.state.sqlite,
        "SELECT id FROM documents WHERE id = ? AND type = 'file'",
        (doc_id,),
    ):
        raise HTTPException(status_code=404, detail="Document not found.")
    ingestion.delete_document_vectors(request.app.state, doc_id)
    ingestion.delete_document_rows(request.app.state, doc_id)
    await request.app.state.event_bus.publish("document", {"id": doc_id, "status": "deleted"})
    return {"status": "success"}
