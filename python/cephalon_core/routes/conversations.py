from fastapi import APIRouter, HTTPException, Request

from .. import storage
from ..services import retrieval


router = APIRouter()


@router.get("/conversations")
def list_conversations(request: Request):
    return {"conversations": storage.list_conversations(request.app.state.sqlite)}


@router.post("/conversations")
def create_conversation(request: Request):
    return storage.create_conversation(request.app.state.sqlite)


@router.get("/conversations/{conversation_id}")
def get_conversation(
    request: Request,
    conversation_id: str,
    limit: int = 100,
    before: int | None = None,
):
    payload = storage.get_conversation(
        request.app.state.sqlite,
        conversation_id,
        message_limit=limit,
        before=before,
    )
    if not payload:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return payload


@router.patch("/conversations/{conversation_id}")
async def rename_conversation(request: Request, conversation_id: str, body: dict):
    payload = storage.rename_conversation(
        request.app.state.sqlite,
        conversation_id,
        str(body.get("title", "")),
    )
    if not payload:
        raise HTTPException(status_code=400, detail="Conversation title is required.")
    await request.app.state.event_bus.publish(
        "conversation",
        {"id": conversation_id, "status": "renamed"},
    )
    return payload


@router.delete("/conversations/{conversation_id}")
async def delete_conversation(request: Request, conversation_id: str):
    retrieval.delete_conversation_memory(request.app.state, conversation_id)
    storage.archive_conversation(request.app.state.sqlite, conversation_id)
    await request.app.state.event_bus.publish(
        "conversation",
        {"id": conversation_id, "status": "deleted"},
    )
    return {"status": "success"}
