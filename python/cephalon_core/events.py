import asyncio
import json
import time
from collections.abc import AsyncIterator

from . import storage


class EventBus:
    def __init__(self, sqlite_conn=None) -> None:
        self.sqlite_conn = sqlite_conn
        self._subscribers: set[asyncio.Queue[dict]] = set()

    async def publish(self, event_type: str, payload: dict, job_id: str | None = None) -> None:
        event = {"type": event_type, "payload": payload, "job_id": job_id, "created_at": int(time.time())}
        if self.sqlite_conn is not None:
            storage.execute(
                self.sqlite_conn,
                "INSERT INTO job_events (job_id, event_type, payload, created_at) VALUES (?, ?, ?, ?)",
                (job_id, event_type, json.dumps(payload), event["created_at"]),
            )
        for queue in list(self._subscribers):
            await queue.put(event)

    async def stream(self) -> AsyncIterator[str]:
        queue: asyncio.Queue[dict] = asyncio.Queue(maxsize=100)
        self._subscribers.add(queue)
        try:
            yield "event: ready\ndata: {}\n\n"
            while True:
                try:
                    event = await asyncio.wait_for(queue.get(), timeout=15)
                    yield f"event: {event['type']}\ndata: {json.dumps(event)}\n\n"
                except asyncio.TimeoutError:
                    yield "event: heartbeat\ndata: {}\n\n"
        finally:
            self._subscribers.discard(queue)
