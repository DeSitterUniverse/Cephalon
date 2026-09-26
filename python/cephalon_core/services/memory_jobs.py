"""Durable, best-effort embedding of completed conversation turns."""

import asyncio
import time

from .. import storage
from . import retrieval


class MemoryJobManager:
    def __init__(self, app_state) -> None:
        self.app_state = app_state
        self._wake = asyncio.Event()
        self._task: asyncio.Task | None = None

    async def start(self) -> None:
        self._task = asyncio.create_task(self._worker())
        self.wake()

    async def stop(self) -> None:
        if self._task is not None:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    def wake(self) -> None:
        self._wake.set()

    async def _worker(self) -> None:
        while True:
            job = storage.fetchone(
                self.app_state.sqlite,
                """
                SELECT memory_jobs.message_id, memory_jobs.conversation_id,
                       memory_jobs.prompt, memory_jobs.attempts,
                       memory_jobs.next_attempt_at, messages.content AS answer_text
                FROM memory_jobs JOIN messages ON messages.id = memory_jobs.message_id
                ORDER BY memory_jobs.next_attempt_at, memory_jobs.rowid LIMIT 1
                """,
            )
            self._wake.clear()
            if job is None:
                await self._wake.wait()
                continue
            delay = max(0, job["next_attempt_at"] - time.time())
            if delay:
                try:
                    await asyncio.wait_for(self._wake.wait(), timeout=delay)
                except asyncio.TimeoutError:
                    pass
                continue
            try:
                await retrieval.save_permanent_memory(
                    self.app_state,
                    job["conversation_id"],
                    job["message_id"],
                    job["prompt"],
                    job["answer_text"],
                )
            except asyncio.CancelledError:
                raise
            except Exception as exc:
                attempts = job["attempts"] + 1
                storage.execute(
                    self.app_state.sqlite,
                    """
                    UPDATE memory_jobs
                    SET attempts = ?, next_attempt_at = ?, last_error = ?
                    WHERE message_id = ?
                    """,
                    (attempts, int(time.time()) + min(300, 2 ** min(attempts, 8)), str(exc)[:500], job["message_id"]),
                )
                self.app_state.last_memory_error = str(exc)
            else:
                storage.execute(self.app_state.sqlite, "DELETE FROM memory_jobs WHERE message_id = ?", (job["message_id"],))
                self.app_state.last_memory_error = None
