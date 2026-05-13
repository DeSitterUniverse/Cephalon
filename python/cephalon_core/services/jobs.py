import asyncio
import os
import time
import uuid

from .. import storage
from ..events import EventBus
from .documents import collect_supported_files
from .ingestion import delete_document_vectors, process_single_file


class JobManager:
    def __init__(self, app_state, event_bus: EventBus) -> None:
        self.app_state = app_state
        self.event_bus = event_bus
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.worker_task: asyncio.Task | None = None
        self.running = False

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        self.worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        self.running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    async def enqueue_ingest(self, path: str, kind: str = "ingest", *, target_doc_id: str | None = None, force_text: bool = False) -> dict:
        now = int(time.time())
        job_id = str(uuid.uuid4())
        storage.execute(
            self.app_state.sqlite,
            """
            INSERT INTO jobs (id, kind, path, status, total_files, processed_files, skipped_files, created_at, updated_at, target_doc_id, force_text)
            VALUES (?, ?, ?, 'queued', 0, 0, 0, ?, ?, ?, ?)
            """,
            (job_id, kind, path, now, now, target_doc_id, 1 if force_text else 0),
        )
        await self.event_bus.publish("job", self.get_job(job_id), job_id)
        await self.queue.put(job_id)
        return self.get_job(job_id)

    def list_jobs(self) -> list[dict]:
        rows = storage.fetchall(self.app_state.sqlite, "SELECT * FROM jobs ORDER BY created_at DESC LIMIT 100")
        return [self._job_payload(row) for row in rows]

    def get_job(self, job_id: str) -> dict:
        row = storage.fetchone(self.app_state.sqlite, "SELECT * FROM jobs WHERE id = ?", (job_id,))
        if not row:
            raise KeyError(job_id)
        return self._job_payload(row)

    def _job_payload(self, row) -> dict:
        return {key: row[key] for key in row.keys()}

    async def _worker(self) -> None:
        while self.running:
            job_id = await self.queue.get()
            try:
                await self._run_job(job_id)
            finally:
                self.queue.task_done()

    async def _run_job(self, job_id: str) -> None:
        job = self.get_job(job_id)
        path = job["path"]
        force_text = bool(job.get("force_text"))
        files = collect_supported_files(path, force_text=force_text)
        total = len(files)
        await self._update_job(job_id, status="running", total_files=total, current_file=os.path.basename(path))

        if total == 0:
            await self._update_job(job_id, status="failed", error="No supported files found.", current_file=None)
            return

        processed = 0
        skipped = 0
        failures: list[str] = []
        rag_settings = storage.get_rag_settings(self.app_state.sqlite)

        for file_path in files:
            await self._update_job(job_id, current_file=file_path)
            existing_doc_id = job.get("target_doc_id") if job["kind"] == "reindex" and total == 1 else None
            if existing_doc_id:
                delete_document_vectors(self.app_state, existing_doc_id)
                storage.delete_document_fts(self.app_state.sqlite, existing_doc_id)
                storage.execute(self.app_state.sqlite, "DELETE FROM chunks WHERE doc_id = ?", (existing_doc_id,))
            result = await process_single_file(
                self.app_state,
                file_path,
                rag_settings,
                force_text=force_text,
                existing_doc_id=existing_doc_id,
            )
            processed += 1
            if result["status"] == "skipped":
                skipped += 1
            elif result["status"] == "failed":
                failures.append(f"{os.path.basename(file_path)}: {result.get('error', 'failed')}")
            await self._update_job(job_id, processed_files=processed, skipped_files=skipped)
            await self.event_bus.publish("document", result, job_id)

        if failures:
            await self._update_job(job_id, status="failed", error="; ".join(failures[:3]), current_file=None)
        else:
            await self._update_job(job_id, status="succeeded", current_file=None)

    async def _update_job(self, job_id: str, **fields) -> None:
        fields["updated_at"] = int(time.time())
        assignments = ", ".join(f"{key} = ?" for key in fields)
        params = tuple(fields.values()) + (job_id,)
        storage.execute(self.app_state.sqlite, f"UPDATE jobs SET {assignments} WHERE id = ?", params)
        await self.event_bus.publish("job", self.get_job(job_id), job_id)
