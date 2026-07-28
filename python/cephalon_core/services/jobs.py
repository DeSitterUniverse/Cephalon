import asyncio
import os
import time
import uuid

from .. import storage
from ..events import EventBus
from .documents import collect_obsidian_files, collect_supported_files
from .ingestion import process_single_file, refresh_document_staleness


class JobManager:
    def __init__(self, app_state, event_bus: EventBus) -> None:
        self.app_state = app_state
        self.event_bus = event_bus
        self.queue: asyncio.Queue[str] = asyncio.Queue()
        self.worker_task: asyncio.Task | None = None
        self.running = False
        self.cancelled_jobs: set[str] = set()

    async def start(self) -> None:
        if self.running:
            return
        self.running = True
        for job_id in self.recover_interrupted_jobs():
            await self.queue.put(job_id)
        self.worker_task = asyncio.create_task(self._worker())

    async def stop(self) -> None:
        self.running = False
        if self.worker_task:
            self.worker_task.cancel()
            try:
                await self.worker_task
            except asyncio.CancelledError:
                pass

    async def enqueue_ingest(self, path: str, kind: str = "ingest", *, target_doc_id: str | None = None, force_text: bool = False, reindex_run_id: str | None = None) -> dict:
        now = int(time.time())
        job_id = str(uuid.uuid4())
        storage.execute(
            self.app_state.sqlite,
            """
            INSERT INTO jobs (id, kind, path, status, total_files, processed_files, skipped_files, created_at, updated_at, target_doc_id, force_text, reindex_run_id)
            VALUES (?, ?, ?, 'queued', 0, 0, 0, ?, ?, ?, ?, ?)
            """,
            (job_id, kind, path, now, now, target_doc_id, 1 if force_text else 0, reindex_run_id),
        )
        await self.event_bus.publish("job", self.get_job(job_id), job_id)
        await self.queue.put(job_id)
        return self.get_job(job_id)

    def recover_interrupted_jobs(self) -> list[str]:
        now = int(time.time())
        storage.execute(
            self.app_state.sqlite,
            """
            UPDATE jobs
            SET status = 'failed', error = 'Job interrupted by backend restart.', current_file = NULL, updated_at = ?
            WHERE status = 'running'
            """,
            (now,),
        )
        rows = storage.fetchall(
            self.app_state.sqlite,
            "SELECT id FROM jobs WHERE status = 'queued' ORDER BY created_at, rowid",
        )
        return [row["id"] for row in rows]

    async def cancel_job(self, job_id: str) -> dict:
        job = self.get_job(job_id)
        if job["status"] not in {"queued", "running"}:
            return job
        self.cancelled_jobs.add(job_id)
        await self._update_job(job_id, status="cancelled", current_file=None, error=None)
        return self.get_job(job_id)

    async def retry_job(self, job_id: str) -> dict:
        job = self.get_job(job_id)
        if job["status"] not in {"failed", "cancelled"}:
            raise ValueError("Only failed or cancelled jobs can be retried.")
        self.cancelled_jobs.discard(job_id)
        await self._update_job(
            job_id,
            status="queued",
            processed_files=0,
            skipped_files=0,
            current_file=None,
            error=None,
        )
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
        if job_id in self.cancelled_jobs or job["status"] == "cancelled":
            return
        path = job["path"]
        force_text = bool(job.get("force_text"))
        files = collect_obsidian_files(path) if job["kind"] == "obsidian" else collect_supported_files(path, force_text=force_text)
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
            if job_id in self.cancelled_jobs:
                await self._update_job(job_id, status="cancelled", current_file=None)
                return
            await self._update_job(job_id, current_file=file_path)
            existing_doc_id = job.get("target_doc_id") if job["kind"] == "reindex" and total == 1 else None

            async def report_progress(stage: str, percent: int) -> None:
                await self._update_job(job_id, stage=stage, stage_progress=percent)

            result = await process_single_file(
                self.app_state,
                file_path,
                rag_settings,
                force_text=force_text or job["kind"] == "obsidian",
                existing_doc_id=existing_doc_id,
                progress=report_progress,
            )
            processed += 1
            if result["status"] == "skipped":
                skipped += 1
            elif result["status"] == "failed":
                failures.append(f"{os.path.basename(file_path)}: {result.get('error', 'failed')}")
            await self._update_job(job_id, processed_files=processed, skipped_files=skipped)
            await self.event_bus.publish("document", result, job_id)

        if failures:
            await self._update_job(
                job_id,
                status="failed",
                error="; ".join(failures[:3]),
                current_file=None,
                stage="failed",
                stage_progress=100,
            )
        else:
            await self._update_job(
                job_id,
                status="succeeded",
                current_file=None,
                stage="complete",
                stage_progress=100,
            )
            if job["kind"] == "reindex":
                stale = refresh_document_staleness(self.app_state)
                self.app_state.reindex_required = bool(stale["stale_document_count"])

    async def _update_job(self, job_id: str, **fields) -> None:
        fields["updated_at"] = int(time.time())
        assignments = ", ".join(f"{key} = ?" for key in fields)
        params = tuple(fields.values()) + (job_id,)
        storage.execute(self.app_state.sqlite, f"UPDATE jobs SET {assignments} WHERE id = ?", params)
        job = self.get_job(job_id)
        if job.get("reindex_run_id"):
            storage.refresh_reindex_run(self.app_state.sqlite, job["reindex_run_id"])
        await self.event_bus.publish("job", job, job_id)
