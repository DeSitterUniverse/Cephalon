import os
import time
import tempfile
from pathlib import Path

import httpx


BASE_URL = os.getenv("CEPHALON_TEST_BASE_URL", "http://127.0.0.1:8765")
MODEL = os.getenv("CEPHALON_TEST_MODEL", "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf")


def build_fixture_docs() -> tuple[str, tempfile.TemporaryDirectory[str] | None]:
    configured = os.getenv("CEPHALON_TEST_DOCS")
    if configured:
        return os.path.abspath(configured), None
    temp_dir = tempfile.TemporaryDirectory()
    root = Path(temp_dir.name)
    (root / "breathing.md").write_text(
        "# Breathing note\n\nThe 4-7-8 method means inhaling for 4 seconds, holding for 7 seconds, and exhaling for 8 seconds.",
        encoding="utf-8",
    )
    (root / "stress.md").write_text(
        "# Stress note\n\nAshwagandha, rhodiola, magnesium, omega-3, and creatine are common supplement topics in stress notes.",
        encoding="utf-8",
    )
    return str(root), temp_dir


def wait_for_job(client: httpx.Client, job_id: str, timeout_s: int = 120) -> dict:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        jobs = client.get(f"{BASE_URL}/jobs").json().get("jobs", [])
        job = next((item for item in jobs if item["id"] == job_id), None)
        if job and job["status"] in {"succeeded", "failed"}:
            return job
        time.sleep(1)
    raise TimeoutError(f"Job {job_id} did not finish.")


def run_test():
    fixture_docs, temp_docs = build_fixture_docs()
    with httpx.Client(timeout=30) as client:
        print(f"Ingesting {fixture_docs}...")
        ingest = client.post(f"{BASE_URL}/ingest", json={"path": fixture_docs})
        ingest.raise_for_status()
        job = wait_for_job(client, ingest.json()["job_id"])
        print("Job:", job)

        if job["status"] != "succeeded":
            raise RuntimeError(f"Ingestion failed: {job}")

        print("\nQuerying: 'Tell me about the 4-7-8 method.'")
        with client.stream(
            "POST",
            f"{BASE_URL}/query",
            json={"prompt": "Tell me about the 4-7-8 method.", "history": [], "model": MODEL},
            timeout=180,
        ) as response:
            response.raise_for_status()
            for chunk in response.iter_text():
                print(chunk, end="", flush=True)
    if temp_docs:
        temp_docs.cleanup()


if __name__ == "__main__":
    run_test()
