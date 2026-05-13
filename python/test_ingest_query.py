import os
import time

import httpx


BASE_URL = os.getenv("CEPHALON_TEST_BASE_URL", "http://127.0.0.1:8765")
MODEL = os.getenv("CEPHALON_TEST_MODEL", "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf")
TEST_DOCS = os.getenv("CEPHALON_TEST_DOCS", os.path.abspath("test_docs"))


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
    with httpx.Client(timeout=30) as client:
        print(f"Ingesting {TEST_DOCS}...")
        ingest = client.post(f"{BASE_URL}/ingest", json={"path": TEST_DOCS})
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


if __name__ == "__main__":
    run_test()
