import os

import httpx


BASE_URL = os.getenv("CEPHALON_TEST_BASE_URL", "http://127.0.0.1:8765")
MODEL = os.getenv("CEPHALON_TEST_MODEL", "NVIDIA-Nemotron3-Nano-4B-Q4_K_M.gguf")


def run_test():
    print("\nQuerying: 'Tell me about the 4-7-8 method.'")
    with httpx.stream(
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
