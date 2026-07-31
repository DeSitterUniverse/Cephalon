"""Dispatch isolated reranker workers from the frozen backend executable.

PyInstaller runtime hooks execute before the bundled FastAPI entry point. The
normal source checkout can launch workers with ``python -m``; a frozen engine
cannot. These private command modes let the same signed backend executable
serve as its own worker without importing or starting FastAPI in the child.
"""

from __future__ import annotations

import sys


if len(sys.argv) > 1 and sys.argv[1] == "--cephalon-jina-gguf-worker":
    from cephalon_core.services.jina_reranker_worker import main

    raise SystemExit(main(*sys.argv[2:7]))

if len(sys.argv) > 1 and sys.argv[1] == "--cephalon-jina-transformers-worker":
    from cephalon_core.services.jina_reranker_transformers_worker import main

    raise SystemExit(main(sys.argv[2]))
