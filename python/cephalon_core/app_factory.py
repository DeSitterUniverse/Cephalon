import os
import sys
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from . import storage
from .config import Settings, settings
from .events import EventBus
from .runtime import ModelRuntime
from .routes import router
from .services.jobs import JobManager
from .services import ingestion, jina_runtime, retrieval


def load_architecture_context() -> str:
    try:
        if getattr(sys, "frozen", False):
            base_dir = sys._MEIPASS
        else:
            base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
        target = os.path.join(base_dir, "AI_SYSTEM_AWARENESS.md")
        with open(target, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"[Error loading internal architecture specs: {e}]"


def create_app(app_settings: Settings | None = None) -> FastAPI:
    active_settings = app_settings or settings

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        os.makedirs(active_settings.data_dir, exist_ok=True)
        os.makedirs(active_settings.model_dir, exist_ok=True)
        os.environ["HF_HOME"] = os.path.expanduser("~/.cephalon/models")
        app.state.settings = active_settings
        app.state.architecture_context = load_architecture_context()
        app.state.llm = None
        app.state.model_runtime = ModelRuntime()
        app.state.embedding_runtime = ModelRuntime()
        app.state.reranker_runtime = ModelRuntime()
        app.state.active_model_name = None
        app.state.sqlite = storage.connect_sqlite(active_settings)
        app.state.llama_server_settings = storage.get_llama_server_settings(app.state.sqlite, active_settings)
        app.state.lance = storage.connect_lance(active_settings)
        app.state.startup_error = None
        jina_runtime.start(app.state)
        app.state.generated_index_backup = storage.clean_generated_vector_state(active_settings, app.state.lance)
        app.state.retrieval_index = retrieval.ensure_retrieval_index(app.state)
        try:
            app.state.index_staleness = ingestion.refresh_document_staleness(app.state)
            app.state.reindex_required = bool(app.state.index_staleness.get("stale_document_count"))
        except Exception as exc:
            app.state.index_staleness = {"error": str(exc)}
            app.state.reindex_required = True
        app.state.event_bus = EventBus(app.state.sqlite)
        app.state.job_manager = JobManager(app.state, app.state.event_bus)
        await app.state.job_manager.start()
        try:
            yield
        finally:
            await app.state.job_manager.stop()
            jina_runtime.stop(app.state)
            app.state.sqlite.close()

    app = FastAPI(lifespan=lifespan, title="Cephalon API")
    app.add_middleware(
        CORSMiddleware,
        allow_origins=active_settings.cors_origins,
        allow_methods=["GET", "POST", "PATCH", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["*"],
    )
    app.include_router(router)
    return app


app = create_app()
