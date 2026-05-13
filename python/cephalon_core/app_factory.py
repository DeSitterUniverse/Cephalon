import os
import shutil
import sys
import json
from contextlib import asynccontextmanager

import onnxruntime as ort
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer

from . import storage
from .config import EMBEDDING_DIMENSION, EMBEDDING_MODEL_ID, RERANKER_MODEL_ID, Settings, settings
from .events import EventBus
from .routes import router
from .services.jobs import JobManager
from .services import retrieval


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


def load_onnx_engines(app_state) -> str | None:
    onnx_path = os.path.join(app_state.settings.model_dir, "reranker")
    embed_path = os.path.join(app_state.settings.model_dir, "embedder")
    model_file = os.path.join(onnx_path, "model.onnx")
    embed_file = os.path.join(embed_path, "model.onnx")

    if not os.path.exists(model_file) or not os.path.exists(embed_file):
        if getattr(sys, "frozen", False):
            bundled_base = os.path.join(sys._MEIPASS, "onnx_models")
            bundled_reranker = os.path.join(bundled_base, "reranker")
            bundled_embed = os.path.join(bundled_base, "embedder")
            if os.path.exists(bundled_reranker) and os.path.exists(bundled_embed):
                if not os.path.exists(onnx_path):
                    shutil.copytree(bundled_reranker, onnx_path)
                if not os.path.exists(embed_path):
                    shutil.copytree(bundled_embed, embed_path)
            else:
                return "Bundled ONNX models were not found."
        else:
            return "Native ONNX models were not found. Run export_onnx.py once to generate them."

    if not _reranker_export_validated(onnx_path):
        return "Jina reranker ONNX export exists but has not passed validation. Run scripts\\validate_onnx_models.py --mark."

    try:
        opts = ort.SessionOptions()
        app_state.reranker = ort.InferenceSession(model_file, sess_options=opts)
        app_state.tokenizer = AutoTokenizer.from_pretrained(onnx_path)
        app_state.embedder = ort.InferenceSession(embed_file, sess_options=opts)
        app_state.embed_tokenizer = AutoTokenizer.from_pretrained(embed_path)
        output_shape = app_state.embedder.get_outputs()[0].shape
        output_dim = output_shape[-1] if output_shape and isinstance(output_shape[-1], int) else EMBEDDING_DIMENSION
        if output_dim != EMBEDDING_DIMENSION:
            return f"Embedding dimension mismatch: got {output_dim}, expected {EMBEDDING_DIMENSION}. Re-export Jina v5 small and rebuild indexes."
        app_state.embedding_dim = output_dim
        meta_file = os.path.join(embed_path, "cephalon_onnx_meta.json")
        embed_meta = {}
        if os.path.exists(meta_file):
            with open(meta_file, "r", encoding="utf-8") as f:
                embed_meta = json.load(f)
        app_state.embedding_model_id = embed_meta.get("model_id") or EMBEDDING_MODEL_ID
        app_state.embedding_pooling = embed_meta.get("pooling", "embedding" if len(output_shape) == 2 else "cls")
        app_state.embedding_fixed_sequence_length = embed_meta.get("fixed_sequence_length")
        reranker_meta = _read_model_meta(onnx_path)
        app_state.reranker_model_id = reranker_meta.get("model_id") or RERANKER_MODEL_ID
        app_state.reranker_score_mode = reranker_meta.get("score_mode", "auto")
        return None
    except Exception as exc:
        return f"Failed to load ONNX engines: {exc}"


def _reranker_export_validated(model_dir: str) -> bool:
    meta_file = os.path.join(model_dir, "cephalon_onnx_meta.json")
    if not os.path.exists(meta_file):
        return False
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            return bool(json.load(f).get("validated"))
    except Exception:
        return False


def _read_model_meta(model_dir: str) -> dict:
    meta_file = os.path.join(model_dir, "cephalon_onnx_meta.json")
    if not os.path.exists(meta_file):
        return {}
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return {}


def create_app(app_settings: Settings | None = None) -> FastAPI:
    active_settings = app_settings or settings
    os.makedirs(active_settings.data_dir, exist_ok=True)
    os.makedirs(active_settings.model_dir, exist_ok=True)
    os.environ["HF_HOME"] = os.path.expanduser("~/.cephalon/models")

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.settings = active_settings
        app.state.architecture_context = load_architecture_context()
        app.state.llm = None
        app.state.active_model_name = None
        app.state.sqlite = storage.connect_sqlite(active_settings)
        app.state.lance = storage.connect_lance(active_settings)
        app.state.startup_error = load_onnx_engines(app.state)
        app.state.generated_index_backup = storage.clean_generated_vector_state(active_settings, app.state.lance)
        app.state.retrieval_index = retrieval.ensure_retrieval_index(app.state)
        app.state.event_bus = EventBus(app.state.sqlite)
        app.state.job_manager = JobManager(app.state, app.state.event_bus)
        await app.state.job_manager.start()
        try:
            yield
        finally:
            await app.state.job_manager.stop()
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
