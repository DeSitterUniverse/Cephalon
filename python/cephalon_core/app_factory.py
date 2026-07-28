import os
import sys
import json
import re
from contextlib import asynccontextmanager
import time

import numpy as np
import onnxruntime as ort
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from transformers import AutoTokenizer

from . import storage
from .config import EMBEDDING_DIMENSION, EMBEDDING_MODEL_ID, RERANKER_MODEL_ID, Settings, settings
from .events import EventBus
from .runtime import ModelRuntime
from .routes import router
from .services.jobs import JobManager
from .services import ingestion, jina_runtime, onnx_setup, retrieval


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
        return "Embedding and reranker models are not set up. Open Settings to download prepared ONNX models or select local ONNX folders."

    embed_meta = _read_model_meta(embed_path)
    reranker_meta = _read_model_meta(onnx_path)
    embed_error = _validate_embedder_meta(embed_meta)
    if embed_error:
        return embed_error

    reranker_error = _validate_reranker_meta(reranker_meta)
    if reranker_error:
        return reranker_error

    if not _reranker_export_validated(onnx_path):
        return "Jina reranker ONNX export exists but has not passed validation. Run scripts\\validate_onnx_models.py --mark."

    try:
        opts = ort.SessionOptions()
        app_state.reranker = ort.InferenceSession(model_file, sess_options=opts)
        app_state.tokenizer = AutoTokenizer.from_pretrained(onnx_path, fix_mistral_regex=True)
        app_state.embedder = ort.InferenceSession(embed_file, sess_options=opts)
        app_state.embed_tokenizer = AutoTokenizer.from_pretrained(embed_path, fix_mistral_regex=True)
        output_shape = app_state.embedder.get_outputs()[0].shape
        output_dim = output_shape[-1] if output_shape and isinstance(output_shape[-1], int) else EMBEDDING_DIMENSION
        if not isinstance(output_dim, int) or output_dim < 1:
            return "Embedding ONNX output does not expose a usable vector dimension."
        app_state.embedding_dim = output_dim
        app_state.embedding_model_id = str(embed_meta.get("model_id") or EMBEDDING_MODEL_ID)
        app_state.embedding_pooling = embed_meta.get("pooling", "embedding" if len(output_shape) == 2 else "auto")
        app_state.embedding_normalized = bool(embed_meta.get("normalized", True))
        app_state.embedding_fixed_sequence_length = embed_meta.get("fixed_sequence_length")
        app_state.active_vector_table = _vector_table_name(app_state.embedding_model_id, output_dim)
        app_state.reranker_model_id = str(reranker_meta.get("model_id") or RERANKER_MODEL_ID)
        app_state.reranker_score_mode = reranker_meta.get("score_mode", "auto")
        # Many otherwise compatible reranker exports have an internal reshape
        # fixed to one query/document pair.  Treat an unspecified value as one
        # so we do not intentionally trigger an ONNX error before retrying.
        app_state.reranker_batch_size = _reranker_batch_size(reranker_meta)
        app_state.onnx_warmup = _warm_onnx_engines(app_state)
        return None
    except Exception as exc:
        return f"Failed to load ONNX engines: {exc}"


def _warm_onnx_engines(app_state) -> dict:
    started = time.perf_counter()
    embed_kwargs = {"truncation": True, "return_tensors": "np"}
    fixed_length = getattr(app_state, "embedding_fixed_sequence_length", None)
    if fixed_length:
        embed_kwargs.update({"padding": "max_length", "max_length": int(fixed_length)})
    else:
        embed_kwargs["padding"] = True
    embed_inputs = app_state.embed_tokenizer("Cephalon warmup text", **embed_kwargs)
    embed_names = {item.name for item in app_state.embedder.get_inputs()}
    embed_ort = {"input_ids": embed_inputs["input_ids"].astype(np.int64)}
    if "attention_mask" in embed_inputs and "attention_mask" in embed_names:
        embed_ort["attention_mask"] = embed_inputs["attention_mask"].astype(np.int64)
    if "token_type_ids" in embed_inputs and "token_type_ids" in embed_names:
        embed_ort["token_type_ids"] = embed_inputs["token_type_ids"].astype(np.int64)
    app_state.embedder.run(None, embed_ort)

    rerank_inputs = app_state.tokenizer(
        [["warmup query", "warmup document"]],
        padding=True,
        truncation=True,
        return_tensors="np",
    )
    rerank_names = {item.name for item in app_state.reranker.get_inputs()}
    rerank_ort = {"input_ids": rerank_inputs["input_ids"].astype(np.int64)}
    if "attention_mask" in rerank_inputs and "attention_mask" in rerank_names:
        rerank_ort["attention_mask"] = rerank_inputs["attention_mask"].astype(np.int64)
    if "token_type_ids" in rerank_inputs and "token_type_ids" in rerank_names:
        rerank_ort["token_type_ids"] = rerank_inputs["token_type_ids"].astype(np.int64)
    app_state.reranker.run(None, rerank_ort)
    return {"ready": True, "warmup_ms": round((time.perf_counter() - started) * 1000, 2)}


def _reranker_export_validated(model_dir: str) -> bool:
    meta_file = _find_model_meta_file(model_dir)
    if not os.path.exists(meta_file):
        return False
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            return bool(json.load(f).get("validated"))
    except Exception:
        return False


def _read_model_meta(model_dir: str) -> dict:
    meta_file = _find_model_meta_file(model_dir)
    if not os.path.exists(meta_file):
        return {}
    try:
        with open(meta_file, "r", encoding="utf-8") as f:
            meta = json.load(f)
            return meta if isinstance(meta, dict) else {}
    except Exception:
        return {}


def _find_model_meta_file(model_dir: str) -> str:
    for filename in ("onnx_profile.json", "cephalon_onnx_meta.json"):
        candidate = os.path.join(model_dir, filename)
        if os.path.exists(candidate):
            return candidate
    return os.path.join(model_dir, "onnx_profile.json")


def _validate_embedder_meta(meta: dict) -> str | None:
    if meta.get("kind") not in {None, "embedder"}:
        return "ONNX profile is not an embedder profile."
    if not str(meta.get("model_id") or "").strip():
        return "Embedder metadata is missing model_id."
    dimension = meta.get("dimension")
    if dimension is not None and (not isinstance(dimension, int) or isinstance(dimension, bool) or dimension < 1):
        return "Embedder metadata has an invalid dimension."
    if meta.get("validated") is not True:
        return "Embedder ONNX export exists but has not passed validation. Run scripts\\validate_onnx_models.py --mark."
    return None


def _validate_reranker_meta(meta: dict) -> str | None:
    if meta.get("kind") not in {None, "reranker"}:
        return "ONNX profile is not a reranker profile."
    if not str(meta.get("model_id") or "").strip():
        return "Reranker metadata is missing model_id."
    if meta.get("validated") is not True:
        return "Reranker ONNX export exists but has not passed validation. Run scripts\\validate_onnx_models.py --mark."
    if not meta.get("score_mode"):
        return "Reranker validation metadata is missing score_mode. Run scripts\\validate_onnx_models.py --mark."
    batch_size = meta.get("max_batch_size")
    if batch_size is not None and (not isinstance(batch_size, int) or isinstance(batch_size, bool) or batch_size < 1):
        return "Jina reranker validation metadata has an invalid max_batch_size."
    return None


def _reranker_batch_size(meta: dict) -> int:
    """Return the validated reranker batch size, safely defaulting to one."""
    batch_size = meta.get("max_batch_size", 1)
    if not isinstance(batch_size, int) or isinstance(batch_size, bool):
        return 1
    return max(1, batch_size)


def _vector_table_name(model_id: str, dimension: int) -> str:
    if model_id == EMBEDDING_MODEL_ID and dimension == EMBEDDING_DIMENSION:
        return "vectors_jina_v5_small_1024"
    slug = re.sub(r"[^a-z0-9]+", "_", model_id.lower()).strip("_")[:48] or "embedding"
    return f"vectors_{slug}_{dimension}"


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
        # Legacy ONNX setup is intentionally not started.  It remains in
        # onnx_setup.py solely to document the migration from 1024-dim Jina
        # Small/pairwise v3 indexes; the fixed Jina Nano + v3.5 stack below is
        # the only retrieval runtime.
        app.state.startup_error = None
        jina_runtime.start(app.state)
        app.state.onnx_setup = {"legacy": True, "active": False}
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
