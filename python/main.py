import uvicorn

from cephalon_core.app_factory import app, create_app, load_architecture_context
from cephalon_core.config import Settings, settings
from cephalon_core.routes import chat_and_remember, delete_document, get_documents, get_models, health, ingest_endpoint
from cephalon_core.schemas import IngestRequest, Message, QueryRequest, RagSettings
from cephalon_core.services.documents import extract_text, find_existing_doc_by_hash, get_file_hash
from cephalon_core.services.models import load_llm
from cephalon_core.services.retrieval import get_embedding, save_permanent_memory
from cephalon_core.storage import (
    SQLITE_LOCK,
    VECTOR_SCHEMA as schema,
    connect_lance,
    connect_sqlite,
    fetchall as sqlite_fetchall,
    fetchone as sqlite_fetchone,
    run_migrations,
)
from cephalon_core.validators import is_supported_file, validate_document_id
from cephalon_core.validators import validate_model_filename as _validate_model_filename

DB_PATH = settings.data_dir
MODEL_DIR = settings.model_dir
ARCHITECTURE_CONTEXT = load_architecture_context()


def validate_model_filename(model_filename: str) -> str:
    return _validate_model_filename(model_filename, MODEL_DIR)


def _init_db(conn):
    return run_migrations(conn, settings)


if __name__ == "__main__":
    uvicorn.run(app, host=settings.host, port=settings.port, log_level="info")
