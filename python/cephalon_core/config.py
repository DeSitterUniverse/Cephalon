import os
import re
from dataclasses import dataclass


SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".csv",
    ".txt", ".md", ".json", ".py", ".js", ".ts", ".html",
}

ACTIVE_VECTOR_TABLE = "vectors_jina_v5_small_1024"
EMBEDDING_MODEL_ID = "jinaai/jina-embeddings-v5-text-small"
RERANKER_MODEL_ID = "jinaai/jina-reranker-v3"
EMBEDDING_DIMENSION = 1024

DOCUMENT_ID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


@dataclass
class RagDefaults:
    top_k: int = 20
    rerank_top_n: int = 3
    max_tokens: int = 512
    temperature: float = 0.4
    chunk_size: int = 1500
    chunk_overlap: int = 150
    context_tokens: int = 32768
    full_context: bool = False
    trace_persistence: bool = True
    no_answer_min_confidence: float = 0.35
    no_answer_min_rerank_score: float = 0.15
    no_answer_min_vector_score: float = 0.05
    no_answer_min_source_count: int = 1


class Settings:
    def __init__(self) -> None:
        self.data_dir = os.path.abspath(os.path.expanduser(os.getenv("CEPHALON_DATA_DIR", "~/cephalon-data")))
        self.model_dir = os.path.abspath(os.path.expanduser(os.getenv("CEPHALON_MODEL_DIR", os.path.join(self.data_dir, "models"))))
        self.host = os.getenv("CEPHALON_HOST", "127.0.0.1")
        self.port = int(os.getenv("CEPHALON_PORT", "8765"))
        self.rag_defaults = RagDefaults(
            top_k=int(os.getenv("CEPHALON_TOP_K", "20")),
            rerank_top_n=int(os.getenv("CEPHALON_RERANK_TOP_N", "3")),
            max_tokens=int(os.getenv("CEPHALON_MAX_TOKENS", "512")),
            temperature=float(os.getenv("CEPHALON_TEMPERATURE", "0.4")),
            chunk_size=int(os.getenv("CEPHALON_CHUNK_SIZE", "1500")),
            chunk_overlap=int(os.getenv("CEPHALON_CHUNK_OVERLAP", "150")),
            context_tokens=int(os.getenv("CEPHALON_CONTEXT_TOKENS", "32768")),
            full_context=os.getenv("CEPHALON_FULL_CONTEXT", "0") == "1",
            trace_persistence=os.getenv("CEPHALON_TRACE_PERSISTENCE", "1") != "0",
            no_answer_min_confidence=float(os.getenv("CEPHALON_NO_ANSWER_MIN_CONFIDENCE", "0.35")),
            no_answer_min_rerank_score=float(os.getenv("CEPHALON_NO_ANSWER_MIN_RERANK_SCORE", "0.15")),
            no_answer_min_vector_score=float(os.getenv("CEPHALON_NO_ANSWER_MIN_VECTOR_SCORE", "0.05")),
            no_answer_min_source_count=int(os.getenv("CEPHALON_NO_ANSWER_MIN_SOURCE_COUNT", "1")),
        )
        self.max_tokens = self.rag_defaults.max_tokens
        self.metrics_dir = os.path.abspath(os.path.expanduser(
            os.getenv("CEPHALON_METRICS_DIR", "~/Documents/Cephalon Metrics")
        ))
        self.cors_origins = self._parse_cors_origins(os.getenv("CEPHALON_CORS_ORIGINS"))

    @staticmethod
    def _parse_cors_origins(raw: str | None) -> list[str]:
        if raw:
            return [origin.strip() for origin in raw.split(",") if origin.strip()]
        return [
            "http://localhost:1420",
            "http://127.0.0.1:1420",
            "http://tauri.localhost",
            "https://tauri.localhost",
            "tauri://localhost",
        ]


settings = Settings()
