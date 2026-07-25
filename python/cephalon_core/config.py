import os
import re
from dataclasses import dataclass


SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".csv",
    ".txt", ".md", ".json", ".canvas", ".py", ".js", ".ts", ".html",
}

ACTIVE_VECTOR_TABLE = "vectors_jina_v5_small_1024"
EMBEDDING_MODEL_ID = "jinaai/jina-embeddings-v5-text-small"
RERANKER_MODEL_ID = "jinaai/jina-reranker-v3"
EMBEDDING_DIMENSION = 1024
DEFAULT_EMBEDDER_ONNX_REPO = "s-lorin/jina-embeddings-v5-small-onnx"
DEFAULT_RERANKER_ONNX_REPO = "s-lorin/jina-reranker-v3-onnx"

DOCUMENT_ID_PATTERN = re.compile(
    r"^[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}$"
)


@dataclass
class RagDefaults:
    top_k: int = 20
    rerank_top_n: int = 3
    max_tokens: int = 4096
    temperature: float = 0.4
    parent_target_tokens: int = 520
    parent_max_tokens: int = 650
    child_target_tokens: int = 110
    child_max_tokens: int = 150
    child_overlap_tokens: int = 0
    context_tokens: int = 32768
    evidence_required: bool = False
    conversation_memory: bool = True
    trace_persistence: bool = True
    no_answer_min_confidence: float = 0.35
    no_answer_min_rerank_score: float = 0.15
    no_answer_min_vector_score: float = 0.05
    no_answer_min_source_count: int = 1


class Settings:
    def __init__(self) -> None:
        self.data_dir = os.path.abspath(os.path.expanduser(os.getenv("CEPHALON_DATA_DIR", "~/cephalon-data")))
        self.model_dir = os.path.abspath(os.path.expanduser(os.getenv("CEPHALON_MODEL_DIR", os.path.join(self.data_dir, "models"))))
        self.embedder_onnx_repo = os.getenv("CEPHALON_EMBEDDER_ONNX_REPO", DEFAULT_EMBEDDER_ONNX_REPO)
        self.reranker_onnx_repo = os.getenv("CEPHALON_RERANKER_ONNX_REPO", DEFAULT_RERANKER_ONNX_REPO)
        self.embedder_onnx_subfolder = os.getenv("CEPHALON_EMBEDDER_ONNX_SUBFOLDER", "")
        self.reranker_onnx_subfolder = os.getenv("CEPHALON_RERANKER_ONNX_SUBFOLDER", "")
        self.obsidian_vault_dir = os.path.abspath(os.path.expanduser(
            os.getenv("CEPHALON_OBSIDIAN_VAULT_DIR", "~/Documents/Obsidian Vault")
        ))
        self.host = os.getenv("CEPHALON_HOST", "127.0.0.1")
        self.port = int(os.getenv("CEPHALON_PORT", "8765"))
        self.llama_server_url = os.getenv("CEPHALON_LLAMA_SERVER_URL", "http://127.0.0.1:8080").rstrip("/")
        self.llama_server_model = os.getenv("CEPHALON_LLAMA_SERVER_MODEL", "External llama.cpp server").strip() or "External llama.cpp server"
        raw_server_context = os.getenv("CEPHALON_LLAMA_SERVER_CONTEXT_TOKENS", "").strip()
        self.llama_server_context_tokens = max(4096, int(raw_server_context)) if raw_server_context.isdigit() else None
        self.rag_defaults = RagDefaults(
            top_k=int(os.getenv("CEPHALON_TOP_K", "20")),
            rerank_top_n=int(os.getenv("CEPHALON_RERANK_TOP_N", "3")),
            max_tokens=int(os.getenv("CEPHALON_MAX_TOKENS", "4096")),
            temperature=float(os.getenv("CEPHALON_TEMPERATURE", "0.4")),
            parent_target_tokens=int(os.getenv("CEPHALON_PARENT_TARGET_TOKENS", "520")),
            parent_max_tokens=int(os.getenv("CEPHALON_PARENT_MAX_TOKENS", "650")),
            child_target_tokens=int(os.getenv("CEPHALON_CHILD_TARGET_TOKENS", "110")),
            child_max_tokens=int(os.getenv("CEPHALON_CHILD_MAX_TOKENS", "150")),
            child_overlap_tokens=int(os.getenv("CEPHALON_CHILD_OVERLAP_TOKENS", "0")),
            context_tokens=int(os.getenv("CEPHALON_CONTEXT_TOKENS", "32768")),
            evidence_required=os.getenv("CEPHALON_EVIDENCE_REQUIRED", "0") == "1",
            conversation_memory=os.getenv("CEPHALON_CONVERSATION_MEMORY", "1") != "0",
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
