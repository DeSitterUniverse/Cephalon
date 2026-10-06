import hashlib
import json
import os
import re
from dataclasses import dataclass


SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".csv",
    ".txt", ".md", ".json", ".canvas", ".py", ".js", ".ts", ".html",
}

# A new table is mandatory: vectors from distinct models cannot be compared.
ACTIVE_VECTOR_TABLE = "vectors_embeddinggemma2_text_270m_q8_0_768"
EMBEDDING_MODEL_ID = "google/embeddinggemma-2"
EMBEDDING_MODEL_REVISION = "914f7f89142e33e77833254d9c9b90c3cef7303b"
EMBEDDING_DIMENSION = 768
EMBEDDING_QUANTIZATION = "Q8_0"
EMBEDDING_POOLING = "mean"
EMBEDDING_PROMPT_SCHEME = "embeddinggemma2-retrieval-v1"
EMBEDDING_MAX_TOKENS = 8192
EMBEDDING_GGUF_REPO = "unsloth/embeddinggemma-2-GGUF"
EMBEDDING_GGUF_REVISION = "ba3888272494be64ed88c9eb536ddc61a1be73d5"
EMBEDDER_MODEL_FILE = "embeddinggemma-2-Q8_0.gguf"
EMBEDDING_GGUF_SHA256 = "6f1bd4ac6c5df7444f9cca7ca36cafe6cfa34cd6f49fefb1e0b4be8143aed8bc"
EMBEDDING_LLAMA_RELEASE = "b11456"
RERANKER_MODEL_ID = "jinaai/jina-reranker-v3.5"
RERANKER_REPO = "jinaai/jina-reranker-v3.5-GGUF"
RERANKER_REVISION = "884f7c67aa3ac24edb89064da8c7bfd03f4a90f5"
RERANKER_LLAMA_CPP_REVISION = "491219a2bfa0a5a51acfc6a233bea911515b2134"
RERANKER_GGUF_FILE = "jina-reranker-v3.5-Q8_0.gguf"
RERANKER_PRECISION = "Q8_0"
RERANKER_SCORE_TYPE = "cosine"
RERANKER_FILE_SHA256 = {
    RERANKER_GGUF_FILE: "bedbedd688d18665448241f1aad78afb23a4476b89ae0867243e1c79aa4357b8",
    "projector.safetensors": "b14c3d97315ca33490e630218c821640f183180fd971c5c3242f5b81aadcedf9",
    "tokenizer.json": "4e95945ab0cef486709f760b81efcc7a6e75747f9165d13ead29159737455803",
}
RERANKER_WORKER_SHA256 = "ea5e122fdcb2891fc0675568dbb0bf0f641046d7bd2a1438b2eacc0d354fc46e"
RERANKER_PACKAGES = {"numpy": "2.5.1", "tokenizers": "0.22.2"}


def embedding_vector_space() -> dict[str, str | int | bool]:
    return {
        "model_id": EMBEDDING_MODEL_ID,
        "revision": EMBEDDING_MODEL_REVISION,
        "native_dimension": EMBEDDING_DIMENSION,
        "index_dimension": EMBEDDING_DIMENSION,
        "quantization": EMBEDDING_QUANTIZATION,
        "gguf_repo": EMBEDDING_GGUF_REPO,
        "gguf_revision": EMBEDDING_GGUF_REVISION,
        "gguf_sha256": EMBEDDING_GGUF_SHA256,
        "runtime": f"llama.cpp/{EMBEDDING_LLAMA_RELEASE}",
        "pooling": EMBEDDING_POOLING,
        "normalization": "l2",
        "tokenizer": "gguf",
        "modalities": "text",
        "max_tokens": EMBEDDING_MAX_TOKENS,
        "include_prompt": True,
        "add_special_tokens": True,
        "parse_special_tokens": False,
        "truncation": "right",
        "pooling_accumulation": "fp32",
        "prompt_scheme": EMBEDDING_PROMPT_SCHEME,
    }


def embedding_config_hash() -> str:
    return hashlib.sha256(json.dumps(embedding_vector_space(), sort_keys=True).encode()).hexdigest()

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
    hierarchical_context: bool = True
    layout_evidence: bool = True
    evidence_ledger: bool = True
    coverage_selection: bool = True
    gap_retrieval: bool = True
    verified_answer_repair: bool = True
    no_answer_min_confidence: float = 0.35
    no_answer_min_rerank_score: float = 0.15
    no_answer_min_vector_score: float = 0.05
    no_answer_min_source_count: int = 1


class Settings:
    def __init__(self) -> None:
        self.data_dir = os.path.abspath(os.path.expanduser(os.getenv("CEPHALON_DATA_DIR", "~/cephalon-data")))
        self.model_dir = os.path.abspath(os.path.expanduser(os.getenv("CEPHALON_MODEL_DIR", os.path.join(self.data_dir, "models"))))
        self.embedder_model_dir = os.path.join(self.model_dir, "embeddinggemma-2-text-270m-q8_0")
        self.reranker_model_dir = os.path.join(self.model_dir, "jina-reranker-v3.5-gguf-q8_0")
        self.embedder_server_url = os.getenv("CEPHALON_EMBEDDER_SERVER_URL", "http://127.0.0.1:8090").rstrip("/")
        self.embedder_server_port = int(os.getenv("CEPHALON_EMBEDDER_SERVER_PORT", "8090"))
        runtime_dir = os.path.join(self.data_dir, "runtimes", f"llama-{EMBEDDING_LLAMA_RELEASE}-vulkan")
        self.embedder_llama_server_bin = os.path.abspath(os.path.expanduser(os.getenv(
            "CEPHALON_EMBEDDER_LLAMA_SERVER_BIN", os.path.join(runtime_dir, "llama-server.exe" if os.name == "nt" else "llama-server")
        )))
        self.embedder_gpu_layers = int(os.getenv("CEPHALON_EMBEDDER_GPU_LAYERS", "99"))
        self.embedder_device = os.getenv("CEPHALON_EMBEDDER_DEVICE", "").strip()
        self.embedder_batch_size = max(1, int(os.getenv("CEPHALON_EMBEDDER_BATCH_SIZE", "16")))
        self.reranker_worker_url = os.getenv("CEPHALON_RERANKER_WORKER_URL", "http://127.0.0.1:8091").rstrip("/")
        self.reranker_worker_port = int(os.getenv("CEPHALON_RERANKER_WORKER_PORT", "8091"))
        worker_dir = os.path.join(self.data_dir, "runtimes", "jina35-python")
        self.reranker_python_bin = os.path.abspath(os.path.expanduser(os.getenv(
            "CEPHALON_RERANKER_PYTHON_BIN",
            os.path.join(worker_dir, "Scripts", "python.exe") if os.name == "nt" else os.path.join(worker_dir, "bin", "python"),
        )))
        jina_runtime_dir = os.path.join(self.data_dir, "runtimes", f"jina35-pr26286-{RERANKER_LLAMA_CPP_REVISION[:8]}")
        self.reranker_llama_embedding_bin = os.path.abspath(os.path.expanduser(os.getenv(
            "CEPHALON_RERANKER_LLAMA_EMBEDDING_BIN",
            os.path.join(jina_runtime_dir, "llama-embedding.exe" if os.name == "nt" else "llama-embedding"),
        )))
        self.reranker_device = os.getenv("CEPHALON_RERANKER_DEVICE", "Vulkan0").strip()
        self.reranker_gpu_layers = max(0, int(os.getenv("CEPHALON_RERANKER_GPU_LAYERS", "99")))
        self.reranker_max_context_tokens = int(os.getenv("CEPHALON_RERANKER_MAX_CONTEXT_TOKENS", "32768"))
        if not 16384 <= self.reranker_max_context_tokens <= 131072:
            raise ValueError("CEPHALON_RERANKER_MAX_CONTEXT_TOKENS must be between 16384 and 131072")
        self.typed_tables = os.getenv("CEPHALON_TYPED_TABLES", "1") != "0"
        self.table_execution = os.getenv("CEPHALON_TABLE_EXECUTION", "1") != "0"
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
            hierarchical_context=os.getenv("CEPHALON_HIERARCHICAL_CONTEXT", "1") != "0",
            layout_evidence=os.getenv("CEPHALON_LAYOUT_EVIDENCE", "1") != "0",
            evidence_ledger=os.getenv("CEPHALON_EVIDENCE_LEDGER", "1") != "0",
            coverage_selection=os.getenv("CEPHALON_COVERAGE_SELECTION", "1") != "0",
            gap_retrieval=os.getenv("CEPHALON_GAP_RETRIEVAL", "1") != "0",
            verified_answer_repair=os.getenv("CEPHALON_VERIFIED_ANSWER_REPAIR", "1") != "0",
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
        return []


settings = Settings()
