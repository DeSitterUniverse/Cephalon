import os
import re
from dataclasses import dataclass


SUPPORTED_EXTENSIONS = {
    ".pdf", ".docx", ".pptx", ".xlsx", ".csv",
    ".txt", ".md", ".json", ".canvas", ".py", ".js", ".ts", ".html",
}

# Fixed retrieval stack.  These values are deliberately not user-configurable:
# mixing the former 1024-dim Jina Small vectors with Nano's 768-dim vectors
# makes LanceDB distances meaningless.
ACTIVE_VECTOR_TABLE = "vectors_jina_v5_nano_retrieval_768"
LEGACY_VECTOR_TABLE = "vectors_jina_v5_small_1024"
EMBEDDING_MODEL_ID = "jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF"
RERANKER_MODEL_ID = "jinaai/jina-reranker-v3.5"
EMBEDDING_DIMENSION = 768
EMBEDDER_GGUF_REPO = "jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF"
EMBEDDER_GGUF_FILE = "v5-nano-retrieval-Q8_0.gguf"
EMBEDDER_GGUF_SHA256 = "86b6e6279e9b9e71389f02a082764a2ac2b15a50e37482c26f98d69092f12442"
RERANKER_REPO = "jinaai/jina-reranker-v3.5-GGUF"
RERANKER_TRANSFORMERS_REPO = "jinaai/jina-reranker-v3.5"
RERANKER_REVISION = "884f7c67aa3ac24edb89064da8c7bfd03f4a90f5"
RERANKER_LLAMA_CPP_REVISION = "80c940e5a80555167c4ec37652deca6528810f91"
RERANKER_GGUF_FILE = "jina-reranker-v3.5-Q8_0.gguf"
RERANKER_GGUF_PRECISION = "Q8_0"
RERANKER_PROJECTOR_FILE = "projector.safetensors"
RERANKER_TOKENIZER_FILE = "tokenizer.json"
# Q8_0 preserves the fixed scientific retrieval metrics while reducing model
# storage and request wall time. BF16 remains the controlled rollback artifact.
RERANKER_FILE_SHA256 = {
    RERANKER_GGUF_FILE: "bedbedd688d18665448241f1aad78afb23a4476b89ae0867243e1c79aa4357b8",
    RERANKER_PROJECTOR_FILE: "b14c3d97315ca33490e630218c821640f183180fd971c5c3242f5b81aadcedf9",
    RERANKER_TOKENIZER_FILE: "4e95945ab0cef486709f760b81efcc7a6e75747f9165d13ead29159737455803",
}

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
        self.embedder_model_dir = os.path.join(self.model_dir, "jina-v5-nano-retrieval-q8_0")
        self.reranker_model_dir = os.path.join(self.model_dir, "jina-reranker-v3.5-gguf-q8_0")
        # Existing installations remain usable until the GGUF assets and a
        # feature-compatible llama-embedding binary are available.
        self.legacy_reranker_model_dir = os.path.join(self.model_dir, "jina-reranker-v3.5")
        self.embedder_server_url = os.getenv("CEPHALON_EMBEDDER_SERVER_URL", "http://127.0.0.1:8090").rstrip("/")
        self.embedder_server_port = int(os.getenv("CEPHALON_EMBEDDER_SERVER_PORT", "8090"))
        self.llama_server_bin = os.getenv("CEPHALON_LLAMA_SERVER_BIN", r"C:\\AI\\llama.cpp\\build\\bin\\Release\\llama-server.exe")
        # This workstation exposes the RX 6700 XT as Vulkan0. Keep the
        # settings overridable for a different local llama.cpp device name.
        self.embedder_device = os.getenv("CEPHALON_EMBEDDER_DEVICE", "Vulkan0").strip() or "Vulkan0"
        self.embedder_gpu_layers = max(0, int(os.getenv("CEPHALON_EMBEDDER_GPU_LAYERS", "999")))
        self.embedder_batch_size = max(1, int(os.getenv("CEPHALON_EMBEDDER_BATCH_SIZE", "16")))
        # Nano batches 16 document chunks per request. llama.cpp's default
        # physical batch of 512 tokens rejects valid combined requests.
        self.embedder_physical_batch_size = max(512, int(os.getenv("CEPHALON_EMBEDDER_PHYSICAL_BATCH_SIZE", "4096")))
        default_embedding_bin = os.path.join(os.path.dirname(self.llama_server_bin), "llama-embedding.exe")
        self.reranker_llama_embedding_bin = os.getenv(
            "CEPHALON_RERANKER_LLAMA_EMBEDDING_BIN",
            default_embedding_bin,
        )
        self.reranker_device = os.getenv("CEPHALON_RERANKER_DEVICE", self.embedder_device).strip() or self.embedder_device
        self.reranker_gpu_layers = max(0, int(os.getenv("CEPHALON_RERANKER_GPU_LAYERS", "999")))
        # 65,536 matches Jina's GGUF reference default. It is large enough for
        # Cephalon's bounded candidate set without paying the KV-cache cost of
        # the model's full 131,072-token architectural limit.
        self.reranker_max_context_tokens = min(
            131072,
            max(4096, int(os.getenv("CEPHALON_RERANKER_MAX_CONTEXT_TOKENS", "65536"))),
        )
        self.reranker_backend = os.getenv("CEPHALON_RERANKER_BACKEND", "auto").strip().lower() or "auto"
        if self.reranker_backend not in {"auto", "gguf", "transformers"}:
            self.reranker_backend = "auto"
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
        return [
            "http://localhost:1420",
            "http://127.0.0.1:1420",
            "http://tauri.localhost",
            "https://tauri.localhost",
            "tauri://localhost",
        ]


settings = Settings()
