from typing import Literal

from pydantic import BaseModel, Field, field_validator
from urllib.parse import urlparse


class Message(BaseModel):
    role: str
    content: str


class RagSettings(BaseModel):
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
    # Accepted for one compatibility release, but intentionally omitted from
    # serialized settings and all active chunking/config hashes.
    chunk_size: int | None = Field(default=None, exclude=True, deprecated=True)
    chunk_overlap: int | None = Field(default=None, exclude=True, deprecated=True)
    full_context: bool | None = Field(default=None, exclude=True, deprecated=True)

    @field_validator("top_k")
    @classmethod
    def validate_top_k(cls, value: int) -> int:
        if value < 1 or value > 100:
            raise ValueError("top_k must be between 1 and 100")
        return value

    @field_validator("rerank_top_n")
    @classmethod
    def validate_rerank_top_n(cls, value: int) -> int:
        if value < 1 or value > 20:
            raise ValueError("rerank_top_n must be between 1 and 20")
        return value

    @field_validator("max_tokens")
    @classmethod
    def validate_max_tokens(cls, value: int) -> int:
        if value < 16 or value > 8192:
            raise ValueError("max_tokens must be between 16 and 8192")
        return value

    @field_validator("temperature")
    @classmethod
    def validate_temperature(cls, value: float) -> float:
        if value < 0 or value > 2:
            raise ValueError("temperature must be between 0 and 2")
        return value

    @field_validator("parent_target_tokens", "parent_max_tokens", "child_target_tokens", "child_max_tokens")
    @classmethod
    def validate_chunk_limits(cls, value: int) -> int:
        if value < 32 or value > 4000:
            raise ValueError("chunk token limits must be between 32 and 4000")
        return value

    @field_validator("child_overlap_tokens")
    @classmethod
    def validate_child_overlap(cls, value: int) -> int:
        if value < 0 or value > 1000:
            raise ValueError("child_overlap_tokens must be between 0 and 1000")
        return value

    @field_validator("context_tokens")
    @classmethod
    def validate_context_tokens(cls, value: int) -> int:
        if value < 4096 or value > 131072:
            raise ValueError("context_tokens must be between 4096 and 131072")
        return value

    @field_validator("no_answer_min_confidence", "no_answer_min_rerank_score", "no_answer_min_vector_score")
    @classmethod
    def validate_threshold(cls, value: float) -> float:
        if value < 0 or value > 2:
            raise ValueError("thresholds must be between 0 and 2")
        return value

    @field_validator("no_answer_min_source_count")
    @classmethod
    def validate_source_count(cls, value: int) -> int:
        if value < 0 or value > 10:
            raise ValueError("no_answer_min_source_count must be between 0 and 10")
        return value


class IngestRequest(BaseModel):
    path: str
    force_text: bool = False


class QueryRequest(BaseModel):
    prompt: str
    model: str = ""
    conversation_id: str | None = None
    retrieval_scope: Literal["auto", "off", "low", "medium", "high"] = "medium"
    response_effort: Literal["quick", "balanced", "thorough"] = "balanced"
    history: list[Message] = Field(default_factory=list)
    settings: RagSettings | None = None


class LoadModelRequest(BaseModel):
    model: str = ""


class LlamaServerSettings(BaseModel):
    server_url: str = "http://127.0.0.1:8080"
    model_name: str = "External llama.cpp server"
    context_tokens: int | None = None

    @field_validator("server_url")
    @classmethod
    def validate_server_url(cls, value: str) -> str:
        clean = value.strip().rstrip("/")
        parsed = urlparse(clean)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("server_url must be a complete http:// or https:// URL, including its port.")
        return clean

    @field_validator("model_name")
    @classmethod
    def validate_model_name(cls, value: str) -> str:
        return value.strip()[:160] or "External llama.cpp server"

    @field_validator("context_tokens")
    @classmethod
    def validate_context_tokens(cls, value: int | None) -> int | None:
        if value is not None and not 4096 <= value <= 1_000_000:
            raise ValueError("context_tokens must be between 4096 and 1000000.")
        return value

class OnnxInstallLocalRequest(BaseModel):
    kind: str
    path: str


class OnnxDownloadRequest(BaseModel):
    kind: str = "all"
    repo_id: str | None = None
    subfolder: str | None = None


class DocumentUpdateRequest(BaseModel):
    display_name: str | None = None


class TagRequest(BaseModel):
    tag: str


class SourceAsset(BaseModel):
    asset_id: str
    page_number: int
    bounding_box: tuple[float, float, float, float] | None = None
    mime_type: str
    caption: str | None = None
    width: int | None = None
    height: int | None = None
    url: str


class SourceChunk(BaseModel):
    rank: int
    source_id: str | None = None
    doc_id: str
    doc_name: str
    chunk_id: str
    parent_id: str | None = None
    score: float
    snippet: str
    evidence_text: str | None = None
    vector_score: float | None = None
    lexical_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    subquery_id: str | None = None
    block_type: str | None = None
    section_heading: str | None = None
    heading_path: list[str] = Field(default_factory=list)
    page_number: int | None = None
    page_end: int | None = None
    block_index: int | None = None
    bounding_box: tuple[float, float, float, float] | None = None
    element_ids: list[str] = Field(default_factory=list)
    provenance: dict = Field(default_factory=dict)
    assets: list[SourceAsset] = Field(default_factory=list)


class QueryEnvelope(BaseModel):
    sources: list[SourceChunk]


class EvalItem(BaseModel):
    id: str
    question: str
    expected_doc_ids: list[str] = Field(default_factory=list)
    expected_chunk_ids: list[str] = Field(default_factory=list)
    reference_answer: str | None = None
    tags: list[str] = Field(default_factory=list)


class EvalRunRequest(BaseModel):
    evals: list[EvalItem]
    pipeline: str = "hybrid_rerank"
    top_k: int = 10
