from pydantic import BaseModel, Field, field_validator


class Message(BaseModel):
    role: str
    content: str


class RagSettings(BaseModel):
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

    @field_validator("chunk_size")
    @classmethod
    def validate_chunk_size(cls, value: int) -> int:
        if value < 256 or value > 8000:
            raise ValueError("chunk_size must be between 256 and 8000")
        return value

    @field_validator("chunk_overlap")
    @classmethod
    def validate_chunk_overlap(cls, value: int) -> int:
        if value < 0 or value > 2000:
            raise ValueError("chunk_overlap must be between 0 and 2000")
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
    reasoning_mode: str = "balanced"
    history: list[Message] = Field(default_factory=list)
    settings: RagSettings | None = None


class LoadModelRequest(BaseModel):
    model: str


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


class SourceChunk(BaseModel):
    rank: int
    source_id: str | None = None
    doc_id: str
    doc_name: str
    chunk_id: str
    parent_id: str | None = None
    score: float
    snippet: str
    vector_score: float | None = None
    lexical_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    subquery_id: str | None = None


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
