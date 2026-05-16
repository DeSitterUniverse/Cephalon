export type Message = { role: "user" | "assistant"; content: string };
export type Document = {
  id: string;
  name: string;
  status: string;
  chunks: number;
  path: string;
  type?: string;
  size_bytes?: number;
  modified_at?: number | null;
  last_error?: string | null;
  last_indexed_at?: number | null;
  tags?: string[];
  chunk_preview?: Array<{
    id: string;
    index: number;
    text: string;
    block_type?: string | null;
    token_count?: number | null;
    char_count?: number | null;
    chunking_profile?: string | null;
    embedding_status?: string | null;
  }>;
};
export type Job = {
  id: string;
  kind: string;
  path: string;
  status: string;
  total_files: number;
  processed_files: number;
  skipped_files: number;
  current_file?: string | null;
  error?: string | null;
  created_at: number;
  updated_at: number;
};
export type RagSettings = {
  top_k: number;
  rerank_top_n: number;
  max_tokens: number;
  temperature: number;
  chunk_size: number;
  chunk_overlap: number;
  context_tokens: number;
  full_context: boolean;
  trace_persistence: boolean;
  no_answer_min_confidence: number;
  no_answer_min_rerank_score: number;
  no_answer_min_vector_score: number;
  no_answer_min_source_count: number;
};
export type SourceChunk = {
  rank: number;
  source_id?: string | null;
  doc_id: string;
  doc_name: string;
  chunk_id: string;
  parent_id?: string | null;
  score: number;
  snippet: string;
  vector_score?: number | null;
  lexical_score?: number | null;
  fusion_score?: number | null;
  rerank_score?: number | null;
  subquery_id?: string | null;
};
export type StoredMessage = Message & {
  id: string;
  model?: string | null;
  settings?: Record<string, unknown>;
  meta?: Record<string, unknown>;
  created_at: number;
  sources?: SourceChunk[];
};
export type CitationSupport = {
  chunk_id: string;
  source_id?: string | null;
  doc_id?: string;
  doc_name?: string;
  status: "supported" | "weak" | "unsupported";
  reason: string;
  score?: number | null;
  rerank_score?: number | null;
};
export type AnswerSupport = {
  status: "supported" | "weak" | "unsupported";
  citations: CitationSupport[];
};
export type RetrievalTraceSummary = {
  query_id: string;
  raw_query: string;
  normalized_query: string;
  retrieval_mode?: string;
  created_at: number;
  total_ms?: number | null;
  no_answer?: Record<string, unknown>;
};
export type RetrievalTrace = RetrievalTraceSummary & {
  subqueries: Array<{ id: string; text: string }>;
  latency: Record<string, number>;
  candidates: Record<"vector" | "bm25" | "fused" | "reranked" | "unused", Array<Record<string, unknown>>>;
  final_context: Array<Record<string, unknown>>;
};
export type IndexHealth = {
  document_count: number;
  chunk_count: number;
  embedded_chunk_count: number;
  stale_document_count: number;
  failed_ingestion_count: number;
  parse_warning_count: number;
  duplicate_chunk_count: number;
  duplicate_chunk_rate: number;
  average_chunk_length: number;
  median_chunk_length: number;
  min_chunk_length: number;
  max_chunk_length: number;
  documents_never_retrieved: number;
  index_size_bytes: number;
  embedding_model_counts: Record<string, number>;
  chunking_profile_counts: Record<string, number>;
  top_retrieved_documents: Array<{ id: string; name: string; retrieval_count: number }>;
};
export type EvalRun = {
  id: string;
  pipeline: string;
  top_k: number;
  created_at: number;
  aggregate: Record<string, number>;
  results?: Array<{ eval_id: string; question: string; metrics: Record<string, number> }>;
};
export type Conversation = {
  id: string;
  title: string;
  created_at: number;
  updated_at: number;
  messages?: StoredMessage[];
};
export type HealthResponse = {
  status: string;
  startup_error?: string | null;
  engines_ready: boolean;
  data_dir: string;
  model_dir: string;
  metrics_dir?: string;
  active_model?: string | null;
  active_context_tokens?: number | null;
  active_model_context_tokens?: number | null;
  llama_backend?: {
    vulkan_available?: boolean;
    vulkan_required?: boolean;
    vulkan_dll?: string | null;
    loaded_lib_base_path?: string | null;
    override_lib_path?: string | null;
  };
  retrieval_index?: {
    mode?: string;
    dense_available?: boolean;
    lexical_available?: boolean;
    error?: string | null;
    table?: string;
  };
  embedding?: { model_id: string; dimension: number; table: string };
};

export type ModelsResponse = {
  models: string[];
  auxiliary_gguf?: string[];
  model_dir?: string;
  active_model?: string | null;
  active_context_tokens?: number | null;
  active_model_context_tokens?: number | null;
  llama_backend?: HealthResponse["llama_backend"];
};
export type LoadModelResponse = {
  status: "loaded";
  active_model: string | null;
  active_context_tokens?: number | null;
  active_model_context_tokens?: number | null;
  llama_backend?: HealthResponse["llama_backend"];
};
type DocumentsResponse = { documents: Document[] };
type JobsResponse = { jobs: Job[] };
type ConversationsResponse = { conversations: Conversation[] };
type IngestResponse = { job_id: string; status: string; message?: string };
type RetrievalTracesResponse = { traces: RetrievalTraceSummary[] };
type EvalRunsResponse = { runs: EvalRun[] };

function configuredApiBaseUrl(): string {
  try {
    const storageValue = typeof window !== "undefined" && typeof window.localStorage?.getItem === "function"
      ? window.localStorage.getItem("cephalon.apiBaseUrl")
      : null;
    return storageValue || import.meta.env.VITE_CEPHALON_API_URL || "http://127.0.0.1:8765";
  } catch {
    return import.meta.env.VITE_CEPHALON_API_URL || "http://127.0.0.1:8765";
  }
}

const API_BASE_URL = configuredApiBaseUrl();

export class ApiError extends Error {
  status: number;

  constructor(message: string, status: number) {
    super(message);
    this.name = "ApiError";
    this.status = status;
  }
}

async function parseError(res: Response): Promise<string> {
  try {
    const data = await res.json();
    return data.detail || data.error || res.statusText;
  } catch {
    return res.statusText;
  }
}

async function requestJson<T>(path: string, init?: RequestInit): Promise<T> {
  const res = await fetch(`${API_BASE_URL}${path}`, {
    ...init,
    headers: {
      "Content-Type": "application/json",
      ...init?.headers,
    },
  });

  if (!res.ok) {
    throw new ApiError(await parseError(res), res.status);
  }

  return res.json() as Promise<T>;
}

export async function healthCheck(): Promise<boolean> {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    return res.ok;
  } catch {
    return false;
  }
}

export function getHealth(): Promise<HealthResponse> {
  return requestJson<HealthResponse>("/health");
}

export function getModels(): Promise<ModelsResponse> {
  return requestJson<ModelsResponse>("/models");
}

export function loadModel(model: string): Promise<LoadModelResponse> {
  return requestJson<LoadModelResponse>("/models/load", {
    method: "POST",
    body: JSON.stringify({ model }),
  });
}

export function getDocuments(): Promise<DocumentsResponse> {
  return requestJson<DocumentsResponse>("/documents");
}

export function getDocument(id: string): Promise<Document> {
  return requestJson<Document>(`/documents/${encodeURIComponent(id)}`);
}

export function updateDocument(id: string, display_name: string): Promise<Document> {
  return requestJson<Document>(`/documents/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: JSON.stringify({ display_name }),
  });
}

export function ingestPath(path: string, force_text = false): Promise<IngestResponse> {
  return requestJson<IngestResponse>("/ingest", {
    method: "POST",
    body: JSON.stringify({ path, force_text }),
  });
}

export function reindexDocument(id: string): Promise<IngestResponse> {
  return requestJson<IngestResponse>(`/documents/${encodeURIComponent(id)}/reindex`, {
    method: "POST",
  });
}

export function deleteDocument(id: string): Promise<{ status: string }> {
  return requestJson<{ status: string }>(`/documents/${encodeURIComponent(id)}`, {
    method: "DELETE",
  });
}

export function addDocumentTag(id: string, tag: string): Promise<{ status: string; tag: string }> {
  return requestJson<{ status: string; tag: string }>(`/documents/${encodeURIComponent(id)}/tags`, {
    method: "POST",
    body: JSON.stringify({ tag }),
  });
}

export function deleteDocumentTag(id: string, tag: string): Promise<{ status: string }> {
  return requestJson<{ status: string }>(`/documents/${encodeURIComponent(id)}/tags/${encodeURIComponent(tag)}`, {
    method: "DELETE",
  });
}

export function getJobs(): Promise<JobsResponse> {
  return requestJson<JobsResponse>("/jobs");
}

export function getConversations(): Promise<ConversationsResponse> {
  return requestJson<ConversationsResponse>("/conversations");
}

export function createConversation(): Promise<Conversation> {
  return requestJson<Conversation>("/conversations", { method: "POST" });
}

export function getConversation(id: string): Promise<Conversation> {
  return requestJson<Conversation>(`/conversations/${encodeURIComponent(id)}`);
}

export function renameConversation(id: string, title: string): Promise<Conversation> {
  return requestJson<Conversation>(`/conversations/${encodeURIComponent(id)}`, {
    method: "PATCH",
    body: JSON.stringify({ title }),
  });
}

export function deleteConversation(id: string): Promise<{ status: string }> {
  return requestJson<{ status: string }>(`/conversations/${encodeURIComponent(id)}`, { method: "DELETE" });
}

export function getSettings(): Promise<RagSettings> {
  return requestJson<RagSettings>("/settings");
}

export function updateSettings(settings: RagSettings): Promise<RagSettings> {
  return requestJson<RagSettings>("/settings", {
    method: "PUT",
    body: JSON.stringify(settings),
  });
}

export function exportMetrics(): Promise<{ status: string; path: string | null; error?: string | null }> {
  return requestJson<{ status: string; path: string | null; error?: string | null }>("/metrics/export", { method: "POST" });
}

export function getRetrievalTraces(): Promise<RetrievalTracesResponse> {
  return requestJson<RetrievalTracesResponse>("/retrieval/traces");
}

export function getRetrievalTrace(id: string): Promise<RetrievalTrace> {
  return requestJson<RetrievalTrace>(`/retrieval/traces/${encodeURIComponent(id)}`);
}

export function getIndexHealth(): Promise<IndexHealth> {
  return requestJson<IndexHealth>("/observability/index-health");
}

export function getEvalRuns(): Promise<EvalRunsResponse> {
  return requestJson<EvalRunsResponse>("/eval/runs");
}

export function runEval(evals: Array<{ id: string; question: string; expected_doc_ids: string[]; expected_chunk_ids?: string[] }>, pipeline = "hybrid_rerank", top_k = 10): Promise<EvalRun> {
  return requestJson<EvalRun>("/eval/runs", {
    method: "POST",
    body: JSON.stringify({ evals, pipeline, top_k }),
  });
}

export function eventsUrl(): string {
  return `${API_BASE_URL}/events`;
}

export async function queryModel(
  prompt: string,
  model: string,
  history: Message[],
  settings?: RagSettings,
  conversation_id?: string | null,
  reasoning_mode = "balanced",
): Promise<ReadableStream<Uint8Array>> {
  const res = await fetch(`${API_BASE_URL}/query`, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify({ prompt, model, history, settings, conversation_id, reasoning_mode }),
  });

  if (!res.ok) {
    throw new ApiError(await parseError(res), res.status);
  }
  if (!res.body) {
    throw new ApiError("No response body from local service.", res.status);
  }

  return res.body;
}
