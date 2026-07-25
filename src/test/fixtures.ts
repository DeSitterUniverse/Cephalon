import type { RagSettings } from "../api";

export const ragSettings = {
  top_k: 8,
  rerank_top_n: 3,
  max_tokens: 256,
  temperature: 0.4,
  parent_target_tokens: 520,
  parent_max_tokens: 650,
  child_target_tokens: 110,
  child_max_tokens: 150,
  child_overlap_tokens: 0,
  context_tokens: 32768,
  evidence_required: false,
  conversation_memory: true,
  trace_persistence: true,
  no_answer_min_confidence: 0.35,
  no_answer_min_rerank_score: 0.15,
  no_answer_min_vector_score: 0.05,
  no_answer_min_source_count: 1,
} satisfies RagSettings;
