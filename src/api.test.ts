import { afterEach, describe, expect, it, vi } from "vitest";
import { queryModel, type RagSettings } from "./api";

const settings: RagSettings = {
  top_k: 8,
  rerank_top_n: 3,
  max_tokens: 256,
  temperature: 0.4,
  chunk_size: 1500,
  chunk_overlap: 150,
  context_tokens: 32768,
  full_context: false,
  trace_persistence: true,
  no_answer_min_confidence: 0.35,
  no_answer_min_rerank_score: 0.15,
  no_answer_min_vector_score: 0.05,
  no_answer_min_source_count: 1,
};

describe("api client", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("sends conversation id and reasoning mode with query streams", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("event: done\ndata: {\"ok\":true}\n\n", { status: 200 }),
    );

    const body = await queryModel(
      "summarize this",
      "local.gguf",
      [{ role: "user", content: "previous" }],
      settings,
      "conversation-1",
      "deep",
    );

    expect(body).toBeTruthy();
    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(String(init?.body))).toMatchObject({
      prompt: "summarize this",
      model: "local.gguf",
      conversation_id: "conversation-1",
      reasoning_mode: "deep",
    });
  });
});
