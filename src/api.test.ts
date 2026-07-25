import { afterEach, describe, expect, it, vi } from "vitest";
import { queryModel } from "./api";
import { ragSettings } from "./test/fixtures";

describe("api client", () => {
  afterEach(() => {
    vi.restoreAllMocks();
  });

  it("sends conversation id and retrieval scope with query streams", async () => {
    const fetchMock = vi.spyOn(globalThis, "fetch").mockResolvedValue(
      new Response("event: done\ndata: {\"ok\":true}\n\n", { status: 200 }),
    );

    const body = await queryModel(
      "summarize this",
      "local.gguf",
      [{ role: "user", content: "previous" }],
      ragSettings,
      "conversation-1",
      "high",
      "thorough",
    );

    expect(body).toBeTruthy();
    const [, init] = fetchMock.mock.calls[0];
    expect(JSON.parse(String(init?.body))).toMatchObject({
      prompt: "summarize this",
      model: "local.gguf",
      conversation_id: "conversation-1",
      retrieval_scope: "high",
      response_effort: "thorough",
    });
  });
});
