import { cleanup, render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, describe, expect, it, vi } from "vitest";
import { ChatPanel } from "./ChatPanel";
import type { Conversation, RagSettings } from "../../api";
import { queryModel } from "../../api";
import { useUiStore } from "../../store";

vi.mock("../../api", async importOriginal => {
  const actual = await importOriginal<typeof import("../../api")>();
  return {
    ...actual,
    queryModel: vi.fn(),
  };
});

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

const conversation: Conversation = {
  id: "conversation-1",
  title: "Stress notes",
  created_at: 1778755000,
  updated_at: 1778755300,
  messages: [
    { id: "m1", role: "user", content: "What helps stress?", created_at: 1778755001 },
    {
      id: "m2",
      role: "assistant",
      content: "Breathing helps. [[src:S1]]",
      created_at: 1778755002,
      sources: [{
        rank: 1,
        source_id: "S1",
        doc_id: "doc-1",
        doc_name: "stress.md",
        chunk_id: "doc-1_0",
        score: 0.9,
        snippet: "4-7-8 breathing resets the nervous system.",
      }],
    },
  ],
};

describe("ChatPanel", () => {
  afterEach(() => {
    cleanup();
    vi.mocked(queryModel).mockReset();
  });

  it("restores saved chat messages and exposes per-message source badges", async () => {
    const user = userEvent.setup();
    useUiStore.setState({ selectedSources: [], rightPanel: "jobs" });

    render(
      <ChatPanel
        selectedModel="local.gguf"
        modelReady
        settings={settings}
        conversation={conversation}
        selectedConversationId="conversation-1"
      />,
    );

    expect(screen.getByText("What helps stress?")).toBeInTheDocument();
    expect(screen.getByText("Breathing helps.")).toBeInTheDocument();
    expect(screen.getByText("S1")).toBeInTheDocument();

    await user.click(screen.getByText("1 source"));
    expect(useUiStore.getState().selectedSources[0].doc_name).toBe("stress.md");
    expect(useUiStore.getState().rightPanel).toBe("sources");
  });

  it("shows retrieval scope control next to the composer", () => {
    render(<ChatPanel selectedModel="local.gguf" modelReady settings={settings} />);

    expect(screen.getByLabelText("Retrieval scope")).toHaveValue("medium");
    expect(screen.getByText("Low retrieval")).toBeInTheDocument();
    expect(screen.getByText("High retrieval")).toBeInTheDocument();
    expect(screen.getByLabelText("Response effort")).toHaveValue("balanced");
    expect(screen.getByText("Quick response")).toBeInTheDocument();
    expect(screen.getByText("Thorough response")).toBeInTheDocument();
  });

  it("keeps the first streamed answer visible after the conversation event", async () => {
    const user = userEvent.setup();
    const onConversationSelected = vi.fn();
    const stream = new ReadableStream<Uint8Array>({
      start(controller) {
        controller.enqueue(new TextEncoder().encode([
          "event: conversation",
          "data: {\"conversation_id\":\"new-chat\"}",
          "",
          "event: token",
          "data: {\"text\":\"First answer\"}",
          "",
          "event: done",
          "data: {\"ok\":true}",
          "",
        ].join("\n")));
        controller.close();
      },
    });
    vi.mocked(queryModel).mockResolvedValue(stream);

    render(
      <ChatPanel
        selectedModel="local.gguf"
        modelReady
        settings={settings}
        selectedConversationId={null}
        onConversationSelected={onConversationSelected}
      />,
    );

    await user.type(screen.getByPlaceholderText("Search, compare, summarize..."), "hello");
    await user.click(screen.getByRole("button", { name: /run/i }));

    await waitFor(() => expect(screen.getByText("First answer")).toBeInTheDocument());
    expect(onConversationSelected).toHaveBeenCalledWith("new-chat");
  });

  it("supports multiline input and submits with Enter but not Shift+Enter", async () => {
    const user = userEvent.setup();
    vi.mocked(queryModel).mockResolvedValue(new ReadableStream({
      start(controller) {
        controller.close();
      },
    }));

    render(<ChatPanel selectedModel="local.gguf" modelReady settings={settings} />);
    const composer = screen.getByRole("textbox", { name: "Message" });

    await user.type(composer, "first line{Shift>}{Enter}{/Shift}second line");
    expect(composer).toHaveValue("first line\nsecond line");
    expect(queryModel).not.toHaveBeenCalled();

    await user.type(composer, "{Enter}");
    await waitFor(() => expect(queryModel).toHaveBeenCalledTimes(1));
    expect(queryModel).toHaveBeenCalledWith(
      "first line\nsecond line",
      "local.gguf",
      [],
      settings,
      undefined,
      "medium",
      "balanced",
      expect.any(AbortSignal),
    );
  });

  it("stops an active stream while preserving partial output", async () => {
    const user = userEvent.setup();
    vi.mocked(queryModel).mockImplementation(async (...args) => {
      const signal = args[7];
      return new ReadableStream<Uint8Array>({
        start(controller) {
          controller.enqueue(new TextEncoder().encode("event: token\ndata: {\"text\":\"Partial answer\"}\n\n"));
          signal?.addEventListener("abort", () => controller.error(new DOMException("Stopped", "AbortError")));
        },
      });
    });

    render(<ChatPanel selectedModel="local.gguf" modelReady settings={settings} />);
    await user.type(screen.getByRole("textbox", { name: "Message" }), "long request");
    await user.keyboard("{Enter}");
    await waitFor(() => expect(screen.getByText("Partial answer")).toBeInTheDocument());

    await user.click(screen.getByRole("button", { name: "Stop response" }));
    await waitFor(() => expect(screen.queryByRole("button", { name: "Stop response" })).not.toBeInTheDocument());
    expect(screen.getByText("Partial answer")).toBeInTheDocument();
  });

  it("copies and regenerates an assistant answer", async () => {
    const user = userEvent.setup();
    const writeText = vi.fn().mockResolvedValue(undefined);
    Object.defineProperty(navigator, "clipboard", { configurable: true, value: { writeText } });
    vi.mocked(queryModel).mockResolvedValue(new ReadableStream({
      start(controller) {
        controller.enqueue(new TextEncoder().encode("event: token\ndata: {\"text\":\"Updated answer\"}\n\n"));
        controller.close();
      },
    }));

    render(
      <ChatPanel
        selectedModel="local.gguf"
        modelReady
        settings={settings}
        conversation={conversation}
        selectedConversationId="conversation-1"
      />,
    );

    await user.click(screen.getByRole("button", { name: "Copy answer" }));
    expect(writeText).toHaveBeenCalledWith("Breathing helps. [[src:S1]]");

    await user.click(screen.getByRole("button", { name: "Regenerate answer" }));
    await waitFor(() => expect(screen.getByText("Updated answer")).toBeInTheDocument());
    expect(queryModel).toHaveBeenCalledTimes(1);
  });
});
