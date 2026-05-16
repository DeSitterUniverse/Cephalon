import { fireEvent, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SettingsPanel } from "./SettingsPanel";
import type { RagSettings } from "../../api";

const settings: RagSettings = {
  top_k: 20,
  rerank_top_n: 3,
  max_tokens: 512,
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

describe("SettingsPanel", () => {
  it("updates selected model and retrieval settings", async () => {
    const user = userEvent.setup();
    const setSelectedModel = vi.fn();
    const updateSettings = vi.fn();

    render(
      <SettingsPanel
        models={["small.gguf", "large.gguf"]}
        selectedModel="small.gguf"
        setSelectedModel={setSelectedModel}
        settings={settings}
        updateSettings={updateSettings}
      />,
    );

    await user.selectOptions(screen.getByLabelText("Model"), "large.gguf");
    expect(setSelectedModel).toHaveBeenCalledWith("large.gguf");

    fireEvent.change(screen.getByDisplayValue("512"), { target: { value: "1024" } });
    expect(updateSettings).toHaveBeenCalledWith({ ...settings, max_tokens: 1024 });
    expect(screen.getByText("Top K")).toBeInTheDocument();
  });
});
