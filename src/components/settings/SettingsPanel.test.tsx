import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SettingsPanel } from "./SettingsPanel";

describe("SettingsPanel", () => {
  it("keeps settings focused on model setup and appearance", async () => {
    const user = userEvent.setup();
    const setSelectedModel = vi.fn();

    render(
      <SettingsPanel
        models={["small.gguf", "large.gguf"]}
        selectedModel="small.gguf"
        setSelectedModel={setSelectedModel}
        onnxStatus={{
          model_dir: "C:\\models",
          engines_ready: false,
          download_sources: {
            embedder: { repo_id: "example/embedder" },
            reranker: { repo_id: "example/reranker" },
          },
          embedder: { kind: "embedder", ok: true, path: "C:\\models\\embedder", missing: [], runtime_loaded: false },
          reranker: { kind: "reranker", ok: true, path: "C:\\models\\reranker", missing: [], runtime_loaded: false },
        }}
      />,
    );

    await user.selectOptions(screen.getByLabelText("Model"), "large.gguf");
    expect(setSelectedModel).toHaveBeenCalledWith("large.gguf");
    expect(screen.getByText("Appearance")).toBeInTheDocument();
    expect(screen.getByText("Embedding and reranking")).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Download default" })).toHaveLength(2);
    expect(screen.getAllByRole("button", { name: "Use local folder" })).toHaveLength(2);
    expect(screen.getByText("Default source: example/embedder")).toBeInTheDocument();
    expect(screen.getByText("C:\\models\\embedder")).toBeInTheDocument();
    expect(screen.queryByText(/Source:/)).not.toBeInTheDocument();
    expect(screen.queryByText("Temperature")).not.toBeInTheDocument();
    expect(screen.queryByText("Top K")).not.toBeInTheDocument();
  });
});
