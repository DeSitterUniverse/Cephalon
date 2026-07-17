import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SettingsPanel } from "./SettingsPanel";

describe("SettingsPanel", () => {
  it("keeps settings focused on model setup and appearance", async () => {
    const user = userEvent.setup();
    const onSaveLlamaServer = vi.fn();

    render(
      <SettingsPanel
        llamaServer={{ server_url: "http://127.0.0.1:8080", model_name: "External llama.cpp server", context_tokens: 32768 }}
        onSaveLlamaServer={onSaveLlamaServer}
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

    await user.clear(screen.getByLabelText("llama.cpp URL"));
    await user.type(screen.getByLabelText("llama.cpp URL"), "http://127.0.0.1:8081");
    await user.click(screen.getByRole("button", { name: "Save server settings" }));
    expect(onSaveLlamaServer).toHaveBeenCalledWith(expect.objectContaining({ server_url: "http://127.0.0.1:8081" }));
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
