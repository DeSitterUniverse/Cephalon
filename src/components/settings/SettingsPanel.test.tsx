import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { SettingsPanel } from "./SettingsPanel";

describe("SettingsPanel", () => {
  it("keeps settings focused on model setup and appearance", async () => {
    const user = userEvent.setup();
    const onSaveLlamaServer = vi.fn();
    const onDownloadModel = vi.fn();
    const onVerifyModel = vi.fn();
    const onOpenModel = vi.fn();
    const onDeleteModel = vi.fn();
    const onReindex = vi.fn();

    render(
      <SettingsPanel
        llamaServer={{ server_url: "http://127.0.0.1:8080", model_name: "External llama.cpp server", context_tokens: 32768 }}
        onSaveLlamaServer={onSaveLlamaServer}
        retrievalStatus={{
          fixed_stack: true,
          reindex_required: true,
          embedder: {
            kind: "embedder",
            name: "Jina Embeddings v5 Nano Retrieval",
            model_id: "jinaai/jina-embeddings-v5-text-nano-retrieval-GGUF",
            revision: "main",
            path: "C:\\models\\jina-v5-nano-retrieval-q8_0",
            installed: true,
            dimension: 768,
            runtime: { status: "running", port: 8090, pid: 101 },
          },
          reranker: {
            kind: "reranker",
            name: "Jina Reranker v3.5",
            model_id: "jinaai/jina-reranker-v3.5",
            revision: "main",
            path: "C:\\models\\jina-reranker-v3.5",
            installed: true,
            trust_remote_code: true,
            runtime: { status: "running", pid: 102, queue_size: 0 },
          },
        }}
        reindexProgress={{ status: "completed", processed: 20, total: 20, succeeded: 20, failed: 0, cancelled: 0, stale_document_count: 0, reindex_required: false }}
        onDownloadModel={onDownloadModel}
        onVerifyModel={onVerifyModel}
        onOpenModel={onOpenModel}
        onDeleteModel={onDeleteModel}
        onReindex={onReindex}
      />,
    );

    await user.clear(screen.getByLabelText("llama.cpp URL"));
    await user.type(screen.getByLabelText("llama.cpp URL"), "http://127.0.0.1:8081");
    await user.click(screen.getByRole("button", { name: "Save server settings" }));
    expect(onSaveLlamaServer).toHaveBeenCalledWith(expect.objectContaining({ server_url: "http://127.0.0.1:8081" }));
    expect(screen.getByText("Appearance")).toBeInTheDocument();
    expect(screen.getByText("Embedding and reranking")).toBeInTheDocument();
    expect(screen.getAllByRole("button", { name: "Download" })).toHaveLength(2);
    expect(screen.getAllByRole("button", { name: "Verify" })).toHaveLength(2);
    expect(screen.getByText("Jina Embeddings v5 Nano Retrieval")).toBeInTheDocument();
    expect(screen.getByText(/Reindex required: old 1024-dimensional vectors are blocked/)).toBeInTheDocument();
    expect(screen.getByText(/Reindex completed: 20 \/ 20/)).toBeInTheDocument();
    await user.click(screen.getAllByRole("button", { name: "Download" })[0]);
    await user.click(screen.getAllByRole("button", { name: "Verify" })[1]);
    await user.click(screen.getByRole("button", { name: "Reindex all documents" }));
    expect(onDownloadModel).toHaveBeenCalledWith("embedder");
    expect(onVerifyModel).toHaveBeenCalledWith("reranker");
    expect(onReindex).toHaveBeenCalledWith("full");
    expect(screen.queryByText("Temperature")).not.toBeInTheDocument();
    expect(screen.queryByText("Top K")).not.toBeInTheDocument();
  });
});
