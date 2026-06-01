import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { LibraryPanel } from "./LibraryPanel";
import type { Document } from "../../api";

const docs: Document[] = [
  { id: "1", name: "stress.md", path: "C:/docs/stress.md", status: "ready", chunks: 2, tags: ["health"] },
  { id: "2", name: "broken.md", path: "C:/docs/broken.md", status: "failed", chunks: 0, tags: [] },
];

describe("LibraryPanel", () => {
  it("filters by status and search text", async () => {
    const user = userEvent.setup();
    const setSearch = vi.fn();
    const setStatusFilter = vi.fn();

    render(
      <LibraryPanel
        documents={docs}
        search=""
        setSearch={setSearch}
        statusFilter="all"
        setStatusFilter={setStatusFilter}
        onImportFolder={vi.fn()}
        onImportText={vi.fn()}
        onDelete={vi.fn()}
        onReindex={vi.fn()}
      />,
    );

    expect(screen.getByText("stress.md")).toBeInTheDocument();
    expect(screen.getByText("broken.md")).toBeInTheDocument();
    await user.click(screen.getByText("failed"));
    expect(setStatusFilter).toHaveBeenCalledWith("failed");
    await user.type(screen.getByPlaceholderText("Search path, name, tag"), "stress");
    expect(setSearch).toHaveBeenCalled();
  });
});
