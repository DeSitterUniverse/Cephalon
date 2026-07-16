import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ChatHistoryPanel } from "./ChatHistoryPanel";

describe("ChatHistoryPanel", () => {
  it("selects, creates, renames, deletes, and filters saved conversations", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    const onDelete = vi.fn();
    const onRename = vi.fn().mockResolvedValue(undefined);

    render(
      <ChatHistoryPanel
        conversations={[
          {
            id: "conversation-1",
            title: "Stress notes",
            created_at: 1778755000,
            updated_at: 1778755300,
          },
          {
            id: "conversation-2",
            title: "Project plan",
            created_at: 1778754000,
            updated_at: 1778754300,
          },
        ]}
        selectedId="conversation-1"
        onSelect={onSelect}
        onDelete={onDelete}
        onRename={onRename}
      />,
    );

    await user.type(screen.getByRole("searchbox", { name: "Search saved chats" }), "stress");
    expect(screen.getByText("Stress notes")).toBeInTheDocument();
    expect(screen.queryByText("Project plan")).not.toBeInTheDocument();

    await user.click(screen.getByText("Stress notes"));
    expect(onSelect).toHaveBeenCalledWith("conversation-1");

    await user.click(screen.getByTitle("Delete chat"));
    expect(onDelete).toHaveBeenCalledWith("conversation-1");

    await user.click(screen.getByTitle("Rename chat"));
    const titleInput = screen.getByRole("textbox", { name: "Chat title" });
    await user.clear(titleInput);
    await user.type(titleInput, "Calming techniques{Enter}");
    await waitFor(() => expect(onRename).toHaveBeenCalledWith("conversation-1", "Calming techniques"));

    expect(document.querySelector("button button")).toBeNull();
  });
});
