import { render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { describe, expect, it, vi } from "vitest";
import { ChatHistoryPanel } from "./ChatHistoryPanel";

describe("ChatHistoryPanel", () => {
  it("selects, creates, and deletes saved conversations", async () => {
    const user = userEvent.setup();
    const onSelect = vi.fn();
    const onNew = vi.fn();
    const onDelete = vi.fn();

    render(
      <ChatHistoryPanel
        conversations={[{
          id: "conversation-1",
          title: "Stress notes",
          created_at: 1778755000,
          updated_at: 1778755300,
        }]}
        selectedId="conversation-1"
        onSelect={onSelect}
        onNew={onNew}
        onDelete={onDelete}
      />,
    );

    await user.click(screen.getByText("Stress notes"));
    expect(onSelect).toHaveBeenCalledWith("conversation-1");

    await user.click(screen.getByTitle("Start new chat"));
    expect(onNew).toHaveBeenCalled();

    await user.click(screen.getByTitle("Delete chat"));
    expect(onDelete).toHaveBeenCalledWith("conversation-1");
  });
});
