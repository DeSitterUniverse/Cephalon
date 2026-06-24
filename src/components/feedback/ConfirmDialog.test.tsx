import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useUiStore } from "../../store";
import { ConfirmDialog } from "./ConfirmDialog";

describe("ConfirmDialog", () => {
  beforeEach(() => {
    useUiStore.setState({ confirmation: null });
  });

  afterEach(cleanup);

  it("focuses the destructive action and confirms from the keyboard", async () => {
    const onConfirm = vi.fn();
    useUiStore.getState().requestConfirmation({
      title: "Delete document?",
      message: "This removes indexed content.",
      confirmLabel: "Delete",
      onConfirm,
    });

    const user = userEvent.setup();
    render(<ConfirmDialog />);

    expect(screen.getByRole("button", { name: "Delete" })).toHaveFocus();
    await user.keyboard("{Enter}");

    expect(onConfirm).toHaveBeenCalledOnce();
    expect(useUiStore.getState().confirmation).toBeNull();
  });

  it("dismisses without confirming on Escape", async () => {
    const onConfirm = vi.fn();
    useUiStore.getState().requestConfirmation({
      title: "Delete chat?",
      message: "This removes saved messages.",
      onConfirm,
    });

    const user = userEvent.setup();
    render(<ConfirmDialog />);
    await user.keyboard("{Escape}");

    expect(onConfirm).not.toHaveBeenCalled();
    expect(useUiStore.getState().confirmation).toBeNull();
  });
});
