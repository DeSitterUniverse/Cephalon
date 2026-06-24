import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { useUiStore } from "../../store";
import { NotificationCenter } from "./NotificationCenter";

describe("NotificationCenter", () => {
  beforeEach(() => {
    vi.useFakeTimers();
    useUiStore.setState({ notifications: [] });
  });

  afterEach(() => {
    cleanup();
    vi.useRealTimers();
  });

  it("renders queued notifications and keeps errors until dismissed", async () => {
    useUiStore.getState().notify("Indexed document", "success");
    useUiStore.getState().notify("Backend unavailable", "error");
    render(<NotificationCenter />);

    expect(screen.getByText("Indexed document")).toBeInTheDocument();
    expect(screen.getByText("Backend unavailable")).toBeInTheDocument();

    await vi.advanceTimersByTimeAsync(7000);

    expect(screen.queryByText("Indexed document")).not.toBeInTheDocument();
    expect(screen.getByText("Backend unavailable")).toBeInTheDocument();
  });

  it("allows a queued notice to be dismissed directly", async () => {
    vi.useRealTimers();
    useUiStore.getState().notify("Saved settings", "success");
    render(<NotificationCenter />);

    await userEvent.setup().click(screen.getByRole("button", { name: "Dismiss Saved settings" }));

    expect(screen.queryByText("Saved settings")).not.toBeInTheDocument();
  });
});
