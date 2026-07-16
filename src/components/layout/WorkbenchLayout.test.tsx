import { cleanup, render, screen } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";
import { WorkbenchLayout } from "./WorkbenchLayout";
import { useUiStore } from "../../store";

vi.mock("@tauri-apps/api/core", () => ({
  invoke: vi.fn(),
}));

describe("WorkbenchLayout", () => {
  beforeEach(() => {
    useUiStore.setState({
      rightPanel: "history",
      leftPanelOpen: true,
      rightPanelOpen: true,
      leftPanelWidth: 300,
      rightPanelWidth: 340,
    });
  });

  afterEach(cleanup);

  it("uses visible labeled navigation grouped by work and diagnostics", async () => {
    const user = userEvent.setup();
    render(
      <WorkbenchLayout
        left={<div>Library content</div>}
        center={<div>Chat content</div>}
        right={<div>Panel content</div>}
      />,
    );

    expect(screen.getByText("Work")).toBeInTheDocument();
    expect(screen.getByText("Diagnostics")).toBeInTheDocument();
    expect(screen.queryByRole("button", { name: "Jobs" })).not.toBeInTheDocument();
    expect(screen.getByRole("button", { name: "Settings" })).toHaveTextContent("Settings");

    await user.click(screen.getByRole("button", { name: "Settings" }));
    expect(useUiStore.getState().rightPanel).toBe("settings");
    expect(useUiStore.getState().rightPanelOpen).toBe(true);
  });

  it("closes the detail panel from its close button and Escape", async () => {
    const user = userEvent.setup();
    render(
      <WorkbenchLayout
        left={<div>Library content</div>}
        center={<div>Chat content</div>}
        right={<div>Panel content</div>}
      />,
    );

    await user.click(screen.getByRole("button", { name: "Hide details" }));
    expect(useUiStore.getState().rightPanelOpen).toBe(false);

    await user.click(screen.getByRole("button", { name: "Chats" }));
    expect(useUiStore.getState().rightPanelOpen).toBe(true);
    await user.keyboard("{Escape}");
    expect(useUiStore.getState().rightPanelOpen).toBe(false);
  });

  it("closes the detail drawer when the responsive backdrop is clicked", async () => {
    const user = userEvent.setup();
    render(
      <WorkbenchLayout
        left={<div>Library content</div>}
        center={<div>Chat content</div>}
        right={<div>Panel content</div>}
      />,
    );

    await user.click(screen.getByTestId("detail-panel-backdrop"));

    expect(useUiStore.getState().rightPanelOpen).toBe(false);
  });
});
