import type { CSSProperties, MouseEvent, ReactNode } from "react";
import { useEffect } from "react";
import { Maximize2, MessageSquarePlus, Minus, PanelLeftClose, PanelLeftOpen, X } from "lucide-react";
import { invoke } from "@tauri-apps/api/core";
import logoUrl from "../../assets/cephalon.svg";
import { useUiStore } from "../../store";
import { RightPanel } from "./RightPanel";
import { WorkbenchNav } from "./WorkbenchNav";

type Props = {
  left: ReactNode;
  center: ReactNode;
  right: ReactNode;
  modelControl?: ReactNode;
  onNewChat?: () => void;
  newChatDisabled?: boolean;
};

export function WorkbenchLayout({ left, center, right, modelControl, onNewChat, newChatDisabled }: Props) {
  const rightPanel = useUiStore(state => state.rightPanel);
  const theme = useUiStore(state => state.theme);
  const leftPanelOpen = useUiStore(state => state.leftPanelOpen);
  const rightPanelOpen = useUiStore(state => state.rightPanelOpen);
  const leftPanelWidth = useUiStore(state => state.leftPanelWidth);
  const rightPanelWidth = useUiStore(state => state.rightPanelWidth);
  const setLeftPanelOpen = useUiStore(state => state.setLeftPanelOpen);
  const setRightPanelOpen = useUiStore(state => state.setRightPanelOpen);

  useEffect(() => {
    const closeDetails = (event: KeyboardEvent) => {
      if (event.key === "Escape") setRightPanelOpen(false);
    };
    window.addEventListener("keydown", closeDetails);
    return () => window.removeEventListener("keydown", closeDetails);
  }, [setRightPanelOpen]);

  const stopWindowDrag = (event: MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
  };
  const windowCommand = (command: "minimize_window" | "toggle_maximize_window" | "close_window") => {
    invoke(command).catch(error => console.error(`Window command failed: ${command}`, error));
  };

  return (
    <div className={`app-frame theme-${theme}`}>
      <div className="app-titlebar">
        <div className="titlebar-drag-area" data-tauri-drag-region onDoubleClick={() => windowCommand("toggle_maximize_window")}>
          <div className="window-title" data-tauri-drag-region>Cephalon</div>
        </div>
        <div className="window-controls">
          <button type="button" onMouseDown={stopWindowDrag} onClick={() => windowCommand("minimize_window")} title="Minimize"><Minus size={14} /></button>
          <button type="button" onMouseDown={stopWindowDrag} onClick={() => windowCommand("toggle_maximize_window")} title="Maximize"><Maximize2 size={13} /></button>
          <button type="button" onMouseDown={stopWindowDrag} onClick={() => windowCommand("close_window")} title="Close"><X size={15} /></button>
        </div>
      </div>
      <div
        className={`workbench panel-${rightPanel} ${leftPanelOpen ? "left-open" : "left-closed"} ${rightPanelOpen ? "right-open" : "right-closed"}`}
        style={{
          "--left-panel-width": `${leftPanelWidth}px`,
          "--right-panel-width": `${rightPanelWidth}px`,
        } as CSSProperties}
      >
        <aside className="panel panel-left" aria-label="Library">{left}</aside>
        <WorkbenchNav />
        <main className="workspace">
          <header className="topbar">
            <div className="brand-block">
              <img src={logoUrl} alt="" />
              <div>
                <div className="brand">Cephalon</div>
              </div>
              {onNewChat && <button className="title-new-chat" type="button" onClick={onNewChat} disabled={newChatDisabled} title="Start new chat" aria-label="New chat"><MessageSquarePlus size={15} /></button>}
            </div>
            <div className="topbar-actions">
              {modelControl}
              <button
                type="button"
                className="topbar-panel-toggle"
                onClick={() => setLeftPanelOpen(!leftPanelOpen)}
                aria-label={leftPanelOpen ? "Hide library" : "Show library"}
                title={leftPanelOpen ? "Hide library" : "Show library"}
              >
                {leftPanelOpen ? <PanelLeftClose size={16} /> : <PanelLeftOpen size={16} />}
              </button>
            </div>
          </header>
          {center}
        </main>
        <RightPanel
          open={rightPanelOpen}
          panel={rightPanel}
          onClose={() => setRightPanelOpen(false)}
        >
          {right}
        </RightPanel>
      </div>
    </div>
  );
}
