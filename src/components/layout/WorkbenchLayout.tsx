import type { MouseEvent, ReactNode } from "react";
import { BarChart3, Circle, FileText, ListChecks, Maximize2, MessageSquareText, Minus, MoreHorizontal, SearchCode, ShieldCheck, SlidersHorizontal, X } from "lucide-react";
import { useState } from "react";
import { invoke } from "@tauri-apps/api/core";
import logoUrl from "../../assets/cephalon.svg";
import { useUiStore } from "../../store";

type Props = {
  left: ReactNode;
  center: ReactNode;
  right: ReactNode;
  modelControl?: ReactNode;
};

export function WorkbenchLayout({ left, center, right, modelControl }: Props) {
  const [panelMenuOpen, setPanelMenuOpen] = useState(false);
  const rightPanel = useUiStore(state => state.rightPanel);
  const setRightPanel = useUiStore(state => state.setRightPanel);
  const eventStatus = useUiStore(state => state.eventStatus);
  const liveLabel = eventStatus === "connected" ? "Live" : eventStatus === "offline" ? "Offline" : "Reconnecting";
  const liveTitle = eventStatus === "connected"
    ? "Live updates are connected."
    : eventStatus === "offline"
      ? "The local event stream is offline; cached data will refresh when the backend returns."
    : "Connecting to live updates; cached data refreshes periodically while reconnecting.";
  const selectPanel = (panel: Parameters<typeof setRightPanel>[0]) => {
    setRightPanel(panel);
    setPanelMenuOpen(false);
  };
  const stopWindowDrag = (event: MouseEvent<HTMLButtonElement>) => {
    event.stopPropagation();
  };
  const windowCommand = (command: "minimize_window" | "toggle_maximize_window" | "close_window") => {
    invoke(command).catch(error => console.error(`Window command failed: ${command}`, error));
  };

  return (
    <div className="app-frame">
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
      <div className="workbench">
        <aside className="panel panel-left">{left}</aside>
        <main className="workspace">
          <header className="topbar">
            <div className="brand-block">
              <img src={logoUrl} alt="" />
              <div>
                <div className="brand">Cephalon</div>
              </div>
            </div>
            <div className="topbar-actions">
              {modelControl}
              <button type="button" className={rightPanel === "jobs" ? "icon-button active" : "icon-button"} onClick={() => selectPanel("jobs")} title="Jobs">
                <ListChecks size={16} />
              </button>
              <button type="button" className={rightPanel === "history" ? "icon-button active" : "icon-button"} onClick={() => selectPanel("history")} title="Chat history">
                <MessageSquareText size={16} />
              </button>
              <button type="button" className={rightPanel === "document" ? "icon-button active" : "icon-button"} onClick={() => selectPanel("document")} title="Document details">
                <FileText size={16} />
              </button>
              <button type="button" className={rightPanel === "settings" ? "icon-button active" : "icon-button"} onClick={() => selectPanel("settings")} title="Search and model controls">
                <SlidersHorizontal size={16} />
              </button>
              <div className="panel-menu">
                <button type="button" className={["trace", "health", "eval", "support"].includes(rightPanel) ? "icon-button active" : "icon-button"} onClick={() => setPanelMenuOpen(value => !value)} title="Diagnostics">
                  <MoreHorizontal size={16} />
                </button>
                {panelMenuOpen && (
                  <div className="panel-menu-list">
                    <button type="button" className={rightPanel === "trace" ? "active" : ""} onClick={() => selectPanel("trace")}><SearchCode size={15} />Retrieval trace</button>
                    <button type="button" className={rightPanel === "health" ? "active" : ""} onClick={() => selectPanel("health")}><BarChart3 size={15} />Index health</button>
                    <button type="button" className={rightPanel === "eval" ? "active" : ""} onClick={() => selectPanel("eval")}><ListChecks size={15} />Evaluation</button>
                    <button type="button" className={rightPanel === "support" ? "active" : ""} onClick={() => selectPanel("support")}><ShieldCheck size={15} />Answer support</button>
                  </div>
                )}
              </div>
              <span className={eventStatus === "connected" ? "status-pill ok" : eventStatus === "offline" ? "status-pill danger" : "status-pill warn"} title={liveTitle}>
                <Circle size={9} fill="currentColor" />
                {liveLabel}
              </span>
            </div>
          </header>
          {center}
        </main>
        <aside className="panel panel-right">{right}</aside>
      </div>
    </div>
  );
}
