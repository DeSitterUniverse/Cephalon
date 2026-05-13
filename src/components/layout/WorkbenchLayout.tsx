import type { ReactNode } from "react";
import { Activity, Circle, Database, SlidersHorizontal } from "lucide-react";
import logoUrl from "../../assets/cephalon.svg";
import { useUiStore } from "../../store";

type Props = {
  left: ReactNode;
  center: ReactNode;
  right: ReactNode;
  modelControl?: ReactNode;
};

export function WorkbenchLayout({ left, center, right, modelControl }: Props) {
  const rightPanel = useUiStore(state => state.rightPanel);
  const setRightPanel = useUiStore(state => state.setRightPanel);
  const eventStatus = useUiStore(state => state.eventStatus);
  const liveLabel = eventStatus === "connected" ? "Live" : eventStatus === "offline" ? "Offline" : "Reconnecting";
  const liveTitle = eventStatus === "connected"
    ? "Live updates are connected."
    : eventStatus === "offline"
      ? "The local event stream is offline; cached data will refresh when the backend returns."
      : "Connecting to live updates; cached data refreshes periodically while reconnecting.";

  return (
    <div className="workbench">
      <aside className="panel panel-left">{left}</aside>
      <main className="workspace">
        <header className="topbar">
          <div className="brand-block">
            <img src={logoUrl} alt="" />
            <div>
            <div className="brand">Cephalon</div>
            <div className="subtle">Local document search</div>
            </div>
          </div>
          <div className="topbar-actions">
            {modelControl}
            <button className={rightPanel === "jobs" ? "icon-button active" : "icon-button"} onClick={() => setRightPanel("jobs")} title="Jobs">
              <Activity size={16} />
            </button>
            <button className={rightPanel === "document" ? "icon-button active" : "icon-button"} onClick={() => setRightPanel("document")} title="Document details">
              <Database size={16} />
            </button>
            <button className={rightPanel === "settings" ? "icon-button active" : "icon-button"} onClick={() => setRightPanel("settings")} title="Search and model controls">
              <SlidersHorizontal size={16} />
            </button>
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
  );
}
