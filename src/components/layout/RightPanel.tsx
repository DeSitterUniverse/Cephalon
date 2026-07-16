import type { ReactNode } from "react";
import { X } from "lucide-react";
import type { RightPanel as RightPanelName } from "../../store";

type Props = {
  open: boolean;
  panel: RightPanelName;
  children: ReactNode;
  onClose: () => void;
};

export function RightPanel({ open, panel, children, onClose }: Props) {
  return (
    <>
      <div
        className="detail-panel-backdrop"
        data-testid="detail-panel-backdrop"
        aria-hidden="true"
        onMouseDown={onClose}
      />
      <aside className="panel panel-right" aria-label="Details" aria-hidden={!open}>
        <div className="detail-panel-header">
          <span>{panelTitle(panel)}</span>
          <button type="button" onClick={onClose} aria-label="Hide details" title="Hide details">
            <X size={16} />
          </button>
        </div>
        <div className="detail-panel-content">{children}</div>
      </aside>
    </>
  );
}

function panelTitle(panel: RightPanelName) {
  return {
    settings: "Settings",
    document: "Document",
    sources: "Sources",
    history: "Chats",
    trace: "Retrieval trace",
    health: "Index health",
    eval: "Evaluation",
    support: "Answer support",
  }[panel];
}
