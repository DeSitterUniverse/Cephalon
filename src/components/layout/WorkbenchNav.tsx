import {
  BarChart3,
  FileText,
  History,
  Library,
  SearchCode,
  Settings2,
  ShieldCheck,
} from "lucide-react";
import type { ComponentType } from "react";
import type { RightPanel } from "../../store";
import { useUiStore } from "../../store";

type NavItem = {
  panel: RightPanel;
  label: string;
  icon: ComponentType<{ size?: number }>;
};

const workItems: NavItem[] = [
  { panel: "history", label: "Chats", icon: History },
  { panel: "document", label: "Document", icon: FileText },
  { panel: "sources", label: "Sources", icon: Library },
  { panel: "settings", label: "Settings", icon: Settings2 },
];

const diagnosticItems: NavItem[] = [
  { panel: "trace", label: "Trace", icon: SearchCode },
  { panel: "health", label: "Health", icon: BarChart3 },
  { panel: "eval", label: "Evaluation", icon: BarChart3 },
  { panel: "support", label: "Support", icon: ShieldCheck },
];

function NavGroup({ label, items }: { label: string; items: NavItem[] }) {
  const activePanel = useUiStore(state => state.rightPanel);
  const panelOpen = useUiStore(state => state.rightPanelOpen);
  const setRightPanel = useUiStore(state => state.setRightPanel);

  return (
    <div className="workbench-nav-group">
      <div className="workbench-nav-label">{label}</div>
      {items.map(({ panel, label: itemLabel, icon: Icon }) => (
        <button
          key={panel}
          type="button"
          className={activePanel === panel && panelOpen ? "workbench-nav-item active" : "workbench-nav-item"}
          aria-pressed={activePanel === panel && panelOpen}
          onClick={() => setRightPanel(panel)}
        >
          <Icon size={16} />
          <span>{itemLabel}</span>
        </button>
      ))}
    </div>
  );
}

export function WorkbenchNav() {
  return (
    <nav className="workbench-nav" aria-label="Workbench">
      <NavGroup label="Work" items={workItems} />
      <NavGroup label="Diagnostics" items={diagnosticItems} />
    </nav>
  );
}
