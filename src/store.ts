import { create } from "zustand";
import type { AnswerSupport, SourceChunk } from "./api";

export type RightPanel = "settings" | "document" | "sources" | "history" | "trace" | "health" | "eval" | "support";
export type NotificationKind = "info" | "success" | "error";
export type AppNotification = { id: number; message: string; kind: NotificationKind };
export type ConfirmationRequest = {
  title: string;
  message: string;
  confirmLabel?: string;
  onConfirm: () => void;
};

type UiState = {
  theme: "black" | "graphite";
  selectedModel: string;
  selectedDocumentId: string | null;
  selectedConversationId: string | null;
  selectedSources: SourceChunk[];
  selectedSupport: AnswerSupport | null;
  rightPanel: RightPanel;
  leftPanelOpen: boolean;
  rightPanelOpen: boolean;
  leftPanelWidth: number;
  rightPanelWidth: number;
  notifications: AppNotification[];
  confirmation: ConfirmationRequest | null;
  eventStatus: "connecting" | "connected" | "reconnecting" | "offline";
  setSelectedModel: (model: string) => void;
  setSelectedDocumentId: (id: string | null) => void;
  setSelectedConversationId: (id: string | null) => void;
  setSelectedSources: (sources: SourceChunk[]) => void;
  setSelectedSupport: (support: AnswerSupport | null) => void;
  setRightPanel: (panel: RightPanel) => void;
  setLeftPanelOpen: (open: boolean) => void;
  setRightPanelOpen: (open: boolean) => void;
  setLeftPanelWidth: (width: number) => void;
  setRightPanelWidth: (width: number) => void;
  notify: (message: string, kind?: NotificationKind) => void;
  dismissNotification: (id: number) => void;
  requestConfirmation: (request: ConfirmationRequest) => void;
  closeConfirmation: () => void;
  setEventStatus: (status: UiState["eventStatus"]) => void;
  setTheme: (theme: UiState["theme"]) => void;
};

type StoredLayout = Pick<UiState, "leftPanelOpen" | "rightPanelOpen" | "leftPanelWidth" | "rightPanelWidth">;

const defaultLayout: StoredLayout = {
  leftPanelOpen: true,
  rightPanelOpen: true,
  leftPanelWidth: 300,
  rightPanelWidth: 340,
};

function storedTheme(): UiState["theme"] {
  try {
    return typeof window !== "undefined" && typeof window.localStorage?.getItem === "function" && window.localStorage.getItem("cephalon.theme") === "graphite"
      ? "graphite"
      : "black";
  } catch {
    return "black";
  }
}

function storedLayout(): StoredLayout {
  try {
    if (typeof window === "undefined" || typeof window.localStorage?.getItem !== "function") return defaultLayout;
    const saved = JSON.parse(window.localStorage.getItem("cephalon.layout") || "{}") as Partial<StoredLayout>;
    return {
      leftPanelOpen: typeof saved.leftPanelOpen === "boolean" ? saved.leftPanelOpen : defaultLayout.leftPanelOpen,
      rightPanelOpen: typeof saved.rightPanelOpen === "boolean" ? saved.rightPanelOpen : defaultLayout.rightPanelOpen,
      leftPanelWidth: clampWidth(saved.leftPanelWidth, 240, 460, defaultLayout.leftPanelWidth),
      rightPanelWidth: clampWidth(saved.rightPanelWidth, 300, 520, defaultLayout.rightPanelWidth),
    };
  } catch {
    return defaultLayout;
  }
}

function clampWidth(value: unknown, minimum: number, maximum: number, fallback: number) {
  return typeof value === "number" && Number.isFinite(value)
    ? Math.min(maximum, Math.max(minimum, value))
    : fallback;
}

function persistLayout(layout: StoredLayout) {
  try {
    if (typeof window.localStorage?.setItem === "function") {
      window.localStorage.setItem("cephalon.layout", JSON.stringify(layout));
    }
  } catch {}
}

const initialLayout = storedLayout();
let nextNotificationId = 1;

export const useUiStore = create<UiState>((set) => ({
  theme: storedTheme(),
  selectedModel: "",
  selectedDocumentId: null,
  selectedConversationId: null,
  selectedSources: [],
  selectedSupport: null,
  rightPanel: "history",
  ...initialLayout,
  notifications: [],
  confirmation: null,
  eventStatus: "connecting",
  setSelectedModel: (selectedModel) => set({ selectedModel }),
  setSelectedDocumentId: (selectedDocumentId) => set(state => ({
    selectedDocumentId,
    rightPanel: selectedDocumentId ? "document" : state.rightPanel,
    rightPanelOpen: selectedDocumentId ? true : state.rightPanelOpen,
  })),
  setSelectedConversationId: (selectedConversationId) => set({ selectedConversationId }),
  setSelectedSources: (selectedSources) => set({ selectedSources }),
  setSelectedSupport: (selectedSupport) => set({ selectedSupport }),
  setRightPanel: (rightPanel) => set({ rightPanel, rightPanelOpen: true }),
  setLeftPanelOpen: (leftPanelOpen) => set(state => {
    const layout = { ...state, leftPanelOpen };
    persistLayout(layout);
    return { leftPanelOpen };
  }),
  setRightPanelOpen: (rightPanelOpen) => set(state => {
    const layout = { ...state, rightPanelOpen };
    persistLayout(layout);
    return { rightPanelOpen };
  }),
  setLeftPanelWidth: (width) => set(state => {
    const leftPanelWidth = clampWidth(width, 240, 460, state.leftPanelWidth);
    persistLayout({ ...state, leftPanelWidth });
    return { leftPanelWidth };
  }),
  setRightPanelWidth: (width) => set(state => {
    const rightPanelWidth = clampWidth(width, 300, 520, state.rightPanelWidth);
    persistLayout({ ...state, rightPanelWidth });
    return { rightPanelWidth };
  }),
  notify: (message, kind = "info") => set(state => ({
    notifications: [...state.notifications, { id: nextNotificationId++, message, kind }],
  })),
  dismissNotification: (id) => set(state => ({
    notifications: state.notifications.filter(notification => notification.id !== id),
  })),
  requestConfirmation: (confirmation) => set({ confirmation }),
  closeConfirmation: () => set({ confirmation: null }),
  setEventStatus: (eventStatus) => set({ eventStatus }),
  setTheme: (theme) => {
    try {
      if (typeof window.localStorage?.setItem === "function") window.localStorage.setItem("cephalon.theme", theme);
    } catch {}
    set({ theme });
  },
}));
