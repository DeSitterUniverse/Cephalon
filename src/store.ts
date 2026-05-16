import { create } from "zustand";
import type { AnswerSupport, SourceChunk } from "./api";

type UiState = {
  selectedModel: string;
  selectedDocumentId: string | null;
  selectedConversationId: string | null;
  selectedSources: SourceChunk[];
  selectedSupport: AnswerSupport | null;
  rightPanel: "jobs" | "settings" | "document" | "sources" | "history" | "trace" | "health" | "eval" | "support";
  eventStatus: "connecting" | "connected" | "reconnecting" | "offline";
  setSelectedModel: (model: string) => void;
  setSelectedDocumentId: (id: string | null) => void;
  setSelectedConversationId: (id: string | null) => void;
  setSelectedSources: (sources: SourceChunk[]) => void;
  setSelectedSupport: (support: AnswerSupport | null) => void;
  setRightPanel: (panel: UiState["rightPanel"]) => void;
  setEventStatus: (status: UiState["eventStatus"]) => void;
};

export const useUiStore = create<UiState>((set) => ({
  selectedModel: "",
  selectedDocumentId: null,
  selectedConversationId: null,
  selectedSources: [],
  selectedSupport: null,
  rightPanel: "jobs",
  eventStatus: "connecting",
  setSelectedModel: (selectedModel) => set({ selectedModel }),
  setSelectedDocumentId: (selectedDocumentId) => set({ selectedDocumentId, rightPanel: selectedDocumentId ? "document" : "jobs" }),
  setSelectedConversationId: (selectedConversationId) => set({ selectedConversationId }),
  setSelectedSources: (selectedSources) => set({ selectedSources, rightPanel: selectedSources.length ? "sources" : "jobs" }),
  setSelectedSupport: (selectedSupport) => set({ selectedSupport, rightPanel: selectedSupport ? "support" : "jobs" }),
  setRightPanel: (rightPanel) => set({ rightPanel }),
  setEventStatus: (eventStatus) => set({ eventStatus }),
}));
