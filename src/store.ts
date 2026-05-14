import { create } from "zustand";
import type { SourceChunk } from "./api";

type UiState = {
  selectedModel: string;
  selectedDocumentId: string | null;
  selectedConversationId: string | null;
  selectedSources: SourceChunk[];
  rightPanel: "jobs" | "settings" | "document" | "sources" | "history";
  eventStatus: "connecting" | "connected" | "reconnecting" | "offline";
  setSelectedModel: (model: string) => void;
  setSelectedDocumentId: (id: string | null) => void;
  setSelectedConversationId: (id: string | null) => void;
  setSelectedSources: (sources: SourceChunk[]) => void;
  setRightPanel: (panel: UiState["rightPanel"]) => void;
  setEventStatus: (status: UiState["eventStatus"]) => void;
};

export const useUiStore = create<UiState>((set) => ({
  selectedModel: "",
  selectedDocumentId: null,
  selectedConversationId: null,
  selectedSources: [],
  rightPanel: "jobs",
  eventStatus: "connecting",
  setSelectedModel: (selectedModel) => set({ selectedModel }),
  setSelectedDocumentId: (selectedDocumentId) => set({ selectedDocumentId, rightPanel: selectedDocumentId ? "document" : "jobs" }),
  setSelectedConversationId: (selectedConversationId) => set({ selectedConversationId }),
  setSelectedSources: (selectedSources) => set({ selectedSources, rightPanel: selectedSources.length ? "sources" : "jobs" }),
  setRightPanel: (rightPanel) => set({ rightPanel }),
  setEventStatus: (eventStatus) => set({ eventStatus }),
}));
