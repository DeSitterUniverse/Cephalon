import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { listen } from "@tauri-apps/api/event";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import {
  addDocumentTag,
  deleteDocument,
  deleteDocumentTag,
  exportMetrics,
  getDocument,
  getDocuments,
  getJobs,
  getModels,
  getSettings,
  healthCheck,
  ingestPath,
  reindexDocument,
  updateDocument,
  updateSettings,
  type Document,
  type RagSettings,
} from "./api";
import { ChatPanel } from "./components/chat/ChatPanel";
import { JobsPanel } from "./components/jobs/JobsPanel";
import { LibraryPanel } from "./components/library/LibraryPanel";
import { DocumentDetails } from "./components/library/DocumentDetails";
import { WorkbenchLayout } from "./components/layout/WorkbenchLayout";
import { SettingsPanel } from "./components/settings/SettingsPanel";
import { SourcesPanel } from "./components/sources/SourcesPanel";
import { useEventStream } from "./hooks/useEventStream";
import { useUiStore } from "./store";
import "./App.css";

type FileDropPayload = string[];
type DragDropPayload = { paths?: string[] };

function isTauriRuntime() {
  return "__TAURI_INTERNALS__" in window;
}

export default function App() {
  const queryClient = useQueryClient();
  const [bootReady, setBootReady] = useState(false);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [toast, setToast] = useState<string | null>(null);
  const selectedModel = useUiStore(state => state.selectedModel);
  const setSelectedModel = useUiStore(state => state.setSelectedModel);
  const selectedDocumentId = useUiStore(state => state.selectedDocumentId);
  const selectedSources = useUiStore(state => state.selectedSources);
  const rightPanel = useUiStore(state => state.rightPanel);

  useEventStream();

  const modelsQuery = useQuery({ queryKey: ["models"], queryFn: getModels, enabled: bootReady });
  const documentsQuery = useQuery({ queryKey: ["documents"], queryFn: getDocuments, enabled: bootReady });
  const jobsQuery = useQuery({ queryKey: ["jobs"], queryFn: getJobs, enabled: bootReady });
  const settingsQuery = useQuery({ queryKey: ["settings"], queryFn: getSettings, enabled: bootReady });
  const documentQuery = useQuery({
    queryKey: ["document", selectedDocumentId],
    queryFn: () => getDocument(selectedDocumentId as string),
    enabled: Boolean(selectedDocumentId),
  });

  const ingestMutation = useMutation({
    mutationFn: (path: string) => ingestPath(path),
    onSuccess: data => {
      setToast(data.message || "Ingestion queued.");
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
    onError: error => setToast(error instanceof Error ? error.message : "Failed to queue ingestion."),
  });

  const deleteMutation = useMutation({
    mutationFn: (doc: Document) => deleteDocument(doc.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const reindexMutation = useMutation({
    mutationFn: (doc: Document) => reindexDocument(doc.id),
    onSuccess: data => {
      setToast(data.message || "Reindex queued.");
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    },
  });

  const settingsMutation = useMutation({
    mutationFn: updateSettings,
    onSuccess: data => queryClient.setQueryData(["settings"], data),
  });

  const metricsMutation = useMutation({
    mutationFn: exportMetrics,
    onSuccess: data => setToast(data.status === "success" && data.path ? `Metrics exported: ${data.path}` : `Metrics export failed: ${data.error || "metrics directory is unavailable"}`),
    onError: error => setToast(error instanceof Error ? error.message : "Failed to export metrics."),
  });

  useEffect(() => {
    let active = true;
    const poll = async () => {
      while (active) {
        if (await healthCheck()) {
          setBootReady(true);
          return;
        }
        await new Promise(resolve => setTimeout(resolve, 750));
      }
    };
    poll();
    return () => { active = false; };
  }, []);

  useEffect(() => {
    const names = modelsQuery.data?.models || [];
    if (!selectedModel && names.length > 0) setSelectedModel(names[0]);
  }, [modelsQuery.data, selectedModel, setSelectedModel]);

  useEffect(() => {
    const unlistenFileDrop = isTauriRuntime()
      ? listen<FileDropPayload>("tauri://file-drop", event => {
        if (Array.isArray(event.payload) && event.payload[0]) ingestMutation.mutate(event.payload[0]);
      })
      : Promise.resolve(() => {});
    const unlistenDragDrop = isTauriRuntime()
      ? listen<DragDropPayload>("tauri://drag-drop", event => {
        if (event.payload?.paths?.[0]) ingestMutation.mutate(event.payload.paths[0]);
      })
      : Promise.resolve(() => {});
    const preventDefault = (event: DragEvent) => event.preventDefault();
    document.addEventListener("dragenter", preventDefault);
    document.addEventListener("dragover", preventDefault);
    document.addEventListener("drop", preventDefault);
    return () => {
      unlistenFileDrop.then(unlisten => unlisten()).catch(() => {});
      unlistenDragDrop.then(unlisten => unlisten()).catch(() => {});
      document.removeEventListener("dragenter", preventDefault);
      document.removeEventListener("dragover", preventDefault);
      document.removeEventListener("drop", preventDefault);
    };
  }, [ingestMutation]);

  async function importFolder() {
    if (!isTauriRuntime()) {
      setToast("Folder import is available in the Tauri app. Use drag/drop or the API in browser dev mode.");
      return;
    }
    const selectedPath = await openDialog({ directory: true, multiple: false });
    if (selectedPath && typeof selectedPath === "string") ingestMutation.mutate(selectedPath);
  }

  async function importTextFile() {
    if (!isTauriRuntime()) {
      setToast("Text import is available in the Tauri app. Use the API in browser dev mode.");
      return;
    }
    const selectedPath = await openDialog({ directory: false, multiple: false });
    if (selectedPath && typeof selectedPath === "string") ingestPath(selectedPath, true)
      .then(data => {
        setToast(data.message || "Text ingestion queued.");
        queryClient.invalidateQueries({ queryKey: ["jobs"] });
      })
      .catch(error => setToast(error instanceof Error ? error.message : "Failed to queue text import."));
  }

  function removeDocument(doc: Document) {
    if (window.confirm(`Delete ${doc.name} from the library?`)) deleteMutation.mutate(doc);
  }

  const right = rightPanel === "settings"
    ? <SettingsPanel models={modelsQuery.data?.models || []} selectedModel={selectedModel} setSelectedModel={setSelectedModel} settings={settingsQuery.data} updateSettings={(value: RagSettings) => settingsMutation.mutate(value)} onExportMetrics={() => metricsMutation.mutate()} />
    : rightPanel === "sources"
      ? <SourcesPanel sources={selectedSources} />
      : rightPanel === "document"
        ? (
          <DocumentDetails
            document={documentQuery.data}
            onRename={(displayName) => selectedDocumentId && updateDocument(selectedDocumentId, displayName).then(data => {
              queryClient.setQueryData(["document", selectedDocumentId], data);
              queryClient.invalidateQueries({ queryKey: ["documents"] });
            })}
            onAddTag={(tag) => selectedDocumentId && addDocumentTag(selectedDocumentId, tag).then(() => {
              queryClient.invalidateQueries({ queryKey: ["document", selectedDocumentId] });
              queryClient.invalidateQueries({ queryKey: ["documents"] });
            })}
            onDeleteTag={(tag) => selectedDocumentId && deleteDocumentTag(selectedDocumentId, tag).then(() => {
              queryClient.invalidateQueries({ queryKey: ["document", selectedDocumentId] });
              queryClient.invalidateQueries({ queryKey: ["documents"] });
            })}
            onReindex={() => documentQuery.data && reindexMutation.mutate(documentQuery.data)}
            onDelete={() => documentQuery.data && removeDocument(documentQuery.data)}
          />
        )
        : <JobsPanel jobs={jobsQuery.data?.jobs || []} />;

  if (!bootReady) {
    return (
      <div className="boot-screen">
        <div className="boot-card">
          <strong>Cephalon</strong>
          <span>Opening local service...</span>
        </div>
      </div>
    );
  }

  const backendLabel = modelsQuery.data?.llama_backend?.vulkan_available ? "Vulkan" : "CPU";
  const contextLabel = modelsQuery.data?.active_context_tokens
    ? `${Math.round(modelsQuery.data.active_context_tokens / 1024)}k ctx`
    : backendLabel;

  return (
    <>
      <WorkbenchLayout
        left={
          <LibraryPanel
            documents={documentsQuery.data?.documents || []}
            search={search}
            setSearch={setSearch}
            statusFilter={statusFilter}
            setStatusFilter={setStatusFilter}
            onImportFolder={importFolder}
            onImportText={importTextFile}
            onDelete={removeDocument}
            onReindex={(doc) => reindexMutation.mutate(doc)}
          />
        }
        center={<ChatPanel selectedModel={selectedModel} settings={settingsQuery.data} />}
        modelControl={(
          <label className="top-model-picker" title="Local GGUF model">
            <span>Model</span>
            <select value={selectedModel} onChange={event => setSelectedModel(event.target.value)} disabled={modelsQuery.isLoading}>
              <option value="">No GGUF model found</option>
              {(modelsQuery.data?.models || []).map(model => <option key={model} value={model}>{model}</option>)}
            </select>
            <small>{contextLabel}</small>
          </label>
        )}
        right={right}
      />
      {toast && <button className="toast" onClick={() => setToast(null)}>{toast}</button>}
    </>
  );
}
