import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { listen } from "@tauri-apps/api/event";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import {
  addDocumentTag,
  createConversation,
  deleteConversation,
  deleteDocument,
  deleteDocumentTag,
  exportMetrics,
  getConversation,
  getConversations,
  getDocument,
  getDocuments,
  getJobs,
  getHealth,
  getIndexHealth,
  getModels,
  getEvalRuns,
  getRetrievalTrace,
  getRetrievalTraces,
  getSettings,
  ingestPath,
  loadModel,
  reindexDocument,
  runEval,
  updateDocument,
  updateSettings,
  type Document,
  type HealthResponse,
  type RagSettings,
} from "./api";
import { ChatPanel } from "./components/chat/ChatPanel";
import { AnswerSupportPanel } from "./components/diagnostics/AnswerSupportPanel";
import { EvaluationPanel } from "./components/diagnostics/EvaluationPanel";
import { IndexHealthPanel } from "./components/diagnostics/IndexHealthPanel";
import { RetrievalTracePanel } from "./components/diagnostics/RetrievalTracePanel";
import { ChatHistoryPanel } from "./components/chat/ChatHistoryPanel";
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
  const [bootStatus, setBootStatus] = useState("Starting local service...");
  const [bootHealth, setBootHealth] = useState<HealthResponse | null>(null);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const [toast, setToast] = useState<string | null>(null);
  const selectedModel = useUiStore(state => state.selectedModel);
  const setSelectedModel = useUiStore(state => state.setSelectedModel);
  const selectedDocumentId = useUiStore(state => state.selectedDocumentId);
  const selectedConversationId = useUiStore(state => state.selectedConversationId);
  const setSelectedConversationId = useUiStore(state => state.setSelectedConversationId);
  const selectedSources = useUiStore(state => state.selectedSources);
  const selectedSupport = useUiStore(state => state.selectedSupport);
  const rightPanel = useUiStore(state => state.rightPanel);
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null);

  useEventStream();

  const modelsQuery = useQuery({ queryKey: ["models"], queryFn: getModels, enabled: bootReady });
  const documentsQuery = useQuery({ queryKey: ["documents"], queryFn: getDocuments, enabled: bootReady });
  const jobsQuery = useQuery({ queryKey: ["jobs"], queryFn: getJobs, enabled: bootReady });
  const conversationsQuery = useQuery({ queryKey: ["conversations"], queryFn: getConversations, enabled: bootReady });
  const settingsQuery = useQuery({ queryKey: ["settings"], queryFn: getSettings, enabled: bootReady });
  const tracesQuery = useQuery({ queryKey: ["retrieval-traces"], queryFn: getRetrievalTraces, enabled: bootReady && rightPanel === "trace" });
  const traceQuery = useQuery({
    queryKey: ["retrieval-trace", selectedTraceId],
    queryFn: () => getRetrievalTrace(selectedTraceId as string),
    enabled: Boolean(selectedTraceId),
  });
  const indexHealthQuery = useQuery({ queryKey: ["index-health"], queryFn: getIndexHealth, enabled: bootReady && rightPanel === "health" });
  const evalRunsQuery = useQuery({ queryKey: ["eval-runs"], queryFn: getEvalRuns, enabled: bootReady && rightPanel === "eval" });
  const documentQuery = useQuery({
    queryKey: ["document", selectedDocumentId],
    queryFn: () => getDocument(selectedDocumentId as string),
    enabled: Boolean(selectedDocumentId),
  });
  const conversationQuery = useQuery({
    queryKey: ["conversation", selectedConversationId],
    queryFn: () => getConversation(selectedConversationId as string),
    enabled: Boolean(selectedConversationId),
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

  const loadModelMutation = useMutation({
    mutationFn: (model: string) => loadModel(model),
    onMutate: () => setToast("Loading model into llama.cpp..."),
    onSuccess: data => {
      setToast(data.active_model ? `Loaded ${data.active_model}` : "Model loaded.");
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
    onError: error => setToast(error instanceof Error ? error.message : "Failed to load model."),
  });

  const metricsMutation = useMutation({
    mutationFn: exportMetrics,
    onSuccess: data => setToast(data.status === "success" && data.path ? `Metrics exported: ${data.path}` : `Metrics export failed: ${data.error || "metrics directory is unavailable"}`),
    onError: error => setToast(error instanceof Error ? error.message : "Failed to export metrics."),
  });

  const newConversationMutation = useMutation({
    mutationFn: createConversation,
    onSuccess: data => {
      setSelectedConversationId(data.id);
      queryClient.invalidateQueries({ queryKey: ["conversations"] });
    },
  });

  const evalMutation = useMutation({
    mutationFn: ({ question, expectedDoc }: { question: string; expectedDoc: string }) => runEval([{
      id: `manual-${Date.now()}`,
      question,
      expected_doc_ids: [expectedDoc],
    }]),
    onSuccess: () => {
      setToast("Eval run saved.");
      queryClient.invalidateQueries({ queryKey: ["eval-runs"] });
    },
    onError: error => setToast(error instanceof Error ? error.message : "Eval run failed."),
  });

  const deleteConversationMutation = useMutation({
    mutationFn: deleteConversation,
    onSuccess: (_data, id) => {
      if (selectedConversationId === id) setSelectedConversationId(null);
      queryClient.invalidateQueries({ queryKey: ["conversations"] });
    },
  });

  useEffect(() => {
    let active = true;
    const poll = async () => {
      let attempts = 0;
      while (active) {
        attempts += 1;
        try {
          const health = await getHealth();
          setBootHealth(health);
          setBootStatus(health.startup_error ? "Backend is reachable with startup warnings." : "Backend is ready.");
          setBootReady(true);
          return;
        } catch {
          setBootStatus(`Waiting for local backend (${attempts})...`);
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
      : rightPanel === "trace"
        ? (
          <RetrievalTracePanel
            traces={tracesQuery.data?.traces || []}
            selected={traceQuery.data}
            selectedId={selectedTraceId}
            onSelect={setSelectedTraceId}
          />
        )
      : rightPanel === "health"
        ? <IndexHealthPanel health={indexHealthQuery.data} isLoading={indexHealthQuery.isLoading} />
      : rightPanel === "eval"
        ? <EvaluationPanel runs={evalRunsQuery.data?.runs || []} isRunning={evalMutation.isPending} onRun={(question, expectedDoc) => evalMutation.mutate({ question, expectedDoc })} />
      : rightPanel === "support"
        ? <AnswerSupportPanel support={selectedSupport} />
      : rightPanel === "history"
        ? (
          <ChatHistoryPanel
            conversations={conversationsQuery.data?.conversations || []}
            selectedId={selectedConversationId}
            onSelect={setSelectedConversationId}
            onNew={() => newConversationMutation.mutate()}
            onDelete={(id) => deleteConversationMutation.mutate(id)}
          />
        )
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
          <div className="boot-mark" aria-hidden="true" />
          <strong>Cephalon</strong>
          <span>{bootStatus}</span>
          {bootHealth?.startup_error && <small>{bootHealth.startup_error}</small>}
        </div>
      </div>
    );
  }

  const activeModel = modelsQuery.data?.active_model || "";
  const modelReady = Boolean(selectedModel && activeModel === selectedModel);
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
        center={(
          <ChatPanel
            selectedModel={selectedModel}
            modelReady={modelReady}
            settings={settingsQuery.data}
            conversation={conversationQuery.data}
            selectedConversationId={selectedConversationId}
            onConversationSelected={(id) => {
              setSelectedConversationId(id);
              queryClient.invalidateQueries({ queryKey: ["conversations"] });
            }}
          />
        )}
        modelControl={(
          <div className={modelReady ? "top-model-picker ready" : "top-model-picker"} title="Local GGUF model">
            <label>
              <span>Model</span>
              <select value={selectedModel} onChange={event => setSelectedModel(event.target.value)} disabled={modelsQuery.isLoading || loadModelMutation.isPending}>
                <option value="">{modelsQuery.isLoading ? "Scanning models..." : (modelsQuery.data?.models?.length ? "Select model" : "No chat GGUF models found")}</option>
                {(modelsQuery.data?.models || []).map(model => <option key={model} value={model}>{model}</option>)}
              </select>
            </label>
            <small>{modelReady ? `Loaded / ${contextLabel}` : backendLabel}</small>
            <button
              type="button"
              onClick={() => selectedModel && loadModelMutation.mutate(selectedModel)}
              disabled={!selectedModel || modelReady || loadModelMutation.isPending}
              title={modelReady ? "Selected model is loaded." : "Load selected GGUF model into memory."}
            >
              {loadModelMutation.isPending ? "Loading" : modelReady ? "Loaded" : "Load"}
            </button>
          </div>
        )}
        right={right}
      />
      {toast && <button className="toast" onClick={() => setToast(null)}>{toast}</button>}
    </>
  );
}
