import { useEffect, useState } from "react";
import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { listen } from "@tauri-apps/api/event";
import { invoke } from "@tauri-apps/api/core";
import { open as openDialog } from "@tauri-apps/plugin-dialog";
import {
  addDocumentTag,
  createConversation,
  deleteConversation,
  deleteDocument,
  deleteDocumentTag,
  downloadOnnxModels,
  exportMetrics,
  getConversation,
  getConversations,
  getDocument,
  getDocuments,
  getHealth,
  getIndexHealth,
  getLlamaServerSettings,
  getModels,
  getOnnxSetupStatus,
  getEvalRuns,
  getRetrievalTrace,
  getRetrievalTraces,
  getSettings,
  ingestPath,
  installLocalOnnxModel,
  loadModel,
  reindexDocument,
  renameConversation,
  runEval,
  updateDocument,
  updateLlamaServerSettings,
  updateSettings,
  type Document,
  type HealthResponse,
  type OnnxModelKind,
} from "./api";
import { ChatPanel } from "./components/chat/ChatPanel";
import { ConfirmDialog } from "./components/feedback/ConfirmDialog";
import { NotificationCenter } from "./components/feedback/NotificationCenter";
import { AnswerSupportPanel } from "./components/diagnostics/AnswerSupportPanel";
import { EvaluationPanel } from "./components/diagnostics/EvaluationPanel";
import { IndexHealthPanel } from "./components/diagnostics/IndexHealthPanel";
import { RetrievalTracePanel } from "./components/diagnostics/RetrievalTracePanel";
import { ChatHistoryPanel } from "./components/chat/ChatHistoryPanel";
import { LibraryPanel } from "./components/library/LibraryPanel";
import { DocumentDetails } from "./components/library/DocumentDetails";
import { WorkbenchLayout } from "./components/layout/WorkbenchLayout";
import { ModelPicker } from "./components/model/ModelPicker";
import { SettingsPanel } from "./components/settings/SettingsPanel";
import { SourcesPanel } from "./components/sources/SourcesPanel";
import { useEventStream } from "./hooks/useEventStream";
import { useNotifications } from "./hooks/useNotifications";
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
  const [bootError, setBootError] = useState<string | null>(null);
  const [bootCycle, setBootCycle] = useState(0);
  const [isRetryingBackend, setIsRetryingBackend] = useState(false);
  const [search, setSearch] = useState("");
  const [statusFilter, setStatusFilter] = useState("all");
  const { notify, confirm } = useNotifications();
  const selectedModel = useUiStore(state => state.selectedModel);
  const setSelectedModel = useUiStore(state => state.setSelectedModel);
  const selectedDocumentId = useUiStore(state => state.selectedDocumentId);
  const setSelectedDocumentId = useUiStore(state => state.setSelectedDocumentId);
  const selectedConversationId = useUiStore(state => state.selectedConversationId);
  const setSelectedConversationId = useUiStore(state => state.setSelectedConversationId);
  const selectedSources = useUiStore(state => state.selectedSources);
  const selectedSupport = useUiStore(state => state.selectedSupport);
  const rightPanel = useUiStore(state => state.rightPanel);
  const setRightPanel = useUiStore(state => state.setRightPanel);
  const [selectedTraceId, setSelectedTraceId] = useState<string | null>(null);

  useEventStream();

  const modelsQuery = useQuery({ queryKey: ["models"], queryFn: getModels, enabled: bootReady, refetchInterval: 4000, refetchOnWindowFocus: true });
  const llamaServerQuery = useQuery({ queryKey: ["llama-server"], queryFn: getLlamaServerSettings, enabled: bootReady });
  const documentsQuery = useQuery({ queryKey: ["documents"], queryFn: getDocuments, enabled: bootReady });
  const conversationsQuery = useQuery({ queryKey: ["conversations"], queryFn: getConversations, enabled: bootReady });
  const settingsQuery = useQuery({ queryKey: ["settings"], queryFn: getSettings, enabled: bootReady });
  const onnxStatusQuery = useQuery({ queryKey: ["onnx-setup"], queryFn: getOnnxSetupStatus, enabled: bootReady && rightPanel === "settings" });
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
      notify(data.message || "Ingestion queued.", "success");
    },
    onError: error => notify(error instanceof Error ? error.message : "Failed to queue ingestion.", "error"),
  });

  const deleteMutation = useMutation({
    mutationFn: (doc: Document) => deleteDocument(doc.id),
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
    },
  });

  const reindexMutation = useMutation({
    mutationFn: (doc: Document) => reindexDocument(doc.id),
    onSuccess: data => {
      notify(data.message || "Reindex queued.", "success");
    },
  });

  const loadModelMutation = useMutation({
    mutationFn: () => loadModel(),
    onMutate: () => notify("Connecting to external llama.cpp server..."),
    onSuccess: data => {
      notify(data.active_model ? `Connected to ${data.active_model}` : "Connected to llama.cpp server.", "success");
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
    onError: error => notify(error instanceof Error ? error.message : "Failed to connect to the external llama.cpp server.", "error"),
  });

  const llamaServerMutation = useMutation({
    mutationFn: updateLlamaServerSettings,
    onSuccess: data => {
      notify(`Saved llama.cpp endpoint: ${data.server_url}`, "success");
      queryClient.setQueryData(["llama-server"], data);
      queryClient.invalidateQueries({ queryKey: ["models"] });
    },
    onError: error => notify(error instanceof Error ? error.message : "Failed to save llama.cpp server settings.", "error"),
  });

  const settingsMutation = useMutation({
    mutationFn: updateSettings,
    onSuccess: data => {
      queryClient.setQueryData(["settings"], data);
      notify("Retrieval settings saved. Reindex documents after changing chunk boundaries.", "success");
    },
    onError: error => notify(error instanceof Error ? error.message : "Failed to save retrieval settings.", "error"),
  });

  const metricsMutation = useMutation({
    mutationFn: exportMetrics,
    onSuccess: data => notify(
      data.status === "success" && data.path ? `Metrics exported: ${data.path}` : `Metrics export failed: ${data.error || "metrics directory is unavailable"}`,
      data.status === "success" ? "success" : "error",
    ),
    onError: error => notify(error instanceof Error ? error.message : "Failed to export metrics.", "error"),
  });

  const onnxDownloadMutation = useMutation({
    mutationFn: (kind: OnnxModelKind) => downloadOnnxModels(kind),
    onMutate: (kind) => notify(kind === "all" ? "Downloading ONNX engines..." : `Downloading ${kind} ONNX engine...`),
    onSuccess: () => {
      notify("ONNX model setup finished. Restart the backend to load the new engines.", "success");
      queryClient.invalidateQueries({ queryKey: ["onnx-setup"] });
      queryClient.invalidateQueries({ queryKey: ["health"] });
    },
    onError: error => notify(error instanceof Error ? error.message : "Failed to set up ONNX models.", "error"),
  });

  const onnxInstallMutation = useMutation({
    mutationFn: ({ kind, path }: { kind: "embedder" | "reranker"; path: string }) => installLocalOnnxModel(kind, path),
    onSuccess: () => {
      notify("ONNX model folder installed. Restart the backend to load the new engine.", "success");
      queryClient.invalidateQueries({ queryKey: ["onnx-setup"] });
      queryClient.invalidateQueries({ queryKey: ["health"] });
    },
    onError: error => notify(error instanceof Error ? error.message : "Failed to install ONNX model folder.", "error"),
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
      notify("Eval run saved.", "success");
      queryClient.invalidateQueries({ queryKey: ["eval-runs"] });
    },
    onError: error => notify(error instanceof Error ? error.message : "Eval run failed.", "error"),
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
      const startedAt = Date.now();
      setBootReady(false);
      setBootError(null);
      while (active) {
        attempts += 1;
        try {
          const health = await getHealth();
          setBootHealth(health);
          setBootStatus(health.startup_error ? "Local backend reached; retrieval setup needs attention." : "Local backend is ready.");
          if (!health.engines_ready || health.onnx_setup?.startup_error) {
            setRightPanel("settings");
            notify("Embedding and reranking setup needed.", "error");
          }
          setBootReady(true);
          return;
        } catch (error) {
          const elapsed = Date.now() - startedAt;
          if (elapsed >= 15_000) {
            const detail = error instanceof Error && error.message ? ` ${error.message}` : "";
            let processDetail = "";
            if (isTauriRuntime()) {
              try {
                const backend = await invoke<{ managed_process: boolean; exited: boolean; exit_code?: number | null; address: string }>("get_backend_status");
                if (backend.exited) processDetail = ` The managed backend exited${backend.exit_code == null ? "" : ` with code ${backend.exit_code}`}.`;
                else if (!backend.managed_process) processDetail = ` No managed backend process is running for ${backend.address}.`;
              } catch {
                processDetail = "";
              }
            }
            setBootStatus("Local backend is unavailable.");
            setBootError(`Cephalon could not reach its local backend after ${attempts} attempts.${detail}${processDetail}`);
            return;
          }
          setBootStatus(`Starting local backend… attempt ${attempts}`);
        }
        await new Promise(resolve => setTimeout(resolve, 750));
      }
    };
    poll();
    return () => { active = false; };
  }, [bootCycle, notify, setRightPanel]);

  async function retryBackendStartup() {
    setIsRetryingBackend(true);
    try {
      if (isTauriRuntime()) await invoke("restart_backend");
      setBootStatus("Retrying local backend...");
      setBootCycle(value => value + 1);
    } catch (error) {
      setBootStatus("Local backend could not be restarted.");
      setBootError(error instanceof Error ? error.message : "Cephalon could not restart its local backend.");
    } finally {
      setIsRetryingBackend(false);
    }
  }

  useEffect(() => {
    const configuredName = llamaServerQuery.data?.model_name || modelsQuery.data?.llama_backend?.model_name || "External llama.cpp server";
    if (selectedModel !== configuredName) setSelectedModel(configuredName);
  }, [llamaServerQuery.data?.model_name, modelsQuery.data?.llama_backend?.model_name, selectedModel, setSelectedModel]);

  useEffect(() => {
    const firstConversation = conversationsQuery.data?.conversations?.[0];
    if (!selectedConversationId && firstConversation?.id) setSelectedConversationId(firstConversation.id);
  }, [conversationsQuery.data, selectedConversationId, setSelectedConversationId]);

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
      notify("Folder import is available in the Tauri app. Use drag/drop or the API in browser dev mode.");
      return;
    }
    const selectedPath = await openDialog({ directory: true, multiple: false });
    if (selectedPath && typeof selectedPath === "string") ingestMutation.mutate(selectedPath);
  }

  async function importTextFile() {
    if (!isTauriRuntime()) {
      notify("Text import is available in the Tauri app. Use the API in browser dev mode.");
      return;
    }
    const selectedPath = await openDialog({ directory: false, multiple: false });
    if (selectedPath && typeof selectedPath === "string") ingestPath(selectedPath, true)
      .then(data => {
        notify(data.message || "Text ingestion queued.", "success");
      })
      .catch(error => notify(error instanceof Error ? error.message : "Failed to queue text import.", "error"));
  }

  async function browseOnnxFolder(kind: "embedder" | "reranker") {
    if (!isTauriRuntime()) {
      notify("Model folder browsing is available in the Tauri app.");
      return;
    }
    const defaultPath = onnxStatusQuery.data?.[kind]?.path || onnxStatusQuery.data?.model_dir;
    const selectedPath = await openDialog({ directory: true, multiple: false, defaultPath });
    if (selectedPath && typeof selectedPath === "string") onnxInstallMutation.mutate({ kind, path: selectedPath });
  }

  function removeDocument(doc: Document) {
    confirm({
      title: "Delete document?",
      message: `Delete ${doc.name} from the library and remove its indexed content?`,
      confirmLabel: "Delete",
      onConfirm: () => deleteMutation.mutate(doc),
    });
  }

  const right = rightPanel === "settings"
    ? (
      <SettingsPanel
        llamaServer={llamaServerQuery.data}
        onSaveLlamaServer={(settings) => llamaServerMutation.mutate(settings)}
        isSavingLlamaServer={llamaServerMutation.isPending}
        ragSettings={settingsQuery.data}
        onSaveRagSettings={(settings) => settingsMutation.mutate(settings)}
        isSavingRagSettings={settingsMutation.isPending}
        onnxStatus={onnxStatusQuery.data}
        isDownloadingModels={onnxDownloadMutation.isPending || onnxInstallMutation.isPending}
        onDownloadOnnx={(kind) => onnxDownloadMutation.mutate(kind)}
        onBrowseOnnx={browseOnnxFolder}
        onExportMetrics={() => metricsMutation.mutate()}
      />
    )
    : rightPanel === "sources"
      ? <SourcesPanel sources={selectedSources} onOpenDocument={setSelectedDocumentId} />
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
            onRename={(id, title) => renameConversation(id, title).then(() => {
              queryClient.invalidateQueries({ queryKey: ["conversations"] });
              queryClient.invalidateQueries({ queryKey: ["conversation", id] });
            })}
            onDelete={(id) => {
              const conversation = conversationsQuery.data?.conversations.find(item => item.id === id);
              confirm({
                title: "Delete chat?",
                message: `Delete ${conversation?.title || "this chat"} and its saved messages?`,
                confirmLabel: "Delete",
                onConfirm: () => deleteConversationMutation.mutate(id),
              });
            }}
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
        : null;

  if (!bootReady) {
    return (
      <div className="boot-screen">
        <div className="boot-card">
          <div className="boot-mark" aria-hidden="true" />
          <strong>Cephalon</strong>
          <span>{bootStatus}</span>
          {bootHealth?.startup_error && <small>{bootHealth.startup_error}</small>}
          {bootError && <small className="boot-error">{bootError}</small>}
          {bootError && <button type="button" className="boot-retry" onClick={() => void retryBackendStartup()} disabled={isRetryingBackend}>{isRetryingBackend ? "Retrying…" : isTauriRuntime() ? "Retry local backend" : "Retry connection"}</button>}
        </div>
      </div>
    );
  }

  const activeModel = modelsQuery.data?.active_model || "";
  const serverAvailable = modelsQuery.data?.llama_backend?.server_available;
  const modelReady = Boolean(activeModel && serverAvailable);
  const backendLabel = modelsQuery.data?.llama_backend?.backend_label || "External llama.cpp server";

  return (
    <>
      <WorkbenchLayout
        onNewChat={() => newConversationMutation.mutate()}
        newChatDisabled={newConversationMutation.isPending}
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
            onLoadOlder={() => {
              const current = conversationQuery.data;
              if (!selectedConversationId || !current?.next_before) return;
              getConversation(selectedConversationId, 100, current.next_before).then(page => {
                queryClient.setQueryData(["conversation", selectedConversationId], {
                  ...current,
                  messages: [...(page.messages || []), ...(current.messages || [])],
                  has_more: page.has_more,
                  next_before: page.next_before,
                });
              }).catch(error => notify(error instanceof Error ? error.message : "Failed to load older messages.", "error"));
            }}
          />
        )}
        modelControl={(
          <ModelPicker
            activeModel={activeModel}
            backendLabel={backendLabel}
            serverUrl={modelsQuery.data?.llama_backend?.server_url}
            serverAvailable={serverAvailable}
            serverError={modelsQuery.data?.llama_backend?.server_error}
            contextTokens={modelsQuery.data?.active_context_tokens}
            isLoading={loadModelMutation.isPending}
            onLoad={() => loadModelMutation.mutate()}
          />
        )}
        right={right}
      />
      <NotificationCenter />
      <ConfirmDialog />
    </>
  );
}
