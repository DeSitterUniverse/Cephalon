import { Download, FolderOpen, Save, ShieldCheck, Trash2 } from "lucide-react";
import { useEffect, useState } from "react";
import type { FixedModelInfo, FixedModelKind, FixedRetrievalStatus, LlamaServerSettings, RagSettings, ReindexProgress } from "../../api";
import { useUiStore } from "../../store";

type Props = {
  llamaServer?: LlamaServerSettings;
  onSaveLlamaServer?: (settings: LlamaServerSettings) => void;
  isSavingLlamaServer?: boolean;
  ragSettings?: RagSettings;
  onSaveRagSettings?: (settings: RagSettings) => void;
  isSavingRagSettings?: boolean;
  retrievalStatus?: FixedRetrievalStatus;
  reindexProgress?: ReindexProgress;
  /** Deprecated test-harness compatibility; ONNX is never rendered or used. */
  onnxStatus?: unknown;
  isDownloadingModels?: boolean;
  onDownloadModel?: (kind: FixedModelKind) => void;
  onVerifyModel?: (kind: FixedModelKind) => void;
  onOpenModel?: (kind: FixedModelKind) => void;
  onDeleteModel?: (kind: FixedModelKind) => void;
  onReindex?: (mode: "full" | "stale") => void;
  onExportMetrics?: () => void;
};

export function SettingsPanel({
  llamaServer,
  onSaveLlamaServer,
  isSavingLlamaServer,
  ragSettings,
  onSaveRagSettings,
  isSavingRagSettings,
  retrievalStatus,
  reindexProgress,
  isDownloadingModels,
  onDownloadModel,
  onVerifyModel,
  onOpenModel,
  onDeleteModel,
  onReindex,
  onExportMetrics,
}: Props) {
  const theme = useUiStore(state => state.theme);
  const setTheme = useUiStore(state => state.setTheme);
  const [serverUrl, setServerUrl] = useState(llamaServer?.server_url || "http://127.0.0.1:8080");
  const [modelName, setModelName] = useState(llamaServer?.model_name || "External llama.cpp server");
  const [contextTokens, setContextTokens] = useState(llamaServer?.context_tokens ? String(llamaServer.context_tokens) : "");
  const [draftRagSettings, setDraftRagSettings] = useState<RagSettings | undefined>(ragSettings);

  useEffect(() => {
    if (!llamaServer) return;
    setServerUrl(llamaServer.server_url);
    setModelName(llamaServer.model_name);
    setContextTokens(llamaServer.context_tokens ? String(llamaServer.context_tokens) : "");
  }, [llamaServer]);

  useEffect(() => setDraftRagSettings(ragSettings), [ragSettings]);

  return (
    <section className="side-section settings-screen">
      <div className="panel-header">
        <div>
          <h2>Settings</h2>
          <span>Appearance, models, retrieval</span>
        </div>
      </div>

      <div className="settings-scroll">
        <section className="settings-section">
          <h3>Appearance</h3>
          <div className="theme-grid">
            <button type="button" className={theme === "black" ? "theme-choice active" : "theme-choice"} onClick={() => setTheme("black")}>
              <strong>Black</strong>
              <span>Pure black workspace</span>
            </button>
            <button type="button" className={theme === "graphite" ? "theme-choice active" : "theme-choice"} onClick={() => setTheme("graphite")}>
              <strong>Graphite</strong>
              <span>Dark gray, white text</span>
            </button>
          </div>
        </section>

        <section className="settings-section">
          <h3>Retrieval behavior</h3>
          <p className="settings-note">These controls affect future queries. Changing chunk boundaries requires reindexing documents before the new settings take effect.</p>
          <label className="checkbox-field"><input type="checkbox" checked={draftRagSettings?.evidence_required || false} onChange={event => setDraftRagSettings(current => current && { ...current, evidence_required: event.target.checked })} />Require local evidence for answers</label>
          <label className="checkbox-field"><input type="checkbox" checked={draftRagSettings?.conversation_memory ?? true} onChange={event => setDraftRagSettings(current => current && { ...current, conversation_memory: event.target.checked })} />Use past chats as local retrieval memory</label>
          <details className="onnx-guide">
            <summary>Advanced chunking</summary>
            <div className="settings-grid">
              <NumberSetting label="Parent target" value={draftRagSettings?.parent_target_tokens} onChange={value => setDraftRagSettings(current => current && { ...current, parent_target_tokens: value })} />
              <NumberSetting label="Parent maximum" value={draftRagSettings?.parent_max_tokens} onChange={value => setDraftRagSettings(current => current && { ...current, parent_max_tokens: value })} />
              <NumberSetting label="Child target" value={draftRagSettings?.child_target_tokens} onChange={value => setDraftRagSettings(current => current && { ...current, child_target_tokens: value })} />
              <NumberSetting label="Child maximum" value={draftRagSettings?.child_max_tokens} onChange={value => setDraftRagSettings(current => current && { ...current, child_max_tokens: value })} />
              <NumberSetting label="Child overlap" value={draftRagSettings?.child_overlap_tokens} onChange={value => setDraftRagSettings(current => current && { ...current, child_overlap_tokens: value })} />
            </div>
          </details>
          <div className="settings-actions"><button type="button" disabled={!draftRagSettings || isSavingRagSettings} onClick={() => draftRagSettings && onSaveRagSettings?.(draftRagSettings)}><Save size={14} />{isSavingRagSettings ? "Saving" : "Save retrieval settings"}</button></div>
        </section>

        <section className="settings-section">
          <h3>Chat model</h3>
          <p className="settings-note">Start llama.cpp with the model you want, then save its endpoint here and use Connect in the title bar.</p>
          <label className="field compact-field">
            <span>llama.cpp URL</span>
            <input aria-label="llama.cpp URL" value={serverUrl} onChange={event => setServerUrl(event.target.value)} placeholder="http://127.0.0.1:8080" />
          </label>
          <label className="field compact-field">
            <span>Server label</span>
            <input aria-label="Server label" value={modelName} onChange={event => setModelName(event.target.value)} placeholder="External llama.cpp server" />
          </label>
          <label className="field compact-field">
            <span>Context window (optional)</span>
            <input aria-label="Context window" inputMode="numeric" value={contextTokens} onChange={event => setContextTokens(event.target.value.replace(/[^0-9]/g, ""))} placeholder="For example, 32768" />
          </label>
          <div className="settings-actions"><button type="button" disabled={isSavingLlamaServer} onClick={() => onSaveLlamaServer?.({ server_url: serverUrl, model_name: modelName, context_tokens: contextTokens ? Number(contextTokens) : null })}><Save size={14} />{isSavingLlamaServer ? "Saving" : "Save server settings"}</button></div>
        </section>

        <section className="settings-section">
          <h3>Embedding and reranking</h3>
          <p className="settings-note">Cephalon uses one fixed local retrieval stack. It is separate from your chat-generation server; switching backends or dimensions is not supported.</p>
          <FixedModelRow info={retrievalStatus?.embedder} disabled={isDownloadingModels} onDownload={onDownloadModel} onVerify={onVerifyModel} onOpen={onOpenModel} onDelete={onDeleteModel} />
          <FixedModelRow info={retrievalStatus?.reranker} disabled={isDownloadingModels} onDownload={onDownloadModel} onVerify={onVerifyModel} onOpen={onOpenModel} onDelete={onDeleteModel} />
          <div className={retrievalStatus?.reindex_required ? "runtime-line warn" : "runtime-line ok"}>
            {retrievalStatus?.reindex_required ? "Reindex required: old 1024-dimensional vectors are blocked." : "768-dimensional Jina Nano index is active."}
          </div>
          <div className="settings-actions">
            <button type="button" disabled={isDownloadingModels} onClick={() => onReindex?.("stale")}>Reindex stale only</button>
            <button type="button" disabled={isDownloadingModels} onClick={() => onReindex?.("full")}>Reindex all documents</button>
          </div>
          {reindexProgress && <small>Reindex {reindexProgress.status}: {reindexProgress.processed} / {reindexProgress.total} · {reindexProgress.succeeded} succeeded · {reindexProgress.failed} failed · {reindexProgress.stale_document_count} stale document{reindexProgress.stale_document_count === 1 ? "" : "s"}</small>}
        </section>

        <section className="settings-section">
          <h3>Data</h3>
          <p className="settings-note">
            Generation and retrieval use app defaults plus the Scope selector in the chat bar. Your external llama.cpp server model handles its own response style.
          </p>
          <div className="settings-actions">
            <button type="button" onClick={onExportMetrics}>Export metrics CSV</button>
          </div>
        </section>
      </div>
    </section>
  );
}

function NumberSetting({ label, value, onChange }: { label: string; value?: number; onChange: (value: number) => void }) {
  return <label className="field compact-field"><span>{label} tokens</span><input aria-label={label} inputMode="numeric" value={value ?? ""} onChange={event => onChange(Number(event.target.value.replace(/[^0-9]/g, "")) || 0)} /></label>;
}

function FixedModelRow({
  info,
  disabled,
  onDownload,
  onVerify,
  onOpen,
  onDelete,
}: {
  info?: FixedModelInfo;
  disabled?: boolean;
  onDownload?: (kind: FixedModelKind) => void;
  onVerify?: (kind: FixedModelKind) => void;
  onOpen?: (kind: FixedModelKind) => void;
  onDelete?: (kind: FixedModelKind) => void;
}) {
  if (!info) return <div className="onnx-row"><small>Checking fixed retrieval stack…</small></div>;
  const runtime = info.runtime as { status?: string; last_error?: string; last_failure?: string; port?: number; pid?: number; queue_size?: number };
  return (
    <div className="onnx-row">
      <div className="onnx-main">
        <strong>{info.name}</strong>
        <small>{info.kind === "embedder" ? "768-dimensional Nano Retrieval via dedicated llama.cpp." : "Jina v3.5 listwise custom Transformers worker."}</small>
        <span className={runtime.status === "running" ? "status-text ok" : "status-text warn"}>{runtime.status || (info.installed ? "stopped" : "not installed")}</span>
        <small>Revision: {info.revision} · Path: <code>{info.path}</code></small>
        <small>{info.kind === "embedder" ? `Port ${runtime.port ?? "—"} · PID ${runtime.pid ?? "—"} · fixed ${info.dimension}-dim` : `Worker PID ${runtime.pid ?? "—"} · queue ${runtime.queue_size ?? 0} · trust_remote_code`}</small>
        {(runtime.last_error || runtime.last_failure) && <em>{runtime.last_error || runtime.last_failure}</em>}
      </div>
      <div className="onnx-actions">
        <button type="button" onClick={() => onDownload?.(info.kind)} disabled={disabled}>
          <Download size={14} />
          Download
        </button>
        <button type="button" onClick={() => onVerify?.(info.kind)} disabled={disabled || !info.installed}><ShieldCheck size={14} />Verify</button>
        <button type="button" onClick={() => onOpen?.(info.kind)} disabled={disabled}>
          <FolderOpen size={14} />
          Open folder
        </button>
        <button type="button" onClick={() => onDelete?.(info.kind)} disabled={disabled || !info.installed}><Trash2 size={14} />Delete cache</button>
      </div>
    </div>
  );
}
