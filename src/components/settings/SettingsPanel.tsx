import { Download, FolderOpen, Save } from "lucide-react";
import { useEffect, useState } from "react";
import type { LlamaServerSettings, OnnxSetupStatus, RagSettings } from "../../api";
import { useUiStore } from "../../store";

type Props = {
  llamaServer?: LlamaServerSettings;
  onSaveLlamaServer?: (settings: LlamaServerSettings) => void;
  isSavingLlamaServer?: boolean;
  ragSettings?: RagSettings;
  onSaveRagSettings?: (settings: RagSettings) => void;
  isSavingRagSettings?: boolean;
  onnxStatus?: OnnxSetupStatus;
  isDownloadingModels?: boolean;
  onDownloadOnnx?: (kind: "embedder" | "reranker" | "all") => void;
  onBrowseOnnx?: (kind: "embedder" | "reranker") => void;
  onExportMetrics?: () => void;
};

export function SettingsPanel({
  llamaServer,
  onSaveLlamaServer,
  isSavingLlamaServer,
  ragSettings,
  onSaveRagSettings,
  isSavingRagSettings,
  onnxStatus,
  isDownloadingModels,
  onDownloadOnnx,
  onBrowseOnnx,
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
          <p className="settings-note">Cephalon does not select or load a GGUF. Start llama.cpp with the model you want, then save its endpoint here and use Connect in the title bar.</p>
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
          <p className="settings-note">
            These two ONNX engines power document search. They are separate from your chat model and external llama.cpp server: the Embedder finds relevant text and the Reranker puts the best matches first.
          </p>
          <div className="onnx-guide">
            <strong>Choose one setup method for each engine:</strong>
            <ol>
              <li><b>Download default</b> fetches Cephalon’s configured Hugging Face export and replaces the engine in the shown destination.</li>
              <li><b>Use local folder</b> copies a compatible exported ONNX folder from your computer. It must contain an ONNX model, <code>tokenizer.json</code>, and <code>tokenizer_config.json</code>.</li>
            </ol>
            <span>Restart Cephalon after installing or replacing either engine; the running backend does not reload ONNX models automatically. Rerankers are run one query/document pair at a time unless their export has been explicitly validated for larger batches.</span>
          </div>
          {onnxStatus && (
            <div className={onnxStatus.engines_ready ? "runtime-line ok" : "runtime-line warn"}>
              {onnxStatus.engines_ready ? "Engines loaded in this backend session." : `Engines not loaded${onnxStatus.startup_error ? `: ${onnxStatus.startup_error}` : "."}`}
            </div>
          )}
          <OnnxRow
            title="Embedder"
            info={onnxStatus?.embedder}
            source={onnxStatus?.download_sources.embedder}
            disabled={isDownloadingModels}
            onDownload={() => onDownloadOnnx?.("embedder")}
            onBrowse={() => onBrowseOnnx?.("embedder")}
          />
          <OnnxRow
            title="Reranker"
            info={onnxStatus?.reranker}
            source={onnxStatus?.download_sources.reranker}
            disabled={isDownloadingModels}
            onDownload={() => onDownloadOnnx?.("reranker")}
            onBrowse={() => onBrowseOnnx?.("reranker")}
          />
          <div className="settings-actions">
            <button type="button" onClick={() => onDownloadOnnx?.("all")} disabled={isDownloadingModels}>
              <Download size={14} />
              {isDownloadingModels ? "Installing" : "Download both defaults"}
            </button>
          </div>
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

function OnnxRow({
  title,
  info,
  source,
  disabled,
  onDownload,
  onBrowse,
}: {
  title: string;
  info?: OnnxSetupStatus["embedder"];
  source?: { repo_id: string; subfolder?: string };
  disabled?: boolean;
  onDownload: () => void;
  onBrowse: () => void;
}) {
  const ready = info?.ok;
  const loaded = info?.runtime_loaded;
  return (
    <div className="onnx-row">
      <div className="onnx-main">
        <strong>{title}</strong>
        <small>{title === "Embedder" ? "Finds semantically relevant document chunks." : "Reorders retrieved chunks by relevance."}</small>
        <span className={loaded ? "status-text ok" : ready ? "status-text warn" : "status-text warn"}>{loaded ? "loaded" : ready ? "installed, restart to load" : "setup needed"}</span>
        <small>Default source: {source?.repo_id || "not configured"}{source?.subfolder ? ` / ${source.subfolder}` : ""}</small>
        <small>Installed at: <code>{info?.path || "not checked"}</code></small>
        {title === "Reranker" && <small>Validated batch size: {info?.max_batch_size || 1} pair{(info?.max_batch_size || 1) === 1 ? "" : "s"} per inference.</small>}
        {!ready && <em>{info?.meta_error || (info?.missing?.length ? `missing ${info.missing.join(", ")}` : "status unavailable")}</em>}
      </div>
      <div className="onnx-actions">
        <button type="button" onClick={onDownload} disabled={disabled} title={`Download and install the default ${title.toLowerCase()} engine`}>
          <Download size={14} />
          Download default
        </button>
        <button type="button" onClick={onBrowse} disabled={disabled} title={`Copy an exported ${title.toLowerCase()} ONNX folder into Cephalon`}>
          <FolderOpen size={14} />
          Use local folder
        </button>
      </div>
    </div>
  );
}
