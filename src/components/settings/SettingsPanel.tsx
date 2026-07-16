import { Download, FolderOpen } from "lucide-react";
import type { OnnxSetupStatus } from "../../api";
import { useUiStore } from "../../store";

type Props = {
  models: string[];
  selectedModel: string;
  setSelectedModel: (model: string) => void;
  onnxStatus?: OnnxSetupStatus;
  isDownloadingModels?: boolean;
  onDownloadOnnx?: (kind: "embedder" | "reranker" | "all") => void;
  onBrowseOnnx?: (kind: "embedder" | "reranker") => void;
  onExportMetrics?: () => void;
};

export function SettingsPanel({
  models,
  selectedModel,
  setSelectedModel,
  onnxStatus,
  isDownloadingModels,
  onDownloadOnnx,
  onBrowseOnnx,
  onExportMetrics,
}: Props) {
  const theme = useUiStore(state => state.theme);
  const setTheme = useUiStore(state => state.setTheme);

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
          <h3>Chat model</h3>
          <label className="field compact-field">
            <span>External llama.cpp server<strong>{models.length ? "configured" : "not configured"}</strong></span>
            <select aria-label="Model" value={selectedModel} onChange={event => setSelectedModel(event.target.value)}>
              <option value="">No model selected</option>
              {models.map(model => <option key={model} value={model}>{model}</option>)}
            </select>
          </label>
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
            <span>Restart Cephalon after installing or replacing either engine; the running backend does not reload ONNX models automatically.</span>
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
