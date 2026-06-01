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
            <span>GGUF model<strong>{models.length} found</strong></span>
            <select aria-label="Model" value={selectedModel} onChange={event => setSelectedModel(event.target.value)}>
              <option value="">No model selected</option>
              {models.map(model => <option key={model} value={model}>{model}</option>)}
            </select>
          </label>
        </section>

        <section className="settings-section">
          <h3>Embedding and reranking</h3>
          <p className="settings-note">
            Use ONNX Runtime folders that contain model.onnx, tokenizer files, onnx_profile.json, and any external ONNX data files. Install the defaults or browse to exported local folders.
          </p>
          {onnxStatus && (
            <div className={onnxStatus.engines_ready ? "runtime-line ok" : "runtime-line warn"}>
              {onnxStatus.engines_ready ? "Engines loaded in this backend session." : `Engines not loaded${onnxStatus.startup_error ? `: ${onnxStatus.startup_error}` : "."}`}
            </div>
          )}
          <OnnxRow
            title="Embedder"
            info={onnxStatus?.embedder}
            disabled={isDownloadingModels}
            onDownload={() => onDownloadOnnx?.("embedder")}
            onBrowse={() => onBrowseOnnx?.("embedder")}
          />
          <OnnxRow
            title="Reranker"
            info={onnxStatus?.reranker}
            disabled={isDownloadingModels}
            onDownload={() => onDownloadOnnx?.("reranker")}
            onBrowse={() => onBrowseOnnx?.("reranker")}
          />
          <div className="settings-actions">
            <button type="button" onClick={() => onDownloadOnnx?.("all")} disabled={isDownloadingModels}>
              <Download size={14} />
              {isDownloadingModels ? "Installing" : "Install default engines"}
            </button>
          </div>
        </section>

        <section className="settings-section">
          <h3>Data</h3>
          <p className="settings-note">
            Generation and retrieval use app defaults plus the Scope selector in the chat bar. The local model handles its own response style.
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
  disabled,
  onDownload,
  onBrowse,
}: {
  title: string;
  info?: OnnxSetupStatus["embedder"];
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
        <span className={loaded ? "status-text ok" : ready ? "status-text warn" : "status-text warn"}>{loaded ? "loaded" : ready ? "installed, restart to load" : "setup needed"}</span>
        <code>{info?.path || "not checked"}</code>
        {!ready && <em>{info?.meta_error || (info?.missing?.length ? `missing ${info.missing.join(", ")}` : "status unavailable")}</em>}
      </div>
      <div className="onnx-actions">
        <button type="button" onClick={onDownload} disabled={disabled} title={`Install default ${title.toLowerCase()} engine`}>
          <Download size={14} />
        </button>
        <button type="button" onClick={onBrowse} disabled={disabled} title={`Select exported ${title.toLowerCase()} ONNX folder`}>
          <FolderOpen size={14} />
        </button>
      </div>
    </div>
  );
}
