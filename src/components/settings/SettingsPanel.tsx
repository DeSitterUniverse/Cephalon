import { Download, FolderOpen } from "lucide-react";
import type { OnnxSetupStatus, RagSettings } from "../../api";
import { useUiStore } from "../../store";

type Props = {
  models: string[];
  selectedModel: string;
  setSelectedModel: (model: string) => void;
  settings?: RagSettings;
  onnxStatus?: OnnxSetupStatus;
  isDownloadingModels?: boolean;
  updateSettings: (settings: RagSettings) => void;
  onDownloadOnnx?: (kind: "embedder" | "reranker" | "all") => void;
  onBrowseOnnx?: (kind: "embedder" | "reranker") => void;
  onExportMetrics?: () => void;
};

type SettingKey = keyof RagSettings;

export function SettingsPanel({
  models,
  selectedModel,
  setSelectedModel,
  settings,
  onnxStatus,
  isDownloadingModels,
  updateSettings,
  onDownloadOnnx,
  onBrowseOnnx,
  onExportMetrics,
}: Props) {
  const theme = useUiStore(state => state.theme);
  const setTheme = useUiStore(state => state.setTheme);

  if (!settings) return <div className="empty-state">Settings unavailable.</div>;

  const setValue = (key: SettingKey, value: number) => updateSettings({ ...settings, [key]: value });
  const setBool = (key: SettingKey, value: boolean) => updateSettings({ ...settings, [key]: value });

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
            Use ONNX Runtime folders that contain model.onnx, tokenizer files, onnx_profile.json, and any external ONNX data files. Install default engines pulls the configured embedder and reranker. Browse selects an exported local folder.
          </p>
          {onnxStatus && (
            <div className={onnxStatus.engines_ready ? "runtime-line ok" : "runtime-line warn"}>
              {onnxStatus.engines_ready ? "Engines loaded in this backend session." : `Engines not loaded${onnxStatus.startup_error ? `: ${onnxStatus.startup_error}` : "."}`}
            </div>
          )}
          <OnnxRow
            title="Embedder"
            info={onnxStatus?.embedder}
            source={onnxStatus?.download_sources.embedder.repo_id}
            disabled={isDownloadingModels}
            onDownload={() => onDownloadOnnx?.("embedder")}
            onBrowse={() => onBrowseOnnx?.("embedder")}
          />
          <OnnxRow
            title="Reranker"
            info={onnxStatus?.reranker}
            source={onnxStatus?.download_sources.reranker.repo_id}
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
          <h3>Generation</h3>
          <SettingSlider label="Temperature" value={settings.temperature} min={0} max={2} step={0.05} onChange={value => setValue("temperature", value)} />
          <SettingSlider label="Max tokens" value={settings.max_tokens} min={64} max={4096} step={64} onChange={value => setValue("max_tokens", value)} />
          <SettingSlider label="Context" value={settings.context_tokens} min={4096} max={131072} step={4096} onChange={value => setValue("context_tokens", value)} />
          <label className="field checkbox-field">
            <span>Full model context<strong>{settings.full_context ? "on" : "off"}</strong></span>
            <input type="checkbox" checked={settings.full_context} onChange={event => setBool("full_context", event.target.checked)} />
          </label>
        </section>

        <section className="settings-section">
          <h3>Retrieval</h3>
          <SettingSlider label="Top K" value={settings.top_k} min={1} max={60} step={1} onChange={value => setValue("top_k", value)} />
          <SettingSlider label="Rerank" value={settings.rerank_top_n} min={1} max={10} step={1} onChange={value => setValue("rerank_top_n", value)} />
          <SettingSlider label="Min confidence" value={settings.no_answer_min_confidence} min={0} max={1} step={0.05} onChange={value => setValue("no_answer_min_confidence", value)} />
          <SettingSlider label="Min rerank" value={settings.no_answer_min_rerank_score} min={0} max={2} step={0.05} onChange={value => setValue("no_answer_min_rerank_score", value)} />
          <SettingSlider label="Min dense" value={settings.no_answer_min_vector_score} min={0} max={1} step={0.05} onChange={value => setValue("no_answer_min_vector_score", value)} />
          <SettingSlider label="Min sources" value={settings.no_answer_min_source_count} min={0} max={5} step={1} onChange={value => setValue("no_answer_min_source_count", value)} />
          <label className="field checkbox-field">
            <span>Save retrieval traces<strong>{settings.trace_persistence ? "on" : "off"}</strong></span>
            <input type="checkbox" checked={settings.trace_persistence} onChange={event => setBool("trace_persistence", event.target.checked)} />
          </label>
        </section>

        <section className="settings-section">
          <h3>Indexing</h3>
          <SettingSlider label="Chunk size" value={settings.chunk_size} min={512} max={4000} step={128} onChange={value => setValue("chunk_size", value)} />
          <SettingSlider label="Overlap" value={settings.chunk_overlap} min={0} max={800} step={25} onChange={value => setValue("chunk_overlap", value)} />
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
  source?: string;
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
        {source && (
          <small>
            Source: <a href={`https://huggingface.co/${source}`} target="_blank" rel="noreferrer">{source}</a>
          </small>
        )}
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

function SettingSlider({ label, value, min, max, step, onChange }: { label: string; value: number; min: number; max: number; step: number; onChange: (value: number) => void }) {
  return (
    <label className="field">
      <span>{label}<strong>{value}</strong></span>
      <input type="range" value={value} min={min} max={max} step={step} onChange={event => onChange(Number(event.target.value))} />
    </label>
  );
}
