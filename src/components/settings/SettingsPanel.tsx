import type { RagSettings } from "../../api";

type Props = {
  models: string[];
  selectedModel: string;
  setSelectedModel: (model: string) => void;
  settings?: RagSettings;
  updateSettings: (settings: RagSettings) => void;
  onExportMetrics?: () => void;
};

type SettingKey = keyof RagSettings;

export function SettingsPanel({ models, selectedModel, setSelectedModel, settings, updateSettings, onExportMetrics }: Props) {
  if (!settings) return <div className="empty-state">Settings unavailable.</div>;

  const setValue = (key: SettingKey, value: number) => updateSettings({ ...settings, [key]: value });
  const setBool = (key: SettingKey, value: boolean) => updateSettings({ ...settings, [key]: value });

  return (
    <section className="side-section">
      <div className="panel-header">
        <div>
          <h2>Controls</h2>
          <span>Model and retrieval tuning</span>
        </div>
      </div>
      <label className="field">
        <span>Model</span>
        <select value={selectedModel} onChange={event => setSelectedModel(event.target.value)}>
          <option value="">No model selected</option>
          {models.map(model => <option key={model} value={model}>{model}</option>)}
        </select>
      </label>

      <SettingSlider label="Temperature" value={settings.temperature} min={0} max={2} step={0.05} onChange={value => setValue("temperature", value)} />
      <SettingSlider label="Max tokens" value={settings.max_tokens} min={64} max={4096} step={64} onChange={value => setValue("max_tokens", value)} />
      <SettingSlider label="Context" value={settings.context_tokens} min={4096} max={131072} step={4096} onChange={value => setValue("context_tokens", value)} />
      <label className="field checkbox-field">
        <span>Full model context<strong>{settings.full_context ? "on" : "off"}</strong></span>
        <input type="checkbox" checked={settings.full_context} onChange={event => setBool("full_context", event.target.checked)} />
      </label>
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
      <SettingSlider label="Chunk size" value={settings.chunk_size} min={512} max={4000} step={128} onChange={value => setValue("chunk_size", value)} />
      <SettingSlider label="Overlap" value={settings.chunk_overlap} min={0} max={800} step={25} onChange={value => setValue("chunk_overlap", value)} />
      <div className="action-row">
        <button type="button" onClick={onExportMetrics}>Export metrics CSV</button>
      </div>
    </section>
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
