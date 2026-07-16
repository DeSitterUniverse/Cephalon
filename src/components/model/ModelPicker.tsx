import { ChevronDown, HardDrive, Loader2 } from "lucide-react";
import { useState } from "react";
type Props = {
  models: string[];
  modelDetails?: Array<{ name: string; size_bytes: number }>;
  selectedModel: string;
  activeModel?: string | null;
  backendLabel?: string;
  contextTokens?: number | null;
  isScanning?: boolean;
  isLoading?: boolean;
  onSelect: (model: string) => void;
  onLoad: () => void;
};

function compactName(name: string) {
  return name.replace(/\.gguf$/i, "").replace(/[-_]+/g, " ");
}

export function ModelPicker({ models, selectedModel, activeModel, backendLabel, contextTokens, isScanning, isLoading, onSelect, onLoad }: Props) {
  const [open, setOpen] = useState(false);
  const loaded = Boolean(selectedModel && activeModel === selectedModel);
  const contextLabel = contextTokens ? `${Math.round(contextTokens / 1024)}k ctx` : backendLabel || "llama.cpp";

  return (
    <div className={loaded ? "model-picker loaded" : "model-picker"}>
      <button className="model-trigger" type="button" onClick={() => setOpen(value => !value)} disabled={isLoading} title="Select configured external llama.cpp server">
        <span className="model-title">{selectedModel ? compactName(selectedModel) : isScanning ? "Scanning models" : "Select model"}</span>
        <span className="model-meta">
          <HardDrive size={12} />
          {selectedModel ? `${contextLabel}` : `${models.length} available`}
        </span>
        <ChevronDown size={15} />
      </button>
      <button className="model-load" type="button" onClick={onLoad} disabled={!selectedModel || loaded || isLoading}>
        {isLoading ? <Loader2 size={14} className="spin-icon" /> : loaded ? "Connected" : "Connect"}
      </button>
      {open && (
        <div className="model-menu">
          {models.map(model => {
            return (
              <button
                key={model}
                type="button"
                className={model === selectedModel ? "active" : ""}
                onClick={() => {
                  onSelect(model);
                  setOpen(false);
                }}
              >
                <strong>{compactName(model)}</strong>
                <span>External server / {contextLabel}</span>
              </button>
            );
          })}
          {models.length === 0 && <div className="model-empty">External llama.cpp server is not configured.</div>}
        </div>
      )}
    </div>
  );
}
