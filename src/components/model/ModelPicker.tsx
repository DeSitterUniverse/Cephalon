import { ChevronDown, HardDrive, Loader2 } from "lucide-react";
import { useMemo, useState } from "react";
import type { ModelsResponse } from "../../api";

type Props = {
  models: string[];
  modelDetails?: ModelsResponse["model_details"];
  selectedModel: string;
  activeModel?: string | null;
  backendLabel?: string;
  contextTokens?: number | null;
  isScanning?: boolean;
  isLoading?: boolean;
  onSelect: (model: string) => void;
  onLoad: () => void;
};

function sizeLabel(bytes?: number) {
  if (!bytes) return "size unknown";
  const gb = bytes / 1024 / 1024 / 1024;
  return gb >= 1 ? `${gb.toFixed(1)} GB` : `${Math.round(bytes / 1024 / 1024)} MB`;
}

function compactName(name: string) {
  return name.replace(/\.gguf$/i, "").replace(/[-_]+/g, " ");
}

export function ModelPicker({ models, modelDetails, selectedModel, activeModel, backendLabel, contextTokens, isScanning, isLoading, onSelect, onLoad }: Props) {
  const [open, setOpen] = useState(false);
  const detailByName = useMemo(() => new Map((modelDetails || []).map(item => [item.name, item])), [modelDetails]);
  const selectedDetail = selectedModel ? detailByName.get(selectedModel) : undefined;
  const loaded = Boolean(selectedModel && activeModel === selectedModel);
  const contextLabel = contextTokens ? `${Math.round(contextTokens / 1024)}k ctx` : backendLabel || "llama.cpp";

  return (
    <div className={loaded ? "model-picker loaded" : "model-picker"}>
      <button className="model-trigger" type="button" onClick={() => setOpen(value => !value)} disabled={isLoading} title="Select local GGUF model">
        <span className="model-title">{selectedModel ? compactName(selectedModel) : isScanning ? "Scanning models" : "Select model"}</span>
        <span className="model-meta">
          <HardDrive size={12} />
          {selectedModel ? `${sizeLabel(selectedDetail?.size_bytes)} / ${contextLabel}` : `${models.length} available`}
        </span>
        <ChevronDown size={15} />
      </button>
      <button className="model-load" type="button" onClick={onLoad} disabled={!selectedModel || loaded || isLoading}>
        {isLoading ? <Loader2 size={14} className="spin-icon" /> : loaded ? "Loaded" : "Load"}
      </button>
      {open && (
        <div className="model-menu">
          {models.map(model => {
            const detail = detailByName.get(model);
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
                <span>{sizeLabel(detail?.size_bytes)} / GGUF</span>
              </button>
            );
          })}
          {models.length === 0 && <div className="model-empty">No chat GGUF models found.</div>}
        </div>
      )}
    </div>
  );
}
