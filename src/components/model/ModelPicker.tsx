import { CircleAlert, CircleCheck, Loader2, PlugZap } from "lucide-react";
type Props = {
  activeModel?: string | null;
  backendLabel?: string;
  serverUrl?: string;
  serverAvailable?: boolean | null;
  serverError?: string | null;
  contextTokens?: number | null;
  isLoading?: boolean;
  onLoad: () => void;
};

function compactName(name: string) {
  return name.replace(/\.gguf$/i, "").replace(/[-_]+/g, " ");
}

export function ModelPicker({ activeModel, backendLabel, serverUrl, serverAvailable, serverError, contextTokens, isLoading, onLoad }: Props) {
  const connected = Boolean(serverAvailable && activeModel);
  const state = isLoading ? "Connecting" : connected ? "Connected" : serverAvailable ? "Ready to connect" : "Disconnected";
  const modelLabel = activeModel ? compactName(activeModel) : backendLabel || "External llama.cpp server";
  const contextLabel = contextTokens ? `${Math.round(contextTokens / 1024)}k context` : serverUrl || "Configure server in Settings";

  return (
    <div className={connected ? "model-picker loaded" : "model-picker"}>
      <div className="model-status" title={serverError || serverUrl || "External llama.cpp server"}>
        <span className="model-title">{modelLabel}</span>
        <span className="model-meta">
          {connected ? <CircleCheck size={12} /> : <CircleAlert size={12} />}
          {state} · {contextLabel}
        </span>
      </div>
      <button className="model-load" type="button" onClick={onLoad} disabled={isLoading}>
        {isLoading ? <Loader2 size={14} className="spin-icon" /> : <PlugZap size={14} />}
        {isLoading ? "Connecting" : connected ? "Reconnect" : "Connect"}
      </button>
    </div>
  );
}
