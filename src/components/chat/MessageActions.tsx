import { Copy, RefreshCw } from "lucide-react";
import type { AnswerSupport, SourceChunk } from "../../api";

type Props = {
  content: string;
  sources?: SourceChunk[];
  support?: AnswerSupport | null;
  isError?: boolean;
  onOpenSources: () => void;
  onOpenSupport: () => void;
  onRegenerate: () => void;
};

export function MessageActions({
  content,
  sources = [],
  support,
  isError,
  onOpenSources,
  onOpenSupport,
  onRegenerate,
}: Props) {
  const copyAnswer = () => navigator.clipboard?.writeText(content);

  return (
    <div className="message-actions" aria-label="Answer actions">
      <button type="button" onClick={copyAnswer} aria-label="Copy answer" title="Copy answer">
        <Copy size={13} />Copy
      </button>
      <button type="button" onClick={onRegenerate} aria-label={isError ? "Retry answer" : "Regenerate answer"}>
        <RefreshCw size={13} />{isError ? "Retry" : "Regenerate"}
      </button>
      {sources.length > 0 && (
        <button type="button" onClick={onOpenSources}>
          {sources.length} {sources.length === 1 ? "source" : "sources"}
        </button>
      )}
      {support && (
        <button type="button" onClick={onOpenSupport}>
          Support: {support.status}
        </button>
      )}
    </div>
  );
}
