import { Send, Square } from "lucide-react";
import { FormEvent, KeyboardEvent, useEffect, useRef } from "react";

type Props = {
  value: string;
  onChange: (value: string) => void;
  onSubmit: () => void;
  onStop: () => void;
  isRunning: boolean;
  disabled: boolean;
  placeholder: string;
  retrievalScope: string;
  responseEffort: string;
  onRetrievalScopeChange: (value: string) => void;
  onResponseEffortChange: (value: string) => void;
};

export function Composer({
  value,
  onChange,
  onSubmit,
  onStop,
  isRunning,
  disabled,
  placeholder,
  retrievalScope,
  responseEffort,
  onRetrievalScopeChange,
  onResponseEffortChange,
}: Props) {
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    const textarea = textareaRef.current;
    if (!textarea) return;
    textarea.style.height = "0px";
    textarea.style.height = `${Math.min(160, Math.max(42, textarea.scrollHeight))}px`;
  }, [value]);

  const submit = (event?: FormEvent) => {
    event?.preventDefault();
    if (!disabled && value.trim()) onSubmit();
  };

  const handleKeyDown = (event: KeyboardEvent<HTMLTextAreaElement>) => {
    if (event.key === "Enter" && !event.shiftKey && !event.nativeEvent.isComposing) {
      event.preventDefault();
      submit();
    }
  };

  return (
    <form className="composer" onSubmit={submit}>
      <div className="composer-controls">
        <select
          className="scope-select"
          value={retrievalScope}
          onChange={event => onRetrievalScopeChange(event.target.value)}
          disabled={isRunning}
          aria-label="Retrieval scope"
        >
          <option value="low">Low retrieval</option>
          <option value="medium">Medium retrieval</option>
          <option value="high">High retrieval</option>
        </select>
        <select
          className="scope-select"
          value={responseEffort}
          onChange={event => onResponseEffortChange(event.target.value)}
          disabled={isRunning}
          aria-label="Response effort"
        >
          <option value="quick">Quick response</option>
          <option value="balanced">Balanced response</option>
          <option value="thorough">Thorough response</option>
        </select>
      </div>
      <textarea
        ref={textareaRef}
        aria-label="Message"
        value={value}
        onChange={event => onChange(event.target.value)}
        onKeyDown={handleKeyDown}
        disabled={disabled && !isRunning}
        placeholder={placeholder}
        rows={1}
      />
      {isRunning ? (
        <button type="button" className="composer-stop" onClick={onStop} aria-label="Stop response">
          <Square size={14} />Stop
        </button>
      ) : (
        <button type="submit" disabled={disabled || !value.trim()}>
          <Send size={16} />Run
        </button>
      )}
    </form>
  );
}
