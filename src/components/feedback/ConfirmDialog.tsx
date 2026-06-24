import { useEffect, useRef } from "react";
import { useUiStore } from "../../store";

export function ConfirmDialog() {
  const confirmation = useUiStore(state => state.confirmation);
  const close = useUiStore(state => state.closeConfirmation);
  const confirmRef = useRef<HTMLButtonElement>(null);

  useEffect(() => {
    if (!confirmation) return;
    confirmRef.current?.focus();
    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") close();
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [close, confirmation]);

  if (!confirmation) return null;

  const accept = () => {
    const action = confirmation.onConfirm;
    close();
    action();
  };

  return (
    <div className="dialog-backdrop" onMouseDown={event => {
      if (event.target === event.currentTarget) close();
    }}>
      <section className="confirm-dialog" role="alertdialog" aria-modal="true" aria-labelledby="confirm-title" aria-describedby="confirm-message">
        <h2 id="confirm-title">{confirmation.title}</h2>
        <p id="confirm-message">{confirmation.message}</p>
        <div className="dialog-actions">
          <button type="button" onClick={close}>Cancel</button>
          <button ref={confirmRef} type="button" className="danger" onClick={accept}>
            {confirmation.confirmLabel || "Confirm"}
          </button>
        </div>
      </section>
    </div>
  );
}
