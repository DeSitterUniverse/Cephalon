import { useEffect } from "react";
import { X } from "lucide-react";
import type { AppNotification } from "../../store";
import { useUiStore } from "../../store";

function NotificationItem({ notification }: { notification: AppNotification }) {
  const dismiss = useUiStore(state => state.dismissNotification);

  useEffect(() => {
    if (notification.kind === "error") return;
    const timer = window.setTimeout(() => dismiss(notification.id), notification.kind === "success" ? 4200 : 6000);
    return () => window.clearTimeout(timer);
  }, [dismiss, notification]);

  return (
    <div className={`notification notification-${notification.kind}`} role={notification.kind === "error" ? "alert" : "status"}>
      <span>{notification.message}</span>
      <button type="button" onClick={() => dismiss(notification.id)} aria-label={`Dismiss ${notification.message}`}>
        <X size={14} />
      </button>
    </div>
  );
}

export function NotificationCenter() {
  const notifications = useUiStore(state => state.notifications);
  if (notifications.length === 0) return null;
  return (
    <div className="notification-center" aria-label="Notifications">
      {notifications.map(notification => <NotificationItem key={notification.id} notification={notification} />)}
    </div>
  );
}
