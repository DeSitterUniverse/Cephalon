import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { eventsUrl } from "../api";
import { useUiStore } from "../store";

export function useEventStream() {
  const queryClient = useQueryClient();
  const setEventStatus = useUiStore(state => state.setEventStatus);

  useEffect(() => {
    let fallback: number | undefined;
    let reconnectTimer: number | undefined;
    let source: EventSource | undefined;
    let closed = false;
    const refresh = () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    };

    const clearFallback = () => {
      if (fallback) window.clearInterval(fallback);
      fallback = undefined;
    };

    const connect = () => {
      if (closed) return;
      source?.close();
      source = new EventSource(eventsUrl());
      setEventStatus("connecting");

      source.onopen = () => {
        clearFallback();
        setEventStatus("connected");
      };
      source.addEventListener("ready", () => {
        clearFallback();
        setEventStatus("connected");
      });
      source.addEventListener("heartbeat", () => {
        clearFallback();
        setEventStatus("connected");
      });
      source.addEventListener("job", refresh);
      source.addEventListener("document", refresh);
      source.addEventListener("settings", () => queryClient.invalidateQueries({ queryKey: ["settings"] }));
      source.onerror = () => {
        source?.close();
        setEventStatus(navigator.onLine === false ? "offline" : "reconnecting");
        if (!fallback) fallback = window.setInterval(refresh, 3000);
        if (reconnectTimer) window.clearTimeout(reconnectTimer);
        reconnectTimer = window.setTimeout(connect, 1500);
      };
    };
    connect();

    return () => {
      closed = true;
      source?.close();
      clearFallback();
      if (reconnectTimer) window.clearTimeout(reconnectTimer);
    };
  }, [queryClient, setEventStatus]);
}
