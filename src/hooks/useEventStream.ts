import { useEffect } from "react";
import { useQueryClient } from "@tanstack/react-query";
import { eventsUrl } from "../api";
import { useUiStore } from "../store";

export function useEventStream() {
  const queryClient = useQueryClient();
  const setEventStatus = useUiStore(state => state.setEventStatus);

  useEffect(() => {
    let fallback: number | undefined;
    const refresh = () => {
      queryClient.invalidateQueries({ queryKey: ["documents"] });
      queryClient.invalidateQueries({ queryKey: ["jobs"] });
    };

    const source = new EventSource(eventsUrl());
    setEventStatus("connecting");

    source.addEventListener("ready", () => {
      setEventStatus("connected");
    });
    source.addEventListener("job", refresh);
    source.addEventListener("document", refresh);
    source.addEventListener("settings", () => queryClient.invalidateQueries({ queryKey: ["settings"] }));
    source.onerror = () => {
      setEventStatus(navigator.onLine === false ? "offline" : "reconnecting");
      source.close();
      fallback = window.setInterval(refresh, 3000);
    };

    return () => {
      source.close();
      if (fallback) window.clearInterval(fallback);
    };
  }, [queryClient, setEventStatus]);
}
