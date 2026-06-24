import { useUiStore } from "../store";

export function useNotifications() {
  const notify = useUiStore(state => state.notify);
  const confirm = useUiStore(state => state.requestConfirmation);
  return { notify, confirm };
}
