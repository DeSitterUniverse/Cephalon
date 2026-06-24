import threading
from contextlib import contextmanager


class ModelRuntime:
    def __init__(self) -> None:
        self._lock = threading.RLock()

    @contextmanager
    def exclusive(self):
        with self._lock:
            yield
