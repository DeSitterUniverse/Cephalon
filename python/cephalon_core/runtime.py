import threading
from contextlib import contextmanager


class ModelRuntime:
    """Serialize access to the configured generation server.

    FastAPI advances synchronous streaming iterators through its worker pool,
    so a stream may acquire this guard in one worker and close in another.
    ``threading.RLock`` is thread-affine and raises when that happens, leaving
    the model lane permanently held. A plain ``Lock`` deliberately permits the
    closing worker to release the guard. Model operations must therefore stay
    non-reentrant; Cephalon's completion and connection call sites satisfy that
    invariant and keep at most one request active against the external server.
    """

    def __init__(self) -> None:
        self._lock = threading.Lock()

    @contextmanager
    def exclusive(self):
        with self._lock:
            yield
