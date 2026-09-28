"""Bounded metadata calls and L1 fallback for the optional Mooncake store.

Only initialization and existence queries run here. Never time out a GPU transfer
and report it completed: a late native transfer may still access its buffers.
"""

import contextvars
import logging
import math
import threading
import time
from concurrent.futures import Future, TimeoutError
from queue import SimpleQueue

logger = logging.getLogger(__name__)


class StoreUnavailable(RuntimeError):
    """Remote caching is temporarily unavailable; use a local cache miss."""


def checked_exists(values, count):
    if values is None or isinstance(values, (int, bool)):
        raise StoreUnavailable(f"Invalid Mooncake existence response: {values!r}")
    values = list(values)
    if len(values) != count or any(value not in (0, 1) for value in values):
        raise StoreUnavailable(
            "Mooncake existence query failed or returned a short response"
        )
    return values


class MasterAvailability:
    """One bounded metadata worker per linker, with cooldown and recovery probes.

    Timed-out work is retained until it actually exits. Repeated misses cannot
    create additional blocked threads. Generation checks prevent a stale success
    from undoing a newer failure. No distributed collectives run on this thread.
    """

    def __init__(
        self, factory, probe, *, timeout_s=2.0, retry_s=10.0, clock=time.monotonic
    ):
        if not all(math.isfinite(v) and v > 0 for v in (timeout_s, retry_s)):
            raise ValueError(
                "Mooncake fallback timeout and retry interval must be finite and positive"
            )
        context = contextvars.copy_context()
        self._factory = lambda: context.copy().run(factory)
        self._probe = probe
        self.timeout_s, self.retry_s, self._clock = timeout_s, retry_s, clock
        self._lock = threading.Lock()
        self._job = None
        self._generation = 0
        self._healthy = False
        self._closed = False
        self._retry_at = 0.0
        self.storage = None
        self.failures = 0
        self.recoveries = 0
        self._work = SimpleQueue()
        self._worker = threading.Thread(
            target=self._run, daemon=True, name="mooncake-master-metadata"
        )
        self._worker.start()

    def _trip_locked(self, error):
        if self._healthy or self.failures == 0:
            logger.warning(
                "Mooncake store unavailable; using L1 cache and recomputation: %s",
                error,
            )
        self._healthy = False
        self._generation += 1
        self.failures += 1
        self._retry_at = self._clock() + self.retry_s

    def failed(self, error):
        # A deliberately skipped operation must not extend the cooldown forever.
        with self._lock:
            if not self._closed:
                self._trip_locked(error)

    def _start_locked(self, kind, operation):
        future = Future()
        self._job = future
        # _job gates admission: at most one operation is queued or executing.
        self._work.put((future, kind, operation, self._generation))
        return future

    def _run(self):
        while True:
            task = self._work.get()
            if task is None:
                return
            self._run_operation(*task)
            # Do not retain an SDK closure or its arguments while idle.
            del task

    def _run_operation(self, future, kind, operation, generation):
        try:
            result = operation()
        except BaseException as error:
            # Do not retain SDK traceback frames (or their buffer owners).
            message = f"{type(error).__name__}: {error}"
            with self._lock:
                self._job = None
                if not self._closed:
                    self._trip_locked(message)
            future.set_exception(StoreUnavailable(message))
            return
        with self._lock:
            self._job = None
            if kind == "initialize":
                self.storage = result
            if kind in ("initialize", "probe") and not self._closed:
                if generation == self._generation:
                    if self.failures:
                        self.recoveries += 1
                        logger.info(
                            "Mooncake store probe succeeded; remote caching restored"
                        )
                    self._healthy = True
        future.set_result(result)

    def _recover_locked(self):
        if self._closed or self._job is not None or self._clock() < self._retry_at:
            return None
        if self.storage is None:
            return self._start_locked("initialize", self._factory)
        storage = self.storage
        return self._start_locked("probe", lambda: self._probe(storage))

    def initialize(self):
        with self._lock:
            future = self._recover_locked()
        if future is not None:
            try:
                future.result(timeout=self.timeout_s)
            except TimeoutError:
                self.failed("store initialization timed out")
            except StoreUnavailable:
                pass
        return self.ready()

    def ready(self):
        with self._lock:
            if self._closed:
                return False
            if not self._healthy:
                self._recover_locked()
            return self._healthy

    def query(self, operation):
        deadline = time.monotonic() + self.timeout_s
        while True:
            if not self.ready():
                raise StoreUnavailable("Mooncake circuit is open")
            with self._lock:
                if self._closed or not self._healthy:
                    raise StoreUnavailable("Mooncake circuit is open")
                pending = self._job
                generation = self._generation
                if pending is None:
                    future = self._start_locked("query", operation)
                    break
            # Serialize healthy scheduler/offload queries without spawning an
            # unbounded executor queue. The total wait still has one deadline.
            try:
                pending.result(timeout=max(0.0, deadline - time.monotonic()))
            except TimeoutError:
                self.failed("master metadata query timed out")
                raise StoreUnavailable("master metadata query timed out") from None
        try:
            result = future.result(timeout=max(0.0, deadline - time.monotonic()))
        except TimeoutError:
            self.failed("master metadata query timed out")
            raise StoreUnavailable("master metadata query timed out") from None
        with self._lock:
            if not self._healthy or generation != self._generation:
                raise StoreUnavailable(
                    "discarding result from an older master generation"
                )
        return result

    def close(self):
        with self._lock:
            if self._closed:
                return
            self._closed = True
            self._healthy = False
            self._generation += 1
            self._work.put(None)
