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
from dataclasses import dataclass, field
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


@dataclass
class _QueryTiming:
    operation_source: str
    num_keys: int
    queued_at: float = field(default_factory=time.monotonic)
    started: threading.Event = field(default_factory=threading.Event)
    started_at: float | None = None
    finished_at: float | None = None

    def log(self, event, level, message):
        now = time.monotonic()
        fields = dict(
            metadata_event=event,
            operation_source=self.operation_source,
            num_keys=self.num_keys,
            queue_wait_ms=1000
            * (
                (self.started_at if self.started_at is not None else now)
                - self.queued_at
            ),
            rpc_execution_ms=(
                0.0
                if self.started_at is None
                else 1000
                * (
                    (self.finished_at if self.finished_at is not None else now)
                    - self.started_at
                )
            ),
        )
        logger.log(
            level,
            "Mooncake metadata %s: operation_source=%s num_keys=%d "
            "queue_wait_ms=%.3f rpc_execution_ms=%.3f",
            message,
            fields["operation_source"],
            fields["num_keys"],
            fields["queue_wait_ms"],
            fields["rpc_execution_ms"],
            extra=fields,
        )


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
        self._idle = threading.Event()
        self._idle.set()
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

    def _start_locked(self, kind, operation, timing=None):
        future = Future()
        self._job = future
        self._idle.clear()
        # _job gates admission: at most one operation is queued or executing.
        self._work.put((future, kind, operation, self._generation, timing))
        return future

    def _run(self):
        while True:
            task = self._work.get()
            if task is None:
                return
            self._run_operation(*task)
            # Do not retain an SDK closure or its arguments while idle.
            del task

    def _run_operation(self, future, kind, operation, generation, timing=None):
        # Synchronize dispatch with cancellation of work that never started.
        with self._lock:
            if not future.set_running_or_notify_cancel():
                self._job = None
                self._idle.set()
                return
            if timing is not None:
                timing.started_at = time.monotonic()
                timing.started.set()
        try:
            result = operation()
        except BaseException as error:
            # Do not retain SDK traceback frames (or their buffer owners).
            message = f"{type(error).__name__}: {error}"
            if timing is not None:
                timing.finished_at = time.monotonic()
                timing.log("rpc_error", logging.WARNING, f"RPC failed ({message})")
            with self._lock:
                self._job = None
                self._idle.set()
                if not self._closed:
                    self._trip_locked(message)
            future.set_exception(StoreUnavailable(message))
            return
        if timing is not None:
            timing.finished_at = time.monotonic()
            # A late completion is still useful diagnostically; it does not
            # restore health or make the timed-out result usable by the caller.
            timing.log("rpc_completed", logging.DEBUG, "RPC completed")
        with self._lock:
            self._job = None
            self._idle.set()
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

    def query(self, operation, *, operation_source="unknown", num_keys=0):
        timing = _QueryTiming(operation_source, num_keys)
        queue_deadline = timing.queued_at + self.timeout_s

        def worker_timeout():
            # Local contention is not evidence of a failed master. The linker
            # treats StoreUnavailable as a miss without calling failed() again.
            timing.log(
                "worker_timeout", logging.WARNING, "waiting for worker timed out"
            )
            raise StoreUnavailable("master metadata waiting for worker timed out")

        while True:
            if not self.ready():
                raise StoreUnavailable("Mooncake circuit is open")
            if time.monotonic() >= queue_deadline:
                worker_timeout()
            with self._lock:
                if self._closed or not self._healthy:
                    raise StoreUnavailable("Mooncake circuit is open")
                pending = self._job
                generation = self._generation
                if pending is None:
                    future = self._start_locked("query", operation, timing)
                    break
            # Wait for actual worker availability, including cancelled jobs
            # still awaiting dispatch. A cancelled Future alone is already done.
            if not self._idle.wait(max(0.0, queue_deadline - time.monotonic())):
                worker_timeout()

        if not timing.started.wait(max(0.0, queue_deadline - time.monotonic())):
            with self._lock:
                # If dispatch won this race, use the RPC deadline below.
                cancelled = future.cancel()
            if cancelled:
                worker_timeout()

        # Give an admitted RPC its own budget. Queue contention must not shorten
        # this budget and cause an otherwise healthy call to trip the circuit.
        rpc_deadline = timing.started_at + self.timeout_s
        try:
            result = future.result(timeout=max(0.0, rpc_deadline - time.monotonic()))
        except TimeoutError:
            timing.log(
                "rpc_timeout", logging.WARNING, "waiting for RPC result timed out"
            )
            self.failed("master metadata waiting for RPC result timed out")
            raise StoreUnavailable(
                "master metadata waiting for RPC result timed out"
            ) from None
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
