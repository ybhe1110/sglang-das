"""Track source ownership across queueing, transfer execution and aborts."""

import concurrent.futures
import threading
import time
from itertools import count


class SourceTransferUnresolved(RuntimeError):
    """The synchronous call returned a failure code without completion proof."""


class SourceTransferTimeout(TimeoutError):
    """The caller deadline expired while the native source operation was active.

    A timeout does not cancel the native call.  The operation's hazard is kept
    until the detached call returns so callers can quarantine the source
    buffers instead of releasing memory while RDMA may still be writing it.
    """


class _SourceOp:
    def __init__(self, token, room, generation, chunk, queue):
        self.token = token
        self.room = room
        self.generation = generation
        self.chunk = chunk
        self.queue = queue
        self.state = "pending"
        self.futures = set()
        self.hazards = set()


class SourceDrainTracker:
    def __init__(self):
        self.lock = threading.RLock()
        self._condition = threading.Condition(self.lock)
        self._sequence = count(1)
        self._generations = {}
        self._ops = {}
        self._closed = set()
        self._poisoned = set()
        self._dead_queues = set()
        self._context = threading.local()

    def open(self, room):
        with self.lock:
            if room in self._generations:
                raise RuntimeError(f"Source room {room} has not been forgotten")
            generation = next(self._sequence)
            self._generations[room] = generation
            return generation

    def _key(self, room, generation=None):
        return room, self._generations.get(room) if generation is None else generation

    def _current(self, key):
        return key[1] is not None and self._generations.get(key[0]) == key[1]

    def owns_generation(self, room, generation):
        """Check an explicit owner; None must never resolve a reused room."""
        with self.lock:
            return self._current((room, generation))

    def _op(self, chunk):
        op = self._ops.get(getattr(chunk, "_source_token", None))
        return op if op is not None and op.chunk is chunk else None

    def publish(self, chunk, queue, accept, generation=None):
        with self.lock:
            if generation is None and chunk.room not in self._generations:
                self.open(chunk.room)
            key = self._key(chunk.room, generation)
            if not self._current(key) or key in self._closed or not accept():
                return False
            if queue in self._dead_queues:
                self.poison(*key)
                raise RuntimeError("Source transfer worker has exited")
            if self._op(chunk) is not None:
                raise RuntimeError("Source chunk already published")
            op = _SourceOp(next(self._sequence), *key, chunk, queue)
            chunk._source_token = op.token
            self._ops[op.token] = op
            try:
                queue.put(chunk)
            except BaseException:
                op.state = "cancelled"
                self.poison(*key)
                raise
            return True

    def begin(self, chunk):
        with self.lock:
            op = self._op(chunk)
            if op is None or op.state != "pending":
                return False
            key = (op.room, op.generation)
            if not self._current(key) or key in self._closed:
                op.state = "cancelled"
                return False
            op.state = "active"
            self._context.op = op
            self._condition.notify_all()
            return True

    @staticmethod
    def _safe(op):
        return not op.hazards and all(f.done() for f in op.futures)

    def park(self, chunk):
        with self.lock:
            op = self._op(chunk)
            self._context.op = None
            if op is None or op.state != "active":
                return False
            if not self._safe(op):
                op.state = "unresolved"
                self.poison(op.room, op.generation)
                return False
            op.state = "cancelled" if (op.room, op.generation) in self._closed else "pending"
            self._condition.notify_all()
            return True

    def complete(self, chunk):
        with self.lock:
            op = self._op(chunk)
            if op is None or op.state != "active":
                return False
            if not self._safe(op):
                op.state = "unresolved"
                self.poison(op.room, op.generation)
                return False
            op.state = "done"
            self._context.op = None
            self._condition.notify_all()
            return True

    def mark_unresolved(self, chunk):
        with self.lock:
            op = self._op(chunk)
            if op is not None and op.state == "active":
                op.state = "unresolved"
                self._condition.notify_all()

    def finish_quiesced(self, chunk):
        with self.lock:
            op = self._op(chunk)
            if op is None or op.state != "unresolved" or not self._safe(op):
                return False
            op.state = "done"
            self._condition.notify_all()
            return True

    def run(self, fn, *args, success=None):
        """Track a synchronous call whose invocation has returned."""
        op = getattr(self._context, "op", None)
        if op is None:
            raise RuntimeError("Source access outside a tracked transfer")
        hazard = object()
        with self.lock:
            op.hazards.add(hazard)
        try:
            result = fn(*args)
        finally:
            with self.lock:
                op.hazards.discard(hazard)
                self._condition.notify_all()
        if success is not None and not success(result):
            raise SourceTransferUnresolved("Synchronous transfer did not complete")
        return result

    def run_with_deadline(self, fn, *args, timeout_s, success=None):
        """Run a native source operation without blocking its owner forever.

        The Mooncake Python binding normally enforces ``MC_TRANSFER_TIMEOUT``
        in native code.  This watchdog is a second line of defense for a
        broken transport or CQ path that ignores that deadline.  Python cannot
        safely kill a thread executing a native RDMA call, so a timed-out call
        is deliberately detached.  Its hazard remains active until the call
        returns, which makes ``terminal``/``forget`` retain the source buffers
        and lets the scheduler quarantine the request safely.
        """
        op = getattr(self._context, "op", None)
        if op is None:
            raise RuntimeError("Source access outside a tracked transfer")
        if timeout_s is None or timeout_s <= 0:
            return self.run(fn, *args, success=success)

        future = concurrent.futures.Future()
        hazard = object()
        with self.lock:
            op.hazards.add(hazard)
            op.futures.add(future)

        def invoke():
            try:
                result = fn(*args)
            except BaseException as exc:
                future.set_exception(exc)
            else:
                future.set_result(result)
            finally:
                with self.lock:
                    op.hazards.discard(hazard)
                    self._condition.notify_all()
                # This may complete an unresolved operation after the caller
                # has already quarantined its request.
                self.finish_quiesced(op.chunk)

        thread = threading.Thread(
            target=invoke,
            name=f"source-transfer-{op.room}-{op.token}",
            daemon=True,
        )
        thread.start()
        try:
            result = future.result(timeout=timeout_s)
        except concurrent.futures.TimeoutError as exc:
            with self.lock:
                if op.state == "active":
                    op.state = "unresolved"
                # A timed-out native call remains quarantined until its
                # detached thread returns.  Closing/poisoning here prevents a
                # new chunk from being published into the same room while the
                # source buffer is still potentially in use.
                self._closed.add((op.room, op.generation))
                self._poisoned.add((op.room, op.generation))
                self._context.op = None
                self._condition.notify_all()
            raise SourceTransferTimeout(
                f"Source transfer exceeded watchdog deadline ({timeout_s:.3f}s)"
            ) from exc
        if success is not None and not success(result):
            raise SourceTransferUnresolved("Synchronous transfer did not complete")
        return result

    def submit(self, executor, fn, *args):
        op = getattr(self._context, "op", None)
        if op is None:
            raise RuntimeError("Source submission outside a tracked transfer")

        def invoke():
            self._context.op = op
            try:
                return fn(*args)
            finally:
                self._context.op = None

        hazard = object()
        future = None
        with self.lock:
            op.hazards.add(hazard)
        try:
            future = executor.submit(invoke)
        finally:
            with self.lock:
                op.hazards.discard(hazard)
                # Register the future inside the same lock acquisition that
                # removes the hazard.  Without this there is a window between
                # the two separate `with self.lock` blocks where op.hazards is
                # empty AND op.futures is still empty, making _safe(op) return
                # True prematurely.  A concurrent finish_quiesced callback
                # could then mark the op "done" while this future is still
                # in flight.
                if future is not None:
                    op.futures.add(future)
        if future is None:
            raise RuntimeError("executor.submit did not return a future")
        future.add_done_callback(lambda _: self.finish_quiesced(op.chunk))
        return future

    def attach_futures(self, chunk, futures):
        with self.lock:
            op = self._op(chunk)
            if op is not None:
                op.futures.update(futures)
            self._condition.notify_all()

    def _quiesced(self, key):
        return not any(
            (op.room, op.generation) == key
            and op.state in ("pending", "active", "unresolved")
            for op in self._ops.values()
        )

    def close(self, room, generation=None, *, poisoned=False):
        with self.lock:
            key = self._key(room, generation)
            if not self._current(key):
                return
            self._closed.add(key)
            if poisoned:
                self._poisoned.add(key)
                for op in self._ops.values():
                    if (op.room, op.generation) == key and op.state == "pending":
                        op.state = "cancelled"
            self._condition.notify_all()

    def poison(self, room, generation=None):
        self.close(room, generation, poisoned=True)

    def drained(self, room, generation=None):
        with self.lock:
            key = self._key(room, generation)
            return (
                self._current(key)
                and key in self._closed
                and key not in self._poisoned
                and self._quiesced(key)
            )

    def terminal(self, room, generation=None):
        with self.lock:
            key = self._key(room, generation)
            return self._current(key) and key in self._closed and self._quiesced(key)

    def worker_failed(self, queue):
        with self.lock:
            self._dead_queues.add(queue)
            affected = set()
            for op in self._ops.values():
                if op.queue is queue and op.state in ("pending", "active", "unresolved"):
                    affected.add((op.room, op.generation))
                    if op.state == "active":
                        op.state = "unresolved"
                    self.finish_quiesced(op.chunk)
            for key in affected:
                self.poison(*key)
            self._condition.notify_all()
            return affected

    def shutdown(self, timeout_s=5.0):
        """Poison all rooms and wait for detached native calls to quiesce.

        A Python timeout cannot cancel a native RDMA call safely.  Shutdown
        therefore keeps every room's buffers quarantined until its hazards and
        futures are complete.  The returned keys are still live after the
        deadline and must not be deregistered by the caller.
        """
        timeout_s = max(float(timeout_s or 0.0), 0.0)
        deadline = time.monotonic() + timeout_s
        with self.lock:
            keys = set(self._generations.items())
            for room, generation in keys:
                key = (room, generation)
                self._closed.add(key)
                self._poisoned.add(key)
                for op in self._ops.values():
                    if (op.room, op.generation) == key and op.state == "pending":
                        op.state = "cancelled"
            self._condition.notify_all()

            while True:
                remaining = {
                    key for key in keys if not self._quiesced(key)
                }
                if not remaining:
                    break
                wait_s = deadline - time.monotonic()
                if wait_s <= 0:
                    return sorted(remaining)
                self._condition.wait(timeout=wait_s)

            for room, generation in keys:
                self.forget(room, generation)
            return []

    def forget(self, room, generation=None):
        with self.lock:
            key = self._key(room, generation)
            if not self.terminal(*key):
                return False
            for token, op in list(self._ops.items()):
                if (op.room, op.generation) == key:
                    del self._ops[token]
            self._closed.discard(key)
            self._poisoned.discard(key)
            self._generations.pop(room, None)
            self._condition.notify_all()
            return True
