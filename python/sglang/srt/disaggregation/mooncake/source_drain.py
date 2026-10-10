"""Track source ownership across queueing, transfer execution and aborts."""

import threading
from itertools import count


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
            return True

    def mark_unresolved(self, chunk):
        with self.lock:
            op = self._op(chunk)
            if op is not None and op.state == "active":
                op.state = "unresolved"

    def finish_quiesced(self, chunk):
        with self.lock:
            op = self._op(chunk)
            if op is None or op.state != "unresolved" or not self._safe(op):
                return False
            op.state = "done"
            return True

    def run(self, fn, *args, success=None):
        op = getattr(self._context, "op", None)
        if op is None:
            raise RuntimeError("Source access outside a tracked transfer")
        hazard = object()
        with self.lock:
            op.hazards.add(hazard)
        result = fn(*args)
        if success is None or success(result):
            with self.lock:
                op.hazards.remove(hazard)
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
        with self.lock:
            op.hazards.add(hazard)
        future = executor.submit(invoke)
        with self.lock:
            op.futures.add(future)
            op.hazards.remove(hazard)
        def on_done(done):
            if not done.cancelled():
                try:
                    failed = done.exception() is not None
                except BaseException:
                    failed = True
                if failed:
                    with self.lock:
                        op.hazards.add(done)
            self.finish_quiesced(op.chunk)

        future.add_done_callback(on_done)
        return future

    def attach_futures(self, chunk, futures):
        with self.lock:
            op = self._op(chunk)
            if op is not None:
                op.futures.update(futures)

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
            return affected

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
            return True
