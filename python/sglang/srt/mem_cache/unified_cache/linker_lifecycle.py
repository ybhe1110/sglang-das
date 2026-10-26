"""Ownership barriers for direct external KV loads.

Layer readiness is not batch completion: sessions and native writes must have
ended before the worker publishes terminal status. A timeout never grants
permission to recycle a buffer.
"""

import hashlib
import logging
from concurrent.futures import Future
from concurrent.futures import TimeoutError as FutureTimeoutError
from datetime import timedelta

logger = logging.getLogger(__name__)


class LayerWiseLoadCounter:
    def __init__(self, num_layers, on_layer_ready=None):
        self.num_layers = num_layers
        self.on_layer_ready = on_layer_ready
        self.producer_index = self.consumer_index = -1
        self.futures = {}
        self.terminals = {}
        self.errors = {}

    def update_producer(self):
        self.producer_index += 1
        index = self.producer_index
        self.futures[index] = [Future() for _ in range(self.num_layers)]
        self.terminals[index] = Future()
        return index

    def set_consumer(self, index):
        self.consumer_index = index

    def complete(self, index, layer):
        self.futures[index][layer].set_result(None)

    def fail(self, index, error):
        # Never keep a traceback into a model/native worker alive per layer.
        self.errors[index] = str(error)
        for future in self.futures[index]:
            if not future.done():
                future.set_result(False)

    def wait_until(self, threshold):
        index = self.consumer_index
        futures = self.futures.get(index)
        if futures is None:
            return
        try:
            futures[threshold].result(timeout=300)
        except (TimeoutError, FutureTimeoutError):
            raise  # Unknown writer state: abort the scheduler, retain its buffers.
        except Exception as error:
            self.errors[index] = str(error)
            error.__traceback__ = None
        # Failed loads still follow the same collective/forward sequence on all
        # ranks. Their outputs are discarded at the result boundary.
        if self.on_layer_ready is not None:
            self.on_layer_ready(index, threshold)

    def publish_terminal(self, index):
        """Worker-only: call after the native operation AND session cleanup."""
        self.terminals[index].set_result(None)

    def finish(self, index):
        if index < 0 or index not in self.futures:
            return True
        success = True
        # Do not use all(generator): it short-circuits on an early failure.
        for future in self.futures[index]:
            try:
                if future.result(timeout=300) is False:
                    success = False
            except (TimeoutError, FutureTimeoutError):
                raise  # Retain ownership: later layers may still be writing.
            except Exception as error:
                success = False
                error.__traceback__ = None
        self.terminals[index].result(timeout=300)
        success = success and index not in self.errors
        self.futures.pop(index)
        self.terminals.pop(index)
        self.errors.pop(index, None)
        return success

    def reset(self):
        if any(not terminal.done() for terminal in self.terminals.values()):
            raise RuntimeError("Cannot reset linker with native loads in flight")
        self.futures.clear()
        self.terminals.clear()
        self.errors.clear()
        self.producer_index = self.consumer_index = -1


def identity_words(value):
    digest = hashlib.sha256(repr(value).encode()).digest()
    return [int.from_bytes(digest[i : i + 4], "big") for i in range(0, 16, 4)]


def all_reduce_with_timeout(tensor, op, group):
    import torch.distributed as dist

    try:
        work = dist.all_reduce(tensor, op=op, group=group, async_op=True)
        if work.wait(timeout=timedelta(seconds=300)) is not True:
            raise TimeoutError("collective did not confirm completion")
    except Exception as error:
        abort = getattr(group, "abort", None)
        if callable(abort):
            try:
                abort()
            except Exception:
                logger.exception("Failed to abort external KV collective group")
        raise RuntimeError(
            "External KV collective failed (timeout budget: 300s)"
        ) from error


def preflight(cache, phase, identity, geometry=(), valid=True):
    """Fixed-width identity/geometry agreement before any variable-size reduce."""
    import torch

    values = [phase, *identity_words(identity), *geometry]
    # All callers use three geometry words, including empty batches.
    if len(geometry) != 3:
        raise ValueError("linker preflight requires three geometry fields")
    fields = values + [int(valid)]
    payload = torch.tensor(fields + [-v for v in fields], dtype=torch.int64)
    cache._all_reduce_attn_groups(payload, torch.distributed.ReduceOp.MIN)
    n = len(fields)
    minimum, maximum = payload[:n].tolist(), (-payload[n:]).tolist()
    if minimum[:-1] != maximum[:-1]:
        raise RuntimeError(
            f"External KV rank divergence: phase={phase} identity={identity!r} "
            f"min={minimum} max={maximum}"
        )
    return bool(minimum[-1])


def wait_queue_drained(queue, timeout_s=300):
    """Bound shutdown without treating a still-running native call as cancelled."""
    with queue.all_tasks_done:
        if not queue.all_tasks_done.wait_for(
            lambda: queue.unfinished_tasks == 0, timeout_s
        ):
            raise TimeoutError("External KV worker still active; retaining buffers")
