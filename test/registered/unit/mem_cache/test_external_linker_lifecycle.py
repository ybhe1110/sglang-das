"""CPU regressions for private-load completion and metadata fallback."""

import threading
from concurrent.futures import Future
from queue import Queue
from unittest.mock import patch

from sglang.srt.mem_cache.storage.mooncake_store.master_availability import (
    MasterAvailability,
    StoreUnavailable,
)
from sglang.srt.mem_cache.unified_cache.linker_lifecycle import (
    LayerWiseLoadCounter,
    wait_queue_drained,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class ObservedFuture(Future):
    def __init__(self):
        super().__init__()
        self.waiting = threading.Event()

    def result(self, timeout=None):
        self.waiting.set()
        return super().result(timeout)


class TestExternalLinkerLifecycle(CustomTestCase):
    def test_first_failed_layer_does_not_skip_later_layers_or_terminal(self):
        counter = LayerWiseLoadCounter(2)
        index = counter.update_producer()
        later = ObservedFuture()
        terminal = ObservedFuture()
        counter.futures[index][0].set_result(False)
        counter.futures[index][1] = later
        counter.terminals[index] = terminal
        result = Future()

        def finish():
            try:
                result.set_result(counter.finish(index))
            except BaseException as error:
                result.set_exception(error)

        worker = threading.Thread(target=finish)
        worker.start()
        try:
            self.assertTrue(later.waiting.wait(2))
            self.assertFalse(result.done())
            later.set_result(None)
            self.assertTrue(terminal.waiting.wait(2))
            self.assertFalse(result.done())
            self.assertIn(index, counter.futures)
            terminal.set_result(None)
            self.assertFalse(result.result(timeout=2))
            self.assertNotIn(index, counter.futures)
        finally:
            if not later.done():
                later.set_result(None)
            if not terminal.done():
                terminal.set_result(None)
            worker.join(2)

    def test_model_hook_does_not_consume_completion_or_skip_collective(self):
        callbacks = []
        counter = LayerWiseLoadCounter(2, lambda i, l: callbacks.append((i, l)))
        index = counter.update_producer()
        counter.set_consumer(index)
        counter.fail(index, RuntimeError("first layer failed"))
        counter.wait_until(1)
        self.assertEqual(callbacks, [(index, 1)])
        self.assertIn(index, counter.futures)
        self.assertFalse(counter.terminals[index].done())
        counter.publish_terminal(index)
        self.assertFalse(counter.finish(index))

    def test_timeout_and_reset_cannot_discard_active_writer(self):
        counter = LayerWiseLoadCounter(1)
        index = counter.update_producer()
        counter.complete(index, 0)
        with patch.object(counter.terminals[index], "result", side_effect=TimeoutError):
            with self.assertRaises(TimeoutError):
                counter.finish(index)
        self.assertIn(index, counter.futures)
        with self.assertRaisesRegex(RuntimeError, "native loads in flight"):
            counter.reset()
        counter.publish_terminal(index)
        self.assertTrue(counter.finish(index))
        counter.reset()

    def test_session_cleanup_failure_changes_batch_outcome(self):
        counter = LayerWiseLoadCounter(2)
        index = counter.update_producer()
        for layer in range(2):
            counter.complete(index, layer)
        counter.fail(index, RuntimeError("session cleanup failed"))
        counter.publish_terminal(index)
        self.assertFalse(counter.finish(index))
        self.assertEqual(counter.errors, {})

    def test_metadata_timeout_retains_one_worker_until_native_return(self):
        entered, release = threading.Event(), threading.Event()
        store = object()
        gate = MasterAvailability(
            lambda: store, lambda _: None, timeout_s=0.02, retry_s=10
        )
        self.assertTrue(gate.initialize())
        worker = gate._worker

        def blocked_rpc():
            entered.set()
            release.wait(2)
            return [1]

        try:
            with self.assertRaises(StoreUnavailable):
                gate.query(blocked_rpc)
            self.assertTrue(entered.is_set())
            for _ in range(20):
                self.assertFalse(gate.ready())
            self.assertIs(gate._worker, worker)
            self.assertFalse(gate.wait_closed(0.01))
        finally:
            release.set()
            self.assertTrue(gate.wait_closed(2))
        self.assertIs(gate.storage, store)
        self.assertFalse(gate.ready())

    def test_late_initialization_is_retained_for_shutdown(self):
        release = threading.Event()
        store = object()

        def factory():
            release.wait(2)
            return store

        gate = MasterAvailability(factory, lambda _: None, timeout_s=0.02, retry_s=10)
        try:
            self.assertFalse(gate.initialize())
            self.assertFalse(gate.wait_closed(0.01))
        finally:
            release.set()
            self.assertTrue(gate.wait_closed(2))
        self.assertIs(gate.storage, store)

    def test_shutdown_timeout_does_not_claim_native_queue_is_drained(self):
        queue = Queue()
        queue.put(object())
        queue.get()
        with self.assertRaises(TimeoutError):
            wait_queue_drained(queue, 0.01)
        self.assertEqual(queue.unfinished_tasks, 1)
        queue.task_done()
        wait_queue_drained(queue, 0.01)

    def test_layer_wait_timeout_is_fatal_not_failed_completion(self):
        counter = LayerWiseLoadCounter(1)
        index = counter.update_producer()
        counter.set_consumer(index)
        with patch.object(
            counter.futures[index][0], "result", side_effect=TimeoutError
        ):
            with self.assertRaises(TimeoutError):
                counter.wait_until(0)
        self.assertFalse(counter.terminals[index].done())
        self.assertIn(index, counter.futures)
