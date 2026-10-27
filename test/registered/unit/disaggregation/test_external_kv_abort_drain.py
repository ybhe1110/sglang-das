"""CPU-only regression tests using production method bodies extracted by AST.

These avoid importing CUDA/Mooncake on Windows. They supplement, not replace,
the ordinary prefill/PP tests and real-device transport stress tests on Linux.
"""

import ast
import concurrent.futures
import importlib.util
import pathlib
import sys
import threading
import time
import unittest
from collections import defaultdict
from http import HTTPStatus
from types import SimpleNamespace
from unittest.mock import Mock

ROOT = pathlib.Path(__file__).resolve().parents[4]
# CI registration is parsed statically. Load its stdlib-only implementation
# directly to keep these CPU tests runnable without importing sglang/GPU modules.
ci_spec = importlib.util.spec_from_file_location(
    "abort_cleanup_ci_register", ROOT / "python/sglang/test/ci/ci_register.py"
)
ci_module = importlib.util.module_from_spec(ci_spec)
sys.modules[ci_spec.name] = ci_module
ci_spec.loader.exec_module(ci_module)
register_cpu_ci = ci_module.register_cpu_ci
register_cpu_ci(est_time=3, suite="base-a-test-cpu")

PREFILL = ROOT / "python/sglang/srt/disaggregation/prefill.py"
SCHED = ROOT / "python/sglang/srt/managers/scheduler.py"
CONN = ROOT / "python/sglang/srt/disaggregation/mooncake/conn.py"
TRACKER = ROOT / "python/sglang/srt/disaggregation/mooncake/source_drain.py"
spec = importlib.util.spec_from_file_location("source_drain_under_test", TRACKER)
tracker_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(tracker_module)
SourceDrainTracker = tracker_module.SourceDrainTracker
SourceTransferTimeout = tracker_module.SourceTransferTimeout
KVPoll = SimpleNamespace(Success=1, Failed=2, Transferring=3)


class Reason:
    def __init__(self, message="failure", status_code=500, err_type="external"):
        self.message = message
        self.status_code = status_code
        self.err_type = err_type

    def to_json(self):
        return vars(self).copy()


def load_methods(path, names, env, class_name=None):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    if class_name is not None:
        tree = next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
    methods = []
    for name in names:
        matches = [
            n
            for n in ast.walk(tree)
            if isinstance(n, ast.FunctionDef) and n.name == name
        ]
        if len(matches) != 1:
            raise AssertionError((name, len(matches)))
        node = matches[0]
        node.decorator_list = []
        methods.append(node)
    cls = ast.ClassDef(
        name="UnderTest", bases=[], keywords=[], body=methods, decorator_list=[]
    )
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            cls,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    exec(compile(module, str(path), "exec"), env)
    return env["UnderTest"]


class TestSourceDrain(unittest.TestCase):
    def setUp(self):
        self.tracker = SourceDrainTracker()
        self.chunk = SimpleNamespace(room=7)

    def test_registration_precedes_publication_and_tracks_all_chunks(self):
        queue = Mock()
        queue.put.side_effect = lambda chunk: self.assertFalse(self.tracker.drained(7))
        self.tracker.publish(self.chunk, queue, lambda: True)
        another = SimpleNamespace(room=7)
        self.tracker.publish(another, queue, lambda: True)
        self.tracker.begin(self.chunk)
        self.tracker.begin(another)
        self.tracker.complete(self.chunk)
        self.tracker.complete(self.chunk)
        self.assertFalse(self.tracker.drained(7))
        self.tracker.complete(another)
        self.tracker.close(7)
        self.assertTrue(self.tracker.drained(7))

    def test_deferred_requeue_keeps_original_ownership(self):
        queue = Mock()
        self.tracker.publish(self.chunk, queue, lambda: True)
        queue.put(self.chunk)  # Worker defer / hidden ACK wake, same object.
        self.assertFalse(self.tracker.drained(7))
        self.tracker.begin(self.chunk)
        self.tracker.complete(self.chunk)
        self.tracker.close(7)
        self.assertTrue(self.tracker.drained(7))

    def test_aborted_room_rejects_late_enqueue(self):
        queue = Mock()
        self.tracker.publish(self.chunk, queue, lambda: False)
        queue.put.assert_not_called()
        self.tracker.close(7, poisoned=True)
        self.assertTrue(self.tracker.terminal(7))
        self.assertFalse(self.tracker.drained(7))

    def test_ambiguous_publication_stays_quarantined(self):
        queue = Mock()
        queue.put.side_effect = RuntimeError("after publication unknown")
        with self.assertRaises(RuntimeError):
            self.tracker.publish(self.chunk, queue, lambda: True)
        self.tracker.begin(self.chunk)
        self.tracker.complete(self.chunk)
        self.assertFalse(self.tracker.drained(7))

    def test_worker_exception_never_claims_drain(self):
        self.tracker.poison(7)
        self.assertFalse(self.tracker.drained(7))

    def test_sync_exception_retains_native_hazard_after_call_returns(self):
        queue = Mock()
        self.tracker.publish(self.chunk, queue, lambda: True)
        self.tracker.begin(self.chunk)
        with self.assertRaises(RuntimeError):
            self.tracker.run(
                lambda: (_ for _ in ()).throw(RuntimeError("sync failure"))
            )
        self.tracker.mark_unresolved(self.chunk)
        self.assertFalse(self.tracker.finish_quiesced(self.chunk))
        self.tracker.poison(7)
        self.assertFalse(self.tracker.terminal(7))
        self.assertFalse(self.tracker.forget(7))

    def test_watchdog_detaches_native_call_and_retains_hazard_until_return(self):
        queue = Mock()
        self.tracker.publish(self.chunk, queue, lambda: True)
        self.tracker.begin(self.chunk)
        started = threading.Event()
        finished = threading.Event()
        release = threading.Event()

        def blocked_native_call():
            started.set()
            release.wait(2)
            finished.set()
            return 0

        self.assertRaises(
            SourceTransferTimeout,
            self.tracker.run_with_deadline,
            blocked_native_call,
            timeout_s=0.02,
            success=lambda ret: ret == 0,
        )
        self.assertTrue(started.wait(1))
        # The caller returned, but the source is not safe to release while the
        # detached native operation still owns its RDMA buffer.
        self.tracker.mark_unresolved(self.chunk)
        self.tracker.poison(7)
        self.assertFalse(self.tracker.terminal(7))

        release.set()
        self.assertTrue(finished.wait(1))
        self.assertTrue(self.tracker.terminal(7))
        self.assertTrue(self.tracker.forget(7))

    def test_shutdown_quarantines_timed_out_native_call_then_forgets(self):
        """Manager shutdown must not release a buffer before native return."""
        queue = Mock()
        self.tracker.publish(self.chunk, queue, lambda: True)
        self.tracker.begin(self.chunk)
        started = threading.Event()
        release = threading.Event()

        def blocked_native_call():
            started.set()
            release.wait(2)
            return 0

        with self.assertRaises(SourceTransferTimeout):
            self.tracker.run_with_deadline(
                blocked_native_call, timeout_s=0.02, success=lambda ret: ret == 0
            )
        self.assertTrue(started.wait(1))

        # The first shutdown call times out and leaves the room quarantined.
        self.assertTrue(self.tracker.shutdown(timeout_s=0.01))
        self.assertFalse(self.tracker.terminal(7))

        release.set()
        self.assertTrue(self.tracker.shutdown(timeout_s=1.0) == [])
        self.assertFalse(self.tracker._generations)

    def test_failed_future_completion_does_not_prove_native_drain(self):
        queue = Mock()
        self.tracker.publish(self.chunk, queue, lambda: True)
        self.tracker.begin(self.chunk)
        with concurrent.futures.ThreadPoolExecutor(1) as executor:
            future = self.tracker.submit(
                executor,
                lambda: (_ for _ in ()).throw(RuntimeError("future failure")),
            )
            with self.assertRaises(RuntimeError):
                future.result()
        self.tracker.mark_unresolved(self.chunk)
        self.assertFalse(self.tracker.finish_quiesced(self.chunk))

    def test_rejected_submit_does_not_leave_submission_hazard(self):
        class RejectingExecutor:
            def submit(self, fn):
                raise RuntimeError("executor stopped")

        queue = Mock()
        self.tracker.publish(self.chunk, queue, lambda: True)
        self.tracker.begin(self.chunk)
        with self.assertRaises(RuntimeError):
            self.tracker.submit(RejectingExecutor(), lambda: None)
        self.tracker.mark_unresolved(self.chunk)
        self.assertTrue(self.tracker.finish_quiesced(self.chunk))

    def test_sender_probe_wakes_parked_chunks_but_does_not_invent_drain(self):
        env = dict(KVPoll=KVPoll)
        cls = load_methods(CONN, ["source_transfers_drained"], env)
        mgr = cls()
        mgr._source_drain = self.tracker
        mgr.request_status = {7: KVPoll.Failed}
        mgr._wake_pd_hidden_ack_waiters = Mock()
        self.tracker.publish(self.chunk, Mock(), lambda: True)
        self.assertFalse(mgr.source_transfers_drained(7))
        mgr._wake_pd_hidden_ack_waiters.assert_called_once_with(7)
        self.tracker.begin(self.chunk)
        self.tracker.complete(self.chunk)
        self.tracker.poison(7)
        self.assertTrue(mgr._source_drain.terminal(7))
        mgr.request_status.clear()  # failure_exception() clears the room.
        self.assertFalse(mgr.source_transfers_drained(7))

    def test_real_worker_skip_completes_only_its_own_token(self):
        class StopWorker(BaseException):
            pass

        cls = load_methods(
            CONN,
            ["transfer_worker", "_complete_source_chunk"],
            dict(KVPoll=KVPoll, logger=Mock(), concurrent=concurrent),
        )
        mgr = cls()
        mgr.enable_trace = False
        mgr.enable_deferred_decode_kv_release = False
        mgr._source_drain = self.tracker
        mgr._staging_outstanding = defaultdict(int)
        mgr.request_status = {7: KVPoll.Failed}
        mgr.check_status = Mock(return_value=KVPoll.Failed)
        mgr._wake_pd_hidden_ack_waiters = Mock()
        mgr._has_pd_hidden_state = Mock(return_value=False)
        chunk = SimpleNamespace(
            room=7,
            staging_counted=False,
            source_event=None,
            wait_event=None,
            pd_hidden_start=None,
            pd_hidden_sent=False,
            state_indices=None,
        )
        queue = Mock()
        self.tracker.publish(chunk, queue, lambda: True)
        self.tracker.publish(self.chunk, queue, lambda: True)
        queue.get.side_effect = [chunk, StopWorker()]
        with self.assertRaises(StopWorker):
            mgr.transfer_worker(queue, Mock())
        self.assertFalse(self.tracker.drained(7))
        self.tracker.begin(self.chunk)
        self.tracker.complete(self.chunk)
        self.tracker.close(7)
        self.assertTrue(self.tracker.drained(7))

    def test_worker_exit_publishes_failure_to_active_decode_rooms(self):
        class StopWorker(BaseException):
            pass

        cls = load_methods(
            CONN,
            ["_run_transfer_worker"],
            dict(KVPoll=KVPoll, logger=Mock()),
        )
        mgr = cls()
        mgr.transfer_worker = Mock(side_effect=StopWorker())
        mgr._source_drain = Mock()
        mgr._source_drain.worker_failed.return_value = {(7, 1), (8, 2)}
        mgr.request_status = {7: KVPoll.Transferring, 8: KVPoll.Failed}
        mgr.poison_deferred_ack_room = Mock()
        mgr.conclude_failure = Mock()
        with self.assertRaises(StopWorker):
            mgr._run_transfer_worker(Mock(), Mock())
        mgr.conclude_failure.assert_called_once_with(
            bootstrap_room=7,
            failure_reason=(
                "Mooncake transfer worker exited while the KV transfer for room "
                "7 was active"
            ),
        )

    def test_all_running_futures_drain_after_status_failure_or_exception(self):
        cls = load_methods(
            CONN, ["_await_transfer_futures"], dict(concurrent=concurrent)
        )
        for error in (False, True):
            for deferred in (False, True):
                with self.subTest(exception=error, deferred=deferred):
                    mgr = cls()
                    mgr.enable_deferred_decode_kv_release = deferred
                    first = concurrent.futures.Future()
                    running = concurrent.futures.Future()
                    running.set_running_or_notify_cancel()
                    failure_observed = threading.Event()
                    real_cancel = running.cancel

                    def cancel():
                        failure_observed.set()
                        return real_cancel()

                    running.cancel = cancel
                    if error:
                        first.set_exception(RuntimeError("transfer failed"))
                    else:
                        first.set_result(-1)
                    outcome = []

                    def wait():
                        try:
                            outcome.append(
                                mgr._await_transfer_futures([first, running])
                            )
                        except Exception as exc:
                            outcome.append(exc)

                    worker = threading.Thread(target=wait, daemon=True)
                    worker.start()
                    try:
                        self.assertTrue(failure_observed.wait(2))
                        self.assertTrue(worker.is_alive())
                        self.assertEqual(outcome, [])
                    finally:
                        running.set_result(0)
                        worker.join(2)
                    self.assertFalse(worker.is_alive())
                    if error:
                        self.assertIsInstance(outcome[0], RuntimeError)
                    else:
                        self.assertEqual(outcome, [-1])


class TestTransferWorkerFailure(unittest.TestCase):
    def test_source_forget_is_idempotent_after_terminal_clear(self):
        cls = load_methods(
            CONN,
            ["forget_source_transfers"],
            dict(),
        )
        sender = cls()
        sender._source_generation = None
        sender._source_transfers_forgotten = True
        sender.kv_mgr = SimpleNamespace(_source_drain=Mock(lock=threading.RLock()))
        self.assertTrue(cls.forget_source_transfers(sender))
        sender.kv_mgr._source_drain.forget.assert_not_called()

    def test_worker_failure_broadcasts_terminal_status_and_releases_bookkeeping(self):
        cls = load_methods(
            CONN,
            ["_conclude_transfer_worker_failure"],
            dict(KVPoll=KVPoll, logger=Mock()),
        )
        mgr = cls()
        mgr._source_drain = Mock()
        mgr._staging_outstanding = defaultdict(int, {7: 1})
        mgr.enable_deferred_decode_kv_release = False
        mgr._wake_pd_hidden_ack_waiters = Mock()
        mgr._has_pd_hidden_state = Mock(return_value=False)
        mgr.poison_deferred_ack_room = Mock()
        mgr.record_failure = Mock()
        mgr.poison_deferred_ack_room = Mock()
        mgr.conclude_failure = Mock()
        mgr.update_status = Mock()
        chunk = SimpleNamespace(
            room=7,
            pd_hidden_start=None,
            pd_hidden_sent=False,
            state_indices=None,
            staging_counted=True,
        )

        cls._conclude_transfer_worker_failure(
            mgr, chunk, RuntimeError("synchronous transfer failed")
        )

        mgr._source_drain.mark_unresolved.assert_called_once_with(chunk)
        mgr._source_drain.finish_quiesced.assert_called_once_with(chunk)
        mgr._source_drain.poison.assert_called_once_with(7)
        mgr.record_failure.assert_called_once_with(7, "synchronous transfer failed")
        mgr.conclude_failure.assert_called_once_with(
            bootstrap_room=7,
            failure_reason="synchronous transfer failed",
        )
        mgr._wake_pd_hidden_ack_waiters.assert_called_once_with(7)
        self.assertNotIn(7, mgr._staging_outstanding)


if __name__ == "__main__":
    unittest.main()
