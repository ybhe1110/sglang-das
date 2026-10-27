"""Production ownership/collective bodies tested without Torch or native RDMA.

These model transport outcomes, not native DMA completion. Actual wheel fault
injection is still required: a returned failure must never be treated as a fence.
"""

import threading
import unittest
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

from test_collective_timeout import load_collective_helpers
from test_external_kv_abort_drain import (
    CONN,
    ROOT,
    SourceDrainTracker,
    load_methods,
    register_cpu_ci,
)

register_cpu_ci(est_time=3, suite="base-a-test-cpu")
COMMON = ROOT / "python/sglang/srt/disaggregation/common/conn.py"
DECODE = ROOT / "python/sglang/srt/disaggregation/decode.py"


def manager_class():
    return load_methods(
        COMMON,
        [
            "register_deferred_abort_room",
            "note_abort_ack",
            "is_abort_release_safe",
            "clear_deferred_abort_state",
            "_maybe_ack_drained_abort",
            "_send_abort_acks",
            "register_deferred_ack_target",
            "poison_deferred_ack_room",
        ],
        {"DeferredAbortAckState": SimpleNamespace},
    )


def manager():
    m = manager_class()()
    m._deferred_abort_generation = 0
    m._deferred_abort_ack_tracker = {}
    m._deferred_ack_targets = {}
    m._deferred_ack_poisoned_rooms = set()
    m._staging_outstanding = defaultdict(int)
    m._send_abort_ack = Mock()
    return m


def mooncake_manager():
    override = load_methods(CONN, ["_maybe_ack_drained_abort"], {})
    m = type("MooncakeUnderTest", (override, manager_class()), {})()
    m.__dict__.update(manager().__dict__)
    m._source_drain = SourceDrainTracker()
    return m


class TestGenerationAndNativeDrain(unittest.TestCase):
    def test_legacy_abort_is_upgraded_once_after_partial_metadata_publish(self):
        class Notification(SimpleNamespace):
            def to_zmq(self):
                return [self.generation]

        cls = load_methods(
            COMMON,
            ["ensure_abort_notified", "_send_abort_notification"],
            {"AbortNotification": Notification, "logger": Mock()},
            class_name="CommonKVReceiver",
        )
        r = cls()
        r.kv_mgr = m = manager()
        m.enable_deferred_decode_kv_release = True
        m.local_ip, m.rank_port = "peer", 1234
        r.bootstrap_room, r.init_time = 7, None
        r.bootstrap_infos = [{"rank_ip": "p", "rank_port": 1}]
        r.abort_notified, r._abort_generation = False, None
        sent = []

        def send(msg):
            sent.append(msg)
            if msg[0] is not None:
                # Synchronous ACK during send proves accounting was armed first.
                m.note_abort_ack(7, 0, msg[0])

        socket = SimpleNamespace(send_multipart=send)
        r._connect_to_bootstrap_server = lambda info: (socket, threading.Lock())
        r.ensure_abort_notified()
        self.assertEqual(sent, [[None]])
        r.ensure_abort_notified(force_arm=True)
        self.assertTrue(m.is_abort_release_safe(7, 1))
        r.ensure_abort_notified(force_arm=True)
        self.assertEqual(len(sent), 2)
        self.assertTrue(m.is_abort_release_safe(7, 1))

    def test_stale_and_duplicate_acks_cannot_release_reused_room(self):
        m = manager()
        old = m.register_deferred_abort_room(7)
        m.note_abort_ack(7, 0, old)
        m.note_abort_ack(7, 0, old)
        self.assertFalse(m.is_abort_release_safe(7, 2))
        m.clear_deferred_abort_state(7)
        new = m.register_deferred_abort_room(7)
        m.note_abort_ack(7, 1, old)
        self.assertFalse(m.is_abort_release_safe(7, 1))
        for rank in (0, 1):
            m.note_abort_ack(7, rank, new)
        self.assertTrue(m.is_abort_release_safe(7, 2))

    def test_ack_fanout_waits_for_source_tracker_not_just_chunk_counter(self):
        m = mooncake_manager()
        chunk = SimpleNamespace(room=7)
        m._source_drain.publish(chunk, Mock(), lambda: True)
        m._source_drain.begin(chunk)
        for port in (6000, 6001):
            m.register_deferred_ack_target(
                7, SimpleNamespace(ip="peer", port=port, generation=1)
            )
        m._maybe_ack_drained_abort(7)
        m._send_abort_ack.assert_not_called()
        m._source_drain.complete(chunk)
        m._source_drain.close(7)
        m._maybe_ack_drained_abort(7)
        self.assertEqual(m._send_abort_ack.call_count, 2)
        m._maybe_ack_drained_abort(7)
        self.assertEqual(m._send_abort_ack.call_count, 2)

    def test_failed_native_return_never_turns_into_drained_ack(self):
        m = mooncake_manager()
        chunk = SimpleNamespace(room=7)
        m._source_drain.publish(chunk, Mock(), lambda: True)
        m._source_drain.begin(chunk)
        with self.assertRaises(RuntimeError):
            m._source_drain.run(lambda: -1, success=lambda rc: rc == 0)
        m._source_drain.complete(chunk)
        m.register_deferred_ack_target(
            7, SimpleNamespace(ip="peer", port=1, generation=1)
        )
        m._maybe_ack_drained_abort(7)
        m._send_abort_ack.assert_not_called()
        self.assertFalse(m._source_drain.forget(7))

    def test_poison_suppresses_late_abort_and_zero_counter_ack(self):
        m = manager()
        m.poison_deferred_ack_room(7)
        m.register_deferred_ack_target(
            7, SimpleNamespace(ip="peer", port=1, generation=1)
        )
        m._maybe_ack_drained_abort(7)
        m._send_abort_ack.assert_not_called()


def queue(env, mgr, rid="r", ready=False):
    cls = load_methods(
        DECODE,
        [
            "_reduce_deferred_flags",
            "resolve_deferred_releases",
            "_do_release",
            "has_pending_deferred_releases",
            "num_pending_deferred_releases",
            "release_memory_occupation",
        ],
        env,
        class_name="DecodeTransferQueue",
    )
    q = cls()
    q.gloo_group = object()
    q.queue = []
    q.enable_staging = False
    q.tree_cache = object()
    q._release_pd_hidden_rows = Mock()
    q.req_to_metadata_buffer_idx_allocator = Mock()
    q.metadata_buffers = SimpleNamespace(bootstrap_room={3: 7})
    q._observe_deferred_kv_release = Mock()
    q._deferred_release_quarantine = {}
    generation = mgr.register_deferred_abort_room(7)
    if ready:
        mgr.note_abort_ack(7, 0, generation)
    receiver = SimpleNamespace(kv_mgr=mgr, clear=Mock())
    req = SimpleNamespace(
        req=SimpleNamespace(rid=rid, bootstrap_room=7), kv_receiver=receiver
    )
    q._deferred_releases = [(req, 90.0, 110.0, 3, 1)]
    return q, req


class PairMin:
    """Bounded two-rank MIN simulator using real preflight tensor payloads."""

    def __init__(self):
        self.pending = []
        self.lengths = []
        self.lock = threading.Lock()
        self.barrier = threading.Barrier(2, action=self.reduce)

    def reduce(self):
        a, b = self.pending
        if len(a.values) != len(b.values):
            raise AssertionError("variable collective reached with mismatched shape")
        self.lengths.append(len(a.values))
        result = [min(x, y) for x, y in zip(a.values, b.values)]
        a.values[:] = b.values[:] = result
        self.pending.clear()

    def __call__(self, tensor, **kw):
        with self.lock:
            self.pending.append(tensor)
        self.barrier.wait(timeout=5)


class TestDecodeReleaseConsensus(unittest.TestCase):
    def setUp(self):
        self.env = load_collective_helpers()
        self.env["torch"].uint8 = object()
        self.env.update(
            time=SimpleNamespace(monotonic=lambda: 100.0), release_kv_cache=Mock()
        )
        self.env["all_reduce_with_timeout"] = lambda *a, **kw: None

    def test_timeout_preserves_all_destination_owners(self):
        q, dr = queue(self.env, manager())
        q._deferred_releases[0] = (dr, 50.0, 99.0, 3, 1)
        with self.assertRaises(self.env["DisaggregationCollectiveError"]):
            q.resolve_deferred_releases()
        self.assertEqual(q.num_pending_deferred_releases(), 1)
        self.env["release_kv_cache"].assert_not_called()
        q._release_pd_hidden_rows.assert_not_called()
        q.req_to_metadata_buffer_idx_allocator.free.assert_not_called()
        dr.kv_receiver.clear.assert_not_called()
        with self.assertRaises(RuntimeError):
            q.release_memory_occupation()

    def test_complete_acks_release_all_owners_once(self):
        q, dr = queue(self.env, manager(), ready=True)
        receiver = dr.kv_receiver
        q.resolve_deferred_releases()
        q.resolve_deferred_releases()
        self.env["release_kv_cache"].assert_called_once_with(
            dr.req, q.tree_cache, is_insert=False
        )
        q._release_pd_hidden_rows.assert_called_once_with(dr)
        q.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        receiver.clear.assert_called_once_with()
        self.assertFalse(q.has_pending_deferred_releases())

    def test_partial_free_exception_is_quarantined_without_retry(self):
        q, dr = queue(self.env, manager(), ready=True)
        q._release_pd_hidden_rows.side_effect = RuntimeError("partially freed")
        with self.assertRaises(self.env["DisaggregationCollectiveError"]):
            q.resolve_deferred_releases()
        q.resolve_deferred_releases()
        self.assertEqual(q.num_pending_deferred_releases(), 1)
        self.env["release_kv_cache"].assert_called_once()
        q._release_pd_hidden_rows.assert_called_once()
        q.req_to_metadata_buffer_idx_allocator.free.assert_not_called()

    def _run_pair(self, queues):
        pair = PairMin()
        self.env["all_reduce_with_timeout"] = pair
        with ThreadPoolExecutor(2) as pool:
            jobs = [pool.submit(q.resolve_deferred_releases) for q in queues]
            errors = []
            for job in jobs:
                try:
                    job.result(timeout=10)
                except Exception as exc:
                    errors.append(exc)
        return pair, errors

    def test_rank_local_ack_readiness_cannot_release_one_rank_early(self):
        a, _ = queue(self.env, manager(), ready=True)
        b, _ = queue(self.env, manager(), ready=False)
        _, errors = self._run_pair([a, b])
        self.assertEqual(errors, [])
        self.assertEqual(
            [a.num_pending_deferred_releases(), b.num_pending_deferred_releases()],
            [1, 1],
        )
        self.env["release_kv_cache"].assert_not_called()

    def test_count_and_identity_drift_fail_before_variable_reduce(self):
        for mismatch in ("empty", "rid"):
            with self.subTest(mismatch=mismatch):
                a, _ = queue(self.env, manager(), ready=True)
                b, _ = queue(self.env, manager(), rid="other", ready=True)
                if mismatch == "empty":
                    b._deferred_releases = []
                pair, errors = self._run_pair([a, b])
                self.assertEqual(len(errors), 2)
                self.assertTrue(
                    all(
                        isinstance(e, self.env["DisaggregationCollectiveError"])
                        for e in errors
                    )
                )
                self.assertEqual(pair.lengths, [8])
                self.env["release_kv_cache"].assert_not_called()


class TestPDHiddenOwnership(unittest.TestCase):
    def setUp(self):
        path = ROOT / "python/sglang/srt/disaggregation/hidden_events.py"
        cls = load_methods(
            path,
            [
                "_assert_chunk_source_quiesced",
                "free_chunk_rows",
                "release_or_mark_done",
            ],
            {},
        )
        self.h = cls()
        self.tracker = SourceDrainTracker()
        self.h.owner = SimpleNamespace(_source_drain=self.tracker)
        self.h.release_state_indices = lambda chunk: [[1, 2]]
        self.h.free_state_indices = Mock()
        self.h.mark_request_done = Mock()
        self.chunk = SimpleNamespace(room=7, pd_hidden_start=0)
        self.tracker.publish(self.chunk, Mock(), lambda: True)
        self.tracker.begin(self.chunk)

    def test_failed_native_access_keeps_streaming_and_full_hidden_rows(self):
        with self.assertRaises(RuntimeError):
            self.tracker.run(lambda: -1, success=lambda rc: rc == 0)
        for start in (0, None):
            self.chunk.pd_hidden_start = start
            with self.assertRaisesRegex(RuntimeError, "unresolved DMA"):
                self.h.release_or_mark_done(self.chunk)
        self.h.free_state_indices.assert_not_called()
        self.h.mark_request_done.assert_not_called()

    def test_hidden_rows_released_once_on_success(self):
        self.h.free_chunk_rows(self.chunk)
        self.h.free_chunk_rows(self.chunk)
        self.h.free_state_indices.assert_called_once()

    def test_partial_hidden_release_is_never_retried(self):
        self.h.free_state_indices.side_effect = RuntimeError("partial free")
        with self.assertRaisesRegex(RuntimeError, "partial free"):
            self.h.free_chunk_rows(self.chunk)
        with self.assertRaisesRegex(RuntimeError, "quarantined"):
            self.h.free_chunk_rows(self.chunk)
        self.h.free_state_indices.assert_called_once()

    def test_decode_hidden_ack_timeout_does_not_drop_receive_owner(self):
        cls = load_methods(DECODE, ["_release_pd_hidden_rows"], {"logger": Mock()})
        q = cls()
        q.kv_manager = SimpleNamespace(
            wait_pd_hidden_ack_completions=lambda room: False
        )
        q.metadata_buffers = SimpleNamespace(pd_hidden_pool=Mock())
        req = SimpleNamespace(
            req=SimpleNamespace(rid="r", bootstrap_room=7), pd_hidden_dst_indices=[1, 2]
        )
        with self.assertRaisesRegex(RuntimeError, "ACK completion unknown"):
            q._release_pd_hidden_rows(req)
        self.assertEqual(req.pd_hidden_dst_indices, [1, 2])
        q.metadata_buffers.pd_hidden_pool.free.assert_not_called()


class TestRestoreAndOffload(unittest.TestCase):
    def test_restore_failure_and_pending_are_part_of_reduced_poll(self):
        polls = SimpleNamespace(Failed=0, Transferring=1, Success=2)
        restore = SimpleNamespace(FAILED="failed", PENDING="pending", READY="ready")
        path = ROOT / "python/sglang/srt/disaggregation/decode_hicache_mixin.py"
        cls = load_methods(
            path,
            ["poll"],
            {"KVPoll": polls, "HiCacheRestoreResult": restore},
            class_name="HiCacheRestoreGatedKVReceiver",
        )
        for local, state, expected in (
            (2, "failed", 0),
            (2, "pending", 1),
            (2, "ready", 2),
            (1, "failed", 1),
            (0, "pending", 0),
        ):
            with self.subTest(local=local, state=state):
                gate = cls()
                gate.decode_req = SimpleNamespace(
                    kv_receiver=SimpleNamespace(poll=lambda: local),
                    hicache_restore_status=state,
                )
                self.assertEqual(gate.poll(), expected)

    def test_offload_retraction_refuses_local_or_peer_pending_copy(self):
        class Tensor:
            def __init__(self, values):
                self.value = values[0]

            def item(self):
                return self.value

        torch = SimpleNamespace(
            tensor=lambda values, **kw: Tensor(values),
            int=int,
            distributed=SimpleNamespace(ReduceOp=SimpleNamespace(MAX="max")),
        )
        env = {"torch": torch, "all_reduce_with_timeout": Mock()}
        path = (
            ROOT / "python/sglang/srt/disaggregation/decode_kvcache_offload_manager.py"
        )
        cls = load_methods(path, ["drain_before_retraction"], env)
        for local, peer in ((False, False), (True, False), (False, True)):
            with self.subTest(local=local, peer=peer):
                m = cls()
                m.check_offload_progress = Mock()
                m.cache_controller = SimpleNamespace(
                    ack_write_queue=[1] if local else []
                )
                m.ongoing_offload, m.offload_inflight = {}, {}
                m.tp_world_size, m.tp_group = 2, object()

                def reduce(tensor, **kw):
                    m.check_offload_progress.assert_called_once()
                    tensor.value = tensor.value or peer

                env["all_reduce_with_timeout"].side_effect = reduce
                if local or peer:
                    with self.assertRaisesRegex(RuntimeError, "refusing to retract"):
                        m.drain_before_retraction()
                else:
                    m.drain_before_retraction()


if __name__ == "__main__":
    unittest.main()
