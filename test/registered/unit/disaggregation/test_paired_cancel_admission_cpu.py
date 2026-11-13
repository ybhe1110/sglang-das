"""Cancel-before-P-admission regressions using production protocol bodies.

IO and native completion are controlled test boundaries. These tests do not
prove the deployed RDMA wheel's cancellation/completion contract.
"""

import ast
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from types import SimpleNamespace
from unittest.mock import Mock

import test_pd_failure_protocol_cpu as protocol
from test_collective_timeout import UTILS
from test_external_kv_abort_drain import PREFILL, load_methods, register_cpu_ci
from test_pd_failure_protocol_cpu import (
    COMMON,
    CONN,
    PairMin,
    queue,
    wire_managers,
)
from test_source_drain_retirement import extract_class

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def sender_class(env):
    env.update(
        KVTransferMetric=Mock,
        KVTransferError=lambda room, reason, **kw: RuntimeError(reason),
        TraceNullContext=Mock,
        get_parallel=lambda: SimpleNamespace(num_dp_ranks=1),
        mooncake_trace_func=lambda stage: lambda fn: fn,
        MooncakeRequestStage=SimpleNamespace(MOONCAKE_SEND=0),
    )
    extract_class(COMMON, "CommonKVSender", {"__init__", "clear", "abort"}, [], env)
    extract_class(CONN, "MooncakeFailureExceptionMixin", {"failure_exception"}, [], env)
    return extract_class(
        CONN,
        "MooncakeKVSender",
        {
            "__init__",
            "poll",
            "_poll_with_status",
            "send",
            "failure_exception",
            "_init_trace_ctx",
            "clear",
            "abort",
            "is_transfer_terminal",
            "is_transfer_drained",
            "forget_source_transfers",
        },
        ["MooncakeFailureExceptionMixin", "CommonKVSender"],
        env,
    )


def receiver(p, d, env, generation=123):
    cls = load_methods(
        COMMON,
        [
            "ensure_abort_notified",
            "_send_abort_notification",
            "retry_abort_notification",
        ],
        env,
        class_name="CommonKVReceiver",
    )
    r = cls()
    r.kv_mgr = d
    d.local_ip, d.rank_port = "decode", 6000
    r.bootstrap_room, r.init_time = 7, 1.0
    r._bootstrap_pair_key, r._transfer_generation = "cancelled-pair", generation
    r.abort_notified, r._abort_generation = False, None
    r._last_abort_notification_time = None
    r.bootstrap_infos = [{"rank_ip": "prefill", "rank_port": 9000}]
    r._connect_to_bootstrap_server = lambda info, **kw: (
        SimpleNamespace(
            send_multipart=lambda frames, **kw: p._handle_abort_notification(frames)
        ),
        threading.Lock(),
    )
    r.clear = Mock()
    return r


class TestPairedCancellation(protocol.CustomTestCase):
    def setUp(self):
        self.p, self.d, self.sent, self.env = wire_managers()
        self.p.is_dummy_cp_rank = self.p.enable_trace = False
        self.p.failure_records = {}
        self.p.failure_lock = threading.Lock()
        self.p.record_failure = Mock()
        self.p._wake_pd_hidden_ack_waiters = Mock()
        protocol.TestGenerationAndNativeDrain.prepare_destinations(self.p)

    def cancel(self, key="cancelled-pair", generation=123):
        self.p._handle_abort_notification(
            self.env["AbortNotification"](7, "decode", 6000, generation, key).to_zmq()
        )

    def sender(self, key="cancelled-pair"):
        return sender_class(self.env)(self.p, "prefill:9000", 7, bootstrap_pair_key=key)

    def test_wire_roundtrip_and_malformed_pair_do_not_downgrade(self):
        cls = self.env["AbortNotification"]
        msg = cls(7, "decode", 6000, 123, "shared-pair")
        self.assertEqual(cls.from_zmq(msg.to_zmq()), msg)
        self.assertEqual(len(cls(7, "decode", 6000, 123).to_zmq()), 5)
        for pair in (b"", b"x" * 129, b"\xff"):
            with self.subTest(pair=pair[:8]):
                self.p._handle_abort_notification(msg.to_zmq()[:5] + [pair])
        self.assertEqual(self.sent, [])
        self.assertEqual(self.p._cancelled_source_pairs, set())

    def test_cancel_before_owner_purges_only_same_pair_metadata(self):
        metadata = protocol.TestGenerationAndNativeDrain.metadata
        self.p._bind_abort_metadata(7, metadata("cancelled-pair", 123), "old")
        self.p._bind_abort_metadata(7, metadata("new-pair", 456), "new")
        r = receiver(self.p, self.d, self.env)
        r.ensure_abort_notified()
        self.assertTrue(self.d.is_abort_release_safe(7, 123, {0}))
        self.assertEqual(self.p._pending_abort_metadata_count, 1)
        self.assertEqual(set(self.p._pending_abort_metadata[7]), {("new-pair", "new")})
        self.assertEqual(
            set(self.p._pending_abort_metadata_times), {(7, "new-pair", "new")}
        )
        self.assertEqual(self.p.request_status, {})
        self.assertFalse(
            self.p._bind_abort_metadata(7, metadata("cancelled-pair", 123), "late")
        )
        self.assertIsNone(self.p._open_source_room(7, "cancelled-pair"))
        self.p._open_source_room(7, "new-pair")
        self.assertEqual(set(self.p.transfer_infos[7]), {"new"})

    def test_late_sender_is_failed_and_never_touches_reused_room(self):
        self.cancel()
        new_epoch = self.p._open_source_room(7, "new-pair")
        self.p.failure_records[7] = "new owner's failure"
        for _ in range(2):
            late = self.sender()
            self.assertEqual(late.poll(), self.env["KVPoll"].Failed)
            late.send([])
            late.clear()
            late.abort()
            self.assertTrue(late.is_transfer_terminal())
            self.assertTrue(late.is_transfer_drained())
            self.assertTrue(late.forget_source_transfers())
            with self.assertRaisesRegex(
                RuntimeError, "cancelled before source admission"
            ):
                late.failure_exception()
        self.assertEqual(self.p.request_status[7], self.env["KVPoll"].Bootstrapping)
        self.assertEqual(self.p.failure_records[7], "new owner's failure")
        self.assertTrue(self.p._source_drain.owns_generation(7, new_epoch))
        self.assertNotIn((7, new_epoch), self.p._source_drain._closed)
        self.p.record_failure.assert_not_called()

    def test_cancel_between_open_and_sender_initialization_stays_failed(self):
        original = self.p._open_source_room

        def open_then_cancel(*args):
            epoch = original(*args)
            self.cancel()
            return epoch

        self.p._open_source_room = open_then_cancel
        sender = self.sender()
        self.assertEqual(sender.poll(), self.env["KVPoll"].Failed)
        self.assertFalse(
            self.p._source_drain.publish(SimpleNamespace(room=7), Mock(), lambda: True)
        )
        self.assertEqual(len(self.sent), 1)

    def test_paired_abort_waits_for_active_source_even_without_metadata_binding(self):
        self.p._open_source_room(7, "cancelled-pair")
        chunk = SimpleNamespace(room=7)
        self.p._source_drain.publish(chunk, Mock(), lambda: True)
        self.p._source_drain.begin(chunk)
        self.cancel()
        self.assertEqual(self.sent, [])
        self.assertFalse(self.p._source_drain.forget(7))
        self.assertFalse(
            self.p._bind_abort_metadata(
                7,
                protocol.TestGenerationAndNativeDrain.metadata("cancelled-pair", 123),
                "late",
            )
        )
        self.p._source_drain.complete(chunk)
        self.p._maybe_ack_drained_abort(7)
        self.assertEqual(len(self.sent), 1)

    def test_paired_cancel_never_acks_uncertain_native_dma(self):
        self.p._open_source_room(7, "cancelled-pair")
        chunk = SimpleNamespace(room=7)
        tracker = self.p._source_drain
        tracker.publish(chunk, Mock(), lambda: True)
        tracker.begin(chunk)
        with self.assertRaises(RuntimeError):
            tracker.run(lambda: -1, success=lambda rc: rc == 0)
        tracker.complete(chunk)
        for _ in range(3):
            self.cancel()
        self.assertEqual(self.sent, [])
        self.assertFalse(tracker.forget(7))

    def test_cancel_removes_pending_chunk_without_starting_native_transfer(self):
        self.p._open_source_room(7, "cancelled-pair")
        chunk = SimpleNamespace(room=7)
        tracker = self.p._source_drain
        tracker.publish(chunk, Mock(), lambda: True)
        self.cancel()
        self.assertEqual(len(self.sent), 1)
        self.assertFalse(tracker.begin(chunk))
        self.assertTrue(tracker.terminal(7))

    def test_clear_before_cleanup_and_duplicate_cancel_preserve_terminal_fence(self):
        sender = self.sender()
        self.cancel()
        self.assertEqual(sender.poll(), self.env["KVPoll"].Failed)
        sender.clear()
        for _ in range(2):
            self.assertTrue(sender.forget_source_transfers())
            self.assertTrue(sender.is_transfer_terminal())
            self.cancel()
        self.assertEqual(len(self.sent), 3)
        self.assertNotIn(7, self.p._source_drain._generations)
        self.assertIsNone(self.p._open_source_room(7, "cancelled-pair"))
        epoch = self.p._open_source_room(7, "new-pair")
        sender.abort()
        sender.clear()
        self.assertTrue(self.p._source_drain.owns_generation(7, epoch))
        self.assertEqual(self.p.request_status[7], self.env["KVPoll"].Bootstrapping)

    def test_old_pair_cancel_does_not_poison_new_room_or_count_as_new_ack(self):
        self.cancel()
        self.p._open_source_room(7, "new-pair")
        self.d.register_deferred_abort_room(7, generation=456)
        self.cancel()
        self.assertFalse(self.d.is_abort_release_safe(7, 456, {0}))
        self.assertEqual(self.p.request_status[7], self.env["KVPoll"].Bootstrapping)
        self.assertTrue(
            self.p._source_drain.publish(SimpleNamespace(room=7), Mock(), lambda: True)
        )
        self.assertNotIn(7, self.p._deferred_ack_targets)

    def test_conflicting_receiver_generation_cannot_cancel_current_pair(self):
        self.p._open_source_room(7, "cancelled-pair")
        self.p._bind_abort_metadata(
            7, protocol.TestGenerationAndNativeDrain.metadata("cancelled-pair", 456)
        )
        self.cancel(generation=123)
        self.assertEqual(self.sent, [])
        self.assertEqual(self.p._cancelled_source_pairs, set())
        self.assertEqual(self.p.request_status[7], self.env["KVPoll"].Bootstrapping)

    def test_lost_preowner_ack_is_retried_without_rearming(self):
        clock = [100.0]
        self.env["time"] = SimpleNamespace(monotonic=lambda: clock[0])
        r = receiver(self.p, self.d, self.env)
        deliver = self.p._send_multipart_locked
        self.p._send_multipart_locked = Mock(side_effect=BlockingIOError("busy"))
        r.ensure_abort_notified()
        self.assertFalse(self.d.is_abort_release_safe(7, 123, {0}))
        self.assertIn((7, "cancelled-pair"), self.p._cancelled_source_pairs)
        self.p._send_multipart_locked = deliver
        clock[0] += 1.1
        r.retry_abort_notification()
        self.assertTrue(self.d.is_abort_release_safe(7, 123, {0}))
        self.assertEqual(r._abort_generation, 123)
        self.assertEqual(len(self.p._cancelled_source_pairs), 1)

    def test_cancel_fence_precedes_network_send_and_does_not_hold_source_lock(self):
        sending, resume = threading.Event(), threading.Event()

        def send(*args, **kwargs):
            sending.set()
            if not resume.wait(3):
                raise TimeoutError("test send")

        self.p._send_multipart_locked = send
        with ThreadPoolExecutor(2) as pool:
            job = pool.submit(self.cancel)
            try:
                self.assertTrue(sending.wait(2))
                late = pool.submit(self.p._open_source_room, 7, "cancelled-pair")
                self.assertIsNone(late.result(1))
                new = pool.submit(self.p._open_source_room, 7, "new-pair")
                self.assertIsNotNone(new.result(1))
            finally:
                resume.set()
            job.result(3)

    def test_fence_survives_ack_retirement_and_capacity_fails_admission_safely(self):
        self.p._cancelled_source_pair_limit = 1
        self.p._open_source_room(8, "existing")
        self.cancel()
        self.p._retired_abort_rooms.clear()
        self.p._abort_room_bindings.pop(7, None)
        self.p._source_room_pair_keys.pop(7, None)
        self.assertIsNone(self.p._open_source_room(7, "cancelled-pair"))
        self.cancel(key="another-cancel")
        self.assertTrue(self.p._source_admission_fenced)
        self.assertEqual(len(self.p._cancelled_source_pairs), 1)
        self.assertIsNone(self.p._open_source_room(9, "new-work"))
        self.assertTrue(
            self.p._bind_abort_metadata(
                8, protocol.TestGenerationAndNativeDrain.metadata("existing", 456)
            )
        )
        self.assertTrue(
            self.p._source_drain.publish(SimpleNamespace(room=8), Mock(), lambda: True)
        )

    def test_preowner_ack_releases_decode_slots_metadata_and_hidden_once(self):
        collective_case = protocol.TestDecodeReleaseConsensus()
        collective_case.setUp()
        env = collective_case.env
        q, req = queue(env, self.d)
        generation = self.d._deferred_abort_ack_tracker[7].generation
        r = receiver(self.p, self.d, self.env, generation)
        req.kv_receiver = r
        q.resolve_deferred_releases()  # No ACK: all owners remain held.
        env["release_kv_cache"].assert_not_called()
        r.ensure_abort_notified()
        q.resolve_deferred_releases()
        q.resolve_deferred_releases()
        env["release_kv_cache"].assert_called_once_with(
            req.req, q.tree_cache, checkpoint=False
        )
        q._release_pd_hidden_rows.assert_called_once_with(req)
        q.req_to_metadata_buffer_idx_allocator.free.assert_called_once_with(3)
        r.clear.assert_called_once()
        self.assertEqual(q.num_pending_deferred_releases(), 0)

    def test_rank_local_preowner_cancel_preserves_bootstrap_queue_consensus(self):
        other, _, _, other_env = wire_managers()
        other.is_dummy_cp_rank = other.enable_trace = False
        self.cancel()  # Only one rank has received cancellation so far.
        collective_case = protocol.TestDecodeReleaseConsensus()
        collective_case.setUp()
        collective = collective_case.env
        collective["all_reduce_with_timeout"] = PairMin()
        collective["_poll_with_failure_injection"] = lambda pollers: [
            p.poll() for p in pollers
        ]
        nodes = [
            n
            for n in ast.parse(UTILS.read_text(encoding="utf-8")).body
            if isinstance(n, ast.FunctionDef)
            and n.name
            in {"poll_and_all_reduce", "poll_and_all_reduce_attn_cp_tp_group"}
        ]
        module = ast.Module(
            body=[ast.ImportFrom("__future__", [ast.alias("annotations")], 0)] + nodes,
            type_ignores=[],
        )
        exec(compile(ast.fix_missing_locations(module), str(UTILS), "exec"), collective)
        queues = []
        for manager, env in ((self.p, self.env), (other, other_env)):
            cls = sender_class(env)
            env.update(
                _is_hcu=True,
                TransferBackend=SimpleNamespace(FAKE="fake", MOONCAKE="mooncake"),
                KVClassType=SimpleNamespace(SENDER=1),
                FAKE_BOOTSTRAP_HOST="fake",
                get_kv_class=lambda *args, cls=cls: cls,
                is_unadmitted_reject=lambda req: False,
                poll_and_all_reduce_attn_cp_tp_group=collective[
                    "poll_and_all_reduce_attn_cp_tp_group"
                ],
            )
            qcls = load_methods(
                PREFILL,
                ["create_sender", "add", "_process_req", "pop_bootstrapped"],
                env,
                class_name="PrefillBootstrapQueue",
            )
            q = qcls()
            q.queue, q.kv_manager, q.transfer_backend = [], manager, "mooncake"
            q.tp_rank, q.pp_rank, q.pp_size, q.bootstrap_port = 0, 0, 1, 9000
            q.scheduler = SimpleNamespace(
                attn_cp_cpu_group=object(),
                attn_tp_cpu_group=object(),
                handle_bootstrap_failure=Mock(),
            )
            q._check_if_req_exceed_kv_capacity = lambda req: False
            q._aligned_bootstrap_requests = lambda q=q: q.queue
            q._get_consensus_bootstrap_resource_ready = lambda requests, polls: (
                [True] * len(polls)
            )
            req = SimpleNamespace(
                rid="same-request",
                bootstrap_host="prefill",
                bootstrap_room=7,
                bootstrap_pair_key="cancelled-pair",
                disagg_prefill_dp_rank=None,
                sampling_params=SimpleNamespace(max_new_tokens=100),
            )
            q.add(req, 1)
            self.assertEqual(len(q.queue), 1)
            # The peer has bootstrapped and is otherwise ready to allocate/run.
            if manager is other:
                manager.update_status(7, env["KVPoll"].WaitingForInput)
            queues.append(q)
        with ThreadPoolExecutor(2) as pool:
            results = [pool.submit(q.pop_bootstrapped, True) for q in queues]
            for result in results:
                good, failed = result.result(5)
                self.assertEqual(good, [])
                self.assertEqual(len(failed), 1)
        for q in queues:
            self.assertEqual(q.queue, [])
            q.scheduler.handle_bootstrap_failure.assert_called_once()


if __name__ == "__main__":
    unittest.main()
