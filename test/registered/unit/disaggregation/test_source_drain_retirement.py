"""Retired senders stay terminal without touching a reused room (CPU only)."""

import ast
from collections import OrderedDict
import threading
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import test_external_kv_abort_drain as abort_drain_tests
from test_external_kv_abort_drain import (
    CONN,
    KVPoll,
    PREFILL,
    ROOT,
    SCHED,
    SourceDrainTracker,
    load_methods,
)

register_cpu_ci = abort_drain_tests.register_cpu_ci
CustomTestCase = unittest.TestCase

register_cpu_ci(est_time=10, suite="base-a-test-cpu")


def extract_class(path, name, names, bases, env):
    original = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == name
    )
    original.body = [
        n for n in original.body if isinstance(n, ast.FunctionDef) and n.name in names
    ]
    original.bases = [ast.Name(id=b, ctx=ast.Load()) for b in bases]
    tree = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            original,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(tree), str(path), "exec"), env)
    return env[name]


def make_sender():
    # Keep the real sender clear/abort and manager drain methods. Only trace,
    # IPC notifications and failure reporting are external boundary doubles.
    env = dict(KVPoll=KVPoll)
    extract_class(
        ROOT / "python/sglang/srt/disaggregation/common/conn.py",
        "CommonKVSender",
        {"clear", "abort"},
        [],
        env,
    )
    cls = extract_class(
        CONN,
        "MooncakeKVSender",
        {
            "clear",
            "abort",
            "is_transfer_terminal",
            "is_transfer_drained",
            "forget_source_transfers",
        },
        ["CommonKVSender"],
        env,
    )
    manager_cls = load_methods(
        CONN,
        [
            "source_transfers_terminal",
            "source_transfers_drained",
            "_retire_abort_binding",
        ],
        dict(KVPoll=KVPoll),
    )
    mgr = manager_cls()
    mgr._source_drain = SourceDrainTracker()
    mgr._abort_room_bindings = {}
    mgr._source_room_pair_keys = {}
    mgr._retired_abort_rooms = OrderedDict()
    mgr._abort_ack_lock = threading.RLock()
    mgr._deferred_ack_targets = {}
    mgr.request_status = {7: KVPoll.Transferring}
    mgr.record_failure = Mock()
    mgr.deferred_bootstrap = None
    mgr.req_to_decode_prefix_len = {}
    mgr.transfer_infos = {}
    mgr._maybe_ack_drained_abort = Mock()
    mgr._wake_pd_hidden_ack_waiters = Mock()

    def update(room, status):
        mgr.request_status[room] = status
        if status == KVPoll.Failed:
            mgr._source_drain.poison(room)

    mgr.update_status = update
    sender = cls()
    sender.kv_mgr = mgr
    sender.bootstrap_room = 7
    sender._source_generation = mgr._source_drain.open(7)
    sender._source_transfers_forgotten = False
    sender.conclude_state = None
    sender.trace_ctx = Mock()
    sender.poll = lambda: sender.conclude_state
    return sender


class TestSourceDrainRetirement(CustomTestCase):
    def test_bootstrap_abort_clear_before_cleanup_remains_terminal(self):
        sender = make_sender()
        sender.abort()  # No KV chunk has ever been submitted.
        sender.clear()  # failure_exception clears before scheduler cleanup.
        for _ in range(3):
            self.assertTrue(sender.is_transfer_terminal())
            self.assertTrue(sender.is_transfer_drained())
            self.assertTrue(sender.forget_source_transfers())
            sender.abort()
            sender.clear()
        self.assertEqual(sender.kv_mgr.request_status, {})
        self.assertEqual(sender.kv_mgr._source_drain._generations, {})

    def test_forget_before_clear_removes_room_metadata(self):
        sender = make_sender()
        sender.abort()
        self.assertTrue(sender.forget_source_transfers())
        self.assertEqual(sender.kv_mgr.request_status, {})
        sender.clear()
        self.assertTrue(sender.is_transfer_terminal())

    def test_retired_sender_does_not_query_or_poison_reused_room(self):
        sender = make_sender()
        sender.abort()
        sender.clear()
        tracker = sender.kv_mgr._source_drain
        new_generation = tracker.open(7)
        sender.kv_mgr.request_status[7] = KVPoll.Transferring
        for _ in range(2):
            self.assertTrue(sender.is_transfer_terminal())
            self.assertTrue(sender.is_transfer_drained())
            self.assertTrue(sender.forget_source_transfers())
            sender.abort()
            sender.clear()
        self.assertTrue(tracker.owns_generation(7, new_generation))
        self.assertEqual(sender.kv_mgr.request_status[7], KVPoll.Transferring)
        self.assertNotIn((7, new_generation), tracker._closed)
        self.assertNotIn((7, new_generation), tracker._poisoned)

    def test_unknown_or_stale_generation_is_not_retired(self):
        for unknown in (False, True):
            with self.subTest(unknown=unknown):
                sender = make_sender()
                tracker = sender.kv_mgr._source_drain
                generation = sender._source_generation
                tracker.poison(7, generation)
                tracker.forget(7, generation)  # Not acknowledged by this sender.
                new_generation = tracker.open(7)
                if unknown:
                    sender._source_generation = None
                self.assertFalse(sender.is_transfer_terminal())
                self.assertFalse(sender.is_transfer_drained())
                self.assertFalse(sender.forget_source_transfers())
                sender.abort()
                sender.clear()
                self.assertTrue(tracker.owns_generation(7, new_generation))
                self.assertNotIn((7, new_generation), tracker._closed)
                self.assertEqual(sender.kv_mgr.request_status[7], KVPoll.Transferring)

    def test_active_source_cannot_be_forgotten_until_completion(self):
        sender = make_sender()
        tracker = sender.kv_mgr._source_drain
        chunk = SimpleNamespace(room=7)
        tracker.publish(chunk, Mock(), lambda: True, sender._source_generation)
        tracker.begin(chunk)
        sender.abort()
        sender.clear()
        self.assertFalse(sender.is_transfer_terminal())
        self.assertFalse(sender.forget_source_transfers())
        tracker.complete(chunk)
        self.assertTrue(sender.is_transfer_terminal())
        sender.clear()
        self.assertTrue(sender.is_transfer_terminal())

    def test_tp8_quarantine_retires_together_and_frees_once(self):
        barrier = threading.Barrier(8, timeout=15)
        flags = [None] * 8
        fixtures = []
        chunks = []
        for rank in range(8):
            # v4.1 parks aborted ReqKvInfo owners in the common inflight queue.
            # Run that production path instead of the release-only quarantine API.
            sender = make_sender()
            req = SimpleNamespace(
                rid="same",
                bootstrap_room=7,
                disagg_kv_sender=sender,
                inflight_middle_chunks=0,
                pending_bootstrap=False,
                finished_reason=None,
                return_logprob=False,
                time_stats=Mock(),
                kv=SimpleNamespace(holds_kv=True, holds_mamba=False),
                cache_request_handle=object(),
            )

            class FinishedAbort:
                pass

            env = dict(
                KVPoll=SimpleNamespace(**vars(KVPoll), WaitingForInput=9),
                time=SimpleNamespace(monotonic=lambda: 10),
                envs=SimpleNamespace(
                    SGLANG_DISAGGREGATION_COLLECTIVE_TIMEOUT=SimpleNamespace(
                        get=lambda: 30
                    )
                ),
                is_aborted=lambda req: isinstance(req.finished_reason, FinishedAbort),
                FINISH_ABORT=FinishedAbort,
                FINISH_LENGTH=lambda **kw: None,
                logger=Mock(),
                get_parallel=lambda: SimpleNamespace(tp_rank=rank),
                HTTPStatus=SimpleNamespace(INTERNAL_SERVER_ERROR=500),
                prepare_abort=lambda req, *a, abort_type=FinishedAbort, **kw: setattr(
                    req, "finished_reason", abort_type()
                ),
                maybe_release_pd_hidden_rows_on_hidden_done=Mock(),
                release_kv_cache=Mock(),
                maybe_release_metadata_buffer=Mock(),
            )
            cls = load_methods(
                PREFILL,
                [
                    "_retire_aborted_prefill_result",
                    "_prefill_cleanup_flags",
                    "process_disagg_prefill_inflight_queue",
                    "handle_inflight_transfer_failure",
                ],
                env,
            )
            scheduler = cls()
            scheduler.tree_cache = SimpleNamespace(linker=None)
            scheduler.clear_pending_chunk_send = Mock()
            scheduler._release_aborted_request = lambda req: (
                req.disagg_kv_sender.clear()
            )
            scheduler.chunked_req = req
            scheduler.attn_cp_cpu_group = object()
            scheduler.attn_tp_cpu_group = object()
            scheduler.disagg_prefill_inflight_queue = []
            scheduler.output_streamer = Mock()
            scheduler.req_to_metadata_buffer_idx_allocator = Mock()
            scheduler.metrics_reporter = SimpleNamespace(enable_metrics=False)
            fixture = SimpleNamespace(req=req, s=scheduler, env=env)
            chunk = SimpleNamespace(room=7)
            if rank == 7:
                sender.kv_mgr._source_drain.publish(chunk, Mock(), lambda: True)
                sender.kv_mgr._source_drain.begin(chunk)
            sender.abort()
            sender.clear()
            scheduler._retire_aborted_prefill_result(req)
            scheduler._retire_aborted_prefill_result(req)

            def reduce(values, *args, rank=rank, **kwargs):
                flags[rank] = values
                barrier.wait()
                self.assertEqual({len(values) for values in flags}, {len(flags[0])})
                result = [min(pair) for pair in zip(*flags)]
                barrier.wait()
                return result

            fixture.env.update(
                all_reduce_min_flags=reduce,
                poll_and_all_reduce_attn_cp_tp_group=lambda senders, *args: (
                    [KVPoll.Failed] * len(senders)
                ),
                _QuarantinedKVSender=lambda: None,
            )
            queue_cls = load_methods(
                PREFILL, ["process_disagg_prefill_inflight_queue"], fixture.env
            )
            fixture.s.process_disagg_prefill_inflight_queue = (
                queue_cls.process_disagg_prefill_inflight_queue.__get__(fixture.s)
            )
            fixtures.append(fixture)
            chunks.append(chunk)

        def round_trip():
            errors = []

            def run(fixture):
                try:
                    fixture.s.process_disagg_prefill_inflight_queue()
                except BaseException as error:
                    errors.append(error)
                    barrier.abort()

            threads = [threading.Thread(target=run, args=(f,)) for f in fixtures]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(20)
                self.assertFalse(thread.is_alive())
            if errors:
                raise errors[0]

        # Seven ranks finish first; the active eighth rank keeps every queue.
        round_trip()
        for fixture in fixtures:
            self.assertEqual(fixture.s.disagg_prefill_inflight_queue, [fixture.req])
        self.assertFalse(getattr(fixtures[0].req, "_pd_cleanup_done", False))
        self.assertFalse(getattr(fixtures[-1].req, "_pd_cleanup_done", False))
        fixtures[-1].req.disagg_kv_sender.kv_mgr._source_drain.complete(chunks[-1])
        round_trip()
        round_trip()  # Empty-queue consensus and repeated cleanup are harmless.
        for fixture in fixtures:
            self.assertEqual(fixture.s.disagg_prefill_inflight_queue, [])
            self.assertTrue(fixture.req._pd_cleanup_done)
            fixture.env["release_kv_cache"].assert_called_once()
            fixture.env["maybe_release_metadata_buffer"].assert_called_once()

    def test_load_snapshot_updates_without_forward_or_full_idle(self):
        clock = [10.0]
        env = dict(
            time=SimpleNamespace(
                monotonic=lambda: clock[0],
                monotonic_ns=lambda: int(clock[0] * 1e9),
                sleep=Mock(),
            ),
            LOAD_STALL_REFRESH_S=1.0,
            DisaggregationMode=SimpleNamespace(NULL="null"),
        )
        cls = load_methods(SCHED, ["on_idle", "publish_load_snapshot"], env)
        s = cls()
        snapshots = []
        s.load_snapshot_writer = SimpleNamespace(
            publish_counter=0, publish_interval=1, write=snapshots.append
        )
        s.maybe_send_health_check_signal = Mock()
        s.dp_attn_adapter = Mock()
        s.metrics_reporter = Mock()
        s.load_publisher = Mock()
        s._last_stall_publish_ts = 0.0
        s.enable_hicache_storage = False
        s.disaggregation_mode = "prefill"
        s.is_fully_idle = lambda: False
        state = {"inflight": 46}
        s.load_inquirer = SimpleNamespace(get_loads=lambda: dict(state))
        s.dp_budget_source = SimpleNamespace(attach=lambda *args: None)
        s.on_idle()
        state["inflight"] = 0
        clock[0] += 2
        s.on_idle()
        self.assertEqual(snapshots, [{"inflight": 46}, {"inflight": 0}])


if __name__ == "__main__":
    unittest.main()
