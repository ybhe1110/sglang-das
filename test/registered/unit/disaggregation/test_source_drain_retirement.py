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
    mgr.deferred_bootstrap = None
    mgr.req_to_decode_prefix_len = {}
    mgr.req_to_pd_hidden_meta = {}
    mgr.transfer_infos = {}
    mgr._source_drain = SourceDrainTracker()
    mgr._abort_room_bindings = {}
    mgr._source_room_pair_keys = {}
    mgr._retired_abort_rooms = OrderedDict()
    mgr._abort_ack_lock = threading.RLock()
    mgr._deferred_ack_targets = {}
    mgr.request_status = {7: KVPoll.Transferring}
    mgr.record_failure = Mock()
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
        self._check_collective_retirement(quarantined=True)

    def test_tp8_failed_poll_gates_healthy_load_back_until_all_senders_drain(self):
        self._check_collective_retirement(quarantined=False)

    def _check_collective_retirement(self, quarantined):
        barrier = threading.Barrier(8, timeout=15)
        flags = [None] * 8
        calls = [[] for _ in range(8)]
        admitted = [False] * 8
        gate = load_methods(
            ROOT / "python/sglang/srt/mem_cache/prefill_budget.py",
            ["prepare_load_back"],
            {},
            class_name="SharedSWAPrefillBudget",
        ).prepare_load_back

        def collective(rank, name, rids):
            calls[rank].append((name, tuple(rids)))
            barrier.wait()
            self.assertEqual([trace[-1] for trace in calls], [calls[0][-1]] * 8)
            barrier.wait()

        fixtures = []
        chunks = []
        for rank in range(8):
            fixture = abort_drain_tests.TestAbortDrain()
            fixture.setUp()
            sender = make_sender()
            fixture.req.disagg_kv_sender = sender
            fixture.req.external_kv_abort_response_via_chunked = False
            fixture.s._pending_chunked_abort_req = None
            fixture.s.tree_cache = Mock()
            fixture.s.chunked_req = None
            fixture.s.enable_unified_cache_external_linker = False
            fixture.s.attn_cp_cpu_group = object()
            fixture.s.attn_tp_cpu_group = object()
            fixture.s.disagg_prefill_inflight_queue = [fixture.req]
            chunk = SimpleNamespace(room=7)
            if rank == 7:
                sender.kv_mgr._source_drain.publish(chunk, Mock(), lambda: True)
                sender.kv_mgr._source_drain.begin(chunk)
            sender.abort()
            sender.clear()
            if quarantined:
                fixture.s._quarantine_external_kv_request(
                    fixture.req, abort_message="cancelled", emit_response=False
                )
            else:
                fixture.s._external_kv_abort_rids.clear()
                fixture.req.pending_bootstrap = False
                fixture.s.ps = SimpleNamespace(tp_rank=rank)
                fixture.s.metrics_reporter = SimpleNamespace(enable_metrics=False)
                sender.failure_exception = Mock(side_effect=RuntimeError("aborted"))

            def reduce(values, *args, rank=rank, **kwargs):
                collective(rank, "cleanup", kwargs["request_ids"])
                flags[rank] = values
                barrier.wait()
                self.assertEqual({len(values) for values in flags}, {len(flags[0])})
                result = [min(pair) for pair in zip(*flags)]
                barrier.wait()
                return result

            def poll(senders, *groups, rank=rank, scheduler=fixture.s):
                rids = [req.rid for req in scheduler.disagg_prefill_inflight_queue]
                collective(rank, "poll", rids)
                return [
                    KVPoll.Success if rid == "healthy" else KVPoll.Failed
                    for rid in rids
                ]

            fixture.env.update(
                FINISH_LENGTH=lambda length: SimpleNamespace(length=length),
                FAKE_BOOTSTRAP_HOST="fake",
                CacheRequestOutcome=SimpleNamespace(SUCCESS="success"),
                all_reduce_min_flags=reduce,
                poll_and_all_reduce_attn_cp_tp_group=poll,
                _QuarantinedKVSender=lambda: None,
                maybe_release_pd_hidden_rows_on_hidden_done=Mock(),
            )
            queue_cls = load_methods(
                PREFILL,
                ["process_disagg_prefill_inflight_queue", "_retire_aborted_prefill_result"],
                fixture.env,
            )
            fixture.s.process_disagg_prefill_inflight_queue = (
                queue_cls.process_disagg_prefill_inflight_queue.__get__(fixture.s)
            )
            fixture.s._retire_aborted_prefill_result = (
                queue_cls._retire_aborted_prefill_result.__get__(fixture.s)
            )
            fixtures.append(fixture)
            chunks.append(chunk)

        def round_trip():
            errors = []

            def run(rank, fixture):
                try:
                    if not admitted[rank]:
                        # The healthy host hit needs the one page held by the failed RID.
                        # Only the physical allocator and communication are simulated.
                        def reclaim(cache, full, swa):
                            free = 256 * fixture.env["release_kv_cache"].call_count
                            return None if full <= free else "insufficient pages"

                        budget = SimpleNamespace(
                            allocator=SimpleNamespace(reclaim_for_prealloc=reclaim),
                            tree_cache=fixture.s.tree_cache,
                            total_offset=0,
                            swa_offset=0,
                            ceil_paged_tokens=int,
                            swa_tokens=lambda *a, **kw: 0,
                        )
                        if gate(
                            budget,
                            full_tokens=256,
                            extend_input_len=1,
                            max_new_tokens=1,
                            swa_host_hit_length=0,
                            chunk_limit=None,
                        ):
                            collective(rank, "load_back", ["healthy"])
                            admitted[rank] = True
                            fixture.s.disagg_prefill_inflight_queue.append(
                                SimpleNamespace(
                                    rid="healthy",
                                    finished_reason=None,
                                    bootstrap_host="fake",
                                    pending_bootstrap=False,
                                    return_logprob=False,
                                    return_routed_experts=False,
                                    external_kv_response_sent=False,
                                    external_kv_metadata_released=False,
                                    time_stats=Mock(),
                                    disagg_kv_sender=Mock(),
                                    cache_request_handle=None,
                                )
                            )
                    fixture.s.process_disagg_prefill_inflight_queue()
                except BaseException as error:
                    errors.append(error)
                    barrier.abort()

            threads = [
                threading.Thread(target=run, args=(rank, f))
                for rank, f in enumerate(fixtures)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join(20)
                self.assertFalse(thread.is_alive())
            if errors:
                raise errors[0]

        # Keep one writer active across two complete admission passes.
        round_trip()
        round_trip()
        for fixture in fixtures:
            self.assertEqual(fixture.s.disagg_prefill_inflight_queue, [fixture.req])
            fixture.env["release_kv_cache"].assert_not_called()
            fixture.env["maybe_release_metadata_buffer"].assert_not_called()
        self.assertFalse(any(admitted))
        for fixture in fixtures[:-1]:
            self.assertTrue(fixture.req.external_kv_source_drained)
        if not quarantined:
            for fixture in fixtures:
                fixture.s.output_streamer.stream_output.assert_called_once()
                self.assertFalse(fixture.s._retire_aborted_prefill_result(fixture.req))
                fixture.env["release_kv_cache"].assert_not_called()
                fixture.env["maybe_release_metadata_buffer"].assert_not_called()
        self.assertFalse(getattr(fixtures[-1].req, "external_kv_source_drained", False))

        fixtures[-1].req.disagg_kv_sender.kv_mgr._source_drain.complete(chunks[-1])
        for _ in range(5):
            round_trip()
        self.assertTrue(all(admitted))
        self.assertEqual(calls[0].count(("load_back", ("healthy",))), 1)
        for fixture in fixtures:
            self.assertEqual(fixture.s.disagg_prefill_inflight_queue, [])
            self.assertEqual(fixture.s._external_kv_cleanup_quarantine, {})
            self.assertTrue(fixture.req.external_kv_cleanup_done)
            for op in ("release_kv_cache", "maybe_release_metadata_buffer"):
                self.assertEqual(
                    [c.args[0].rid for c in fixture.env[op].call_args_list],
                    ["r", "healthy"],
                )
            outputs = [
                req
                for c in fixture.s.output_streamer.stream_output.call_args_list
                for req in c.args[0]
            ]
            self.assertEqual([req.rid for req in outputs], ["r", "healthy"])
            self.assertIsNotNone(outputs[1].finished_reason)

    def test_load_snapshot_updates_without_forward_or_full_idle(self):
        clock = [1.0]
        env = dict(
            time=SimpleNamespace(monotonic=lambda: clock[0], monotonic_ns=lambda: 0),
            LOAD_STALL_REFRESH_S=0.1,
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
        s.enable_hicache_storage = s.enable_lmcache = False
        s.disaggregation_mode = "null"
        s.is_fully_idle = lambda: False
        state = {"inflight": 46}
        s.load_inquirer = SimpleNamespace(get_loads=lambda: dict(state))
        s.dp_budget_source = SimpleNamespace(attach=lambda *args: None)
        s.on_idle()
        state["inflight"] = 0
        clock[0] += 1.0
        s.on_idle()
        self.assertEqual(snapshots, [{"inflight": 46}, {"inflight": 0}])


if __name__ == "__main__":
    unittest.main()
