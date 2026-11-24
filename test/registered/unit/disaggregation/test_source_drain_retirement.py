"""Retired senders stay terminal without touching a reused room (CPU only)."""

import ast
from collections import OrderedDict
import threading
import time
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

register_cpu_ci(est_time=40, suite="base-a-test-cpu")


def extract_class(path, name, names, bases, env):
    original = next(
        n
        for n in ast.parse(path.read_text(encoding="utf-8")).body
        if isinstance(n, ast.ClassDef) and n.name == name
    )
    if names is not None:
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

    def test_common_budget_caps_continuation_at_immediate_capacity(self):
        budget_file = ROOT / "python/sglang/srt/mem_cache/prefill_budget.py"
        fixed = extract_class(budget_file, "PrefillBudget", None, [], {})
        common = extract_class(budget_file, "CommonPrefillBudget", None, [], {})
        allocator = Mock(page_size=1)
        cache = Mock()
        cache.supports_mamba.return_value = False
        cache.evictable_size.return_value = 0
        local = fixed(allocator, cache)
        local.total_offset = 1024
        budget = common(local, minimum=lambda value: value)
        args = dict(extend_input_len=256, max_new_tokens=1, chunk_limit=256)
        allocator.available_size.return_value = 0
        self.assertIsNone(budget.fit_chunk(**args))
        allocator.available_size.return_value = 128
        self.assertEqual(budget.fit_chunk(**args), 128)

    def test_tp8_quarantine_retires_together_and_frees_once(self):
        self._check_collective_retirement(quarantined=True)

    def test_tp8_failed_poll_gates_healthy_load_back_until_all_senders_drain(self):
        self._check_collective_retirement(quarantined=False)

    def test_tp8_chunked_abort_and_late_result_share_one_owner(self):
        """One active writer retains buffers while the chunked reply can finish."""
        self._check_collective_retirement(quarantined=False, entry="chunked")

    def test_tp8_client_cancels_partially_sent_chunk(self):
        """Client cancellation retains the original abort envelope while ranks drain."""
        self._check_collective_retirement(quarantined=False, entry="client cancel")

    def test_tp8_late_result_retries_abort_after_quarantine(self):
        """Late results keep the RID reachable when abort fails before publication."""
        self._check_collective_retirement(quarantined=False, entry="late result")

    def test_tp8_single_rank_release_retry(self):
        """A persistent local release error permits admission from shared headroom."""
        self._check_collective_retirement(quarantined=False, retry_release=True)

    def test_tp4_cp2_single_rank_release_retry(self):
        self._check_collective_retirement(
            quarantined=False, retry_release=True, tp_size=4
        )

    def _check_collective_retirement(
        self, quarantined, entry="inflight", retry_release=False, tp_size=8
    ):
        barrier = threading.Barrier(8, timeout=15)
        via_chunked = entry in ("chunked", "client cancel")
        flags = [None] * 8
        calls = [[] for _ in range(8)]
        admitted = [False] * 8
        spare_pages = 0
        rows_on_last_rank = 1
        allow_release = False
        allow_reclaim = True
        budget_file = ROOT / "python/sglang/srt/mem_cache/prefill_budget.py"
        common_budget = extract_class(budget_file, "CommonPrefillBudget", None, [], {})
        use_common = load_methods(
            ROOT / "python/sglang/srt/managers/schedule_policy.py",
            ["use_common_capacity"],
            {"CommonPrefillBudget": common_budget},
        ).use_common_capacity

        def parallel():
            return SimpleNamespace(tp_rank=0, pp_size=1, pp_max_micro_batch_size=16)

        gates = load_methods(
            ROOT / "python/sglang/srt/mem_cache/prefill_budget.py",
            ["prepare_load_back", "check_prefill"],
            {},
            class_name="SharedSWAPrefillBudget",
        )

        scheduler_methods = load_methods(
            SCHED,
            ["get_new_batch_prefill", "get_num_allocatable_reqs", "abort_request"],
            dict(
                get_parallel=parallel,
                NextBatchPlan=SimpleNamespace,
                logger=Mock(),
                DisaggregationMode=SimpleNamespace(PREFILL="prefill"),
            ),
        )
        plan = scheduler_methods.get_new_batch_prefill

        def collective(rank, name, rids):
            calls[rank].append((name, tuple(rids)))
            barrier.wait()
            self.assertEqual([trace[-1] for trace in calls], [calls[0][-1]] * 8)
            barrier.wait()

        wire_send = Mock()
        sender_wrapper = load_methods(
            ROOT / "python/sglang/srt/managers/scheduler_components/output_sender.py",
            ["__init__", "send_output"],
            {"BaseReq": SimpleNamespace, "sock_send": wire_send},
        )
        fixtures = []
        chunks = []
        for rank in range(8):
            fixture = abort_drain_tests.TestAbortDrain()
            fixture.setUp()
            sender = make_sender()
            if via_chunked:
                socket = object() if rank == 0 else None
                fixture.s.ipc_channels.send_to_tokenizer = Mock(
                    wraps=sender_wrapper(socket)
                )
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
            if entry in ("late result", "client cancel") and rank == 7:
                # The first abort times out before marking this sender Failed.
                real_abort = sender.abort
                attempts = iter((False, False, True) if via_chunked else (False, True))

                def retry_abort():
                    if not next(attempts):
                        raise RuntimeError("abort was not published")
                    real_abort()

                sender.abort = Mock(side_effect=retry_abort)
                sender.poll = lambda: (
                    KVPoll.Transferring
                    if sender.conclude_state is None
                    else sender.conclude_state
                )
                fixture.req.external_kv_cleanup_started_at = time.monotonic() - 2.0
            else:
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

            def capacity_reduce(tensor, *, op, group, rank=rank):
                # Reduce within each attention TP group, then across its CP peers.
                collective(rank, "capacity_" + group, ())
                flags[rank] = tensor.item()
                barrier.wait()
                start = rank // tp_size * tp_size
                peers = (
                    range(start, start + tp_size)
                    if group == "tp"
                    else range(rank % tp_size, 8, tp_size)
                )
                value = min(flags[i] for i in peers)
                barrier.wait()
                tensor.item = lambda: value

            fixture.s.attn_tp_cpu_group = "tp"
            fixture.s.attn_cp_cpu_group = "cp"
            fixture.s.req_to_token_pool = SimpleNamespace(
                available_size=lambda rank=rank: rows_on_last_rank if rank == 7 else 2
            )
            fixture.s.beam_coordinator = Mock()
            fixture.s.beam_coordinator.pending_member_rows.return_value = 0
            fixture.s.running_batch = object()
            fixture.s.disaggregation_mode = "prefill"
            fixture.s.waiting_queue = []
            fixture.s.disagg_prefill_bootstrap_queue = SimpleNamespace(queue=[])
            fixture.s.dllm_config = fixture.s.mm_receiver = None
            fixture.s.grammar_manager = Mock()
            fixture.s.collect_inflight_reqs = lambda: []
            fixture.env.update(
                get_parallel=parallel,
                torch=SimpleNamespace(
                    float64=float,
                    tensor=lambda values, **kw: SimpleNamespace(item=lambda: values[0]),
                    distributed=SimpleNamespace(
                        ReduceOp=SimpleNamespace(MIN="min"),
                        get_world_size=lambda group: (
                            tp_size if group == "tp" else 8 // tp_size
                        ),
                    ),
                ),
                all_reduce_with_timeout=capacity_reduce,
                FINISH_LENGTH=lambda length: SimpleNamespace(length=length),
                FAKE_BOOTSTRAP_HOST="fake",
                CacheRequestOutcome=SimpleNamespace(SUCCESS="success", ABORT="abort"),
                all_reduce_min_flags=reduce,
                poll_and_all_reduce_attn_cp_tp_group=poll,
                _QuarantinedKVSender=lambda: None,
                maybe_release_pd_hidden_rows_on_hidden_done=Mock(),
            )
            queue_cls = load_methods(
                PREFILL,
                [
                    "_min_prefill_capacity",
                    "prepare_disagg_prefill_transfer",
                    "process_disagg_prefill_inflight_queue",
                    "_retire_aborted_prefill_result",
                ],
                fixture.env,
            )
            fixture.s.process_disagg_prefill_inflight_queue = (
                queue_cls.process_disagg_prefill_inflight_queue.__get__(fixture.s)
            )
            fixture.s._retire_aborted_prefill_result = (
                queue_cls._retire_aborted_prefill_result.__get__(fixture.s)
            )
            fixture.s._min_prefill_capacity = queue_cls._min_prefill_capacity.__get__(
                fixture.s
            )
            fixture.s.prefill_delayer = None
            if entry != "inflight":
                req = fixture.req
                fixture.s.disagg_prefill_inflight_queue.clear()
                fixture.s.prefetch_disagg_page_indices = Mock()
                fixture.s.maybe_send_cached_prefix_chunk = Mock()
                queue_cls.prepare_disagg_prefill_transfer(fixture.s, req)
                req.finished = lambda req=req: req.finished_reason is not None
                req.cache_request_handle = None
                if via_chunked:
                    fixture.s.chunked_req = fixture.s._pending_chunked_abort_req = req
                    fixture.s.result_queue = [(SimpleNamespace(reqs=[req]), object())]
                    if entry == "chunked":
                        fixture.finalize(
                            forward_drained=False,
                            sender_terminal=False,
                            emit_response=False,
                        )
                    else:
                        req.to_finish = req.discard_output_reason = None
                        fixture.env["_make_abort_req"] = Mock(
                            return_value=SimpleNamespace(
                                rid=req.rid, weight_versions=["v1"]
                            )
                        )
                    fixture.s.process_pending_chunked_abort()
                    fixture.s.result_queue.clear()
                self.assertFalse(fixture.s._retire_aborted_prefill_result(req))
                self.assertEqual(fixture.s.disagg_prefill_inflight_queue, [req])
            if retry_release and rank == 7:

                def try_release(req, *args):
                    if req.rid == "r" and not allow_release:
                        raise RuntimeError("metadata remains unavailable")

                fixture.env["maybe_release_metadata_buffer"].side_effect = try_release
            fixtures.append(fixture)
            chunks.append(chunk)

        def round_trip():
            errors = []

            def run(rank, fixture):
                try:

                    def try_admit(*, running_batch, **kwargs):
                        if admitted[rank]:
                            return None, running_batch
                        # The healthy host hit needs the one page held by the failed RID.
                        # Only the physical allocator and communication are simulated.
                        if not scheduler_methods.get_num_allocatable_reqs(fixture.s, 0):
                            return None, running_batch

                        free = 256 * (
                            spare_pages + fixture.env["release_kv_cache"].call_count
                        )

                        def reclaim(cache, full, swa):
                            if rank == 7 and not allow_reclaim:
                                return "pinned page"
                            return None if full <= free else "insufficient pages"

                        budget = SimpleNamespace(
                            allocator=SimpleNamespace(reclaim_for_prealloc=reclaim),
                            tree_cache=fixture.s.tree_cache,
                            total_offset=0,
                            swa_offset=0,
                            ceil_paged_tokens=int,
                            swa_tokens=lambda *a, **kw: 0,
                        )
                        budget.load_back_needs_full_tokens = True
                        budget._fits = lambda full, swa: full <= free
                        budget.check_prefill = gates.check_prefill.__get__(budget)
                        budget.prepare_load_back = gates.prepare_load_back.__get__(
                            budget
                        )
                        adder = SimpleNamespace(
                            memory_budget=budget, rem_mamba_slots=None
                        )
                        if minimum := getattr(fixture.s, "_prefill_capacity_min", None):
                            use_common(adder, minimum)
                        fits, cap = adder.memory_budget.check_prefill(
                            extend_input_len=256,
                            total_tokens=256,
                            max_new_tokens=0,
                            input_tokens=256,
                            swa_host_hit_length=0,
                            chunk_limit=None,
                        )
                        self.assertIsNone(cap)
                        if fits and adder.memory_budget.prepare_load_back(
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
                        return None, running_batch

                    fixture.s._get_new_batch_prefill_raw = try_admit
                    plan(fixture.s, None)
                    if entry == "client cancel":
                        for _ in range(2):
                            scheduler_methods.abort_request(
                                fixture.s,
                                SimpleNamespace(
                                    abort_all=False, rid="r", abort_message=None
                                ),
                            )
                    if via_chunked:
                        fixture.s.process_pending_chunked_abort()
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
        if via_chunked:
            wire_send.assert_called_once()
            self.assertEqual(wire_send.call_args.args[1].rid, "r")
            for f in fixtures:
                f.s.ipc_channels.send_to_tokenizer.send_output.assert_called_once()
                self.assertFalse(f.req.external_kv_cleanup_done)
        if entry == "client cancel":
            self.assertTrue(fixtures[-1].req.external_kv_cleanup_quarantined)
            self.assertEqual(fixtures[-1].req.disagg_kv_sender.abort.call_count, 2)
            round_trip()
            self.assertEqual(fixtures[-1].req.disagg_kv_sender.abort.call_count, 3)
        if entry == "late result":
            self.assertTrue(fixtures[-1].req.external_kv_cleanup_quarantined)
            self.assertEqual(fixtures[-1].req.disagg_kv_sender.abort.call_count, 2)
        for fixture in fixtures[:-1]:
            self.assertTrue(fixture.req.external_kv_source_drained)
        if not quarantined and not via_chunked:
            for fixture in fixtures:
                fixture.s.output_streamer.stream_output.assert_called_once()
                self.assertFalse(fixture.s._retire_aborted_prefill_result(fixture.req))
                fixture.env["release_kv_cache"].assert_not_called()
                fixture.env["maybe_release_metadata_buffer"].assert_not_called()
        self.assertFalse(getattr(fixtures[-1].req, "external_kv_source_drained", False))

        fixtures[-1].req.disagg_kv_sender.kv_mgr._source_drain.complete(chunks[-1])
        round_trip()  # Last sender becomes locally drained.
        round_trip()  # The first agreed release runs on all ranks.
        if retry_release:
            self.assertEqual(
                [f.env["release_kv_cache"].call_count for f in fixtures], [1] * 7 + [0]
            )
            for _ in range(3):
                round_trip()
            self.assertFalse(any(admitted))
            # A spare page now fits everywhere but the weakest rank has no row.
            spare_pages, rows_on_last_rank = 1, 0
            round_trip()
            self.assertFalse(any(admitted))
            rows_on_last_rank = 1
            allow_reclaim = False
            round_trip()
            self.assertFalse(any(admitted))
            allow_reclaim = True
            round_trip()
            self.assertTrue(all(admitted))
            self.assertFalse(fixtures[-1].req.external_kv_cleanup_done)
            for f in fixtures:
                self.assertEqual(f.s.disagg_prefill_inflight_queue, [f.req])
                self.assertEqual(
                    f.s.output_streamer.stream_output.call_args.args[0][0].rid,
                    "healthy",
                )
            allow_release = True
        for _ in range(4):
            round_trip()
        self.assertTrue(all(admitted))
        if via_chunked:
            wire_send.assert_called_once()
        self.assertEqual(calls[0].count(("load_back", ("healthy",))), 1)
        for rank, fixture in enumerate(fixtures):
            self.assertEqual(fixture.s.disagg_prefill_inflight_queue, [])
            self.assertEqual(fixture.s._external_kv_cleanup_quarantine, {})
            self.assertTrue(fixture.req.external_kv_cleanup_done)
            for op in ("release_kv_cache", "maybe_release_metadata_buffer"):
                retry = (
                    retry_release
                    and rank == 7
                    and op == "maybe_release_metadata_buffer"
                )
                rids = [c.args[0].rid for c in fixture.env[op].call_args_list]
                if retry:
                    self.assertGreaterEqual(rids.count("r"), 7)
                    self.assertEqual(rids, ["r"] * (len(rids) - 2) + ["healthy", "r"])
                elif retry_release and rank == 7:
                    self.assertEqual(rids, ["healthy", "r"])
                else:
                    self.assertEqual(rids, ["r", "healthy"])
            outputs = [
                req
                for c in fixture.s.output_streamer.stream_output.call_args_list
                for req in c.args[0]
            ]
            aborts = fixture.s.ipc_channels.send_to_tokenizer.send_output
            if via_chunked:
                aborts.assert_called_once()
                if entry == "client cancel":
                    factory = fixture.env["_make_abort_req"]
                    factory.assert_called_once_with(fixture.req, finished_reason=None)
                    self.assertIs(aborts.call_args.args[0], factory.return_value)
                self.assertEqual(aborts.call_args.args[0].rid, "r")
                self.assertEqual([req.rid for req in outputs], ["healthy"])
                self.assertFalse(fixture.s._external_kv_chunked_abort_reqs)
            else:
                aborts.assert_not_called()
                self.assertEqual([req.rid for req in outputs], ["r", "healthy"])
            self.assertIsNotNone(outputs[-1].finished_reason)

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
