"""CPU fault tests for optional Mooncake master availability and L1 fallback.

Loads production code without importing the GPU runtime; no Mooncake service is
needed. Native data transfers are deliberately never timed out by this guard.
"""

import ast
import importlib.util
import logging
import os
import sys
import threading
import time
import unittest
from concurrent.futures import Future
from pathlib import Path
from queue import Empty, Queue
from types import SimpleNamespace
from unittest.mock import Mock, patch

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "python/sglang/srt/mem_cache/storage/mooncake_store"


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


ci = load_module("fallback_ci", ROOT / "python/sglang/test/ci/ci_register.py")
register_cpu_ci = ci.register_cpu_ci
register_cpu_ci(est_time=5, suite="base-a-test-cpu")
availability = load_module("master_availability_test", BASE / "master_availability.py")
MasterAvailability = availability.MasterAvailability
StoreUnavailable = availability.StoreUnavailable
checked_exists = availability.checked_exists


def eventually(predicate):
    deadline = time.monotonic() + 2
    while time.monotonic() < deadline:
        if predicate():
            return
        threading.Event().wait(0.002)
    raise AssertionError("condition did not become true")


class Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class TestAvailability(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.factory = Mock(return_value=object())
        self.probe = Mock(return_value=[0])
        self.gate = MasterAvailability(
            self.factory, self.probe, timeout_s=0.03, retry_s=5, clock=self.clock
        )
        self.addCleanup(self.gate.close)

    def test_startup_failure_can_recover_without_restarting_service(self):
        self.factory.side_effect = [ConnectionError("master down"), object()]
        self.assertFalse(self.gate.initialize())
        self.assertFalse(self.gate.ready())
        self.assertEqual(self.factory.call_count, 1)
        self.clock.now += 5
        self.gate.ready()
        eventually(self.gate.ready)
        self.assertEqual(self.factory.call_count, 2)
        self.assertEqual(self.gate.recoveries, 1)

    def test_missing_key_is_healthy_but_error_code_or_short_reply_is_not(self):
        self.assertTrue(self.gate.initialize())
        self.assertEqual(self.gate.query(lambda: checked_exists([0], 1)), [0])
        for bad in ([-1], [], None, -1):
            with self.subTest(reply=bad):
                with self.assertRaises(StoreUnavailable):
                    checked_exists(bad, 1)
        with self.assertRaises(StoreUnavailable):
            self.gate.query(lambda: checked_exists([-1], 1))
        self.assertFalse(self.gate.ready())

    def test_hung_metadata_does_not_spawn_more_workers_or_accept_late_success(self):
        self.assertTrue(self.gate.initialize())
        release = threading.Event()
        started = threading.Event()
        calls = []

        def stuck():
            calls.append(1)
            started.set()
            release.wait(2)
            return [1]

        try:
            with self.assertRaises(StoreUnavailable):
                self.gate.query(stuck)
            self.assertTrue(started.is_set())
            for _ in range(20):
                self.clock.now += 5
                self.assertFalse(self.gate.ready())
                with self.assertRaises(StoreUnavailable):
                    self.gate.query(stuck)
            self.assertEqual(calls, [1])
            self.probe.assert_not_called()
        finally:
            release.set()
        eventually(lambda: self.gate._job is None)
        self.assertFalse(self.gate._healthy)
        self.clock.now += 5
        eventually(self.gate.ready)
        self.probe.assert_called_once()

    def test_newer_failure_invalidates_a_query_success(self):
        self.assertTrue(self.gate.initialize())

        def query():
            self.gate.failed("write-back failed concurrently")
            return [1]

        with self.assertRaises(StoreUnavailable):
            self.gate.query(query)
        self.assertFalse(self.gate.ready())

    def test_half_open_failure_restarts_cooldown(self):
        self.assertTrue(self.gate.initialize())
        self.gate.failed("master down")
        self.probe.side_effect = [ConnectionError("still down"), [0]]
        self.clock.now = 5
        self.gate.ready()
        eventually(lambda: self.gate._job is None)
        self.assertFalse(self.gate.ready())
        self.assertEqual(self.probe.call_count, 1)
        self.clock.now = 10
        eventually(self.gate.ready)
        self.assertEqual(self.probe.call_count, 2)

    def test_startup_timeout_is_bounded_and_keeps_one_initialization(self):
        release = threading.Event()
        self.factory.side_effect = lambda: (release.wait(2), object())[1]
        try:
            self.assertFalse(self.gate.initialize())
            self.clock.now = 100
            for _ in range(20):
                self.assertFalse(self.gate.ready())
            self.factory.assert_called_once()
        finally:
            release.set()
        eventually(lambda: self.gate._job is None)
        eventually(self.gate.ready)

    def test_close_does_not_wait_for_stuck_metadata(self):
        self.assertTrue(self.gate.initialize())
        release = threading.Event()
        try:
            with self.assertRaises(StoreUnavailable):
                self.gate.query(lambda: release.wait(2))
            self.gate.close()
            self.clock.now = 100
            self.assertFalse(self.gate.ready())
        finally:
            release.set()
        eventually(lambda: self.gate._job is None)
        self.assertFalse(self.gate.ready())

    def test_healthy_concurrent_queries_share_one_persistent_worker(self):
        self.gate.timeout_s = 1
        self.assertTrue(self.gate.initialize())
        worker = self.gate._worker
        started, release = threading.Event(), threading.Event()
        results, errors, threads = [], [], []

        def first():
            started.set()
            release.wait(2)
            return [1]

        def call(operation):
            try:
                results.append(self.gate.query(operation))
            except Exception as error:
                errors.append(error)

        try:
            threads.append(threading.Thread(target=call, args=(first,)))
            threads[0].start()
            self.assertTrue(started.wait(1))
            threads.append(threading.Thread(target=call, args=(lambda: [0],)))
            threads[1].start()
        finally:
            release.set()
            for thread in threads:
                thread.join(2)
        self.assertFalse(errors)
        self.assertCountEqual(results, [[0], [1]])
        self.assertIs(self.gate._worker, worker)
        self.assertTrue(self.gate.ready())

    def test_reject_non_finite_or_nonpositive_configuration(self):
        for value in (0, -1, float("inf"), float("nan")):
            with self.assertRaises(ValueError):
                MasterAvailability(self.factory, self.probe, timeout_s=value)


def linker_class(clock, pool_group, torch):
    tree = ast.parse((BASE / "mooncake_direct_linker.py").read_text(encoding="utf-8"))
    nodes = []
    for node in tree.body:
        if isinstance(node, (ast.FunctionDef, ast.ClassDef)):
            if isinstance(node, ast.ClassDef) and node.name == "MooncakeDirectLinker":
                node.bases = []
            nodes.append(node)
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    ast.fix_missing_locations(module)
    env = dict(
        logging=logging,
        os=os,
        threading=threading,
        time=time,
        Future=Future,
        Empty=Empty,
        Queue=Queue,
        torch=torch,
        logger=logging.getLogger("linker-fallback-test"),
        device_module=SimpleNamespace(Event=Mock),
        PoolName=SimpleNamespace(KV="kv", MAMBA="mamba"),
        resolve_hybrid_device_pool_group=lambda **kwargs: pool_group,
        HybridCacheController=SimpleNamespace(
            parse_storage_backend_extra_config=lambda _: ({},)
        ),
        HiCacheStorageConfig=lambda **kwargs: SimpleNamespace(**kwargs),
        get_memory=lambda: SimpleNamespace(hicache_storage_backend_extra_config=None),
        get_model=lambda: SimpleNamespace(model_path="model"),
        freeze_gc=lambda *args: None,
        MasterAvailability=lambda *args, **kwargs: MasterAvailability(
            *args, clock=clock, **kwargs
        ),
        StoreUnavailable=StoreUnavailable,
        checked_exists=checked_exists,
        arm_load_failure_injection=lambda *args: None,
    )
    exec(compile(module, "production linker", "exec"), env)
    return env["MooncakeDirectLinker"]


class Verdict:
    def __init__(self, values):
        self.value = values[0]

    def item(self):
        return self.value


class TestLinkerFallback(unittest.TestCase):
    def setUp(self):
        self.clock = Clock()
        self.raw = SimpleNamespace(
            batch_is_exist=Mock(return_value=[1]), register_buffer=Mock(return_value=0)
        )
        buffer = SimpleNamespace(
            untyped_storage=lambda: SimpleNamespace(
                data_ptr=lambda: 10, nbytes=lambda: 16
            )
        )
        self.pools = {"kv": SimpleNamespace(get_hybrid_pool_buffer=lambda: [buffer])}
        self.group = SimpleNamespace(
            entry_map=self.pools,
            num_layers=1,
            rank_replicated=False,
            storage_layout_tag="",
            resolve_transfers=lambda transfers, **kwargs: transfers,
        )
        self.torch = SimpleNamespace(
            distributed=SimpleNamespace(
                is_available=lambda: False,
                is_initialized=lambda: False,
                get_world_size=lambda **kwargs: 2,
                all_reduce=Mock(),
                ReduceOp=SimpleNamespace(MIN="min"),
            ),
            tensor=lambda values, **kwargs: Verdict(values),
            int="int",
        )
        self.cls = linker_class(self.clock, self.group, self.torch)
        self.store = SimpleNamespace(
            store=self.raw,
            _batch_exist=self.raw.batch_is_exist,
            _get_hybrid_page_component_keys=lambda keys, transfer: (keys, 1),
            _tag_keys=lambda keys: keys,
            close=Mock(),
        )

        def exists(keys, transfers, **kwargs):
            values = self.store._batch_exist(keys)
            return SimpleNamespace(
                restorable_prefix_pages=(
                    [len(keys)] if all(v == 1 for v in values) else []
                )
            )

        self.store.batch_exists_v2 = Mock(side_effect=exists)
        self.store.batch_set_v2 = Mock(return_value={"kv": [True]})
        self.args = SimpleNamespace(
            tp_size=1,
            mooncake_page_wise_load_threshold=1,
            mooncake_enable_page_wise_load=False,
        )
        self.params = SimpleNamespace(
            page_size=1,
            token_to_kv_pool_allocator=SimpleNamespace(get_kvcache=lambda: object()),
            attn_cp_cache_group=None,
            attn_tp_cache_group=None,
            tp_cache_group=None,
            pp_rank=0,
            pp_size=1,
            attn_cp_rank=0,
            attn_cp_size=1,
            enable_metrics=False,
        )
        self.transfer = SimpleNamespace(name="kv", keys=["key"])
        self.env = patch.dict(
            os.environ,
            {
                "SGLANG_MOONCAKE_L1_FALLBACK": "1",
                "SGLANG_MOONCAKE_MASTER_TIMEOUT_S": "0.05",
                "SGLANG_MOONCAKE_MASTER_RETRY_S": "5",
                "SGLANG_MOONCAKE_READ_PLAN": "0",
                "SGLANG_MOONCAKE_READ_PLAN_REUSE_RANGES": "0",
            },
        )
        self.env.start()
        self.addCleanup(self.env.stop)

    def make_linker(self):
        linker = self.cls(self.args, self.params, components=(), storage=self.store)
        self.addCleanup(linker.close)
        return linker

    def test_lookup_exception_is_miss_and_recovery_restores_hits(self):
        linker = self.make_linker()
        self.assertEqual(linker.lookup("r", [self.transfer]), [1])
        self.raw.batch_is_exist.side_effect = ConnectionError("master down")
        self.assertEqual(linker.lookup("r", [self.transfer]), [])
        count = self.raw.batch_is_exist.call_count
        for _ in range(10):
            self.assertEqual(linker.lookup("r", [self.transfer]), [])
        self.assertEqual(self.raw.batch_is_exist.call_count, count)
        self.raw.batch_is_exist.side_effect = None
        self.clock.now = 5
        eventually(linker._storage_ready)
        self.assertEqual(linker.lookup("r", [self.transfer]), [1])

    def test_lookup_timeout_returns_miss_without_accepting_late_remote_hit(self):
        linker = self.make_linker()
        release = threading.Event()
        self.raw.batch_is_exist.side_effect = lambda keys: (release.wait(2), [1])[1]
        try:
            self.assertEqual(linker.lookup("r", [self.transfer]), [])
            self.assertFalse(linker._storage_ready())
            self.assertEqual(linker.lookup("next-request", [self.transfer]), [])
            self.raw.batch_is_exist.assert_called_once()
        finally:
            release.set()
        eventually(lambda: linker._availability._job is None)
        self.assertFalse(linker._storage_ready())

    def test_write_back_exception_opens_circuit_without_raising_to_scheduler(self):
        linker = self.make_linker()
        self.store.batch_set_v2.side_effect = ConnectionError("master down on put")
        self.assertTrue(linker.offload([self.transfer]))
        eventually(lambda: linker.num_completed_offloads() == 1)
        self.assertFalse(linker.pop_completed_offload())
        self.assertFalse(linker._storage_ready())
        self.assertEqual(linker.lookup("next-request", [self.transfer]), [])
        self.raw.batch_is_exist.assert_not_called()

    def test_startup_registration_failure_uses_l1_then_recovers(self):
        self.raw.register_buffer.side_effect = ConnectionError("master down at startup")
        linker = self.make_linker()
        self.assertIsNone(linker.storage)
        self.assertEqual(linker.lookup("r", [self.transfer]), [])
        self.raw.register_buffer.side_effect = None
        self.clock.now = 5
        eventually(linker._storage_ready)
        self.assertEqual(linker.lookup("r", [self.transfer]), [1])

    def test_revalidation_exception_is_false_and_still_participates_in_reduction(self):
        self.params.attn_tp_cache_group = object()
        linker = self.make_linker()
        self.raw.batch_is_exist.side_effect = ConnectionError("master down")
        self.assertFalse(linker.revalidate_load([self.transfer]))
        self.torch.distributed.all_reduce.assert_called_once()
        self.assertEqual(self.torch.distributed.all_reduce.call_args.args[0].item(), 0)

    def test_cp_and_tp_both_reduce_failure_even_without_layer_split_layout(self):
        self.params.attn_cp_cache_group = object()
        self.params.attn_tp_cache_group = object()
        linker = self.make_linker()
        linker._availability.failed("master down")
        self.assertFalse(linker.revalidate_load([self.transfer]))
        self.assertEqual(
            [
                call.kwargs["group"]
                for call in self.torch.distributed.all_reduce.call_args_list
            ],
            [self.params.attn_cp_cache_group, self.params.attn_tp_cache_group],
        )

    def test_read_plan_transfer_failure_opens_circuit_and_reports_failed_request(self):
        os.environ["SGLANG_MOONCAKE_READ_PLAN"] = "1"
        plan = Mock()
        plan.run.side_effect = ConnectionError("master failed during transfer")
        plan.wait.side_effect = plan.run.side_effect
        self.raw.create_read_plan = Mock(return_value=plan)
        linker = self.make_linker()
        linker._prepare_read_plan_layouts = Mock(return_value=[])
        self.assertTrue(linker.load("r", [self.transfer]))
        index = linker.start_layer_wise_loading()
        eventually(lambda: linker.num_completed_loads() == 1)
        self.assertEqual(linker.pop_completed_load(), (["r"], False))
        self.assertFalse(linker._storage_ready())
        linker.layer_done_counter.set_consumer(index)
        linker.layer_done_counter.wait_until(0)
        self.assertEqual(linker.lookup("next-request", [self.transfer]), [])
        self.raw.batch_is_exist.assert_not_called()

    def test_remote_rank_failure_disables_local_load(self):
        self.params.attn_tp_cache_group = object()
        linker = self.make_linker()
        self.torch.distributed.all_reduce.side_effect = lambda value, **kwargs: setattr(
            value, "value", 0
        )
        self.assertFalse(linker.revalidate_load([self.transfer]))

    def test_pp_preserves_agreed_shape_then_reports_failed_load(self):
        self.params.pp_size = 2
        linker = self.make_linker()
        linker._availability.failed("master down after PP0 admission")
        self.assertTrue(linker.revalidate_load([self.transfer]))
        self.assertTrue(linker.load("r", [self.transfer]))
        index = linker.start_layer_wise_loading()
        eventually(lambda: linker.num_completed_loads() == 1)
        self.assertEqual(linker.pop_completed_load(), (["r"], False))
        linker.layer_done_counter.set_consumer(index)
        linker.layer_done_counter.wait_until(0)
        self.store.batch_set_v2.assert_not_called()

    def test_degraded_offloads_keep_fifo_failure_completions(self):
        linker = self.make_linker()
        linker._availability.failed("master down")
        self.assertTrue(linker.offload([self.transfer]))
        self.assertTrue(linker.offload([self.transfer]))
        eventually(lambda: linker.num_completed_offloads() == 2)
        self.assertEqual(
            [linker.pop_completed_offload(), linker.pop_completed_offload()],
            [False, False],
        )
        self.store.batch_set_v2.assert_not_called()

    def test_inflight_write_is_not_released_early_on_master_failure(self):
        linker = self.make_linker()
        started, release = threading.Event(), threading.Event()

        def put(transfers):
            started.set()
            release.wait(2)
            return {"kv": [False]}

        self.store.batch_set_v2.side_effect = put
        try:
            linker.offload([self.transfer])
            self.assertTrue(started.wait(1))
            linker._availability.failed("master failed during write")
            self.assertEqual(linker.num_completed_offloads(), 0)
            self.assertEqual(linker.lookup("r", [self.transfer]), [])
        finally:
            release.set()
        eventually(lambda: linker.num_completed_offloads() == 1)
        self.assertFalse(linker.pop_completed_offload())

    def make_wrapper(self, linker, device_hit_len):
        path = (
            ROOT / "python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py"
        )
        tree = ast.parse(path.read_text(encoding="utf-8"))
        cls = next(
            n
            for n in tree.body
            if isinstance(n, ast.ClassDef) and n.name == "UnifiedCacheLinkerWrapper"
        )
        cls.bases = []
        cls.body = [
            n
            for n in cls.body
            if isinstance(n, ast.FunctionDef) and n.name in ("match", "load_back")
        ]
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
        env = dict(
            LinkerTransferPhase=SimpleNamespace(LOOKUP="lookup", LOAD="load"),
            ExternalLinkerLoadPhase=SimpleNamespace(ABORT="abort"),
        )
        exec(compile(module, str(path), "exec"), env)
        wrapper = env[cls.name]()
        wrapper.cache_linker = linker
        wrapper.hit_markers = {}
        component = SimpleNamespace(
            build_external_linker_transfer=Mock(return_value=self.transfer)
        )
        wrapper.cache = SimpleNamespace(
            page_size=1,
            pp_size=1,
            _components_tuple=(component,),
            tree_core=SimpleNamespace(
                empty_match_result=SimpleNamespace(device_indices=[])
            ),
        )
        wrapper._tail_hashes = Mock(return_value=["key"])
        wrapper._sync_restorable_prefix = Mock(return_value=0)
        wrapper._update_load = Mock()
        wrapper._queue_load = Mock()
        result = SimpleNamespace(
            device_indices=SimpleNamespace(numel=lambda: device_hit_len)
        )
        return wrapper, result

    def test_full_l1_hit_survives_master_outage_without_remote_lookup(self):
        linker = self.make_linker()
        linker._availability.failed("master down")
        wrapper, result = self.make_wrapper(linker, 3)
        with patch.object(linker, "lookup", wraps=linker.lookup) as lookup:
            self.assertIs(
                wrapper.match([1, 2, 3], SimpleNamespace(rid="r"), result), result
            )
            lookup.assert_not_called()
        self.raw.batch_is_exist.assert_not_called()

    def test_partial_l1_hit_survives_failed_lookup_and_keeps_collective(self):
        linker = self.make_linker()
        self.raw.batch_is_exist.side_effect = ConnectionError("master down")
        wrapper, result = self.make_wrapper(linker, 2)
        self.assertIs(
            wrapper.match([1, 2, 3], SimpleNamespace(rid="r"), result), result
        )
        wrapper._sync_restorable_prefix.assert_called_once_with(
            [], num_pages=1, device_hit_pages=0
        )
        self.assertFalse(wrapper.hit_markers)
        self.assertEqual(result.device_indices.numel(), 2)

    def test_failure_between_match_and_load_back_does_not_publish_device_slots(self):
        linker = self.make_linker()
        wrapper, _ = self.make_wrapper(linker, 2)
        wrapper.hit_markers["r"] = SimpleNamespace(
            device_hit_len=2, tail_hashes=["key"]
        )
        self.raw.batch_is_exist.side_effect = ConnectionError(
            "master failed after lookup"
        )
        req = SimpleNamespace(rid="r", last_node=5)
        self.assertEqual(wrapper.load_back(req), ([], 5))
        self.assertFalse(wrapper.hit_markers)
        wrapper._queue_load.assert_not_called()
        wrapper._update_load.assert_called_once()
        self.assertEqual(wrapper._update_load.call_args.args[0], "abort")
        self.assertFalse(linker.pending_loads)

    def test_disabled_fallback_keeps_lookup_exception_visible(self):
        os.environ["SGLANG_MOONCAKE_L1_FALLBACK"] = "0"
        linker = self.make_linker()
        self.raw.batch_is_exist.side_effect = ConnectionError("master down")
        with self.assertRaises(ConnectionError):
            linker.lookup("r", [self.transfer])


if __name__ == "__main__":
    unittest.main()
