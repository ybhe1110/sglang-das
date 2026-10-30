"""CPU regressions for the 256K chunk + waiting 64K L3-hit leak.

Execute production method bodies without importing GPU/native modules. The fake
storage/allocator boundary tracks individual FULL/SWA slots and rejects double
free. These tests do not establish native DMA completion guarantees.
"""

import ast
import contextlib
import dataclasses
import enum
import importlib.util
import logging
import sys
import unittest
from copy import copy
from pathlib import Path
from types import SimpleNamespace as NS
from unittest.mock import Mock

ROOT = Path(__file__).resolve().parents[4]
spec = importlib.util.spec_from_file_location(
    "admission_ci_register", ROOT / "python/sglang/test/ci/ci_register.py"
)
ci = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = ci
spec.loader.exec_module(ci)
register_cpu_ci = ci.register_cpu_ci
register_cpu_ci(est_time=3, suite="base-a-test-cpu")

POLICY = ROOT / "python/sglang/srt/managers/schedule_policy.py"
LINKER = ROOT / "python/sglang/srt/mem_cache/unified_cache/unified_cache_linker.py"
BATCH = ROOT / "python/sglang/srt/managers/schedule_batch.py"
ALLOCATION = ROOT / "python/sglang/srt/mem_cache/allocation.py"
SCHEDULER = ROOT / "python/sglang/srt/managers/scheduler.py"
COMPONENTS = LINKER.parent / "components"


def extract(path, class_name, names, scope):
    tree = ast.parse(path.read_text(encoding="utf-8"))
    parent = (
        next(
            n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == class_name
        )
        if class_name
        else tree
    )
    nodes = [n for n in parent.body if getattr(n, "name", None) in names]
    assert len(nodes) == len(names), (path, names)
    if class_name:
        parent.body = nodes
        parent.bases = []
        nodes = [parent]
    module = ast.Module(
        body=[
            ast.ImportFrom(
                module="__future__", names=[ast.alias(name="annotations")], level=0
            ),
            *nodes,
        ],
        type_ignores=[],
    )
    exec(compile(ast.fix_missing_locations(module), str(path), "exec"), scope)
    return scope[class_name] if class_name else None


class Indices(list):
    def __getitem__(self, index):
        value = super().__getitem__(index)
        return Indices(value) if isinstance(index, slice) else value

    def numel(self):
        return len(self)

    def to(self, *args, **kwargs):
        return self

    def tolist(self):
        return list(self)


class Result(enum.Enum):
    CONTINUE = 1
    NO_TOKEN = 2
    OTHER = 3


class Phase(enum.Enum):
    LOAD = 1
    PREPARE = 2
    ABORT = 3
    COMMIT = 4


class Ledger:
    def __init__(self):
        self.live = set()
        self.frees = []
        self.allocations = 0
        self.next_slot = 256

    def alloc(self, count):
        slots = Indices(range(self.next_slot, self.next_slot + count))
        self.next_slot += count
        self.live.update(slots)
        self.allocations += 1
        return slots

    def free(self, slots):
        assert set(slots) <= self.live, "double free / foreign slots"
        self.live.difference_update(slots)
        self.frees.append(tuple(slots))


class TestExternalLoadAdmission(unittest.TestCase):
    def setUp(self):
        self.scope = dict(
            __name__=__name__,
            dataclass=dataclasses.dataclass,
            dataclasses=dataclasses,
            contextmanager=contextlib.contextmanager,
            nullcontext=contextlib.nullcontext,
            copy=copy,
            AddReqResult=Result,
            CLIP_MAX_NEW_TOKENS=4096,
            InitLoadBackParams=NS,
            UnifiedRadixCache=type("UnifiedRadixCache", (), {}),
            PoolName=NS(KV="full", SWA="swa"),
            ComponentType=NS(FULL="full", SWA="swa"),
            LinkerTransferPhase=Phase,
            ExternalLinkerLoadPhase=Phase,
            logger=logging.getLogger(__name__),
            torch=NS(
                tensor=lambda x, **_: Indices(x),
                int="int32",
                distributed=NS(ReduceOp=NS(MIN="min")),
            ),
            preflight=lambda *args: args[4] if len(args) > 4 else True,
            get_schedule=lambda: NS(schedule_policy="fcfs"),
        )
        for path, names in [
            (LINKER, ["_PrivateLoad", "ExternalLinkerCleanupError"]),
            (POLICY, ["_PrefillAdmission"]),
            (BATCH, ["ReqKvInfo"]),
        ]:
            extract(path, None, names, self.scope)
        Wrapper = extract(
            LINKER,
            "UnifiedCacheLinkerWrapper",
            [
                "load_back_admission",
                "load_back",
                "bind_private_load",
                "match",
                "_raise_if_cleanup_quarantined",
                "_abort_transfers",
                "adopt_request",
                "finish_loads",
                "release_request",
                "start_layer_wise_loading",
            ],
            self.scope,
        )
        self.Adder = extract(
            POLICY,
            "PrefillAdder",
            [
                "add_one_req",
                "_load_back_admission",
                "_select_prefill_admission",
                "_commit_prefill_admission",
                "_check_prefill_budget",
                "_swa_new_tokens",
                "ceil_paged_tokens",
            ],
            self.scope,
        )
        self.Scheduler = extract(
            SCHEDULER,
            "Scheduler",
            ["_prepare_prefill_batch"],
            self.scope,
        )
        self.full, self.swa = Ledger(), Ledger()
        self.backend = NS(queued={}, started=[], completed=[])

        def load(rid, transfers):
            self.backend.queued[rid] = transfers
            return True

        def start():
            self.backend.started.extend(self.backend.queued)
            self.backend.queued.clear()
            return 0

        self.backend.load = Mock(side_effect=load)
        self.backend.start_layer_wise_loading = Mock(side_effect=start)
        self.backend.cancel_queued_load = Mock(
            side_effect=lambda rid: self.backend.queued.pop(rid, None) is not None
        )
        self.backend.finish_loads = Mock()
        self.backend.pop_completed_load = lambda: self.backend.completed.pop(0)
        self.cache = NS(
            page_size=256,
            disable=False,
            token_to_kv_pool_allocator=NS(
                swa_attn_allocator=self.swa, set_full_to_swa_mapping=Mock()
            ),
            lock=Mock(return_value="request-lock"),
            _all_reduce_attn_groups=Mock(),
        )
        self.wrapper = Wrapper.__new__(Wrapper)
        self.wrapper.cache = self.cache
        self.wrapper.cache_linker = self.backend
        self.wrapper.private_loads = {}
        self.wrapper.hit_markers = {}
        self.wrapper.cleanup_quarantine = {}
        self.wrapper.load_batches = {}
        self.wrapper._finish_sequence = 0
        self.wrapper._skip_swa = False
        self.cache.linker = self.wrapper
        self.cache.init_load_back = Mock(
            side_effect=lambda params: self.wrapper.load_back(params.req)
        )
        Full = extract(
            COMPONENTS / "full.py",
            "FullComponent",
            ["update_external_linker_load"],
            self.scope,
        )
        Swa = extract(
            COMPONENTS / "swa.py",
            "SWAComponent",
            ["update_external_linker_load"],
            self.scope,
        )
        full, swa = Full(), Swa()
        full._full_allocator = lambda: self.full
        swa.cache, swa.sliding_window_size, swa.component_type = self.cache, 256, "swa"
        full.build_external_linker_transfer = lambda phase, node, keys: NS(
            name="full", keys=keys, device_indices=self.full.alloc(len(keys) * 256)
        )
        swa.build_external_linker_transfer = lambda phase, node, keys: NS(
            name="swa", keys=keys[-1:], device_indices=self.swa.alloc(256)
        )
        self.wrapper._components = (full, swa)
        self.error = self.scope["ExternalLinkerCleanupError"]

    def req(self, rid="waiting"):
        req = NS(
            rid=rid,
            prefix_len=0,
            last_node="root",
            best_match_node="root",
            full_untruncated_fill_ids=range(65536),
            output_ids=[],
            sampling_params=NS(max_new_tokens=8, ignore_eos=False),
            host_hit_length=65280,
            swa_host_hit_length=256,
            host_loaded_length=0,
            kv=self.scope["ReqKvInfo"](),
            retracted_stain=False,
            external_kv_load_pending=False,
            external_kv_load_failed=False,
            skip_radix_cache_insert=False,
        )
        req.needs_host_load_back = lambda: req.host_hit_length > 0
        self.wrapper.hit_markers[rid] = NS(device_hit_len=0, tail_hashes=["page"] * 255)
        return req

    def adder(self, chunk=256):
        adder = self.Adder()
        adder.tree_cache = self.cache
        adder.can_run_list = []
        adder.prefill_max_requests = None
        adder.prefill_delayer_single_pass = None
        adder.page_size = adder.per_req_token_overhead = 256
        adder.rem_chunk_tokens, adder.rem_input_tokens = chunk, 1 << 20
        adder.exact_chunk_fill = False
        adder.dllm_config = None
        adder._mamba_gap_budget_for_req = lambda req: 0
        adder._lock_node = lambda *args, **kwargs: contextlib.nullcontext()
        adder._kv_shard_reserve_scratch = Mock(return_value=True)
        adder._check_prefill_tile_budget = Mock(return_value=None)
        adder._update_prefill_budget = Mock()
        adder._account_prefill_cache_admission = Mock()
        adder.budget_state = lambda: Result.CONTINUE
        adder.memory_budget = NS(
            check_prefill=Mock(side_effect=lambda **_: (True, adder.rem_chunk_tokens)),
            prepare_load_back=Mock(return_value=True),
        )
        return adder

    def bind(self, req):
        req.kv.req_pool_idx = 0
        req.kv.kv_allocated_len = req.prefix_len
        self.wrapper.adopt_request(req)
        self.wrapper.bind_private_load(req)

    def test_seven_exhausted_chunks_do_not_allocate_waiting_hit(self):
        req = self.req()
        for _ in range(7):
            adder = self.adder(chunk=0)
            self.assertEqual(adder.add_one_req(req, True, None), Result.OTHER)
        self.assertEqual(self.full.allocations, 0)
        self.assertEqual(self.swa.allocations, 0)
        self.cache.init_load_back.assert_not_called()
        self.assertFalse(self.wrapper.private_loads)
        self.assertEqual(self.adder().add_one_req(req, False, None), Result.CONTINUE)
        self.assertEqual(self.full.allocations, 1)
        self.assertEqual(req.prefix_len, 65280)

    def test_budget_alignment_and_tile_rejections_precede_load(self):
        for reason in ("budget", "alignment", "tile", "reclaim"):
            with self.subTest(reason=reason):
                req, adder = self.req(reason), self.adder()
                align = 512 if reason == "alignment" else None
                if reason == "budget":
                    adder.memory_budget.check_prefill.side_effect = None
                    adder.memory_budget.check_prefill.return_value = (False, 256)
                if reason == "alignment":
                    req.full_untruncated_fill_ids = range(66048)
                if reason == "tile":
                    adder._check_prefill_tile_budget.return_value = Result.OTHER
                if reason == "reclaim":
                    adder.memory_budget.prepare_load_back.return_value = False
                self.assertNotEqual(
                    adder.add_one_req(req, False, align), Result.CONTINUE
                )
        self.assertEqual(self.full.allocations, 0)
        self.assertFalse(self.backend.queued)

    def test_late_rejection_restores_state_and_releases_each_pool_once(self):
        req = self.req()
        for _ in range(3):
            self.wrapper.hit_markers[req.rid] = NS(
                device_hit_len=0, tail_hashes=["page"] * 255
            )
            adder = self.adder()
            adder._kv_shard_reserve_scratch.return_value = False
            self.assertEqual(adder.add_one_req(req, False, None), Result.OTHER)
            self.assertEqual((req.prefix_len, req.host_loaded_length), (0, 0))
            self.assertEqual(req.kv.component_evicted_seqlens, {})
            self.assertFalse(req.external_kv_load_pending)
            self.assertFalse(self.wrapper.private_loads)
        self.assertEqual(len(self.full.frees), 3)
        self.assertEqual(len(self.swa.frees), 3)
        self.assertFalse(self.full.live | self.swa.live)
        self.backend.start_layer_wise_loading.assert_not_called()

    def test_key_evicted_at_queueing_preserves_l1_recompute(self):
        req = self.req()
        self.backend.load.side_effect = None
        self.backend.load.return_value = False
        adder = self.adder(chunk=4096)
        self.assertEqual(adder.add_one_req(req, False, None), Result.CONTINUE)
        self.assertEqual((req.prefix_len, req.extend_end), (0, 4096))
        self.assertFalse(self.wrapper.private_loads)
        self.assertFalse(self.full.live | self.swa.live)
        self.assertEqual(req.kv.component_evicted_seqlens, {})

    def test_later_component_allocation_failure_unwinds_full(self):
        req = self.req()
        self.wrapper._components[1].build_external_linker_transfer = Mock(
            side_effect=RuntimeError("SWA allocation failed")
        )
        with self.assertRaisesRegex(RuntimeError, "SWA allocation failed"):
            self.adder().add_one_req(req, False, None)
        self.assertFalse(self.full.live)
        self.assertEqual(len(self.full.frees), 1)
        self.backend.load.assert_not_called()

    def test_rollback_partial_free_quarantines_without_retry(self):
        req = self.req()
        original = self.swa.free

        def uncertain(slots):
            original(slots)
            raise RuntimeError("after side effect")

        self.swa.free = Mock(side_effect=uncertain)
        adder = self.adder()
        adder._kv_shard_reserve_scratch.return_value = False
        with self.assertRaises(self.error):
            adder.add_one_req(req, False, None)
        for action in (
            lambda: self.wrapper.release_request(req.rid),
            lambda: self.wrapper.start_layer_wise_loading(),
        ):
            with self.assertRaises(self.error):
                action()
        self.swa.free.assert_called_once()
        self.assertIn(req.rid, self.wrapper.cleanup_quarantine)
        self.assertTrue(self.full.live)

    def test_uncertain_cancel_retains_slots_and_blocks_rematch(self):
        req = self.req()
        self.backend.cancel_queued_load.side_effect = None
        self.backend.cancel_queued_load.return_value = False
        adder = self.adder()
        adder._kv_shard_reserve_scratch.return_value = False
        with self.assertRaises(self.error):
            adder.add_one_req(req, False, None)
        self.assertTrue(self.full.live and self.swa.live)
        self.assertIn(req.rid, self.wrapper.private_loads)
        with self.assertRaisesRegex(self.error, "rematch"):
            self.wrapper.match(None, req, None)

    def test_request_row_number_alone_does_not_allow_io(self):
        req = self.req()
        self.adder().add_one_req(req, False, None)
        req.kv.req_pool_idx = 0
        with self.assertRaises(self.error):
            self.wrapper.bind_private_load(req)
        self.wrapper.adopt_request(req)
        with self.assertRaises(self.error):
            self.wrapper.bind_private_load(req)
        with self.assertRaises(self.error):
            self.wrapper.start_layer_wise_loading()
        self.backend.start_layer_wise_loading.assert_not_called()

    def test_success_cannot_retire_without_binding(self):
        req = self.req()
        self.adder().add_one_req(req, False, None)
        self.wrapper.load_batches[0] = [req.rid]
        self.backend.completed.append([(req.rid, True)])
        with self.assertRaisesRegex(RuntimeError, "no request owner"):
            self.wrapper.finish_loads(0)
        self.assertIn(req.rid, self.wrapper.private_loads)
        self.assertTrue(self.full.live and self.swa.live)

    def test_success_hands_off_and_failure_waits_for_request_cleanup(self):
        for ok in (True, False):
            with self.subTest(ok=ok):
                req = self.req(str(ok))
                self.adder().add_one_req(req, False, None)
                self.bind(req)
                private = self.wrapper.private_loads[req.rid]
                index = self.wrapper.start_layer_wise_loading()
                if not ok:
                    with self.assertRaisesRegex(RuntimeError, "before load drain"):
                        self.wrapper.release_request(req.rid)
                self.backend.completed.append([(req.rid, ok)])
                self.assertEqual(self.wrapper.finish_loads(index), [] if ok else [req])
                self.wrapper.release_request(req.rid)
                # Linker must leave adopted FULL/SWA to the request owner.
                self.assertFalse(self.full.frees or self.swa.frees)
                self.assertFalse(req.external_kv_load_pending)
                self.assertEqual(req.skip_radix_cache_insert, not ok)
                self.assertNotIn(req.rid, self.wrapper.private_loads)
                self.assertTrue(set(private.device_indices) <= self.full.live)

    def test_same_rid_different_object_cannot_adopt(self):
        req = self.req()
        self.adder().add_one_req(req, False, None)
        with self.assertRaises(self.error):
            self.wrapper.adopt_request(copy(req))
        with self.assertRaises(self.error):
            self.wrapper.bind_private_load(copy(req))
        with self.assertRaises(self.error):
            self.wrapper.load_back(req)

    def test_scheduler_binds_before_l3_and_preserves_l2_order(self):
        for l3, expected in (
            (True, ["prepare", "start"]),
            (False, ["start", "prepare"]),
        ):
            with self.subTest(l3=l3):
                events = []
                scheduler = self.Scheduler()
                scheduler.enable_hierarchical_cache = not l3
                scheduler.enable_unified_cache_external_linker = l3
                scheduler.tree_cache = NS(
                    ready_to_load_host_cache=lambda: events.append("start")
                )
                batch = NS(prepare_for_extend=lambda: events.append("prepare"))
                scheduler._prepare_prefill_batch(batch)
                self.assertEqual(events, expected)

    def test_scheduler_does_not_start_l3_after_preparation_failure(self):
        scheduler = self.Scheduler()
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_unified_cache_external_linker = True
        scheduler.tree_cache = NS(ready_to_load_host_cache=Mock())
        batch = NS(
            prepare_for_extend=Mock(side_effect=RuntimeError("row write failed"))
        )
        with self.assertRaisesRegex(RuntimeError, "row write failed"):
            scheduler._prepare_prefill_batch(batch)
        scheduler.tree_cache.ready_to_load_host_cache.assert_not_called()

    def test_binding_follows_row_write_and_extent_recording(self):
        tree = ast.parse(ALLOCATION.read_text(encoding="utf-8"))
        func = next(
            n for n in tree.body if getattr(n, "name", None) == "alloc_for_extend"
        )
        calls = {
            n.func.id: n.lineno
            for n in ast.walk(func)
            if isinstance(n, ast.Call) and isinstance(n.func, ast.Name)
        }
        binding = next(
            n.lineno
            for n in ast.walk(func)
            if isinstance(n, ast.Call)
            and isinstance(n.func, ast.Attribute)
            and n.func.attr == "bind_private_load"
        )
        extent = next(
            n.lineno
            for n in ast.walk(func)
            if isinstance(n, ast.Assign)
            and any(
                isinstance(t, ast.Attribute) and t.attr == "kv_allocated_len"
                for t in n.targets
            )
        )
        self.assertLess(calls["write_cache_indices"], extent)
        self.assertLess(extent, binding)


if __name__ == "__main__":
    unittest.main()
