"""CPU regressions for the 256K chunk + waiting 64K L3-hit leak.

Execute production method bodies without importing GPU/native modules. The fake
storage/allocator boundary tracks individual FULL/SWA slots and rejects double
free. These tests do not establish native DMA completion guarantees.
"""

import ast
import contextlib
import enum
import importlib.util
import logging
import sys
import unittest
from copy import copy
from pathlib import Path
from types import SimpleNamespace as NS
from typing import NamedTuple
from unittest.mock import Mock, patch

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
            torch=NS(
                cat=lambda xs: Indices(x for s in xs for x in s),
                tensor=lambda x, **_: Indices(x),
                int64="int64",
            ),
            contextmanager=contextlib.contextmanager,
            copy=copy,
            AddReqResult=Result,
            CLIP_MAX_NEW_TOKENS=4096,
            InitLoadBackParams=NS,
            MatchPrefixParams=NS,
            RadixKey=NS,
            PoolName=NS(KV="full", SWA="swa"),
            LinkerTransferPhase=Phase,
            ExternalLinkerLoadPhase=Phase,
            NamedTuple=NamedTuple,
            logger=logging.getLogger(__name__),
            envs=NS(SGLANG_RADIX_FORCE_MISS=NS(get=lambda: False)),
        )
        # Preserve NamedTuple inheritance when loading the production descriptor.
        tree = ast.parse(LINKER.read_text(encoding="utf-8"))
        nodes = [
            n
            for n in tree.body
            if getattr(n, "name", None)
            in ("_PrivateLoad", "ExternalLinkerCleanupError")
        ]
        module = ast.Module(
            body=[
                ast.ImportFrom(
                    module="__future__", names=[ast.alias(name="annotations")], level=0
                ),
                *nodes,
            ],
            type_ignores=[],
        )
        exec(
            compile(ast.fix_missing_locations(module), str(LINKER), "exec"), self.scope
        )
        Wrapper = extract(
            LINKER,
            "UnifiedCacheLinkerWrapper",
            [
                "load_back_admission",
                "load_back",
                "bind_private_load",
                "match",
                "_raise_if_cleanup_quarantined",
                "has_pending_operations",
                "_remember_private_slot_range",
                "_clear_private_slot_ownership",
                "_mark_private_slots_released",
                "_handoff_private_slots",
                "_abort_component_transfers",
                "_abort_private_load",
                "_update_load",
                "_abort_transfers_with_quarantine",
                "commit_completed_loads",
                "release_request",
                "start_layer_wise_loading",
            ],
            self.scope,
        )
        self.Adder = extract(
            POLICY,
            "PrefillAdder",
            [
                "add_chunked_req",
                "add_one_req",
                "_load_back_admission",
                "_get_prefill_admission_tokens",
                "_update_prefill_budget",
                "ceil_paged_tokens",
                "_swa_budget_for_req",
                "_swa_reserved_tokens",
                "_swa_new_tokens",
            ],
            self.scope,
        )
        self.Req = extract(BATCH, "Req", ["init_next_round_input"], self.scope)
        self.Scheduler = extract(
            SCHEDULER, "Scheduler", ["_prepare_prefill_batch"], self.scope
        )
        Full = extract(
            COMPONENTS / "full_component.py",
            "FullComponent",
            ["update_external_linker_load"],
            self.scope,
        )
        Swa = extract(
            COMPONENTS / "swa_component.py",
            "SWAComponent",
            ["update_external_linker_load"],
            self.scope,
        )
        self.error = self.scope["ExternalLinkerCleanupError"]
        self.full, self.swa = Ledger(), Ledger()
        self.backend = NS(
            queued={}, started=[], revalidate_load=Mock(return_value=True)
        )

        def load(rid, transfers):
            self.backend.queued[rid] = transfers
            return True

        def start():
            self.backend.started.extend(self.backend.queued)
            self.backend.queued.clear()
            return 0

        self.backend.load = load
        self.backend.start_layer_wise_loading = Mock(side_effect=start)
        self.backend.cancel_queued_load = Mock(
            side_effect=lambda rid: self.backend.queued.pop(rid, None) is not None
        )
        self.cache = NS(
            page_size=256,
            disable=False,
            sliding_window_size=128,
            tree_core=NS(empty_match_result=NS(device_indices=Indices())),
            token_to_kv_pool_allocator=NS(
                swa_attn_allocator=self.swa, set_full_to_swa_mapping=Mock()
            ),
            supports_mamba=lambda: False,
            swa_reprefill_tail_tokens=lambda: 0,
        )
        self.wrapper = Wrapper.__new__(Wrapper)
        self.wrapper.cache = self.cache
        self.wrapper.cache_linker = self.backend
        for name in [
            "private_loads",
            "pending_loads",
            "hit_markers",
            "failed_chains",
            "cleanup_quarantine",
        ]:
            setattr(self.wrapper, name, {})
        self.wrapper.taken_loads = []
        self.wrapper.pending_offloads = []
        self.cache.linker = self.wrapper
        self.cache.init_load_back = lambda params: self.wrapper.load_back(params.req)
        full, swa = Full.__new__(Full), Swa.__new__(Swa)
        full._full_allocator = lambda: self.full
        swa.cache = self.cache
        swa.sliding_window_size = 128
        full.build_external_linker_transfer = lambda phase, node, keys: NS(
            name="full", keys=keys, device_indices=self.full.alloc(len(keys) * 256)
        )
        swa.build_external_linker_transfer = lambda phase, node, keys: NS(
            name="swa", keys=keys[-1:], device_indices=self.swa.alloc(256)
        )
        self.cache._components_tuple = (full, swa)
        self.cache.match_prefix = self.match
        # Only ReqKvInfo is imported inside the component / allocator methods.
        self.modules = patch.dict(
            sys.modules, {"sglang.srt.managers.schedule_batch": NS(ReqKvInfo=NS)}
        )
        self.modules.start()
        self.addCleanup(self.modules.stop)

    def match(self, params):
        self.wrapper.hit_markers[params.req.rid] = NS(
            device_hit_len=0, tail_hashes=["page"] * 255
        )
        return NS(
            device_indices=Indices(),
            last_device_node="root",
            last_host_node="root",
            best_match_node="root",
            host_hit_length=65280,
            swa_host_hit_length=256,
            mamba_host_hit_length=0,
            mamba_branching_seqlen=0,
            cache_protected_len=0,
        )

    def req(self, rid="waiting", tokens=65536):
        req = self.Req()
        req.__dict__.update(
            rid=rid,
            prefix_indices=Indices(),
            last_node="root",
            best_match_node="root",
            full_untruncated_fill_ids=range(tokens),
            output_ids=[],
            sampling_params=NS(max_new_tokens=8, ignore_eos=False),
            host_hit_length=65280,
            swa_host_hit_length=256,
            storage_hit_length=0,
            host_loaded_length=0,
            cache_protected_len=0,
            external_cache_hit_length=65280,
            external_kv_private_slots_start=None,
            external_kv_private_slots_end=None,
            external_kv_private_slots_released=False,
            external_kv_private_swa_slots_released=False,
            req_pool_idx=None,
            kv=None,
            retracted_stain=False,
            is_retracted=False,
            positional_embed_overrides=None,
            extra_key=None,
            cache_salt=None,
            session=None,
        )
        req.needs_host_load_back = lambda: req.host_hit_length > 0
        req.set_extend_range = lambda start, end: setattr(
            req, "extend_range", NS(start=start, end=end, length=end - start)
        )
        req.is_dllm = lambda: False
        req._refresh_fill_ids = lambda: None
        req._compute_max_prefix_len = lambda n: n - 1
        req.init_next_round_input(self.cache)
        return req

    def adder(self, budget=32768):
        a = self.Adder()
        a.__dict__.update(
            tree_cache=self.cache,
            page_size=256,
            is_hybrid_swa=True,
            dllm_config=None,
            rem_chunk_tokens=budget,
            rem_total_tokens=10_000_000,
            rem_swa_tokens=200_000,
            rem_input_tokens=131072,
            rem_mamba_slots=None,
            prefill_delayer_single_pass=None,
            can_run_list=[],
            dsa_prefill_cp_in_seq_split=False,
            prefill_max_requests=None,
        )
        for name in [
            "rem_total_token_offset",
            "cur_rem_token_offset",
            "rem_swa_token_offset",
            "log_hit_tokens",
            "log_input_tokens",
            "log_device_hit_tokens",
            "log_host_hit_tokens",
            "log_storage_hit_tokens",
        ]:
            setattr(a, name, 0)
        a._mamba_gap_budget_for_req = lambda req: 0
        a._lock_node = lambda node: contextlib.nullcontext()
        a._req_inc_lock_ref = Mock()
        a._check_prefill_tile_budget = Mock(return_value=None)
        a.budget_state = lambda: Result.CONTINUE
        self.scope["split_cached_prefix_by_tier"] = lambda **_: (0, 0, 0)
        return a

    def bind_row(self, req, write_raises=False):
        """Execute real alloc_for_extend with only its device allocator mocked."""
        scope = dict(self.scope)
        scope.update(
            _is_npu=False,
            _alloc_page_size=lambda b: 1,
            alloc_token_slots=lambda cache, n: Indices(range(n)),
        )
        row = {}

        def alloc_req(pool, reqs, cache):
            for i, r in enumerate(reqs):
                r.req_pool_idx = i
            return list(range(len(reqs)))

        def write(*args):
            if write_raises:
                raise RuntimeError("row write failed")
            row[req.req_pool_idx] = Indices(req.prefix_indices)

        scope.update(alloc_req_slots=alloc_req, write_cache_indices=write)
        extract(ALLOCATION, None, ["alloc_for_extend"], scope)
        length = len(req.prefix_indices) + 256
        batch = NS(
            reqs=[req],
            tree_cache=self.cache,
            maybe_evict_swa=lambda: None,
            is_dllm=lambda: False,
            prefix_lens=[len(req.prefix_indices)],
            extend_lens=[256],
            device="cpu",
            req_to_token_pool=object(),
            extend_num_tokens=256,
            seq_lens=Indices([length]),
            seq_lens_cpu=Indices([length]),
        )
        scope["alloc_for_extend"](batch)
        self.assertEqual(row[req.req_pool_idx], req.prefix_indices)

    def test_seven_exhausted_chunks_do_not_allocate_waiting_hit(self):
        waiting = self.req()
        for _ in range(7):
            a = self.adder()
            a.add_chunked_req(self.req("active", 262144))
            self.assertEqual(a.rem_chunk_tokens, 0)
            waiting.init_next_round_input(self.cache)
            self.assertIs(a.add_one_req(waiting, True, None), Result.OTHER)
            self.assertNotIn(waiting, a.can_run_list)
            self.assertFalse(self.full.live or self.swa.live)
        self.assertEqual(self.full.allocations, 0)
        a = self.adder()
        waiting.init_next_round_input(self.cache)
        a.add_one_req(waiting, False, None)
        self.assertIn(waiting, a.can_run_list)
        self.assertEqual((len(self.full.live), len(self.swa.live)), (65280, 256))
        self.bind_row(waiting)
        self.wrapper.start_layer_wise_loading()
        self.wrapper.taken_loads = [([waiting.rid], True)]
        self.wrapper.commit_completed_loads([True])
        self.assertFalse(self.wrapper.has_pending_operations())
        self.assertIsNone(waiting.external_kv_private_slots_start)
        # Request/radix now owns the only allocation. Linker cleanup must not free it.
        self.wrapper.release_request(waiting.rid)
        self.assertEqual(self.full.allocations, 1)
        self.assertFalse(self.full.frees or self.swa.frees)

    def test_budget_rejections_do_not_allocate(self):
        for kind in ("chunk", "alignment", "tile", "dllm"):
            with self.subTest(kind=kind):
                req = self.req(kind, tokens=131072)
                a = self.adder(0 if kind == "chunk" else 256)
                align = 1024 if kind == "alignment" else None
                if kind == "tile":
                    a._check_prefill_tile_budget.return_value = Result.OTHER
                if kind == "dllm":
                    a.dllm_config = object()
                    a.rem_dllm_tokens = 0
                self.assertIs(a.add_one_req(req, False, align), Result.OTHER)
                self.assertFalse(self.full.live or self.swa.live)

    def test_late_rejection_restores_state_and_releases_each_pool_once(self):
        req = self.req()
        req.kv = NS(kv_allocated_len=0, swa_evicted_seqlen=0)
        a = self.adder()
        a._check_prefill_tile_budget.side_effect = [None, Result.OTHER]
        self.assertIs(a.add_one_req(req, False, None), Result.OTHER)
        self.assertFalse(self.wrapper.private_loads or self.backend.queued)
        self.assertEqual(req.prefix_indices, [])
        self.assertEqual(vars(req.kv), dict(kv_allocated_len=0, swa_evicted_seqlen=0))
        self.assertIsNone(req.external_kv_private_slots_start)
        self.assertFalse(req.external_kv_private_slots_released)
        self.wrapper.release_request(req.rid)
        self.wrapper.commit_completed_loads([])
        self.assertEqual((len(self.full.frees), len(self.swa.frees)), (1, 1))
        self.assertFalse(self.full.live or self.swa.live)
        # Retry with the same RID gets a new allocation, without stale exclusions.
        req.init_next_round_input(self.cache)
        self.adder().add_one_req(req, False, None)
        self.wrapper.release_request(req.rid)
        self.assertFalse(self.full.live or self.swa.live)
        self.assertEqual((len(self.full.frees), len(self.swa.frees)), (2, 2))

    def test_revalidation_miss_preserves_l1_recompute(self):
        req = self.req()
        self.backend.revalidate_load.return_value = False
        a = self.adder()
        a.add_one_req(req, False, None)
        self.assertIn(req, a.can_run_list)
        self.assertEqual(req.extend_range.length, 32768)
        self.assertFalse(self.full.live or self.swa.live)
        self.assertFalse(req.external_kv_private_slots_released)

    def test_revalidation_exception_releases_prequeue_allocations(self):
        req = self.req()
        self.backend.revalidate_load.side_effect = RuntimeError("collective failed")
        with self.assertRaisesRegex(RuntimeError, "collective failed"):
            self.adder().add_one_req(req, False, None)
        self.assertFalse(self.full.live or self.swa.live)
        self.assertEqual(req.prefix_indices, [])
        self.assertIsNone(req.kv)

    def test_later_component_allocation_exception_releases_earlier_component(self):
        req = self.req()
        self.cache._components_tuple[1].build_external_linker_transfer = Mock(
            side_effect=RuntimeError("SWA allocation failed")
        )
        with self.assertRaisesRegex(RuntimeError, "SWA allocation failed"):
            self.adder().add_one_req(req, False, None)
        self.assertFalse(self.full.live or self.swa.live)
        self.assertEqual(len(self.full.frees), 1)
        self.assertFalse(self.backend.queued)

    def test_rollback_free_exception_keeps_quarantine_and_never_retries(self):
        req = self.req()
        a = self.adder()
        a._check_prefill_tile_budget.side_effect = [None, Result.OTHER]
        real_free = self.swa.free

        def partial_free(slots):
            real_free(slots[:128])
            raise RuntimeError("partial SWA free")

        self.swa.free = Mock(side_effect=partial_free)
        with self.assertLogs(level="ERROR"):
            with self.assertRaisesRegex(self.error, "release uncertain"):
                a.add_one_req(req, False, None)
        self.assertIn(req.rid, self.wrapper.cleanup_quarantine)
        self.assertTrue(self.wrapper.has_pending_operations())
        self.assertTrue(req.external_kv_private_slots_released)
        self.assertEqual((len(self.full.live), len(self.swa.live)), (65280, 128))
        for retry in (
            lambda: self.wrapper.release_request(req.rid),
            lambda: self.wrapper.commit_completed_loads([]),
            lambda: self.adder().add_one_req(req, False, None),
        ):
            with self.assertRaises(self.error):
                retry()
        self.swa.free.assert_called_once()
        self.assertFalse(self.full.frees)

    def test_scheduler_establishes_row_owner_before_starting_l3(self):
        req = self.req()
        self.adder().add_one_req(req, False, None)
        scheduler = self.Scheduler()
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_unified_cache_external_linker = True
        scheduler.tree_cache = NS(
            ready_to_load_host_cache=self.wrapper.start_layer_wise_loading
        )
        batch = NS(prepare_for_extend=lambda: self.bind_row(req))
        scheduler._prepare_prefill_batch(batch)
        self.assertTrue(self.wrapper.private_loads[req.rid].bound)
        self.assertEqual(self.backend.started, [req.rid])
        self.assertEqual(batch.hicache_consumer_index, 0)

    def test_scheduler_does_not_start_l3_when_batch_preparation_fails(self):
        req = self.req()
        self.adder().add_one_req(req, False, None)
        scheduler = self.Scheduler()
        scheduler.enable_hierarchical_cache = False
        scheduler.enable_unified_cache_external_linker = True
        scheduler.tree_cache = NS(
            ready_to_load_host_cache=self.wrapper.start_layer_wise_loading
        )
        batch = NS(prepare_for_extend=lambda: self.bind_row(req, write_raises=True))
        with self.assertRaisesRegex(RuntimeError, "row write failed"):
            scheduler._prepare_prefill_batch(batch)
        self.backend.start_layer_wise_loading.assert_not_called()
        self.assertIn(req.rid, self.wrapper.private_loads)
        # Still queued, so cancellation can safely reclaim it exactly once.
        self.wrapper.release_request(req.rid)
        self.wrapper.release_request(req.rid)
        self.assertEqual((len(self.full.frees), len(self.swa.frees)), (1, 1))

    def test_scheduler_preserves_l2_start_order(self):
        events = []
        scheduler = self.Scheduler()
        scheduler.enable_hierarchical_cache = True
        scheduler.enable_unified_cache_external_linker = False
        scheduler.tree_cache = NS(
            ready_to_load_host_cache=lambda: events.append("load") or 3
        )
        batch = NS(prepare_for_extend=lambda: events.append("prepare"))
        scheduler._prepare_prefill_batch(batch)
        self.assertEqual(events, ["load", "prepare"])
        self.assertEqual(batch.hicache_consumer_index, 3)

    def test_uncertain_cancel_retains_owner_and_blocks_rematch(self):
        req = self.req()
        a = self.adder()
        a._check_prefill_tile_budget.side_effect = [None, Result.OTHER]
        self.backend.cancel_queued_load.side_effect = None
        self.backend.cancel_queued_load.return_value = False
        with self.assertRaisesRegex(self.error, "private slots retained"):
            a.add_one_req(req, False, None)
        self.assertIn(req.rid, self.wrapper.private_loads)
        with self.assertRaisesRegex(self.error, "cannot rematch"):
            self.wrapper.match(None, req, None)
        self.assertEqual((len(self.full.live), len(self.swa.live)), (65280, 256))
        self.assertFalse(self.full.frees or self.swa.frees)
        # A later failed completion is a terminal fence and can reclaim once.
        self.wrapper.taken_loads = [([req.rid], False)]
        self.wrapper.commit_completed_loads([False])
        self.wrapper.release_request(req.rid)
        self.assertFalse(self.full.live or self.swa.live)

    def test_unbound_load_cannot_start_or_retire_on_success(self):
        req = self.req()
        self.adder().add_one_req(req, False, None)
        with self.assertRaisesRegex(self.error, "cannot start"):
            self.wrapper.start_layer_wise_loading()
        self.backend.start_layer_wise_loading.assert_not_called()
        self.wrapper.taken_loads = [([req.rid], True)]
        with self.assertRaisesRegex(self.error, "no request-row owner"):
            self.wrapper.commit_completed_loads([True])
        self.assertEqual(self.wrapper.taken_loads, [([req.rid], True)])
        self.assertIn(req.rid, self.wrapper.private_loads)

    def test_request_index_without_successful_row_write_is_not_handoff(self):
        req = self.req()
        self.adder().add_one_req(req, False, None)
        with self.assertRaisesRegex(RuntimeError, "row write failed"):
            self.bind_row(req, write_raises=True)
        self.assertIsNotNone(req.req_pool_idx)
        self.assertFalse(self.wrapper.private_loads[req.rid].bound)

    def test_row_binding_does_not_free_a_failed_inflight_load_early(self):
        req = self.req()
        self.adder().add_one_req(req, False, None)
        self.bind_row(req)
        self.wrapper.start_layer_wise_loading()
        self.wrapper.release_request(req.rid)
        self.assertFalse(self.full.frees or self.swa.frees)
        self.wrapper.taken_loads = [([req.rid], False)]
        self.wrapper.commit_completed_loads([False])
        self.wrapper.release_request(req.rid)
        self.assertEqual((len(self.full.frees), len(self.swa.frees)), (1, 1))


if __name__ == "__main__":
    unittest.main()
