"""Rank-divergence regressions for external-cache matching, without storage IO.

The rendezvous executes real production match methods on every rank. It checks
collective shapes/order and computes MIN across TP/CP groups; it is not Gloo.
"""

import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, replace
from types import SimpleNamespace
from unittest.mock import Mock

import torch

from sglang.srt.disaggregation.utils import DisaggregationCollectiveError
from sglang.srt.mem_cache.hicache_storage import PoolName
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    ExternalCacheHitMarker,
    UnifiedCacheLinkerWrapper,
)
from sglang.test.ci.ci_register import register_cpu_ci
from sglang.test.test_utils import CustomTestCase

register_cpu_ci(est_time=5, suite="base-a-test-cpu")


class _Group:
    def __init__(self, members):
        self.members = members
        self.tensors = {}
        self.history = []
        self.error = None
        self.barrier = threading.Barrier(len(members), action=self._reduce, timeout=2)

    def _reduce(self):
        tensors = [self.tensors[rank] for rank in self.members]
        shapes = [(tuple(t.shape), t.dtype) for t in tensors]
        self.history.append(shapes)
        if len(set(shapes)) != 1:
            self.error = AssertionError(f"collective shape/dtype mismatch: {shapes}")
            return
        result = tensors[0].clone()
        for tensor in tensors[1:]:
            result = torch.minimum(result, tensor)
        for tensor in tensors:
            tensor.copy_(result)

    def reduce(self, rank, tensor, op):
        assert op == torch.distributed.ReduceOp.MIN
        self.tensors[rank] = tensor
        self.barrier.wait()
        if self.error is not None:
            raise self.error


class _Key(list):
    extra_key = cache_salt = None
    is_bigram = False

    def __getitem__(self, index):
        value = super().__getitem__(index)
        return _Key(value) if isinstance(index, slice) else value

    def maybe_to_bigram_view(self, is_eagle):
        return self, False

    def raw_token_ids(self):
        return list(self)


@dataclass
class _Result:
    device_prefix_len: int = 0
    last_device_node: int = 0
    best_match_node: int = 0
    last_host_node: int = 0
    host_hit_length: int = 0
    swa_host_hit_length: int = 0
    mamba_host_hit_length: int = 0

    @property
    def device_indices(self):
        return torch.arange(self.device_prefix_len)

    def _replace(self, **changes):
        return replace(self, **changes)


class TestLinkerMatchCollectives(CustomTestCase):
    def _ranks(self, count=2):
        groups = (
            [_Group((0, 1)), _Group((2, 3)), _Group((0, 2)), _Group((1, 3))]
            if count == 4
            else [_Group((0, 1))]
        )
        wrappers = []
        for rank in range(count):

            def reduce(tensor, op, rank=rank):
                for group in groups:
                    if rank in group.members:
                        group.reduce(rank, tensor, op)

            component = SimpleNamespace(
                build_external_linker_transfer=Mock(
                    side_effect=lambda phase, node, keys: SimpleNamespace(
                        name=PoolName.KV, keys=list(keys)
                    )
                )
            )
            wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
            wrapper.cache = SimpleNamespace(
                page_size=2,
                pp_size=1,
                pp_rank=0,
                tree_core=SimpleNamespace(is_eagle=False),
                _components_tuple=(component,),
                _all_reduce_attn_groups=reduce,
            )
            wrapper._components = (component,)
            wrapper.cache_linker = SimpleNamespace(
                lookup=Mock(return_value=[1, 2]), cancel_host_prefetch=Mock()
            )
            wrapper._match_sequence = 0
            wrapper.private_loads = {}
            wrapper.cleanup_quarantine = {}
            wrapper.hit_markers = {}
            wrapper.host_prefetch_hits = {}
            wrapper.pending_host_prefetch_submissions = {}
            wrapper._tail_hashes = Mock(
                side_effect=lambda key, result, hit: [
                    str(i) for i in range((len(key) - hit) // 2)
                ]
            )
            wrappers.append(wrapper)
        return wrappers, groups

    def _run(self, wrappers, operation):
        def invoke(rank):
            try:
                return operation(rank, wrappers[rank])
            except Exception as error:
                return error

        with ThreadPoolExecutor(max_workers=len(wrappers)) as pool:
            futures = [pool.submit(invoke, rank) for rank in range(len(wrappers))]
            return [future.result(timeout=5) for future in futures]

    def _match(self, wrappers, *, known=None, rids=None):
        return self._run(
            wrappers,
            lambda rank, wrapper: wrapper.match(
                _Key(range(4)),
                SimpleNamespace(
                    rid=rids[rank] if rids else "r",
                    external_cache_hit_length=known[rank] if known else None,
                ),
                _Result(),
            ),
        )

    def _assert_collective_errors(self, outcomes, groups):
        for outcome in outcomes:
            self.assertIsInstance(outcome, DisaggregationCollectiveError)
        # No rank-dependent int32 page mask can be submitted after rejection.
        for group in groups:
            self.assertTrue(group.history)
            self.assertTrue(
                all(
                    dtype == torch.int64
                    for shapes in group.history
                    for _, dtype in shapes
                )
            )

    def test_mask_geometry_mismatch_is_rejected_before_variable_reduce(self):
        for pages, lower_bounds in (
            ((2, 14), (0, 0)),
            ((2, 20), (0, 0)),
            ((2, 2), (0, 1)),
        ):
            with self.subTest(pages=pages, lower_bounds=lower_bounds):
                wrappers, groups = self._ranks()
                outcomes = self._run(
                    wrappers,
                    lambda rank, w: w._sync_restorable_prefix(
                        [1, 2],
                        num_pages=pages[rank],
                        device_hit_pages=lower_bounds[rank],
                    ),
                )
                self._assert_collective_errors(outcomes, groups)

    def test_sparse_intersection_uses_both_mesh_axes(self):
        wrappers, groups = self._ranks(4)
        candidates = ([1, 2, 4], [2, 3, 4], [2, 4], [2, 3])
        outcomes = self._run(
            wrappers,
            lambda rank, w: w._sync_restorable_prefix(
                candidates[rank], num_pages=4, device_hit_pages=0
            ),
        )
        self.assertEqual(outcomes, [2] * 4)
        for group in groups:
            self.assertIsNone(group.error)

    def test_lookup_failure_does_not_leave_peer_in_mask_reduce(self):
        wrappers, groups = self._ranks()
        wrappers[0].cache_linker.lookup.side_effect = OSError("lookup unavailable")
        self._assert_collective_errors(self._match(wrappers), groups)

    def test_component_failure_does_not_skip_match_preflight(self):
        wrappers, groups = self._ranks()
        wrappers[0]._components[
            0
        ].build_external_linker_transfer.side_effect = ValueError("bad layout")
        self._assert_collective_errors(self._match(wrappers), groups)

    def test_tail_hash_failure_does_not_skip_match_preflight(self):
        wrappers, groups = self._ranks()
        wrappers[0]._tail_hashes.side_effect = ValueError("bad hash anchor")
        self._assert_collective_errors(self._match(wrappers), groups)

    def test_pp_boundary_mode_mismatch_is_rejected_on_all_ranks(self):
        wrappers, groups = self._ranks()
        for wrapper in wrappers:
            wrapper.cache.pp_size = 2
        self._assert_collective_errors(self._match(wrappers, known=[None, 4]), groups)

    def test_pp_known_boundary_still_joins_preflight(self):
        wrappers, groups = self._ranks()
        for wrapper in wrappers:
            wrapper.cache.pp_size = 2
        outcomes = self._match(wrappers, known=[4, 4])
        self.assertEqual([o.host_hit_length for o in outcomes], [4, 4])
        self.assertEqual(len(groups[0].history), 1)
        for wrapper in wrappers:
            wrapper.cache_linker.lookup.assert_not_called()

    def test_different_requests_fail_even_with_identical_shapes(self):
        wrappers, groups = self._ranks()
        self._assert_collective_errors(self._match(wrappers, rids=["a", "b"]), groups)

    def test_mixed_validity_skips_lookup_on_every_rank(self):
        wrappers, groups = self._ranks()
        wrappers[0]._components[0].build_external_linker_transfer.side_effect = None
        wrappers[0]._components[0].build_external_linker_transfer.return_value = None
        outcomes = self._match(wrappers)
        self.assertEqual([o.host_hit_length for o in outcomes], [0, 0])
        self.assertEqual(len(groups[0].history), 1)
        for wrapper in wrappers:
            wrapper.cache_linker.lookup.assert_not_called()

    def test_quarantined_slots_stop_all_ranks_without_releasing_ownership(self):
        wrappers, groups = self._ranks()
        retained = {"states": ("uncertain",)}
        wrappers[0].cleanup_quarantine["r"] = retained
        self._assert_collective_errors(self._match(wrappers), groups)
        self.assertIs(wrappers[0].cleanup_quarantine["r"], retained)

    def test_empty_or_disjoint_restorable_sets_are_consistent_misses(self):
        for pages, candidates in ((0, ([], [])), (4, ([1, 3], [2, 4])), (4, ([], [2]))):
            with self.subTest(pages=pages, candidates=candidates):
                wrappers, _ = self._ranks()
                outcomes = self._run(
                    wrappers,
                    lambda rank, w: w._sync_restorable_prefix(
                        candidates[rank], num_pages=pages, device_hit_pages=0
                    ),
                )
                self.assertEqual(outcomes, [0, 0])

    def test_match_geometry_mismatch_is_rejected_before_lookup(self):
        wrappers, groups = self._ranks()
        wrappers[0]._tail_hashes.side_effect = None
        wrappers[0]._tail_hashes.return_value = ["0"]
        self._assert_collective_errors(self._match(wrappers), groups)
        for wrapper in wrappers:
            wrapper.cache_linker.lookup.assert_not_called()

    def test_pp_later_stage_without_boundary_joins_once_and_skips_lookup(self):
        wrappers, groups = self._ranks()
        for wrapper in wrappers:
            wrapper.cache.pp_size = 2
            wrapper.cache.pp_rank = 1
        outcomes = self._match(wrappers)
        self.assertEqual([o.host_hit_length for o in outcomes], [0, 0])
        self.assertEqual(len(groups[0].history), 1)
        for wrapper in wrappers:
            wrapper.cache_linker.lookup.assert_not_called()

    def test_reusable_prefetch_publishes_pp_boundary_without_lookup(self):
        if not hasattr(UnifiedCacheLinkerWrapper, "cancel_waiting_queue_prefetch"):
            self.skipTest("branch has no waiting-queue host prefetch")
        wrappers, groups = self._ranks()
        requests = [
            SimpleNamespace(rid="r", external_cache_hit_length=None) for _ in wrappers
        ]
        for wrapper in wrappers:
            wrapper.cache.pp_size = 2
            wrapper.host_prefetch_hits["r"] = ExternalCacheHitMarker(
                prefix_key=_Key(range(4)),
                tail_hashes=["0", "1"],
                device_hit_len=0,
            )
        outcomes = self._run(
            wrappers, lambda rank, w: w.match(_Key(range(4)), requests[rank], _Result())
        )
        self.assertEqual([o.host_hit_length for o in outcomes], [4, 4])
        self.assertEqual([req.external_cache_hit_length for req in requests], [4, 4])
        self.assertEqual(len(groups[0].history), 1)
        for wrapper in wrappers:
            wrapper.cache_linker.lookup.assert_not_called()
            wrapper.cache_linker.cancel_host_prefetch.assert_not_called()

    def test_prefetch_cancel_failure_does_not_skip_mask_preflight(self):
        if not hasattr(UnifiedCacheLinkerWrapper, "cancel_waiting_queue_prefetch"):
            self.skipTest("branch has no waiting-queue host prefetch")
        wrappers, groups = self._ranks()
        wrappers[0].host_prefetch_hits["r"] = ExternalCacheHitMarker(
            prefix_key=_Key(range(2)),
            tail_hashes=["0"],
            device_hit_len=0,
        )
        wrappers[0].cache_linker.cancel_host_prefetch.side_effect = OSError(
            "cancel unavailable"
        )
        self._assert_collective_errors(self._match(wrappers), groups)

    def test_prefetch_length_mismatch_rechecks_on_every_rank(self):
        if not hasattr(UnifiedCacheLinkerWrapper, "cancel_waiting_queue_prefetch"):
            self.skipTest("branch has no waiting-queue host prefetch")
        wrappers, groups = self._ranks()
        for wrapper, pages in zip(wrappers, (1, 2)):
            wrapper.host_prefetch_hits["r"] = ExternalCacheHitMarker(
                prefix_key=_Key(range(pages * 2)),
                tail_hashes=[str(i) for i in range(pages)],
                device_hit_len=0,
            )
        outcomes = self._match(wrappers)
        self.assertEqual([o.host_hit_length for o in outcomes], [4, 4])
        for wrapper in wrappers:
            self.assertEqual(wrapper.hit_markers["r"].tail_hashes, ["0", "1"])
            wrapper.cache_linker.lookup.assert_called_once()
            wrapper.cache_linker.cancel_host_prefetch.assert_called_once_with("r")


if __name__ == "__main__":
    unittest.main()
