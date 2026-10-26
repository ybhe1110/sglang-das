from array import array
from types import SimpleNamespace

import pytest
import torch

from sglang.srt.managers.schedule_batch import Req
from sglang.srt.disaggregation.utils import DisaggregationCollectiveError
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import PoolName, PoolTransfer
from sglang.srt.mem_cache.memory_pool import ReqToTokenPool
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.swa_memory_pool import SWAKVPool
from sglang.srt.mem_cache.unified_cache.cache_action import (
    ReplaceWriteThroughOnNodeSplit,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.full_component import FullComponent
from sglang.srt.mem_cache.unified_cache.components.swa_component import SWAComponent
from sglang.srt.mem_cache.unified_cache.components.tree_component import (
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    _PrivateLoad,
    UnifiedCacheLinker,
    UnifiedCacheLinkerWrapper,
)
from sglang.srt.mem_cache.common import (
    maybe_cache_unfinished_req,
    _request_kv_segments_excluding_released_private,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.srt.sampling.sampling_params import SamplingParams
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeLinker(UnifiedCacheLinker):
    def __init__(self):
        self.layer_done_counter = object()
        self.restorable = []
        self.lookup_calls = []
        self.queued_loads = {}
        self.queued_offloads = []
        self.completed_loads = []
        self.completed_offloads = []
        self.reset_count = 0
        self.closed = False

    def lookup(self, rid, transfers):
        self.lookup_calls.append((rid, list(transfers)))
        return list(self.restorable)

    def load(self, rid, transfers):
        self.queued_loads[rid] = list(transfers)
        return True

    def start_layer_wise_loading(self):
        return 3

    def cancel_queued_load(self, rid):
        if rid not in self.queued_loads:
            return False
        del self.queued_loads[rid]
        return True

    def num_completed_loads(self):
        return len(self.completed_loads)

    def pop_completed_load(self):
        return self.completed_loads.pop(0)

    def offload(self, transfers):
        self.queued_offloads.append(list(transfers))
        return True

    def num_completed_offloads(self):
        return len(self.completed_offloads)

    def pop_completed_offload(self):
        return self.completed_offloads.pop(0)

    def reset(self):
        self.reset_count += 1

    def close(self):
        self.closed = True


class _MappingRecorder:
    def __init__(self):
        self.mapping = []

    def set_full_to_swa_mapping(self, full, swa):
        self.mapping.append((full.clone(), swa.clone()))


def _cache_for_wrapper(**kwargs):
    defaults = {
        "tree_core": SimpleNamespace(enable_external_cache_linker=False),
        "write_through_threshold": 256,
        "pp_size": 1,
        "pp_rank": 0,
        "pp_group": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def test_cache_linker_attachment_is_backend_independent():
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.tree_core = SimpleNamespace(
        enable_external_cache_linker=False,
        write_through_threshold=256,
    )
    cache.linker = None
    linker = _FakeLinker()

    cache.init_cache_linker(linker)

    assert cache.linker.cache_linker is linker
    assert cache.tree_core.enable_external_cache_linker
    assert cache.write_through_threshold == 1
    assert cache.linker.layer_done_counter is linker.layer_done_counter


def test_restorable_prefix_intersects_sparse_rank_results():
    remote_mask = torch.tensor([0, 0, 1, 0, 0], dtype=torch.int)

    def intersect_remote_mask(mask, op):
        assert op == torch.distributed.ReduceOp.MIN
        mask.copy_(torch.minimum(mask, remote_mask))

    cache = _cache_for_wrapper(_all_reduce_attn_groups=intersect_remote_mask)
    wrapper = UnifiedCacheLinkerWrapper(cache, _FakeLinker())

    hit_pages = wrapper._sync_restorable_prefix([2, 4], num_pages=4, device_hit_pages=0)

    assert hit_pages == 2


@pytest.mark.parametrize("remote_tail_pages", [14, 20])
def test_match_preflight_rejects_rank_shape_mismatch_without_variable_collective(
    remote_tail_pages,
):
    """The 60/12 and 84/12 shapes fail before the page-mask reduce."""

    class _Component:
        def build_external_linker_transfer(self, phase, node, keys):
            assert phase == LinkerTransferPhase.LOOKUP
            return PoolTransfer(name=PoolName.KV, keys=list(keys))

    reduce_shapes = []

    def mismatching_reduce(tensor, op):
        assert op == torch.distributed.ReduceOp.MIN
        reduce_shapes.append((tensor.numel(), tensor.dtype))
        # Simulate two ranks whose page tails are 2 and 14/20 pages long.
        tensor.copy_(
            torch.tensor(
                [8, -8, 0, 0, 2, -remote_tail_pages, 0, -1],
                dtype=torch.int64,
            )
        )

    cache = _cache_for_wrapper(
        page_size=2,
        pp_size=1,
        _components_tuple=(_Component(),),
        _all_reduce_attn_groups=mismatching_reduce,
        get_last_hash_value=lambda node: "ab" * 32,
    )
    backend = _FakeLinker()
    wrapper = UnifiedCacheLinkerWrapper(cache, backend)
    key = RadixKey(array("q", range(8)))
    result = MatchResult(
        device_indices=torch.empty(0, dtype=torch.int64),
        last_device_node=0,
        last_host_node=0,
        best_match_node=0,
    )
    req = SimpleNamespace(rid="shape-mismatch", external_cache_hit_length=None)

    with pytest.raises(DisaggregationCollectiveError, match="geometry"):
        wrapper.match(key, req, result)
    assert backend.lookup_calls == []
    # The preflight is fixed at eight int64 values; no rank-dependent mask
    # tensor (3, 15, or 21 int32 values) is ever submitted.
    assert reduce_shapes == [(8, torch.int64)]


def test_match_preflight_validity_mismatch_falls_back_safely(caplog):
    """A validity-only disagreement skips external lookup without aborting."""

    class _Component:
        def build_external_linker_transfer(self, phase, node, keys):
            return PoolTransfer(name=PoolName.KV, keys=list(keys))

    reduce_shapes = []

    def validity_mismatch_reduce(tensor, op):
        assert op == torch.distributed.ReduceOp.MIN
        reduce_shapes.append(tensor.numel())
        # Geometry is identical (two pages), but one rank cannot build a
        # transfer.  The group must take the safe local-cache fallback.
        tensor.copy_(
            torch.tensor(
                [8, -8, 0, 0, 2, -2, 0, -1], dtype=torch.int64
            )
        )

    cache = _cache_for_wrapper(
        page_size=2,
        pp_size=1,
        _components_tuple=(_Component(),),
        _all_reduce_attn_groups=validity_mismatch_reduce,
        get_last_hash_value=lambda node: "ab" * 32,
    )
    backend = _FakeLinker()
    wrapper = UnifiedCacheLinkerWrapper(cache, backend)
    key = RadixKey(array("q", range(8)))
    result = MatchResult(
        device_indices=torch.empty(0, dtype=torch.int64),
        last_device_node=0,
        last_host_node=0,
        best_match_node=0,
    )
    req = SimpleNamespace(rid="validity-mismatch", external_cache_hit_length=None)

    assert wrapper.match(key, req, result) is result
    assert backend.lookup_calls == []
    assert reduce_shapes == [8]
    assert "External KV match preflight rejected" in caplog.text


@pytest.mark.parametrize("key_len", [0, 4, 26112])
def test_match_preflight_is_called_for_local_early_return(caplog, key_len):
    """A full local hit still participates so peers cannot enter a new call."""

    calls = []

    def identity_reduce(tensor, op):
        calls.append((tensor.numel(), tensor.dtype, op))

    cache = _cache_for_wrapper(
        page_size=2,
        pp_size=1,
        _components_tuple=(),
        _all_reduce_attn_groups=identity_reduce,
    )
    backend = _FakeLinker()
    wrapper = UnifiedCacheLinkerWrapper(cache, backend)
    key = RadixKey(array("q", range(key_len)))
    result = MatchResult(
        device_indices=torch.arange(len(key), dtype=torch.int64),
        last_device_node=0,
        last_host_node=0,
        best_match_node=0,
    )
    req = SimpleNamespace(rid="full-local-hit", external_cache_hit_length=None)

    assert wrapper.match(key, req, result) is result
    assert len(calls) == 1
    assert calls[0][0:2] == (8, torch.int64)
    assert backend.lookup_calls == []
    assert "External KV match preflight rejected" not in caplog.text


@pytest.mark.parametrize("hit_pages", [0, 1, 2])
@pytest.mark.parametrize("device_hit_len", [0, 2, 4])
def test_pp0_queries_and_later_stage_reuses_hit_boundary(hit_pages, device_hit_len):
    class _Component:
        def build_external_linker_transfer(self, phase, node, keys):
            assert phase == LinkerTransferPhase.LOOKUP
            return PoolTransfer(name=PoolName.KV, keys=list(keys))

    def make_cache(pp_rank):
        return _cache_for_wrapper(
            page_size=2,
            pp_size=2,
            pp_rank=pp_rank,
            _components_tuple=(_Component(),),
            _all_reduce_attn_groups=lambda tensor, op: None,
            get_last_hash_value=lambda node: "ab" * 32,
        )

    key = RadixKey(array("q", [1, 2, 3, 4]))
    empty_match = MatchResult(
        device_indices=torch.empty(0, dtype=torch.int64),
        last_device_node=0,
        last_host_node=0,
        best_match_node=0,
    )

    pp0_backend = _FakeLinker()
    pp0_backend.restorable = [hit_pages] if hit_pages else []
    pp0 = UnifiedCacheLinkerWrapper(make_cache(pp_rank=0), pp0_backend)
    pp0_req = SimpleNamespace(rid="rid", external_cache_hit_length=None)
    pp0_result = pp0.match(key, pp0_req, empty_match)

    assert pp0_req.external_cache_hit_length == hit_pages * 2
    assert pp0_result.host_hit_length == hit_pages * 2
    assert len(pp0_backend.lookup_calls) == 1

    pp1_backend = _FakeLinker()
    pp1 = UnifiedCacheLinkerWrapper(make_cache(pp_rank=1), pp1_backend)
    pp1_req = SimpleNamespace(
        rid="rid", external_cache_hit_length=pp0_req.external_cache_hit_length
    )
    local_match = empty_match._replace(device_indices=torch.arange(device_hit_len))
    pp1_result = pp1.match(key, pp1_req, local_match)

    assert pp1_result.host_hit_length == max(0, hit_pages * 2 - device_hit_len)
    assert pp1_backend.lookup_calls == []

    # Repeated matching on PP0 also reuses its query, including a cached miss.
    assert pp0.match(key, pp0_req, local_match).host_hit_length == (
        pp1_result.host_hit_length
    )
    assert len(pp0_backend.lookup_calls) == 1
    assert pp0.has_hit("rid") == (hit_pages * 2 > device_hit_len)


@pytest.fixture
def pp_cache():
    pool = SWAKVPool(
        size=32,
        size_swa=32,
        page_size=1,
        dtype=torch.float32,
        head_num=1,
        head_dim=1,
        swa_attention_layer_ids=[0],
        full_attention_layer_ids=[1],
        device="cpu",
    )
    allocator = SWATokenToKVPoolAllocator(
        32, 32, 1, torch.float32, "cpu", pool, need_sort=False
    )
    req_pool = ReqToTokenPool(2, 16, "cpu", enable_memory_saver=False)
    cache = UnifiedRadixCache(
        CacheInitParams(
            disable=False,
            req_to_token_pool=req_pool,
            token_to_kv_pool_allocator=allocator,
            page_size=1,
            pp_size=2,
            sliding_window_size=4,
            tree_components=(ComponentType.FULL, ComponentType.SWA),
        )
    )
    backend = _FakeLinker()
    backend.restorable = [4]
    cache.init_cache_linker(backend)
    req = Req(
        rid="rid",
        origin_input_text="",
        origin_input_ids=array("q", [1, 2, 3, 4, 5]),
        sampling_params=SamplingParams(temperature=0, max_new_tokens=1),
    )
    req_pool.alloc([req])
    req.init_next_round_input(cache)
    return cache, allocator, req, backend


@pytest.mark.parametrize("overlap", [0, 2, 4])
@pytest.mark.parametrize("outcome", ["finished", "unfinished", "abort"])
def test_pp_load_uses_normal_request_insert_and_release(pp_cache, overlap, outcome):
    cache, allocator, req, backend = pp_cache
    key = RadixKey(req.origin_input_ids)
    loaded, last_node = cache.linker.load_back(req)

    assert len(loaded) == 4
    assert last_node == req.last_node == cache.root_node_handle()
    # Upstream reads req.kv.cache_protected_len; on this branch the field lives
    # directly on Req (ReqKvInfo carries only kv_allocated_len/swa_evicted_seqlen).
    assert req.cache_protected_len == 0
    assert cache.match_prefix(MatchPrefixParams(key=key)).device_indices.numel() == 0

    # A different microbatch may publish overlapping KV before this PP result returns.
    existing = allocator.alloc(overlap)
    if overlap:
        cache.insert(InsertParams(key=key[:overlap], value=existing))
    req.prefix_indices = loaded
    req.set_extend_range(4, 5)
    values = torch.cat([loaded, allocator.alloc(1)])
    cache.req_to_token_pool.write((req.kv.req_pool_idx, slice(0, 5)), values)
    # Upstream drains with linker.drain_loads(1) and a bare rid list. On this
    # branch (#38352) that single call is split into take/commit around the
    # cross-rank MIN reduction, and pop_completed_load carries the verdict, so
    # a completed batch is (rids, success).
    backend.completed_loads.append(([req.rid], True))
    cache.linker.commit_completed_loads(cache.linker.take_completed_loads(1))
    assert allocator.full_available_size() == 32 - 5 - overlap

    if outcome == "unfinished":
        cache.cache_unfinished_req(req)
    else:
        cache.cache_finished_req(
            req, is_insert=outcome == "finished", kv_len_to_handle=5
        )

    matched = cache.match_prefix(MatchPrefixParams(key=key)).device_indices
    expected = (
        existing if outcome == "abort" else torch.cat([existing, values[overlap:]])
    )
    assert torch.equal(matched, expected)
    assert allocator.full_available_size() == 32 - len(expected)
    assert allocator.swa_available_size() == 32 - len(expected)
    if outcome != "abort":
        assert torch.all(allocator.translate_loc_from_full_to_swa(matched) > 0)


@pytest.mark.parametrize("raises", [False, True])
def test_pp_load_queue_failure_releases_full_and_swa_slots(pp_cache, raises):
    cache, allocator, req, backend = pp_cache

    def fail_load(*args):
        if raises:
            # Model a backend that inserted the request before surfacing an
            # exception. cancel_queued_load() must succeed before the private
            # full/SWA slots may be returned.
            backend.queued_loads[req.rid] = [object()]
            raise RuntimeError("load failed")
        return False

    backend.load = fail_load
    with pytest.raises(RuntimeError):
        cache.linker.load_back(req)
    assert allocator.full_available_size() == allocator.swa_available_size() == 32
    assert not cache.linker.pending_loads
    assert not cache.tree_core.root_node.children
    if not raises:
        # A synchronous cache miss falls back to recomputation. Its old
        # private interval is gone from the request row, so the marker must be
        # cleared before request-owned slots are allocated at those positions.
        assert not req.external_kv_private_slots_released


def test_pp_load_exception_with_failed_cancel_retains_private_slots(pp_cache):
    cache, allocator, req, backend = pp_cache

    def fail_load(*args):
        raise RuntimeError("load worker accepted request")

    backend.load = fail_load
    with pytest.raises(RuntimeError):
        cache.linker.load_back(req)

    # A false cancellation result means the backend may still be writing the
    # request-private buffers. Keep ownership until a terminal batch arrives.
    assert req.rid in cache.linker.private_loads
    assert allocator.full_available_size() < 32
    assert allocator.swa_available_size() < 32

    cache.linker.taken_loads = [([req.rid], False)]
    assert cache.linker.commit_completed_loads([False]) == [req.rid]
    assert allocator.full_available_size() == allocator.swa_available_size() == 32


def test_async_offload_pins_node_until_completion():
    class _Component:
        def build_external_linker_transfer(self, phase, node, keys):
            assert phase == LinkerTransferPhase.OFFLOAD
            return PoolTransfer(name=PoolName.KV, keys=["page"])

    linker = _FakeLinker()
    lock_params = object()
    locks = []
    unlocks = []

    def inc_lock_ref(node):
        locks.append(node)
        return SimpleNamespace(to_dec_params=lambda: lock_params)

    node_id = 7
    node = SimpleNamespace(
        id=node_id,
        external_cache_stored=False,
        write_through_pending_id=None,
        detached=False,
    )
    cache = _cache_for_wrapper(
        tree_core=SimpleNamespace(
            enable_external_cache_linker=False,
            mark_write_through_pending=lambda value: setattr(
                node, "write_through_pending_id", value
            ),
        ),
        _components_tuple=(_Component(),),
        inc_lock_ref=inc_lock_ref,
        dec_lock_ref=lambda node, params: unlocks.append((node, params)),
        resolve_node_handle=lambda value: node if value == node_id else None,
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)

    wrapper.offload_nodes([node_id])

    assert locks == [node_id]
    assert node.external_cache_stored
    assert not unlocks

    linker.completed_offloads.append(False)
    completed = wrapper.take_completed_offloads(finish_count=1)
    wrapper.commit_completed_offloads(completed)

    assert not node.external_cache_stored
    assert unlocks == [(node_id, lock_params)]


def test_async_load_pins_node_until_completion():
    linker = _FakeLinker()
    lock_params = object()
    locks = []
    unlocks = []

    def inc_lock_ref(node):
        locks.append(node)
        return SimpleNamespace(to_dec_params=lambda: lock_params)

    node_id = 7
    cache = _cache_for_wrapper(
        inc_lock_ref=inc_lock_ref,
        dec_lock_ref=lambda node, params: unlocks.append((node, params)),
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)

    wrapper._queue_load("rid", node_id, [object()], anchor=0)

    assert locks == [node_id]
    assert not unlocks

    linker.completed_loads.append((["rid"], True))
    wrapper.commit_completed_loads(wrapper.take_completed_loads(1))

    assert unlocks == [(node_id, lock_params)]


def test_release_request_cancels_queued_load():
    linker = _FakeLinker()
    lock_params = object()
    unlocks = []
    cache = _cache_for_wrapper(
        dec_lock_ref=lambda node, params: unlocks.append((node, params))
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper.hit_markers["rid"] = object()
    wrapper.pending_loads["rid"] = (7, lock_params, 0)
    linker.queued_loads["rid"] = [object()]

    wrapper.release_request("rid")

    assert wrapper.hit_markers == {}
    assert wrapper.pending_loads == {}
    assert "rid" not in linker.queued_loads
    assert unlocks == [(7, lock_params)]


def _private_load_wrapper(queued: bool):
    """Build a wrapper with one request-private load and a traceable abort."""

    calls = []

    class _Component:
        def update_external_linker_load(
            self, phase, req, full_transfer, transfer, prefix_len, **kwargs
        ):
            calls.append((phase, req.rid))

    linker = _FakeLinker()
    if queued:
        linker.queued_loads["private-rid"] = [object()]
    wrapper = UnifiedCacheLinkerWrapper.__new__(UnifiedCacheLinkerWrapper)
    wrapper.cache = SimpleNamespace()
    wrapper.cache_linker = linker
    wrapper.hit_markers = {}
    wrapper.pending_loads = {}
    wrapper.failed_chains = {}
    wrapper.taken_loads = []
    wrapper.private_loads = {}
    wrapper.pending_offloads = []
    req = SimpleNamespace(rid="private-rid")
    req.external_kv_private_slots_start = 0
    req.external_kv_private_slots_end = 1
    req.external_kv_private_slots_released = False
    transfer = PoolTransfer(
        name=PoolName.KV,
        keys=["page"],
        device_indices=torch.tensor([3], dtype=torch.int64),
    )
    wrapper.private_loads[req.rid] = _PrivateLoad(
        req=req,
        transfers=[(_Component(), transfer)],
        prev_prefix_len=0,
        prefix_len=1,
        device_indices=transfer.device_indices,
    )
    return wrapper, linker, calls, req


def test_private_slot_range_uses_request_geometry_not_component_index_count():
    """SWA-sized indices must not shorten the FULL request-row exclusion."""
    req = SimpleNamespace()

    UnifiedCacheLinkerWrapper._remember_private_slot_range(req, 1024, 4096)

    assert req.external_kv_private_slots_start == 1024
    assert req.external_kv_private_slots_end == 4096


def test_private_load_abort_before_worker_start_releases_once():
    wrapper, linker, calls, req = _private_load_wrapper(queued=True)

    wrapper.release_request("private-rid")
    wrapper.release_request("private-rid")

    assert linker.queued_loads == {}
    assert wrapper.private_loads == {}
    assert calls == [(ExternalLinkerLoadPhase.ABORT, "private-rid")]
    assert req.external_kv_private_slots_released


def test_private_load_abort_during_worker_waits_for_completion():
    wrapper, linker, calls, req = _private_load_wrapper(queued=False)

    # start_layer_wise_loading moved the backend entry out of its pending map;
    # release_request must not free slots while the worker can still write.
    wrapper.release_request("private-rid")
    assert wrapper.private_loads.keys() == {"private-rid"}
    assert calls == []

    wrapper.taken_loads = [(["private-rid"], False)]
    assert wrapper.commit_completed_loads([False]) == ["private-rid"]
    wrapper.release_request("private-rid")

    assert wrapper.private_loads == {}
    assert calls == [(ExternalLinkerLoadPhase.ABORT, "private-rid")]
    assert req.external_kv_private_slots_released


def test_private_load_success_releases_ownership_without_abort():
    wrapper, _linker, calls, req = _private_load_wrapper(queued=False)
    wrapper.taken_loads = [(["private-rid"], True)]

    assert wrapper.commit_completed_loads([True]) == []
    wrapper.release_request("private-rid")

    assert wrapper.private_loads == {}
    assert calls == []
    assert not req.external_kv_private_slots_released
    assert req.external_kv_private_slots_start is None
    assert req.external_kv_private_slots_end is None


def test_finish_loads_failure_has_the_only_release_owner():
    wrapper, linker, calls, req = _private_load_wrapper(queued=False)
    linker._completions.append((["private-rid"], False))

    batches = wrapper.finish_loads(1)
    assert batches == [(["private-rid"], False)]
    assert wrapper.commit_completed_loads([False]) == ["private-rid"]
    wrapper.release_request("private-rid")

    assert calls == [(ExternalLinkerLoadPhase.ABORT, "private-rid")]
    assert req.external_kv_private_slots_released


def test_released_private_prefix_is_excluded_from_request_cleanup():
    req = SimpleNamespace(
        external_kv_private_slots_start=2,
        external_kv_private_slots_end=6,
        external_kv_private_slots_released=True,
        external_kv_private_swa_slots_released=False,
    )
    row = torch.arange(8, dtype=torch.int64)

    segments = _request_kv_segments_excluding_released_private(req, row, 0, 8)

    assert [(start, values.tolist()) for values, start in segments] == [
        (0, [0, 1]),
        (6, [6, 7]),
    ]


def test_chunked_unfinished_skips_released_private_prefix():
    calls = []
    req = SimpleNamespace(
        external_kv_private_slots_released=True,
        skip_radix_cache_insert=False,
    )
    tree = SimpleNamespace(
        cache_unfinished_req=lambda *args, **kwargs: calls.append(args)
    )

    maybe_cache_unfinished_req(req, tree, chunked=True)

    assert req.skip_radix_cache_insert
    assert calls == []


def test_radix_cache_unfinished_guard_skips_released_private_prefix():
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    req = SimpleNamespace(
        external_kv_private_slots_released=True,
        skip_radix_cache_insert=False,
    )

    # The guard is deliberately before session/tree access; a failed load can
    # reach this method after its request row has already been returned.
    cache.cache_unfinished_req(req, chunked=True)

    assert req.skip_radix_cache_insert


def test_swa_private_abort_is_idempotent():
    freed = []
    allocator = SimpleNamespace(
        swa_attn_allocator=SimpleNamespace(
            free=lambda indices: freed.append(indices.clone())
        )
    )
    component = SWAComponent.__new__(SWAComponent)
    component.cache = SimpleNamespace(token_to_kv_pool_allocator=allocator)
    req = SimpleNamespace(
        rid="swa-private",
        external_kv_private_swa_slots_released=False,
    )
    transfer = PoolTransfer(
        name=PoolName.SWA,
        keys=["page"],
        device_indices=torch.tensor([7, 8], dtype=torch.int64),
    )

    component.update_external_linker_load(
        ExternalLinkerLoadPhase.ABORT,
        req,
        transfer,
        transfer,
        prefix_len=2,
    )
    component.update_external_linker_load(
        ExternalLinkerLoadPhase.ABORT,
        req,
        transfer,
        transfer,
        prefix_len=2,
    )

    assert len(freed) == 1
    assert freed[0].tolist() == [7, 8]


def test_failed_offload_rolls_back_split_fragments():
    class _Component:
        def build_external_linker_transfer(self, phase, node, keys):
            return PoolTransfer(name=PoolName.KV, keys=["page"])

    linker = _FakeLinker()
    lock_params = object()
    unlocks = []
    child = SimpleNamespace(
        id=7,
        external_cache_stored=False,
        write_through_pending_id=None,
        detached=False,
    )
    parent = SimpleNamespace(
        id=8,
        external_cache_stored=False,
        write_through_pending_id=None,
        detached=False,
    )
    nodes = {child.id: child, parent.id: parent}

    def mark_pending(node_id):
        nodes[node_id].write_through_pending_id = node_id

    cache = _cache_for_wrapper(
        tree_core=SimpleNamespace(
            enable_external_cache_linker=False,
            mark_write_through_pending=mark_pending,
        ),
        _components_tuple=(_Component(),),
        inc_lock_ref=lambda node_id: SimpleNamespace(to_dec_params=lambda: lock_params),
        dec_lock_ref=lambda node_id, params: unlocks.append((node_id, params)),
        resolve_node_handle=nodes.__getitem__,
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper.offload_nodes([child.id])

    parent.external_cache_stored = child.external_cache_stored
    parent.write_through_pending_id = child.write_through_pending_id
    wrapper.replace_pending_offload_node(child.id, child.id, [parent.id, child.id])
    linker.completed_offloads.append(False)
    wrapper.commit_completed_offloads(wrapper.take_completed_offloads(finish_count=1))

    assert not parent.external_cache_stored
    assert not child.external_cache_stored
    assert parent.write_through_pending_id is None
    assert child.write_through_pending_id is None
    assert unlocks == [(child.id, lock_params)]


def test_split_action_retargets_pending_external_offload():
    calls = []
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.linker = SimpleNamespace(
        replace_pending_offload_node=lambda *args: calls.append(("linker", *args))
    )
    cache._replace_pending_write_through_node = lambda *args: calls.append(
        ("hicache", *args)
    )
    action = ReplaceWriteThroughOnNodeSplit(
        ack_id=7,
        old_node_id=7,
        new_node_id=8,
        new_child_node_id=7,
    )

    cache._apply_cache_action(action)

    assert calls == [
        ("hicache", 7, 7, [8, 7]),
        ("linker", 7, 7, [8, 7]),
    ]


def test_reset_quiesces_backend_before_releasing_pending_locks():
    class _Component:
        def build_external_linker_transfer(self, phase, node, keys):
            return PoolTransfer(name=PoolName.KV, keys=["page"])

    events = []

    class _QuiescentFakeLinker(_FakeLinker):
        def reset(self):
            events.append("backend")
            super().reset()

    linker = _QuiescentFakeLinker()
    node = SimpleNamespace(
        id=7,
        external_cache_stored=False,
        write_through_pending_id=None,
        detached=False,
    )
    cache = _cache_for_wrapper(
        tree_core=SimpleNamespace(
            enable_external_cache_linker=False,
            mark_write_through_pending=lambda value: setattr(
                node, "write_through_pending_id", value
            ),
        ),
        _components_tuple=(_Component(),),
        inc_lock_ref=lambda node_id: SimpleNamespace(to_dec_params=object),
        dec_lock_ref=lambda node_id, params: events.append(("unlock", node_id)),
        resolve_node_handle=lambda node_id: node,
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper._queue_load("rid", node.id, [object()], anchor=0)
    wrapper.offload_nodes([node.id])

    wrapper.reset()

    assert events == ["backend", ("unlock", node.id), ("unlock", node.id)]
    assert wrapper.pending_loads == {}
    assert wrapper.pending_offloads == []
    assert not node.external_cache_stored
    assert node.write_through_pending_id is None


def test_close_quiesces_backend_before_releasing_pending_loads():
    events = []

    class _ClosingFakeLinker(_FakeLinker):
        def close(self):
            events.append("backend")
            super().close()

    linker = _ClosingFakeLinker()
    cache = _cache_for_wrapper(
        dec_lock_ref=lambda node_id, params: events.append(("unlock", node_id))
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper.pending_loads["rid"] = (7, object(), 0)

    wrapper.close()

    assert events == ["backend", ("unlock", 7)]
    assert linker.closed
    assert wrapper.pending_loads == {}


def test_check_hicache_events_commits_common_rank_results():
    committed = []
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache._failed_linker_rids = []

    def commit_loads(successes):
        committed.append(("load", successes))
        return []

    cache.linker = SimpleNamespace(
        num_completed_loads=lambda: 1,
        take_completed_loads=lambda count: [True] * count,
        commit_completed_loads=commit_loads,
        num_completed_offloads=lambda: 3,
        take_completed_offloads=lambda count: [True] * count,
        commit_completed_offloads=committed.append,
    )

    reduce_calls = 0

    def reduce_to_common_state(value, op):
        nonlocal reduce_calls
        assert op == torch.distributed.ReduceOp.MIN
        reduce_calls += 1
        if reduce_calls == 1:
            value.copy_(torch.tensor([1, 1]))
        else:
            value.fill_(0)

    cache._all_reduce_attn_groups = reduce_to_common_state

    cache.check_hicache_events()

    assert committed == [("load", [False]), [False]]


def test_completed_external_load_preflight_rejects_rid_order_mismatch():
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)

    def mismatch(payload, op):
        # Simulate another rank reporting a different first batch descriptor.
        payload[0] = payload[0] - 1

    cache._all_reduce_attn_groups = mismatch
    with pytest.raises(DisaggregationCollectiveError):
        cache._preflight_linker_load_batches(
            [(["rid-a", "rid-b"], True), (["rid-c"], False)]
        )


def test_component_commit_keeps_only_adopted_pages():
    mapping = _MappingRecorder()
    cache = _cache_for_wrapper(
        page_size=2,
        token_to_kv_pool_allocator=SimpleNamespace(
            set_full_to_swa_mapping=mapping.set_full_to_swa_mapping
        ),
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, _FakeLinker())
    full_component = FullComponent.__new__(FullComponent)
    full_component.cache = cache
    full_component.component_type = ComponentType.FULL
    swa_component = SWAComponent.__new__(SWAComponent)
    swa_component.cache = cache
    swa_component.component_type = ComponentType.SWA
    full = PoolTransfer(
        name=PoolName.KV,
        keys=["a", "b", "c", "d"],
        device_indices=torch.tensor([100, 101, 102, 103, 104, 105, 106, 107]),
    )
    canonical_tail = torch.tensor([10, 11, 102, 103, 14, 15, 106, 107])
    swa = PoolTransfer(
        name=PoolName.SWA,
        keys=["a", "b", "c", "d"],
        device_indices=torch.tensor([200, 201, 202, 203, 204, 205, 206, 207]),
    )
    insert_result = InsertResult(
        prefix_len=0,
        adopted_ranges={
            ComponentType.FULL: [(2, 4), (6, 8)],
            ComponentType.SWA: [(2, 4), (6, 8)],
        },
    )

    filtered = wrapper._update_load(
        ExternalLinkerLoadPhase.COMMIT,
        SimpleNamespace(),
        [(full_component, full), (swa_component, swa)],
        prefix_len=8,
        insert_result=insert_result,
        canonical_full=canonical_tail,
    )

    assert filtered == [full, swa]
    assert full.keys == ["b", "d"]
    assert full.device_indices.tolist() == [102, 103, 106, 107]
    assert swa.keys == ["b", "d"]
    assert swa.device_indices.tolist() == [202, 203, 206, 207]
    mapped_full, mapped_swa = mapping.mapping[0]
    assert mapped_full.tolist() == [102, 103, 106, 107]
    assert mapped_swa.tolist() == [202, 203, 206, 207]


def test_release_donated_swa_slots_skips_released_private_interval():
    """Normal SWA cleanup must not free linker-released private positions."""
    from types import SimpleNamespace

    import torch

    from sglang.srt.mem_cache.common import _release_donated_swa_slots

    freed = []
    allocator = SimpleNamespace(
        free_swa=lambda indices: freed.append(indices.detach().cpu().tolist())
    )
    tree_cache = SimpleNamespace(
        token_to_kv_pool_allocator=allocator,
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.tensor([[10, 11, 12, 13, 14, 15, 16, 17]])
        ),
    )
    req = SimpleNamespace(
        dsv4_donated_swa_len=8,
        kv=object(),
        req_pool_idx=0,
        external_kv_private_slots_start=2,
        external_kv_private_slots_end=6,
        external_kv_private_slots_released=True,
    )

    _release_donated_swa_slots(req, tree_cache)
    _release_donated_swa_slots(req, tree_cache)

    assert freed == [[10, 11, 16, 17]]
    assert req.dsv4_donated_swa_len == 0


def test_release_donated_swa_slots_skips_fully_private_prefix():
    """A donated prefix wholly owned by a failed linker load is not freed."""
    from types import SimpleNamespace

    import torch

    from sglang.srt.mem_cache.common import _release_donated_swa_slots

    freed = []
    tree_cache = SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(
            free_swa=lambda indices: freed.append(indices.detach().cpu().tolist())
        ),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.tensor([[20, 21, 22, 23]])
        ),
    )
    req = SimpleNamespace(
        dsv4_donated_swa_len=4,
        kv=object(),
        req_pool_idx=0,
        external_kv_private_slots_start=0,
        external_kv_private_slots_end=8,
        external_kv_private_slots_released=True,
    )

    _release_donated_swa_slots(req, tree_cache)

    assert freed == []
    assert req.dsv4_donated_swa_len == 0


def test_release_donated_swa_slots_honors_swa_release_marker():
    """The SWA component's own release marker also prevents a second free."""
    from types import SimpleNamespace

    import torch

    from sglang.srt.mem_cache.common import _release_donated_swa_slots

    freed = []
    tree_cache = SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(
            free_swa=lambda indices: freed.append(indices.detach().cpu().tolist())
        ),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.tensor([[30, 31, 32, 33]])
        ),
    )
    req = SimpleNamespace(
        dsv4_donated_swa_len=4,
        kv=object(),
        req_pool_idx=0,
        external_kv_private_slots_start=1,
        external_kv_private_slots_end=3,
        external_kv_private_slots_released=False,
        external_kv_private_swa_slots_released=True,
    )

    _release_donated_swa_slots(req, tree_cache)

    assert freed == [[30, 33]]


def test_release_donated_swa_slots_keeps_non_overlapping_prefix_unchanged():
    """A private interval after the donation must not alter normal cleanup."""
    from sglang.srt.mem_cache.common import _release_donated_swa_slots

    freed = []
    tree_cache = SimpleNamespace(
        token_to_kv_pool_allocator=SimpleNamespace(
            free_swa=lambda indices: freed.append(indices.detach().cpu().tolist())
        ),
        req_to_token_pool=SimpleNamespace(
            req_to_token=torch.tensor([[40, 41]])
        ),
    )
    req = SimpleNamespace(
        dsv4_donated_swa_len=2,
        kv=object(),
        req_pool_idx=0,
        external_kv_private_slots_start=4,
        external_kv_private_slots_end=8,
        external_kv_private_slots_released=True,
    )

    _release_donated_swa_slots(req, tree_cache)

    assert freed == [[40, 41]]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
