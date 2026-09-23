from array import array
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch

from sglang.srt.managers.schedule_batch import Req, ReqKvInfo
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    InsertParams,
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
    LayerWiseLoadCounter,
    UnifiedCacheLinker,
    UnifiedCacheLinkerWrapper,
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
        self.load_landed = True
        self.finished_loads = []
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

    def finish_layer_wise_loading(self, counter_index):
        self.finished_loads.append(counter_index)
        return self.load_landed

    def cancel_queued_load(self, rid):
        if rid not in self.queued_loads:
            return False
        del self.queued_loads[rid]
        return True

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
    assert cache.linker.start_layer_wise_loading() == 3
    assert cache.linker.finish_loads() == []
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
            raise RuntimeError("load failed")
        return False

    backend.load = fail_load
    if raises:
        with pytest.raises(RuntimeError):
            cache.linker.load_back(req)
    else:
        # A refused load is undone before PREPARE and falls back to recompute.
        loaded, last_node = cache.linker.load_back(req)
        assert loaded.numel() == 0
        assert last_node == cache.root_node_handle()
    assert allocator.full_available_size() == allocator.swa_available_size() == 32
    assert not cache.linker.inflight_load_rids
    assert not cache.tree_core.root_node.children


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


def test_layer_counter_reports_failure_after_forward_instead_of_raising():
    ready = []
    counter = LayerWiseLoadCounter(
        2, on_layer_ready=lambda index, layer: ready.append((index, layer))
    )
    failed = counter.update_producer()
    counter.set_consumer(failed)
    counter.complete(failed, 0)
    counter.fail(failed, RuntimeError("storage read failed"))

    counter.wait_until(0)
    counter.wait_until(1)

    # A dedup source still broadcasts the failed layer its peers wait for.
    assert ready == [(failed, 0), (failed, 1)]
    assert counter.finish(failed) is False
    assert counter.finish(failed) is True

    landed = counter.update_producer()
    counter.complete(landed, 0)
    counter.complete(landed, 1)
    assert counter.finish(landed) is True


@pytest.mark.parametrize("peer_landed", [True, False], ids=["landed", "peer-failed"])
@pytest.mark.parametrize("local_landed", [True, False], ids=["local", "local-failed"])
def test_finish_loads_agrees_on_the_verdict_across_ranks(local_landed, peer_landed):
    def reduce_with_peer(value, op):
        assert op == torch.distributed.ReduceOp.MIN
        value.copy_(torch.minimum(value, torch.tensor([int(peer_landed)])))

    cache = _cache_for_wrapper(_all_reduce_attn_groups=reduce_with_peer)
    linker = _FakeLinker()
    linker.load_landed = local_landed
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper.inflight_load_rids = ["a", "b"]
    assert wrapper.start_layer_wise_loading() == 3

    failed = wrapper.finish_loads()

    assert failed == ([] if local_landed and peer_landed else ["a", "b"])
    assert linker.finished_loads == [3]
    assert wrapper.inflight_load_rids == []
    assert wrapper.inflight_load_index == -1

    # A batch that loaded nothing skips the collective entirely.
    cache._all_reduce_attn_groups = MagicMock()
    assert wrapper.finish_loads() == []
    cache._all_reduce_attn_groups.assert_not_called()


def test_release_request_cancels_only_a_queued_load():
    linker = _FakeLinker()
    wrapper = UnifiedCacheLinkerWrapper(_cache_for_wrapper(), linker)
    wrapper.hit_markers["queued"] = object()
    wrapper.inflight_load_rids = ["queued", "started"]
    linker.queued_loads["queued"] = [object()]

    wrapper.release_request("queued")
    wrapper.release_request("started")

    assert wrapper.hit_markers == {}
    assert "queued" not in linker.queued_loads
    # A started load keeps writing until finish_loads waits for it.
    assert wrapper.inflight_load_rids == ["started"]


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
    )
    parent = SimpleNamespace(
        id=8,
        external_cache_stored=False,
        write_through_pending_id=None,
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


def test_reset_quiesces_backend_before_releasing_pending_state():
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
    wrapper.inflight_load_rids = ["rid"]
    wrapper.inflight_load_index = 3
    wrapper.offload_nodes([node.id])

    wrapper.reset()

    assert events == ["backend", ("unlock", node.id)]
    assert wrapper.inflight_load_rids == []
    assert wrapper.inflight_load_index == -1
    assert wrapper.pending_offloads == []
    assert not node.external_cache_stored
    assert node.write_through_pending_id is None


def test_close_quiesces_backend_before_releasing_pending_offloads():
    events = []

    class _ClosingFakeLinker(_FakeLinker):
        def close(self):
            events.append("backend")
            super().close()

    class _Component:
        def build_external_linker_transfer(self, phase, node, keys):
            return PoolTransfer(name=PoolName.KV, keys=["page"])

    linker = _ClosingFakeLinker()
    node = SimpleNamespace(
        id=7,
        external_cache_stored=False,
        write_through_pending_id=None,
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
    wrapper.inflight_load_rids = ["rid"]
    wrapper.offload_nodes([node.id])

    wrapper.close()

    assert events == ["backend", ("unlock", node.id)]
    assert linker.closed
    assert wrapper.inflight_load_rids == []
    assert wrapper.pending_offloads == []


def test_check_hicache_events_commits_common_rank_results():
    committed = []
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.linker = SimpleNamespace(
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
            value.copy_(torch.tensor([1]))
        else:
            value.fill_(0)

    cache._all_reduce_attn_groups = reduce_to_common_state

    cache.check_hicache_events()

    assert committed == [[False]]


def test_component_load_prepare_maps_private_window_and_abort_frees_it():
    mapping = _MappingRecorder()
    freed = []
    cache = _cache_for_wrapper(
        page_size=2,
        is_swa_enabled=True,
        token_to_kv_pool_allocator=SimpleNamespace(
            set_full_to_swa_mapping=mapping.set_full_to_swa_mapping,
            full_attn_allocator=SimpleNamespace(
                free=lambda indices: freed.append(("full", indices.tolist()))
            ),
            swa_attn_allocator=SimpleNamespace(
                free=lambda indices: freed.append(("swa", indices.tolist()))
            ),
        ),
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, _FakeLinker())
    full_component = FullComponent.__new__(FullComponent)
    full_component.cache = cache
    full_component.component_type = ComponentType.FULL
    swa_component = SWAComponent.__new__(SWAComponent)
    swa_component.cache = cache
    swa_component.component_type = ComponentType.SWA
    swa_component.sliding_window_size = 4
    full = PoolTransfer(
        name=PoolName.KV,
        keys=["a", "b", "c", "d"],
        device_indices=torch.tensor([100, 101, 102, 103, 104, 105, 106, 107]),
    )
    swa = PoolTransfer(
        name=PoolName.SWA,
        keys=["c", "d"],
        device_indices=torch.tensor([200, 201, 202, 203]),
    )
    transfers = [(full_component, full), (swa_component, swa)]
    req = SimpleNamespace(kv=ReqKvInfo())

    prepared = wrapper._update_load(
        ExternalLinkerLoadPhase.PREPARE, req, transfers, prefix_len=8
    )

    assert prepared == [full, swa]
    ((mapped_full, mapped_swa),) = mapping.mapping
    assert mapped_full.tolist() == [104, 105, 106, 107]
    assert mapped_swa.tolist() == [200, 201, 202, 203]
    assert req.kv.swa_evicted_seqlen == 4
    assert freed == []

    aborted = wrapper._update_load(
        ExternalLinkerLoadPhase.ABORT, req, transfers, prefix_len=8
    )

    assert aborted == []
    assert freed == [
        ("swa", [200, 201, 202, 203]),
        ("full", [100, 101, 102, 103, 104, 105, 106, 107]),
    ]


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
