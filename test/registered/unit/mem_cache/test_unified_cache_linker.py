import sys
import unittest
from array import array
from collections import defaultdict
from dataclasses import replace
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import test_unified_radix_cache_unittest as shared_cache_suite
import torch
from sglang.srt.managers.schedule_batch import ReqKvInfo
from sglang.srt.mem_cache.allocator.swa import SWATokenToKVPoolAllocator
from sglang.srt.mem_cache.base_prefix_cache import (
    InitLoadBackParams,
    InsertResult,
    MatchPrefixParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolHitPolicy,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.cache_action import (
    ReplaceWriteThroughOnNodeSplit,
)
from sglang.srt.mem_cache.unified_cache.component_type import ComponentType
from sglang.srt.mem_cache.unified_cache.components.base import (
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
)
from sglang.srt.mem_cache.unified_cache.components.swa import SWAComponent
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    ExternalCacheHitMarker,
    UnifiedCacheLinker,
    UnifiedCacheLinkerWrapper,
    _PrivateLoad,
)
from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache
from sglang.test.ci.ci_register import register_cpu_ci, register_cuda_ci
from test_unified_radix_cache_unittest import (
    CacheConfig,
    _device_lock_ref,
    _device_value,
    _InsertWalkSuite,
    _node_children,
    build_fixture,
)

register_cpu_ci(est_time=10, suite="base-a-test-cpu")
register_cuda_ci(est_time=100, stage="base-b", runner_config="1-gpu-small")


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

    def finish_loads(self, consumer_index):
        pass

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


class _FakeExternalTreeCore:
    def __init__(self, nodes=None, offload_transfers=None):
        self.enable_external_cache_linker = False
        self.nodes = nodes or {}
        self.offload_transfers = offload_transfers or [
            PoolTransfer(name=PoolName.KV, keys=["page"])
        ]

    def build_external_linker_offload_transfers(self, node_id):
        node = self.nodes[node_id]
        if node.external_cache_stored or node.write_through_pending_id is not None:
            return None
        return list(self.offload_transfers)

    def mark_external_linker_offload_pending(self, node_id):
        node = self.nodes[node_id]
        assert (
            not node.external_cache_stored and node.write_through_pending_id is None
        ), "invalid external offload state"
        node.write_through_pending_id = node_id

    def finish_external_linker_offload(self, node_ids, ack_id, success):
        nodes = [self.nodes[node_id] for node_id in node_ids]
        assert all(
            node.write_through_pending_id == ack_id for node in nodes
        ), "invalid external offload state"
        for node in nodes:
            node.write_through_pending_id = None
            node.external_cache_stored |= success


def _cache_for_wrapper(**kwargs):
    defaults = {
        "_components_tuple": (),
        "components": {},
        "tree_core": SimpleNamespace(enable_external_cache_linker=False),
        "tree_components": (ComponentType.FULL,),
        "write_through_threshold": 256,
        "_all_reduce_attn_groups": lambda tensor, op: None,
        "pp_size": 1,
        "pp_group": None,
    }
    defaults.update(kwargs)
    return SimpleNamespace(**defaults)


def _swa_allocator(swa_req_ring):
    if swa_req_ring is None:
        return SimpleNamespace()
    allocator = SWATokenToKVPoolAllocator.__new__(SWATokenToKVPoolAllocator)
    allocator._swa_req_ring = swa_req_ring
    return allocator


def test_cache_linker_attachment_is_backend_independent():
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.tree_core = SimpleNamespace(
        enable_external_cache_linker=False,
        write_through_threshold=256,
    )
    cache.tree_components = (ComponentType.FULL,)
    cache.linker = None
    cache._components_tuple = ()
    cache.components = {}
    linker = _FakeLinker()

    cache.init_cache_linker(linker)

    assert cache.linker.cache_linker is linker
    assert cache.tree_core.enable_external_cache_linker
    assert cache.write_through_threshold == 1
    assert cache.linker.layer_done_counter is linker.layer_done_counter


@pytest.mark.parametrize("component_type", [ComponentType.MAMBA, ComponentType.C128])
def test_cache_linker_rejects_unsupported_tree_components(component_type):
    cache = _cache_for_wrapper(tree_components=(ComponentType.FULL, component_type))

    with pytest.raises(ValueError, match=component_type.name):
        UnifiedCacheLinkerWrapper(cache, _FakeLinker())

    assert not cache.tree_core.enable_external_cache_linker


class _InMemoryUnifiedCacheLinker(UnifiedCacheLinker):
    """Controllable transport for shared Python/Rust TreeCore tests."""

    def __init__(self, stored_keys=None):
        self.layer_done_counter = object()
        self.stored_keys = defaultdict(set) if stored_keys is None else stored_keys
        self.lookup_calls = []
        self.offload_calls = []
        self.pending_offloads = []
        self.queued_loads = {}
        self.started_loads = []
        self.completed_loads = []
        self.completed_offloads = []

    @staticmethod
    def _clone_transfer(transfer):
        return replace(
            transfer,
            host_indices=(
                None if transfer.host_indices is None else transfer.host_indices.clone()
            ),
            device_indices=(
                None
                if transfer.device_indices is None
                else transfer.device_indices.clone()
            ),
            keys=None if transfer.keys is None else list(transfer.keys),
            nodes_to_load=(
                None if transfer.nodes_to_load is None else list(transfer.nodes_to_load)
            ),
        )

    @classmethod
    def _clone_transfers(cls, transfers):
        return [cls._clone_transfer(transfer) for transfer in transfers]

    def lookup(self, rid, transfers):
        transfers = self._clone_transfers(transfers)
        self.lookup_calls.append((rid, transfers))
        by_pool = {transfer.name: transfer for transfer in transfers}
        kv = by_pool.get(PoolName.KV)
        if kv is None or not kv.keys:
            return []

        restorable = []
        for prefix_pages in range(1, len(kv.keys) + 1):
            if not set(kv.keys[:prefix_pages]) <= self.stored_keys[PoolName.KV]:
                continue
            for transfer in transfers:
                if transfer.name == PoolName.KV:
                    continue
                if transfer.hit_policy == PoolHitPolicy.TRAILING_PAGES:
                    window_pages = max(1, len(transfer.keys or ()))
                    required = kv.keys[
                        max(0, prefix_pages - window_pages) : prefix_pages
                    ]
                else:
                    required = kv.keys[:prefix_pages]
                if not set(required) <= self.stored_keys[transfer.name]:
                    break
            else:
                restorable.append(prefix_pages)
        return restorable

    def load(self, rid, transfers):
        self.queued_loads[rid] = self._clone_transfers(transfers)
        return True

    def start_layer_wise_loading(self):
        if not self.queued_loads:
            return -1
        rids = list(self.queued_loads)
        self.started_loads.append(rids)
        return len(self.started_loads) - 1

    def cancel_queued_load(self, rid):
        return self.queued_loads.pop(rid, None) is not None

    def finish_loads(self, consumer_index):
        pass

    def num_completed_loads(self):
        return len(self.completed_loads)

    def pop_completed_load(self):
        rids = self.completed_loads.pop(0)
        for rid in rids:
            self.queued_loads.pop(rid, None)
        return [(rid, True) for rid in rids]

    def complete_started_loads(self):
        self.completed_loads.append(self.started_loads[-1])

    def offload(self, transfers):
        transfers = self._clone_transfers(transfers)
        self.offload_calls.append(transfers)
        self.pending_offloads.append(transfers)
        return True

    def num_completed_offloads(self):
        return len(self.completed_offloads)

    def pop_completed_offload(self):
        return self.completed_offloads.pop(0)

    def complete_next_offload(self, success):
        transfers = self.pending_offloads.pop(0)
        if success:
            for transfer in transfers:
                self.stored_keys[transfer.name].update(transfer.keys or ())
        self.completed_offloads.append(success)

    def reset(self):
        self.queued_loads.clear()
        self.pending_offloads.clear()
        self.completed_loads.clear()
        self.completed_offloads.clear()

    def close(self):
        self.reset()


class _TreeCoreBackendTestMixin:
    tree_core_backend = "python"

    def setUp(self):
        previous = shared_cache_suite._TREE_CORE_TEST_BACKEND
        self.addCleanup(
            setattr,
            shared_cache_suite,
            "_TREE_CORE_TEST_BACKEND",
            previous,
        )
        shared_cache_suite._TREE_CORE_TEST_BACKEND = self.tree_core_backend
        super().setUp()


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestUnifiedCacheLinkerPythonBackend(_TreeCoreBackendTestMixin, _InsertWalkSuite):
    def test_full_offload_load_round_trip_and_dedup(self):
        cfg = CacheConfig(page_size=2, kv_size=64, max_context_len=64)
        self.cfg = cfg
        stored_keys = defaultdict(set)

        producer, producer_allocator, producer_req_pool = build_fixture(cfg)
        producer_linker = _InMemoryUnifiedCacheLinker(stored_keys)
        producer.init_cache_linker(producer_linker)
        tokens = list(range(1, 9))

        inserted = self._insert(producer, producer_allocator, producer_req_pool, tokens)
        self.assertEqual(len(producer_linker.offload_calls), 1)
        (kv_offload,) = producer_linker.offload_calls[0]
        self.assertEqual(kv_offload.name, PoolName.KV)
        self.assertEqual(
            kv_offload.keys,
            producer.tree_core.get_hash_values(inserted.last_device_node),
        )
        self.assertTrue(
            torch.equal(
                kv_offload.device_indices,
                _device_value(producer, inserted.last_device_node, ComponentType.FULL),
            )
        )
        self.assertEqual(
            _device_lock_ref(producer, inserted.last_device_node, ComponentType.FULL),
            1,
        )
        self.assertFalse(
            producer.tree_core.is_external_cache_stored(inserted.last_device_node)
        )
        self.assertEqual(
            producer.tree_core.get_write_through_pending_id(inserted.last_device_node),
            inserted.last_device_node,
        )

        self._insert(producer, producer_allocator, producer_req_pool, tokens)
        self.assertEqual(len(producer_linker.offload_calls), 1)

        producer_linker.complete_next_offload(True)
        producer.check_hicache_events()
        self.assertEqual(
            _device_lock_ref(producer, inserted.last_device_node, ComponentType.FULL),
            0,
        )
        self.assertTrue(
            producer.tree_core.is_external_cache_stored(inserted.last_device_node)
        )
        self._insert(producer, producer_allocator, producer_req_pool, tokens)
        self.assertEqual(len(producer_linker.offload_calls), 1)

        consumer, _, consumer_req_pool = build_fixture(cfg)
        consumer_linker = _InMemoryUnifiedCacheLinker(stored_keys)
        consumer.init_cache_linker(consumer_linker)
        req = self._make_req(consumer_req_pool)
        match = consumer.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)), req=req)
        )
        self.assertEqual(match.device_prefix_len, 0)
        self.assertEqual(match.host_hit_length, len(tokens))
        self._apply_match_to_req(req, match)

        loaded_len, loaded_node = consumer.init_load_back(
            InitLoadBackParams(
                best_match_node=match.best_match_node,
                host_hit_length=match.host_hit_length,
                req=req,
            )
        )
        self.assertEqual(loaded_len, len(tokens))
        self.assertEqual(loaded_node, consumer.root_node_handle())
        (kv_load,) = consumer_linker.queued_loads[req.rid]
        self.assertEqual(kv_load.name, PoolName.KV)
        self.assertEqual(kv_load.keys, kv_offload.keys)
        self.assertEqual(
            consumer._match_tree(
                MatchPrefixParams(key=RadixKey(array("q", tokens)))
            ).device_prefix_len,
            0,
        )

        indices = consumer.linker.prefix_device_indices(req)
        consumer_req_pool.req_to_token[req.kv.req_pool_idx, : len(tokens)] = indices
        req.origin_input_ids = req.full_untruncated_fill_ids = array("q", tokens)
        req.kv.kv_allocated_len = req.kv.kv_committed_len = len(tokens)
        consumer.linker.adopt_request(req)
        index = consumer.ready_to_load_host_cache()
        consumer_linker.complete_started_loads()
        assert consumer.finish_external_linker_loads(index) == []
        # Completion alone does not publish any node.
        assert (
            consumer._match_tree(
                MatchPrefixParams(key=RadixKey(array("q", tokens)))
            ).device_prefix_len
            == 0
        )
        consumer.checkpoint(req, up_to=len(tokens))
        final_match = consumer.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)))
        )
        self.assertEqual(final_match.device_prefix_len, len(tokens))
        consumer.sanity_check()

    def test_failed_private_swa_load_frees_each_pool_once_without_checkpoint(self):
        from sglang.srt.mem_cache.common import release_kv_cache

        cfg = CacheConfig(
            page_size=1,
            components=(ComponentType.FULL, ComponentType.SWA),
            sliding_window_size=2,
            kv_size=64,
            max_context_len=64,
        )
        self.cfg = cfg
        for chunked in (False, True):
            with self.subTest(chunked=chunked):
                cache, allocator, pool = build_fixture(cfg)
                backend = _FakeLinker()
                cache.init_cache_linker(backend)
                req = self._make_req(pool)
                req.last_node = cache.root_node_handle()
                req.origin_input_ids = req.full_untruncated_fill_ids = array(
                    "q", range(6)
                )
                req.prefix_len = 0
                before = (
                    allocator.full_available_size(),
                    allocator.swa_available_size(),
                )
                cache.linker.hit_markers[req.rid] = ExternalCacheHitMarker(
                    RadixKey(req.origin_input_ids), [str(i) for i in range(6)], 0
                )
                loaded, _ = cache.linker.load_back(req)
                self.assertEqual(loaded, 6)
                pool.write(
                    (req.kv.req_pool_idx, slice(0, 6)),
                    cache.linker.prefix_device_indices(req),
                )
                req.kv.kv_allocated_len = req.kv.kv_committed_len = 6
                cache.linker.adopt_request(req)
                index = cache.ready_to_load_host_cache()
                backend.completed_loads.append([(req.rid, False)])
                self.assertEqual(cache.finish_external_linker_loads(index), [req])
                # A chunk result arriving after failure must not republish freed KV.
                if chunked:
                    cache.checkpoint(req, up_to=6)
                    cache.advance_unpublished_req(req)
                release_kv_cache(req, cache, checkpoint=False)
                cache.linker.release_request(req.rid)
                self.assertEqual(
                    (allocator.full_available_size(), allocator.swa_available_size()),
                    before,
                )
                match = cache._match_tree(
                    MatchPrefixParams(key=RadixKey(req.origin_input_ids))
                )
                self.assertEqual(match.device_prefix_len, 0)
                cache.sanity_check()

    def test_eagle_lookup_uses_bigram_tail_hashes(self):
        cfg = CacheConfig(
            page_size=2,
            is_eagle=True,
            kv_size=64,
            max_context_len=64,
        )
        self.cfg = cfg
        stored_keys = defaultdict(set)
        tokens = list(range(1, 10))

        producer, producer_allocator, producer_req_pool = build_fixture(cfg)
        producer_linker = _InMemoryUnifiedCacheLinker(stored_keys)
        producer.init_cache_linker(producer_linker)
        self._insert(producer, producer_allocator, producer_req_pool, tokens)
        self.assertEqual(len(producer_linker.offload_calls), 1)
        (kv_offload,) = producer_linker.offload_calls[0]
        self.assertEqual(len(kv_offload.keys), 4)
        producer_linker.complete_next_offload(True)
        producer.check_hicache_events()

        consumer, consumer_allocator, consumer_req_pool = build_fixture(cfg)
        consumer_linker = _InMemoryUnifiedCacheLinker(stored_keys)
        consumer.init_cache_linker(consumer_linker)
        consumer.write_through_threshold = sys.maxsize
        self._insert(
            consumer,
            consumer_allocator,
            consumer_req_pool,
            tokens[:5],
        )
        req = self._make_req(consumer_req_pool)
        lookup_key = RadixKey(array("q", tokens))

        match = consumer.match_prefix(MatchPrefixParams(key=lookup_key, req=req))

        self.assertTrue(lookup_key.is_bigram)
        self.assertEqual(match.device_prefix_len, 4)
        self.assertEqual(match.host_hit_length, 4)
        self.assertEqual(len(consumer_linker.lookup_calls), 1)
        _, transfers = consumer_linker.lookup_calls[0]
        (kv_lookup,) = transfers
        self.assertEqual(kv_lookup.keys, kv_offload.keys[2:])

    def test_failed_split_offload_retries_and_reset_releases_locks(self):
        cfg = CacheConfig(page_size=1, kv_size=64, max_context_len=64)
        self.cfg = cfg
        cache, allocator, req_to_token_pool = build_fixture(cfg)
        linker = _InMemoryUnifiedCacheLinker()
        cache.init_cache_linker(linker)
        tokens = [1, 2, 3, 4]

        inserted = self._insert(cache, allocator, req_to_token_pool, tokens)
        original_node = inserted.last_device_node
        self.assertEqual(len(linker.offload_calls), 1)
        original_keys = linker.offload_calls[0][0].keys

        self._insert(cache, allocator, req_to_token_pool, tokens[:2])
        (parent,) = _node_children(cache, cache.root_node_handle())
        (child,) = _node_children(cache, parent)
        self.assertEqual(child, original_node)
        self.assertEqual(
            cache.linker.pending_offloads[0].publish_node_ids, [parent, child]
        )
        self.assertEqual(cache.tree_core.get_write_through_pending_id(parent), child)
        self.assertEqual(cache.tree_core.get_write_through_pending_id(child), child)
        self.assertFalse(cache.tree_core.is_external_cache_stored(parent))
        self.assertFalse(cache.tree_core.is_external_cache_stored(child))

        linker.complete_next_offload(False)
        cache.check_hicache_events()
        for node_id in (parent, child):
            self.assertIsNone(cache.tree_core.get_write_through_pending_id(node_id))
            self.assertFalse(cache.tree_core.is_external_cache_stored(node_id))
            self.assertEqual(_device_lock_ref(cache, node_id, ComponentType.FULL), 0)

        self._insert(cache, allocator, req_to_token_pool, tokens)
        retry_calls = linker.offload_calls[1:]
        self.assertEqual(len(retry_calls), 2)
        self.assertEqual(
            [key for transfers in retry_calls for key in transfers[0].keys],
            original_keys,
        )
        for _ in retry_calls:
            linker.complete_next_offload(True)
        cache.check_hicache_events()
        for node_id in (parent, child):
            self.assertIsNone(cache.tree_core.get_write_through_pending_id(node_id))
            self.assertTrue(cache.tree_core.is_external_cache_stored(node_id))
            self.assertEqual(_device_lock_ref(cache, node_id, ComponentType.FULL), 0)

        self._insert(cache, allocator, req_to_token_pool, tokens)
        self.assertEqual(len(linker.offload_calls), 3)

        extended = self._insert(cache, allocator, req_to_token_pool, tokens + [5, 6])
        pending_node = extended.last_device_node
        self.assertEqual(len(cache.linker.pending_offloads), 1)
        self.assertEqual(
            cache.tree_core.get_write_through_pending_id(pending_node), pending_node
        )
        self.assertEqual(_device_lock_ref(cache, pending_node, ComponentType.FULL), 1)

        cache.linker.reset()
        self.assertEqual(cache.linker.pending_offloads, [])
        self.assertIsNone(cache.tree_core.get_write_through_pending_id(pending_node))
        self.assertEqual(_device_lock_ref(cache, pending_node, ComponentType.FULL), 0)
        cache.sanity_check()
        cache.reset()
        cache.sanity_check()

    def test_swa_private_tail_is_deduplicated_only_after_load_success(self):
        cfg = CacheConfig(
            page_size=1,
            components=(ComponentType.FULL, ComponentType.SWA),
            sliding_window_size=2,
            kv_size=64,
            max_context_len=64,
        )
        self.cfg = cfg
        stored_keys = defaultdict(set)
        tokens = list(range(1, 7))

        producer, producer_allocator, producer_req_pool = build_fixture(cfg)
        producer_linker = _InMemoryUnifiedCacheLinker(stored_keys)
        producer.init_cache_linker(producer_linker)
        self._insert(producer, producer_allocator, producer_req_pool, tokens)
        self.assertGreaterEqual(len(producer_linker.offload_calls), 1)
        all_keys = []
        for transfers in producer_linker.offload_calls:
            offloads = {transfer.name: transfer for transfer in transfers}
            self.assertEqual(set(offloads), {PoolName.KV, PoolName.SWA})
            self.assertEqual(offloads[PoolName.KV].keys, offloads[PoolName.SWA].keys)
            all_keys.extend(offloads[PoolName.KV].keys)
            producer_linker.complete_next_offload(True)
        producer.check_hicache_events()

        stored_keys[PoolName.KV].difference_update(all_keys[-2:])

        consumer, consumer_allocator, consumer_req_pool = build_fixture(cfg)
        consumer_linker = _InMemoryUnifiedCacheLinker(stored_keys)
        consumer.init_cache_linker(consumer_linker)
        consumer.write_through_threshold = sys.maxsize
        self._insert(consumer, consumer_allocator, consumer_req_pool, tokens[:2])

        req = self._make_req(consumer_req_pool)
        match = consumer.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens)), req=req)
        )
        self.assertEqual(match.device_prefix_len, 2)
        self.assertEqual(match.host_hit_length, 2)
        self.assertEqual(match.swa_host_hit_length, 2)
        self._apply_match_to_req(req, match)

        self._insert(consumer, consumer_allocator, consumer_req_pool, tokens[:3])
        raced_match = consumer.match_prefix(
            MatchPrefixParams(key=RadixKey(array("q", tokens[:3])))
        )
        raced_full = consumer.path_device_indices(raced_match.last_device_node)[
            -1:
        ].clone()

        loaded_len, loaded_node = consumer.init_load_back(
            InitLoadBackParams(
                best_match_node=match.best_match_node,
                host_hit_length=match.host_hit_length,
                req=req,
            )
        )
        self.assertEqual(loaded_len, 2)
        loaded = consumer.linker.private_loads[req.rid].device_indices
        self.assertFalse(torch.equal(loaded[:1], raced_full))
        load_by_pool = {t.name: t for t in consumer_linker.queued_loads[req.rid]}
        for transfer in load_by_pool.values():
            self.assertEqual(transfer.keys, all_keys[2:4])
            self.assertEqual(transfer.device_indices.numel(), 2)
        indices = consumer.linker.prefix_device_indices(req)
        consumer_req_pool.req_to_token[req.kv.req_pool_idx, :4] = indices
        req.origin_input_ids = req.full_untruncated_fill_ids = array("q", tokens[:4])
        req.kv.kv_allocated_len = req.kv.kv_committed_len = 4
        req.kv.cache_protected_len = 2
        consumer.linker.adopt_request(req)
        index = consumer.ready_to_load_host_cache()
        consumer_linker.complete_started_loads()
        self.assertEqual(consumer.finish_external_linker_loads(index), [])
        consumer.checkpoint(req, up_to=4)
        final_match = consumer._match_tree(
            MatchPrefixParams(key=RadixKey(array("q", tokens[:4])))
        )
        self.assertEqual(final_match.device_prefix_len, 4)
        consumer.sanity_check()


@unittest.skipUnless(torch.cuda.is_available(), "cache fixtures need CUDA")
class TestUnifiedCacheLinkerTreeCorePythonBackend(
    _TreeCoreBackendTestMixin, shared_cache_suite._InsertWalkSuite
):
    """TreeCore linker contracts shared by the Python and Rust inspectors."""

    def test_builds_opaque_external_offload_transfers(self):
        cfg = shared_cache_suite.CacheConfig(
            page_size=2, kv_size=64, max_context_len=64
        )
        self.cfg = cfg
        cache, allocator, req_to_token_pool = shared_cache_suite.build_fixture(cfg)
        core = cache.tree_core
        core.enable_external_cache_linker = True

        inserted = self._insert(cache, allocator, req_to_token_pool, list(range(1, 9)))
        node_id = inserted.last_device_node
        transfers = core.build_external_linker_offload_transfers(node_id)

        self.assertIsNotNone(transfers)
        (transfer,) = transfers
        self.assertEqual(transfer.name, PoolName.KV)
        self.assertIsNone(transfer.host_indices)
        self.assertIsNotNone(transfer.device_indices)
        self.assertEqual(transfer.keys, core.get_hash_values(node_id))
        self.assertTrue(
            torch.equal(
                transfer.device_indices,
                shared_cache_suite._device_value(cache, node_id, ComponentType.FULL),
            )
        )

        core.mark_external_linker_offload_pending(node_id)
        self.assertFalse(core.is_external_cache_stored(node_id))
        self.assertIsNone(core.build_external_linker_offload_transfers(node_id))
        with self.assertRaisesRegex(AssertionError, "invalid external offload state"):
            core.mark_external_linker_offload_pending(node_id)
        self.assertEqual(core.get_write_through_pending_id(node_id), node_id)
        self.assertFalse(core.is_external_cache_stored(node_id))

        core.finish_external_linker_offload([node_id], node_id, success=True)
        self.assertIsNone(core.get_write_through_pending_id(node_id))
        self.assertTrue(core.is_external_cache_stored(node_id))

    def test_external_state_updates_are_atomic_and_path_scoped(self):
        cfg = shared_cache_suite.CacheConfig(
            page_size=1, kv_size=64, max_context_len=64
        )
        self.cfg = cfg
        cache, allocator, req_to_token_pool = shared_cache_suite.build_fixture(cfg)
        core = cache.tree_core
        core.enable_external_cache_linker = True

        anchor = self._insert(cache, allocator, req_to_token_pool, [1]).last_device_node
        middle = self._insert(
            cache, allocator, req_to_token_pool, [1, 2]
        ).last_device_node
        tail = self._insert(
            cache, allocator, req_to_token_pool, [1, 2, 3, 4]
        ).last_device_node
        unrelated = self._insert(
            cache, allocator, req_to_token_pool, [9]
        ).last_device_node
        self.assertEqual(core.get_parent_node_id(middle), anchor)
        self.assertEqual(core.get_parent_node_id(tail), middle)

        with self.assertRaisesRegex(RuntimeError, "not an ancestor"):
            core.mark_external_cache_stored_path(tail, unrelated)
        self.assertFalse(core.is_external_cache_stored(middle))
        self.assertFalse(core.is_external_cache_stored(tail))

        core.mark_external_cache_stored_path(tail, anchor)
        self.assertTrue(core.is_external_cache_stored(tail))
        self.assertTrue(core.is_external_cache_stored(middle))
        self.assertFalse(core.is_external_cache_stored(anchor))
        self.assertFalse(core.is_external_cache_stored(unrelated))
        with self.assertRaisesRegex(AssertionError, "invalid external offload state"):
            core.mark_external_linker_offload_pending(tail)

        split_tail = self._insert(
            cache, allocator, req_to_token_pool, [9, 10, 11]
        ).last_device_node
        self.assertEqual(core.get_parent_node_id(split_tail), unrelated)
        core.mark_external_linker_offload_pending(split_tail)
        self._insert(cache, allocator, req_to_token_pool, [9, 10])
        split_parent = core.get_parent_node_id(split_tail)
        self.assertIsNotNone(split_parent)
        self.assertNotEqual(split_parent, unrelated)
        self.assertEqual(core.get_parent_node_id(split_parent), unrelated)
        for node_id in (split_parent, split_tail):
            self.assertEqual(core.get_write_through_pending_id(node_id), split_tail)
            self.assertFalse(core.is_external_cache_stored(node_id))

        independent = self._insert(
            cache, allocator, req_to_token_pool, [20]
        ).last_device_node
        core.mark_external_linker_offload_pending(independent)
        with self.assertRaisesRegex(AssertionError, "invalid external offload state"):
            core.finish_external_linker_offload(
                [independent, split_parent], independent, success=False
            )
        self.assertEqual(core.get_write_through_pending_id(independent), independent)
        for node_id in (split_parent, split_tail):
            self.assertEqual(core.get_write_through_pending_id(node_id), split_tail)
            self.assertFalse(core.is_external_cache_stored(node_id))

        core.finish_external_linker_offload([independent], independent, success=False)
        core.finish_external_linker_offload(
            [split_parent, split_tail], split_tail, success=False
        )
        for node_id in (split_parent, split_tail):
            self.assertIsNone(core.get_write_through_pending_id(node_id))
            self.assertFalse(core.is_external_cache_stored(node_id))
        self.assertTrue(core.is_external_cache_stored(middle))
        self.assertTrue(core.is_external_cache_stored(tail))

    def test_failed_offload_preserves_independently_confirmed_state(self):
        cfg = shared_cache_suite.CacheConfig(
            page_size=1, kv_size=64, max_context_len=64
        )
        self.cfg = cfg
        cache, allocator, req_to_token_pool = shared_cache_suite.build_fixture(cfg)
        core = cache.tree_core
        core.enable_external_cache_linker = True

        anchor = self._insert(cache, allocator, req_to_token_pool, [1]).last_device_node
        node_id = self._insert(
            cache, allocator, req_to_token_pool, [1, 2]
        ).last_device_node
        core.mark_external_linker_offload_pending(node_id)

        core.mark_external_cache_stored_path(node_id, anchor)
        self.assertEqual(core.get_write_through_pending_id(node_id), node_id)
        self.assertTrue(core.is_external_cache_stored(node_id))

        core.finish_external_linker_offload([node_id], node_id, success=False)
        self.assertIsNone(core.get_write_through_pending_id(node_id))
        self.assertTrue(core.is_external_cache_stored(node_id))


class TestUnifiedCacheLinkerRustBackend(TestUnifiedCacheLinkerPythonBackend):
    tree_core_backend = "rust"


class TestUnifiedCacheLinkerTreeCoreRustBackend(
    TestUnifiedCacheLinkerTreeCorePythonBackend
):
    tree_core_backend = "rust"


def test_restorable_prefix_intersects_sparse_rank_results():
    remote_mask = torch.tensor([0, 0, 1, 0, 0], dtype=torch.int)

    def intersect_remote_mask(mask, op):
        assert op == torch.distributed.ReduceOp.MIN
        mask.copy_(torch.minimum(mask, remote_mask))

    cache = _cache_for_wrapper(_all_reduce_attn_groups=intersect_remote_mask)
    wrapper = UnifiedCacheLinkerWrapper(cache, _FakeLinker())

    hit_pages = wrapper._sync_restorable_prefix([2, 4], num_pages=4, device_hit_pages=0)

    assert hit_pages == 2


def test_async_offload_pins_node_until_completion():
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
        tree_core=_FakeExternalTreeCore({node_id: node}),
        inc_lock_ref=inc_lock_ref,
        dec_lock_ref=lambda node, params: unlocks.append((node, params)),
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)

    wrapper.offload_nodes([node_id])

    assert locks == [node_id]
    assert not node.external_cache_stored
    assert node.write_through_pending_id == node_id
    assert not unlocks

    linker.completed_offloads.append(False)
    completed = wrapper.take_completed_offloads(finish_count=1)
    wrapper.commit_completed_offloads(completed)

    assert not node.external_cache_stored
    assert node.write_through_pending_id is None
    assert unlocks == [(node_id, lock_params)]


def test_offload_skips_node_already_stored_by_tree_core():
    linker = _FakeLinker()
    node = SimpleNamespace(
        id=7,
        external_cache_stored=True,
        write_through_pending_id=None,
    )
    cache = _cache_for_wrapper(
        tree_core=_FakeExternalTreeCore({node.id: node}),
        inc_lock_ref=lambda node_id: pytest.fail("stored node must not be locked"),
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)

    wrapper.offload_nodes([node.id])

    assert linker.queued_offloads == []


def _private_fixture(*, adopted=False):
    freed = []
    req = SimpleNamespace(
        rid="rid",
        kv=SimpleNamespace(holds_kv=True),
        external_kv_load_pending=True,
        external_kv_load_failed=False,
        skip_radix_cache_insert=False,
    )
    full = PoolTransfer(
        name=PoolName.KV, device_indices=torch.arange(4), keys=["a", "b"]
    )
    component = SimpleNamespace(
        update_external_linker_load=lambda phase, *args: freed.append(phase)
    )
    cache = _cache_for_wrapper()
    linker = _FakeLinker()
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper.private_loads[req.rid] = _PrivateLoad(
        req, [(component, full)], 2, 6, full.device_indices, adopted=adopted
    )
    linker.queued_loads[req.rid] = [full]
    return wrapper, linker, req, freed


def test_release_unadopted_private_slots_once_even_with_preallocated_row():
    wrapper, linker, req, freed = _private_fixture()
    wrapper.release_request(req.rid)
    wrapper.release_request(req.rid)
    assert freed == [ExternalLinkerLoadPhase.ABORT]
    assert not wrapper.private_loads
    assert not req.external_kv_load_pending


def test_release_adopted_private_slots_leaves_request_as_only_free_owner():
    wrapper, linker, req, freed = _private_fixture(adopted=True)
    wrapper.release_request(req.rid)
    wrapper.release_request(req.rid)
    assert not freed
    assert req.external_kv_load_failed and req.skip_radix_cache_insert


def test_started_load_cannot_be_released_by_cancel_retry():
    wrapper, linker, req, freed = _private_fixture()
    linker.queued_loads.clear()  # worker already took the transfer
    with pytest.raises(RuntimeError, match="before load drain"):
        wrapper.release_request(req.rid)
    assert not freed
    assert req.rid in wrapper.private_loads


def test_failed_rank_completion_never_frees_adopted_slots_or_publishes():
    wrapper, linker, req, freed = _private_fixture(adopted=True)
    wrapper.load_batches[0] = [req.rid]
    linker.completed_loads.append([(req.rid, True)])

    def reduce(tensor, op):
        if tensor.numel() == 1:  # remote rank failed its load
            tensor.zero_()

    wrapper.cache._all_reduce_attn_groups = reduce
    assert wrapper.finish_loads(0) == [req]
    assert req.external_kv_load_failed and req.skip_radix_cache_insert
    assert not freed
    wrapper.release_request(req.rid)
    assert not freed


def test_failed_offload_rolls_back_split_fragments():
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

    cache = _cache_for_wrapper(
        tree_core=_FakeExternalTreeCore(nodes),
        inc_lock_ref=lambda node_id: SimpleNamespace(to_dec_params=lambda: lock_params),
        dec_lock_ref=lambda node_id, params: unlocks.append((node_id, params)),
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
        tree_core=_FakeExternalTreeCore({node.id: node}),
        inc_lock_ref=lambda node_id: SimpleNamespace(to_dec_params=object),
        dec_lock_ref=lambda node_id, params: events.append(("unlock", node_id)),
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, linker)
    wrapper.offload_nodes([node.id])

    wrapper.reset()

    assert events == ["backend", ("unlock", node.id)]
    assert wrapper.private_loads == {}
    assert wrapper.pending_offloads == []
    assert not node.external_cache_stored
    assert node.write_through_pending_id is None


def test_close_quiesces_backend_before_releasing_private_slots():
    wrapper, linker, req, freed = _private_fixture()

    def close():
        assert not freed
        linker.closed = True

    linker.close = close
    wrapper.close()
    assert linker.closed
    assert freed == [ExternalLinkerLoadPhase.ABORT]
    assert not wrapper.private_loads


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


def test_completion_identity_mismatch_retains_private_ownership():
    wrapper, linker, req, freed = _private_fixture()
    wrapper.load_batches[0] = [req.rid]
    linker.completed_loads.append([("different-rid", True)])
    with pytest.raises(RuntimeError, match="Out-of-order"):
        wrapper.finish_loads(0)
    assert req.rid in wrapper.private_loads
    assert not freed


@pytest.mark.parametrize(
    "swa_req_ring",
    [None, False, True],
    ids=["other-allocator", "paged", "request-ring"],
)
@pytest.mark.parametrize("enable_hicache", [False, True])
def test_swa_reuse_policy_tracks_layout_without_a_tier_condition(
    monkeypatch, swa_req_ring, enable_hicache
):
    from sglang.kernels.ops.attention.dsv4.unified_kv_kernels import env_gate

    unified_kv = swa_req_ring is True
    monkeypatch.setattr(env_gate, "is_unified_kv_triton", lambda: unified_kv)
    component = SWAComponent.__new__(SWAComponent)
    component.sliding_window_size = 128
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.token_to_kv_pool_allocator = _swa_allocator(swa_req_ring)
    cache.components = {ComponentType.SWA: component}
    cache.cache_controller = object() if enable_hicache else None
    cache.tree_core = SimpleNamespace(
        enable_hicache=enable_hicache,
        has_swa_host_pool=enable_hicache and not unified_kv,
    )
    component.cache = cache
    component.tree_core = cache.tree_core
    # #32759: request-relative SWA needs tail re-prefill even without HiCache.
    assert cache.swa_reprefill_tail_tokens() == (128 if unified_kv else 0)
    node = SimpleNamespace(
        component_data={
            ComponentType.SWA: SimpleNamespace(value=None, host_value=None)
        },
        backuped=False,
        evicted=False,
    )
    assert component.create_match_validator(match_device_only=True)(node) is unified_kv


def test_cache_without_swa_needs_no_reprefill():
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    cache.components = {}
    assert cache.swa_reprefill_tail_tokens() == 0


@pytest.fixture
def full_linker_component():
    def build_transfer(phase, node, keys):
        keys = ["offload"] if phase == LinkerTransferPhase.OFFLOAD else list(keys)
        return PoolTransfer(
            name=PoolName.KV,
            keys=keys,
            device_indices=(
                None
                if phase == LinkerTransferPhase.LOOKUP
                else torch.arange(len(keys) * 2)
            ),
        )

    return SimpleNamespace(
        component_type=ComponentType.FULL,
        build_external_linker_transfer=MagicMock(side_effect=build_transfer),
        update_external_linker_load=lambda phase, req, full_transfer, transfer, prefix_len, **kwargs: (
            transfer
        ),
    )


def test_linker_filters_request_relative_swa_from_lookup(
    full_linker_component,
):
    full = full_linker_component
    swa = SWAComponent.__new__(SWAComponent)
    swa.build_external_linker_transfer = MagicMock(
        side_effect=AssertionError("excluded SWA reached linker")
    )
    node = SimpleNamespace(id=1, external_cache_stored=False)
    cache = _cache_for_wrapper(
        _components_tuple=(full, swa),
        components={ComponentType.FULL: full, ComponentType.SWA: swa},
        token_to_kv_pool_allocator=_swa_allocator(True),
        tree_core=SimpleNamespace(enable_external_cache_linker=False, is_eagle=False),
        page_size=2,
        _all_reduce_attn_groups=lambda value, op: None,
        get_last_hash_value=lambda node: None,
        resolve_node_handle=lambda node_id: node,
        inc_lock_ref=lambda node_id: SimpleNamespace(to_dec_params=lambda: object()),
        dec_lock_ref=MagicMock(),
    )
    backend = _FakeLinker()
    backend.restorable = [2]
    wrapper = UnifiedCacheLinkerWrapper(cache, backend)
    assert wrapper._components == (full,)
    result = MatchResult(
        device_prefix_len=0,
        last_device_node=0,
        last_host_node=0,
        best_match_node=0,
    )
    matched = wrapper.match(
        RadixKey(array("q", [1, 2, 3, 4])), SimpleNamespace(rid="match"), result
    )
    assert matched.host_hit_length == 4
    assert [c.args[0] for c in full.build_external_linker_transfer.call_args_list] == [
        LinkerTransferPhase.LOOKUP,
    ]
    swa.build_external_linker_transfer.assert_not_called()


@pytest.mark.parametrize("swa_req_ring", [False, True], ids=["paged", "request-ring"])
@pytest.mark.parametrize(
    "completion", [True, False, None], ids=["success", "failure", "reset"]
)
def test_offload_filters_tree_core_swa_transfers_and_preserves_lifecycle(
    swa_req_ring, completion
):
    node = SimpleNamespace(
        id=7, external_cache_stored=False, write_through_pending_id=None
    )
    transfers = [
        PoolTransfer(name=PoolName.KV, keys=["page"]),
        PoolTransfer(name=PoolName.SWA, keys=["page"]),
    ]
    core = _FakeExternalTreeCore({node.id: node}, transfers)
    swa = SWAComponent.__new__(SWAComponent)
    lock_params = object()
    cache = _cache_for_wrapper(
        components={ComponentType.SWA: swa},
        _components_tuple=(swa,),
        token_to_kv_pool_allocator=_swa_allocator(swa_req_ring),
        tree_core=core,
        inc_lock_ref=MagicMock(
            return_value=SimpleNamespace(to_dec_params=lambda: lock_params)
        ),
        dec_lock_ref=MagicMock(),
    )
    backend = _FakeLinker()
    wrapper = UnifiedCacheLinkerWrapper(cache, backend)

    wrapper.offload_nodes([node.id, node.id])
    expected = transfers[:1] if swa_req_ring else transfers
    assert backend.queued_offloads == [expected]
    assert core.offload_transfers == transfers
    assert node.write_through_pending_id == node.id
    assert not node.external_cache_stored
    cache.inc_lock_ref.assert_called_once_with(node.id)
    cache.dec_lock_ref.assert_not_called()

    if completion is None:
        wrapper.reset()
        assert backend.reset_count == 1
    else:
        backend.completed_offloads.append(completion)
        wrapper.commit_completed_offloads(wrapper.take_completed_offloads(1))
    assert not wrapper.pending_offloads
    assert node.write_through_pending_id is None
    assert node.external_cache_stored is (completion is True)
    cache.dec_lock_ref.assert_called_once_with(node.id, lock_params)
    if completion is True:
        wrapper.offload_nodes([node.id])
        assert backend.queued_offloads == [expected]
    else:
        wrapper.offload_nodes([node.id])
        assert backend.queued_offloads == [expected, expected]
        wrapper.reset()


@pytest.mark.parametrize(
    "swa_req_ring,previous_boundary,expected_boundary",
    [
        pytest.param(True, None, 4, id="unified-tombstones"),
        pytest.param(True, 8, 8, id="preserve-existing-boundary"),
        pytest.param(False, None, 2, id="paged-prepare-boundary"),
        pytest.param(None, None, 2, id="other-allocator-prepare-boundary"),
    ],
)
def test_linker_load_preserves_swa_boundaries(
    full_linker_component, swa_req_ring, previous_boundary, expected_boundary
):
    full = full_linker_component
    swa = SWAComponent.__new__(SWAComponent)
    participates = not swa_req_ring

    def prepare(phase, req, full_transfer, transfer, prefix_len, **kwargs):
        if phase == ExternalLinkerLoadPhase.PREPARE:
            req.kv = ReqKvInfo(
                kv_allocated_len=prefix_len,
                component_evicted_seqlens={ComponentType.SWA: prefix_len - 2},
            )
        return transfer

    swa.build_external_linker_transfer = MagicMock(
        return_value=PoolTransfer(
            name=PoolName.SWA, keys=["a", "b"], device_indices=torch.arange(20, 24)
        )
    )
    swa.update_external_linker_load = MagicMock(side_effect=prepare)
    full_indices = torch.arange(4, dtype=torch.int64)
    adopted = {ComponentType.FULL: [(0, 4)]}
    if participates:
        adopted[ComponentType.SWA] = [(0, 4)]
    cache = _cache_for_wrapper(
        _components_tuple=(full, swa),
        page_size=2,
        components={ComponentType.FULL: full, ComponentType.SWA: swa},
        token_to_kv_pool_allocator=_swa_allocator(swa_req_ring),
        tree_core=SimpleNamespace(
            collect_full_device_indices=lambda node, ancestor: full_indices,
            mark_external_cache_stored_path=MagicMock(),
        ),
        insert=MagicMock(
            return_value=InsertResult(
                prefix_len=4, total_len=4, last_device_node=0, adopted_ranges=adopted
            )
        ),
        resolve_node_handle=lambda node_id: SimpleNamespace(id=0),
        prefix_device_indices=lambda req: torch.empty(0, dtype=torch.int64),
    )
    wrapper = UnifiedCacheLinkerWrapper(cache, _FakeLinker())
    wrapper.hit_markers["rid"] = ExternalCacheHitMarker(
        prefix_key=RadixKey(array("q", [1, 2, 3, 4])),
        tail_hashes=["a", "b"],
        device_hit_len=0,
    )
    kv = (
        ReqKvInfo()
        if previous_boundary is None
        else ReqKvInfo(
            kv_allocated_len=previous_boundary,
            component_evicted_seqlens={ComponentType.SWA: previous_boundary},
        )
    )
    req = SimpleNamespace(
        rid="rid",
        kv=kv,
        prefix_len=0,
        last_node=0,
        priority=0,
    )
    restored_len, last_node = wrapper.load_back(req)

    assert restored_len == len(full_indices)
    assert last_node == 0
    assert req.kv.get_evicted_seqlen(ComponentType.SWA) == expected_boundary
    cache.insert.assert_not_called()
    cache.tree_core.mark_external_cache_stored_path.assert_not_called()
    assert "rid" in wrapper.private_loads
    assert [c.args[0] for c in full.build_external_linker_transfer.call_args_list] == [
        LinkerTransferPhase.LOAD
    ]
    if not participates:
        swa.build_external_linker_transfer.assert_not_called()
        swa.update_external_linker_load.assert_not_called()


@pytest.mark.parametrize("remote_geometry", [(16, 2, 14), (17, 14, 3)])
def test_match_geometry_divergence_is_fatal_before_variable_mask(remote_geometry):
    from sglang.srt.mem_cache.unified_cache.linker_lifecycle import (
        identity_words,
        preflight,
    )

    calls = []
    identity = ("request", 1)

    def reduce(payload, op):
        calls.append(payload.numel())
        fields = [1, *identity_words(identity), *remote_geometry, 1]
        remote = torch.tensor(fields + [-v for v in fields], dtype=torch.int64)
        payload.copy_(torch.minimum(payload, remote))

    cache = _cache_for_wrapper(_all_reduce_attn_groups=reduce)
    with pytest.raises(RuntimeError, match="rank divergence"):
        preflight(cache, 1, identity, (16, 14, 2), True)
    assert calls == [18]


@pytest.mark.parametrize("remote_rids", [["a"], ["b", "a"]])
def test_inflight_count_or_same_size_rid_reordering_is_fatal(remote_rids):
    from sglang.srt.mem_cache.unified_cache.linker_lifecycle import (
        identity_words,
        preflight,
    )

    def reduce(payload, op):
        fields = [7, *identity_words((1, remote_rids)), len(remote_rids), 0, 0, 1]
        payload.copy_(
            torch.minimum(payload, torch.tensor(fields + [-v for v in fields]))
        )

    cache = _cache_for_wrapper(_all_reduce_attn_groups=reduce)
    with pytest.raises(RuntimeError, match="rank divergence"):
        preflight(cache, 7, (1, ["a", "b"]), (2, 0, 0))


@pytest.mark.parametrize("key_len", [0, 4, 26112])
def test_full_local_hit_still_negotiates_without_warning(caplog, key_len):
    calls = []
    cache = _cache_for_wrapper(
        page_size=256,
        tree_core=SimpleNamespace(is_eagle=False),
        get_last_hash_value=lambda node: None,
        _all_reduce_attn_groups=lambda tensor, op: calls.append(tensor.numel()),
    )
    backend = _FakeLinker()
    wrapper = UnifiedCacheLinkerWrapper(cache, backend)
    key = RadixKey(array("q", range(key_len)))
    result = SimpleNamespace(device_prefix_len=key_len, last_device_node=0)
    assert wrapper.match(key, SimpleNamespace(rid="full-local-hit"), result) is result
    assert calls == [18]
    assert backend.lookup_calls == []
    assert not [record for record in caplog.records if record.levelno >= 30]


def test_valid_only_divergence_is_safe_miss():
    from sglang.srt.mem_cache.unified_cache.linker_lifecycle import preflight

    def reduce(payload, op):
        payload[8] = 0  # geometry and identity unchanged, remote lookup unavailable

    cache = _cache_for_wrapper(_all_reduce_attn_groups=reduce)
    assert not preflight(cache, 1, ("a", 1), (16, 14, 2), True)


@pytest.mark.parametrize(
    "remote_identity",
    [
        ("other-rid", 1, 1, 0),
        ("same-rid", 2, 1, 0),
        ("same-rid", 1, 2, 0),
        ("same-rid", 1, 1, 1),
    ],
)
def test_match_identity_includes_rid_sequence_and_pp_even_for_full_hit(remote_identity):
    from sglang.srt.mem_cache.unified_cache.linker_lifecycle import identity_words

    calls = []

    def reduce(payload, op):
        calls.append(payload.numel())
        fields = [1, *identity_words(remote_identity), 4, 4, 0, 0]
        payload.copy_(
            torch.minimum(payload, torch.tensor(fields + [-v for v in fields]))
        )

    cache = _cache_for_wrapper(
        page_size=1,
        pp_size=1,
        pp_rank=0,
        tree_core=SimpleNamespace(is_eagle=False),
        get_last_hash_value=lambda node: None,
        _all_reduce_attn_groups=reduce,
    )
    backend = _FakeLinker()
    wrapper = UnifiedCacheLinkerWrapper(cache, backend)
    with pytest.raises(RuntimeError, match="rank divergence"):
        wrapper.match(
            RadixKey(array("q", range(4))),
            SimpleNamespace(rid="same-rid"),
            SimpleNamespace(device_prefix_len=4, last_device_node=0),
        )
    assert calls == [18]
    assert backend.lookup_calls == []


@pytest.mark.parametrize("completed", [False, None])
def test_collective_unconfirmed_completion_aborts_group(completed):
    from unittest.mock import patch

    from sglang.srt.mem_cache.unified_cache.linker_lifecycle import (
        all_reduce_with_timeout,
    )

    group = MagicMock()
    work = SimpleNamespace(wait=lambda **kwargs: completed)
    with patch.object(torch.distributed, "all_reduce", return_value=work):
        with pytest.raises(RuntimeError, match="collective failed"):
            all_reduce_with_timeout(
                torch.zeros(1), torch.distributed.ReduceOp.MIN, group
            )
    group.abort.assert_called_once()


@pytest.mark.parametrize("adopted", [False, True])
def test_full_and_swa_share_one_abort_owner(adopted):
    wrapper, linker, req, freed = _private_fixture(adopted=adopted)
    swa_freed = []
    swa = PoolTransfer(name=PoolName.SWA, device_indices=torch.arange(2), keys=["b"])
    component = SimpleNamespace(
        update_external_linker_load=lambda phase, *args: swa_freed.append(phase)
    )
    wrapper.private_loads[req.rid].transfers.append((component, swa))
    wrapper.release_request(req.rid)
    wrapper.release_request(req.rid)
    assert len(freed) == len(swa_freed) == (0 if adopted else 1)


@pytest.mark.parametrize(
    "method",
    [
        "checkpoint",
        "checkpoint_into_tree",
        "advance_unpublished_req",
        "maybe_hand_to_session",
    ],
)
@pytest.mark.parametrize(
    "state", ["external_kv_load_pending", "external_kv_load_failed"]
)
def test_unfinished_or_session_path_cannot_publish_private_load(method, state):
    cache = UnifiedRadixCache.__new__(UnifiedRadixCache)
    req = SimpleNamespace(rid="chunked", **{state: True})
    # Bare cache has no session, tree, or allocator: touching any is an error.
    kwargs = {"up_to": 4} if method.startswith("checkpoint") else {}
    getattr(cache, method)(req, **kwargs)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__]))
