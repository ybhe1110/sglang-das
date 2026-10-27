"""Direct L3 support for :class:`UnifiedRadixCache`.

Links the cache's device pools straight to an external KV store, with no host
tier in between. The transport contract and tree-side wrapper live here, while
each backend owns its device-pool layout and physical I/O.

* :class:`UnifiedCacheLinker` -- the transport interface a backend implements.
* :class:`UnifiedCacheLinkerWrapper` -- the tree-side flow that drives it. The
  cache owns one as a plain attribute, keeping the whole external-cache path out
  of the main tree file.

The tree only needs a handful of guarded hooks:

* ``match_prefix``      -> :meth:`UnifiedCacheLinkerWrapper.match`
* ``init_load_back``    -> :meth:`UnifiedCacheLinkerWrapper.load_back`
* ``BackupKV`` actions  -> :meth:`UnifiedCacheLinkerWrapper.offload_nodes`

"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, NamedTuple

import torch
from sglang.srt.mem_cache.allocator.swa import is_swa_req_ring
from sglang.srt.mem_cache.base_prefix_cache import (
    DecLockRefParams,
    MatchResult,
)
from sglang.srt.mem_cache.hicache_storage import (
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.radix_cache import RadixKey
from sglang.srt.mem_cache.unified_cache.components import (
    ComponentType,
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
)
from sglang.srt.mem_cache.unified_cache.linker_lifecycle import preflight
from sglang.srt.mem_cache.utils import get_storage_hash_str

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


_EXTERNAL_LINKER_SUPPORTED_COMPONENTS = frozenset(
    {
        ComponentType.FULL,
        ComponentType.SWA,
    }
)


class UnifiedCacheLinker(ABC):
    """External KV store reached directly from the device pools."""

    layer_done_counter: object

    @abstractmethod
    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        """Return every prefix length (in pages) that is fully restorable.

        A length is included only when *all* pools satisfy their hit policy at
        that exact boundary (contiguous prefix pools, plus each trailing-window
        pool's window ending there). Trailing-window state (SWA / compress
        state) only exists at offloaded node boundaries, so the set is sparse
        and generally non-contiguous -- returning just the local maximum would
        let the tree pick a length that is invalid on another rank.

        Local to this rank; the tree intersects the sets across ranks.
        """

    @abstractmethod
    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        """Queue a load into the given device indices.

        The transfer is executed by the next ``start_layer_wise_loading`` call,
        not here.
        """

    @abstractmethod
    def start_layer_wise_loading(self) -> int:
        """Start queued loads and return the layer-counter consumer index."""

    @abstractmethod
    def cancel_queued_load(self, rid: str) -> bool:
        """Cancel a load that has not started yet."""

    @abstractmethod
    def num_completed_loads(self) -> int:
        """Return the number of completed load batches waiting to be consumed."""

    @abstractmethod
    def pop_completed_load(self) -> list[tuple[str, bool]]:
        """Consume a terminal batch as (request ID, success) pairs."""

    @abstractmethod
    def finish_loads(self, consumer_index: int) -> None:
        """Wait for worker/session termination and publish (rid, success).

        A timeout/unknown DMA state must raise without publishing completion.
        """

    @abstractmethod
    def offload(self, transfers: list[PoolTransfer]) -> bool:
        """Queue every transfer for atomic persistence."""

    @abstractmethod
    def num_completed_offloads(self) -> int:
        """Return the number of completed offloads waiting to be consumed."""

    @abstractmethod
    def pop_completed_offload(self) -> bool:
        """Consume the oldest completed offload and return its result."""

    @abstractmethod
    def reset(self) -> None:
        """Quiesce all transfers and reset backend state before returning."""

    @abstractmethod
    def close(self) -> None:
        """Quiesce all transfers and release backend resources."""


class ExternalCacheHitMarker(NamedTuple):
    """What ``match`` found in the external store, consumed by ``load_back``.

    ``prefix_key`` covers the device-cached prefix plus the restorable tail, so
    it is what gets inserted once the tail lands. ``tail_hashes`` are the
    per-page storage hashes of that tail alone, starting at ``device_hit_len``.
    """

    prefix_key: RadixKey
    tail_hashes: list[str]
    device_hit_len: int


@dataclass
class _PrivateLoad:
    req: Req
    transfers: list
    prev_prefix_len: int
    prefix_len: int
    device_indices: torch.Tensor
    cancelled: bool = False
    adopted: bool = False


class _PendingOffload(NamedTuple):
    lock_node_id: NodeId
    lock_params: DecLockRefParams
    publish_node_ids: list[NodeId]


class ExternalLinkerCleanupError(RuntimeError):
    """A component release has uncertain side effects; do not retry it."""


class UnifiedCacheLinkerWrapper:
    """Drives an external KV store on behalf of one :class:`UnifiedRadixCache`."""

    def __init__(
        self,
        cache: UnifiedRadixCache,
        cache_linker: UnifiedCacheLinker,
    ):
        unsupported = set(cache.tree_components) - _EXTERNAL_LINKER_SUPPORTED_COMPONENTS
        if unsupported:
            names = ", ".join(
                component.name for component in sorted(unsupported, key=int)
            )
            raise ValueError(
                "External cache linker supports only Full and SWA tree "
                f"components; unsupported: {names}"
            )

        self.cache = cache
        self.cache_linker = cache_linker
        swa = cache.components.get(ComponentType.SWA)
        self._skip_swa = swa is not None and is_swa_req_ring(
            cache.token_to_kv_pool_allocator
        )
        self._components = tuple(
            component
            for component in cache._components_tuple
            if not (self._skip_swa and component is swa)
        )
        # rid -> what match found, consumed by the next init_load_back.
        self.hit_markers: dict[str, ExternalCacheHitMarker] = {}
        # Unpublished slots; after row allocation the request is the sole free owner.
        self.private_loads: dict[str, _PrivateLoad] = {}
        self.cleanup_quarantine: dict[str, dict] = {}
        self.load_batches: dict[int, list[str]] = {}
        self._match_sequence = 0
        self._finish_sequence = 0
        # Offloads in flight, each holding a lock on its node until it lands.
        self.pending_offloads: list[_PendingOffload] = []

        cache.tree_core.enable_external_cache_linker = True
        cache.write_through_threshold = 1

    @property
    def layer_done_counter(self) -> object:
        return self.cache_linker.layer_done_counter

    def has_hit(self, rid: str) -> bool:
        return rid in self.hit_markers

    def _raise_if_cleanup_quarantined(self) -> None:
        if getattr(self, "cleanup_quarantine", {}):
            raise ExternalLinkerCleanupError(
                "External KV cleanup quarantined; process teardown required: "
                f"rids={list(self.cleanup_quarantine)}"
            )

    # ---- match: probe the remote store and report host_hit_length ----

    def match(self, key: RadixKey, req: Req, result: MatchResult) -> MatchResult:
        self._raise_if_cleanup_quarantined()
        cache = self.cache
        key, _ = key.maybe_to_bigram_view(cache.tree_core.is_eagle)
        page = cache.page_size
        device_hit_len = result.device_prefix_len
        self.hit_markers.pop(req.rid, None)
        tail_hashes = self._tail_hashes(key, result, device_hit_len)
        lookup_transfers = []
        valid = bool(tail_hashes) and device_hit_len < len(key)
        if valid:
            for component in self._components:
                transfer = component.build_external_linker_transfer(
                    LinkerTransferPhase.LOOKUP, None, tail_hashes
                )
                if transfer is None:
                    valid = False
                    break
                lookup_transfers.append(transfer)
        self._match_sequence += 1
        if not preflight(
            cache,
            1,
            (
                req.rid,
                self._match_sequence,
                getattr(cache, "pp_size", 1),
                getattr(cache, "pp_rank", 0),
            ),
            (len(key), device_hit_len, max(0, (len(key) - device_hit_len) // page)),
            valid,
        ):
            return result
        by_pool = {transfer.name: transfer for transfer in lookup_transfers}

        # Tail-relative: page 0 of `tail_hashes` is the first uncached page.
        try:
            restorable = self.cache_linker.lookup(req.rid, lookup_transfers)
        except Exception:
            # Still enter the intersection on every rank; local failures are misses.
            logger.exception(
                "External KV lookup failed; falling back to L1: rid=%s", req.rid
            )
            restorable = []
        hit_pages = self._sync_restorable_prefix(
            restorable,
            num_pages=len(tail_hashes),
            device_hit_pages=0,
        )
        if hit_pages == 0:
            return result
        hit_tokens = hit_pages * page

        swa_transfer = by_pool.get(PoolName.SWA)
        swa_host_hit_length = (
            min(len(swa_transfer.keys), hit_pages) * page
            if swa_transfer is not None
            else 0
        )
        # Mamba keeps a single state slot per node, so a hit is worth one slot.
        mamba_host_hit_length = 1 if PoolName.MAMBA in by_pool else 0

        self.hit_markers[req.rid] = ExternalCacheHitMarker(
            prefix_key=key[: device_hit_len + hit_tokens],
            tail_hashes=list(tail_hashes[:hit_pages]),
            device_hit_len=device_hit_len,
        )
        return result._replace(
            last_host_node=result.best_match_node,
            host_hit_length=hit_tokens,
            swa_host_hit_length=max(result.swa_host_hit_length, swa_host_hit_length),
            mamba_host_hit_length=max(
                result.mamba_host_hit_length, mamba_host_hit_length
            ),
        )

    def _sync_restorable_prefix(
        self, restorable: list[int], *, num_pages: int, device_hit_pages: int
    ) -> int:
        """Intersect the per-rank sets of restorable prefix lengths and return the
        longest one, or 0 when the ranks share none beyond the device prefix.

        A rank's set is sparse, so reducing per-rank maxima could land on a
        length that only some ranks can restore. On a 0/1 mask MIN is AND, which
        makes the reduction an intersection.
        """
        mask = torch.zeros(num_pages + 1, dtype=torch.int)
        for pages in restorable:
            if device_hit_pages < pages <= num_pages:
                mask[pages] = 1
        self.cache._all_reduce_attn_groups(mask, torch.distributed.ReduceOp.MIN)
        common = mask.nonzero()
        if common.numel() == 0:
            return 0
        return int(common[-1].item())

    def _tail_hashes(
        self, key: RadixKey, result: MatchResult, device_hit_len: int
    ) -> list[str]:
        """Per-page storage hashes for the device-uncached tail of the prefix."""
        last_hash = None
        if device_hit_len > 0:
            last_hash = self.cache.get_last_hash_value(result.last_device_node)
            if last_hash is None:
                # Without the anchor the tail would hash as if it started at the
                # sequence head, yielding keys that can never match.
                return []
        page = self.cache.page_size
        tail_len = (len(key) - device_hit_len) // page * page
        if tail_len <= 0:
            return []
        return get_storage_hash_str(
            key[device_hit_len : device_hit_len + tail_len],
            last_hash,
            page_size=page,
        )

    # ---- init_load_back: remote -> request-private slots; NEVER insert ----

    def load_back(self, req: Req) -> tuple[int, NodeId]:
        self._raise_if_cleanup_quarantined()
        hit = self.hit_markers.pop(req.rid, None)
        if hit is None:
            return 0, req.last_node
        if req.rid in self.private_loads:
            raise RuntimeError(f"Duplicate private load: {req.rid}")
        end = hit.device_hit_len + len(hit.tail_hashes) * self.cache.page_size
        transfers = []
        try:
            for component in self._components:
                transfer = component.build_external_linker_transfer(
                    LinkerTransferPhase.LOAD, None, hit.tail_hashes
                )
                if transfer is None:
                    break
                transfers.append((component, transfer))
        except Exception:
            self._abort_transfers(req, transfers, end)
            raise
        full = next((t for _, t in transfers if t.name == PoolName.KV), None)
        valid = len(transfers) == len(self._components) and full is not None
        # Allocation failure is rank-local. All ranks either queue or recompute.
        try:
            valid = preflight(
                self.cache,
                2,
                req.rid,
                (hit.device_hit_len, end, len(hit.tail_hashes)),
                valid,
            )
        except Exception:
            self._abort_transfers(req, transfers, end)
            raise
        if not valid:
            self._abort_transfers(req, transfers, end)
            return 0, req.last_node
        saved_evicted = req.kv.component_evicted_seqlens.copy()
        private = _PrivateLoad(
            req, transfers, hit.device_hit_len, end, full.device_indices
        )
        try:
            for component, transfer in transfers:
                component.update_external_linker_load(
                    ExternalLinkerLoadPhase.PREPARE, req, full, transfer, end
                )
            if self._skip_swa:
                req.kv.set_evicted_seqlen(
                    ComponentType.SWA,
                    max(req.kv.get_evicted_seqlen(ComponentType.SWA), end),
                )
        except Exception:
            self._abort_transfers(req, transfers, end)
            raise
        self.private_loads[req.rid] = private
        req.external_kv_load_pending = True
        req.external_kv_load_failed = False
        try:
            queued = self.cache_linker.load(req.rid, [t for _, t in transfers])
        except Exception:
            # An exception is not proof that a native worker never took ownership.
            if self.cache_linker.cancel_queued_load(req.rid):
                self.private_loads.pop(req.rid)
                self._abort_transfers(req, transfers, end)
                req.external_kv_load_pending = False
            raise
        accepted = preflight(
            self.cache, 3, req.rid, (hit.device_hit_len, end, 0), queued
        )
        if not accepted:
            if queued and not self.cache_linker.cancel_queued_load(req.rid):
                raise RuntimeError(f"Cannot cancel rejected external load: {req.rid}")
            self.private_loads.pop(req.rid)
            self._abort_transfers(req, transfers, end)
            req.external_kv_load_pending = False
            req.kv.component_evicted_seqlens = saved_evicted
            return 0, req.last_node
        # The tree lock and cache_protected_len continue to cover L1 ONLY.
        return len(full.device_indices), req.last_node

    def adopt_request(self, req):
        self._raise_if_cleanup_quarantined()
        private = self.private_loads.get(req.rid)
        if private is not None:
            private.adopted = True

    def prefix_device_indices(self, req):
        private = self.private_loads[req.rid]
        path = self.cache.path_device_indices(req.last_node)
        return torch.cat([path[: private.prev_prefix_len], private.device_indices])

    def _abort_transfers(self, req, transfers, prefix_len):
        self._raise_if_cleanup_quarantined()
        if not transfers:
            return
        full = next((t for _, t in transfers if t.name == PoolName.KV), None)
        # Only unadopted allocations use component ABORT. Once indices are adopted,
        # request cleanup owns both FULL and SWA, including donated SWA.
        ordered = list(reversed(transfers))
        states = ["not_attempted"] * len(ordered)
        for index, (component, transfer) in enumerate(ordered):
            states[index] = "uncertain"
            try:
                component.update_external_linker_load(
                    ExternalLinkerLoadPhase.ABORT, req, full, transfer, prefix_len
                )
            except BaseException as error:
                if not hasattr(self, "cleanup_quarantine"):
                    self.cleanup_quarantine = {}
                # Keep indices/request alive even when the caller already popped
                # private_loads. A failed free may have partially changed a pool.
                self.cleanup_quarantine[req.rid] = {
                    "req": req,
                    "transfers": ordered,
                    "prefix_len": prefix_len,
                    "states": states,
                }
                detail = [
                    (
                        str(t.name),
                        (
                            int(t.device_indices.numel())
                            if t.device_indices is not None
                            else 0
                        ),
                        state,
                    )
                    for (_, t), state in zip(ordered, states)
                ]
                logger.error(
                    "External KV release quarantined: rid=%s prefix_len=%d "
                    "components=%s; process teardown required",
                    req.rid,
                    prefix_len,
                    detail,
                )
                raise ExternalLinkerCleanupError(
                    f"External KV release uncertain for rid={req.rid}"
                ) from error
            states[index] = "released"

    def finish_loads(self, consumer_index):
        self._raise_if_cleanup_quarantined()
        rids = self.load_batches.get(consumer_index, [])
        self._finish_sequence += 1
        preflight(
            self.cache,
            4,
            (self._finish_sequence, consumer_index, rids),
            (len(rids), consumer_index, 0),
        )
        if consumer_index not in self.load_batches:
            return []
        # Backends wait for every layer AND worker/session termination. A wait
        # timeout is fatal and leaves slots retained, never a failed completion.
        self.cache_linker.finish_loads(consumer_index)
        batch = self.cache_linker.pop_completed_load()
        identities = [rid for rid, _ in batch]
        preflight(self.cache, 5, (consumer_index, identities), (len(batch), 0, 0))
        if identities != rids:
            raise RuntimeError(
                f"Out-of-order external load completion: {identities} != {rids}"
            )
        statuses = torch.tensor(
            [bool(ok) and not self.private_loads[rid].cancelled for rid, ok in batch],
            dtype=torch.int,
        )
        self.cache._all_reduce_attn_groups(statuses, torch.distributed.ReduceOp.MIN)
        failed = []
        for (rid, _), ok in zip(batch, statuses.tolist()):
            private = self.private_loads[rid]
            if ok and not private.adopted:
                raise RuntimeError(
                    f"Completed external load has no request owner: {rid}"
                )
            self.private_loads.pop(rid)
            req = private.req
            req.external_kv_load_pending = False
            if not ok:
                req.external_kv_load_failed = True
                req.skip_radix_cache_insert = True
                failed.append(req)
                if not private.adopted:
                    self._abort_transfers(req, private.transfers, private.prefix_len)
            # On success ownership passes to the request; its normal checkpoint
            # performs insertion/dedup. On failure an allocated row is ALSO freed
            # only by request cleanup, never by both linker and request.
        self.load_batches.pop(consumer_index)
        return failed

    # ---- offload: device -> remote, driven by the write-through chain ----

    def offload_nodes(self, node_ids: Sequence[NodeId]) -> None:
        """Persist a write-through chain, skipping nodes already in the store."""
        for node_id in node_ids:
            transfers = self.cache.tree_core.build_external_linker_offload_transfers(
                node_id
            )
            if transfers is not None:
                if self._skip_swa:
                    transfers = [t for t in transfers if t.name != PoolName.SWA]
                self._offload_node(node_id, transfers)

    def _offload_node(self, node_id: NodeId, transfers: list[PoolTransfer]) -> None:
        cache = self.cache
        lock_params = cache.inc_lock_ref(node_id).to_dec_params()
        try:
            queued = self.cache_linker.offload(transfers)
        except BaseException:
            cache.dec_lock_ref(node_id, lock_params)
            raise
        if not queued:
            cache.dec_lock_ref(node_id, lock_params)
            return

        cache.tree_core.mark_external_linker_offload_pending(node_id)
        self.pending_offloads.append(_PendingOffload(node_id, lock_params, [node_id]))

    def replace_pending_offload_node(
        self, ack_id: NodeId, old_node_id: NodeId, new_node_ids: list[NodeId]
    ) -> None:
        for index, pending in enumerate(self.pending_offloads):
            if pending.lock_node_id != ack_id:
                continue
            publish_node_ids = []
            for node_id in pending.publish_node_ids:
                if node_id == old_node_id:
                    publish_node_ids.extend(new_node_ids)
                else:
                    publish_node_ids.append(node_id)
            self.pending_offloads[index] = pending._replace(
                publish_node_ids=publish_node_ids
            )
            return

    def num_completed_offloads(self) -> int:
        return min(
            self.cache_linker.num_completed_offloads(), len(self.pending_offloads)
        )

    def num_completed_loads(self) -> int:
        return self.cache_linker.num_completed_loads()

    def take_completed_offloads(self, finish_count: int) -> list[bool]:
        assert finish_count <= len(self.pending_offloads)
        return [self.cache_linker.pop_completed_offload() for _ in range(finish_count)]

    def commit_completed_offloads(self, successes: Sequence[bool]) -> None:
        assert len(successes) <= len(self.pending_offloads)
        for success in successes:
            pending = self.pending_offloads.pop(0)
            self.cache.tree_core.finish_external_linker_offload(
                pending.publish_node_ids, pending.lock_node_id, success
            )
            self.cache.dec_lock_ref(pending.lock_node_id, pending.lock_params)

    def start_layer_wise_loading(self) -> int:
        self._raise_if_cleanup_quarantined()
        scheduled = {rid for rids in self.load_batches.values() for rid in rids}
        rids = [rid for rid in self.private_loads if rid not in scheduled]
        preflight(self.cache, 6, rids, (len(rids), 0, 0))
        index = self.cache_linker.start_layer_wise_loading()
        if index >= 0:
            self.load_batches[index] = rids
        return index

    # ---- lifecycle ----

    def reset(self) -> None:
        self._raise_if_cleanup_quarantined()
        self.cache_linker.reset()
        self.hit_markers.clear()
        self._release_pending_locks()

    def _release_pending_locks(self) -> None:
        self._raise_if_cleanup_quarantined()
        for private in self.private_loads.values():
            if not private.adopted:
                self._abort_transfers(
                    private.req, private.transfers, private.prefix_len
                )
            private.req.external_kv_load_pending = False
            private.req.external_kv_load_failed = True
            private.req.skip_radix_cache_insert = True
        self.private_loads.clear()
        self.load_batches.clear()
        for pending in self.pending_offloads:
            self.cache.tree_core.finish_external_linker_offload(
                pending.publish_node_ids, pending.lock_node_id, False
            )
            self.cache.dec_lock_ref(pending.lock_node_id, pending.lock_params)
        self.pending_offloads.clear()

    def release_request(self, rid: str) -> None:
        self._raise_if_cleanup_quarantined()
        self.hit_markers.pop(rid, None)
        private = self.private_loads.get(rid)
        if private is None:
            return
        private.cancelled = True
        if not self.cache_linker.cancel_queued_load(rid):
            # The caller must not free the request row while the native writer
            # still owns it. Fail closed; normal result processing drains first.
            raise RuntimeError(f"External KV release before load drain: {rid}")
        self.private_loads.pop(rid)
        private.req.external_kv_load_pending = False
        private.req.external_kv_load_failed = True
        private.req.skip_radix_cache_insert = True
        if not private.adopted:
            self._abort_transfers(private.req, private.transfers, private.prefix_len)

    def close(self) -> None:
        self._raise_if_cleanup_quarantined()
        self.cache_linker.close()
        self._release_pending_locks()
