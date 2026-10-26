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
from typing import TYPE_CHECKING, NamedTuple

import torch

from sglang.srt.disaggregation.utils import DisaggregationCollectiveError
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
    ExternalLinkerLoadPhase,
    LinkerTransferPhase,
    TreeComponent,
)
from sglang.srt.mem_cache.utils import get_hash_str

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache


class ExternalLinkerLoadError(RuntimeError):
    """A recoverable external-cache load failure for the current batch."""


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
        """Cancel a request-private load that has not started yet."""

    @abstractmethod
    def num_completed_loads(self) -> int:
        """Return the number of completed load batches waiting to be consumed."""

    @abstractmethod
    def pop_completed_load(self) -> tuple[list[str], bool]:
        """Consume the oldest completed load batch.

        Returns its request IDs and whether every transfer in it landed. A
        failed batch must still be reported here, not raised, so the wrapper
        can release request-private slots and abort the requests that were
        reading them. Failure is per batch, not per request: the plans are
        merged, so one bad key costs every request in the batch its hit.
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


class _PendingOffload(NamedTuple):
    lock_node_id: NodeId
    lock_params: DecLockRefParams
    publish_node_ids: list[NodeId]


class _PrivateLoad(NamedTuple):
    """State for an external load that has not been published to the tree."""

    req: Req
    transfers: list[tuple[TreeComponent, PoolTransfer]]
    prev_prefix_len: int
    prefix_len: int
    device_indices: torch.Tensor


class UnifiedCacheLinkerWrapper:
    """Drives an external KV store on behalf of one :class:`UnifiedRadixCache`."""

    def __init__(
        self,
        cache: UnifiedRadixCache,
        cache_linker: UnifiedCacheLinker,
    ):
        self.cache = cache
        self.cache_linker = cache_linker
        # rid -> what match found, consumed by the next init_load_back.
        self.hit_markers: dict[str, ExternalCacheHitMarker] = {}
        # Legacy publish-then-rollback state.  The external load_back path
        # below never populates these maps; they remain for older callers and
        # focused compatibility tests while the API is retired.
        self.pending_loads: dict[str, tuple[NodeId, DecLockRefParams, NodeId]] = {}
        # rid -> the chain a failed load published for it, endpoint-first.
        # The nodes hold no valid KV and are already cut out of the tree; the
        # cache frees them in that request's own cache_finished_req, which is
        # where its lock on them goes.
        self.failed_chains: dict[str, list[NodeId]] = {}
        # Load batches popped by take_completed_loads and awaiting the reduced
        # verdict in commit_completed_loads.
        self.taken_loads: list[tuple[list[str], bool]] = []
        # External linker loads use request-private slots.  They are deliberately
        # kept separate from pending_loads, which is retained only for legacy
        # callers that still use the publish-then-rollback helper.
        self.private_loads: dict[str, _PrivateLoad] = {}
        # Offloads in flight, each holding a lock on its node until it lands.
        self.pending_offloads: list[_PendingOffload] = []

        cache.tree_core.enable_external_cache_linker = True
        cache.write_through_threshold = 1

    @property
    def layer_done_counter(self) -> object:
        return self.cache_linker.layer_done_counter

    def has_hit(self, rid: str) -> bool:
        return rid in self.hit_markers

    def has_pending_operations(self) -> bool:
        return bool(
            self.pending_loads
            or self.failed_chains
            or self.private_loads
            or self.pending_offloads
            or self.taken_loads
        )

    # ---- match: probe the remote store and report host_hit_length ----

    def match(self, key: RadixKey, req: Req, result: MatchResult) -> MatchResult:
        cache = self.cache
        page = cache.page_size
        device_hit_len = int(result.device_indices.numel())
        self.hit_markers.pop(req.rid, None)

        known_hit_len = req.external_cache_hit_length if cache.pp_size > 1 else None
        if known_hit_len is not None:
            # PP0's absolute boundary also survives later local L1 rematches.
            key = key[:known_hit_len]
        elif cache.pp_size > 1:
            req.external_cache_hit_length = 0
            if cache.pp_rank != 0:
                return result

        # Every rank in an attention group must reach the same collective in
        # the same order.  In particular, do not return early when this rank
        # has a full local hit, no hash anchor, or a component that cannot
        # build a transfer.  Those are local facts; the other ranks still
        # need a fixed-size preflight to learn that the external lookup must
        # be skipped for the whole group.
        tail_hashes = (
            self._tail_hashes(key, result, device_hit_len)
            if device_hit_len < len(key)
            else []
        )
        lookup_transfers = []
        local_valid = bool(tail_hashes)
        if local_valid:
            for component in cache._components_tuple:
                transfer = component.build_external_linker_transfer(
                    LinkerTransferPhase.LOOKUP, None, tail_hashes
                )
                if transfer is None:
                    local_valid = False
                    break
                lookup_transfers.append(transfer)

        if known_hit_len is None:
            local_valid, common_tail_pages = self._match_preflight(
                req.rid,
                key_len=len(key),
                device_hit_len=device_hit_len,
                tail_pages=len(tail_hashes),
                local_valid=local_valid,
            )
            if not local_valid:
                return result
            # The preflight guarantees that every rank has the same geometry,
            # but retain the negotiated bound so a future backend cannot
            # accidentally create different mask lengths.
            tail_hashes = tail_hashes[:common_tail_pages]

        by_pool = {transfer.name: transfer for transfer in lookup_transfers}

        # Tail-relative: page 0 of `tail_hashes` is the first uncached page.
        if known_hit_len is None:
            hit_pages = self._sync_restorable_prefix(
                self.cache_linker.lookup(req.rid, lookup_transfers),
                num_pages=len(tail_hashes),
                device_hit_pages=0,
            )
            if cache.pp_size > 1 and hit_pages:
                req.external_cache_hit_length = device_hit_len + hit_pages * page
        else:
            hit_pages = len(tail_hashes)
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

    def _match_preflight(
        self,
        rid: str,
        *,
        key_len: int,
        device_hit_len: int,
        tail_pages: int,
        local_valid: bool,
    ) -> tuple[bool, int]:
        """Negotiate match geometry before any variable-size collective.

        The payload is always eight int64 values.  The first half carries
        local values and the second half carries their negation, allowing one
        MIN reduction to recover both the group minimum and maximum.  A
        A geometry mismatch means that ranks are looking at different
        radix-tree prefixes.  Continuing with local results would make the
        subsequent forward collectives consume different KV layouts, so that
        condition is fatal.  A validity-only mismatch is safe to handle by
        abandoning the external lookup on every rank.
        """
        preflight = torch.tensor(
            [
                key_len,
                -key_len,
                device_hit_len,
                -device_hit_len,
                tail_pages,
                -tail_pages,
                int(local_valid),
                -int(local_valid),
            ],
            dtype=torch.int64,
        )
        self.cache._all_reduce_attn_groups(
            preflight, torch.distributed.ReduceOp.MIN
        )

        same_geometry = (
            int(preflight[0]) == -int(preflight[1])
            and int(preflight[2]) == -int(preflight[3])
            and int(preflight[4]) == -int(preflight[5])
        )
        all_valid = int(preflight[6]) == 1 and int(preflight[7]) == -1
        mixed_valid = int(preflight[6]) != -int(preflight[7])
        if not same_geometry:
            logger.error(
                "External KV match preflight geometry mismatch: rid=%s "
                "local=(key_len=%d,device_hit_len=%d,tail_pages=%d) "
                "group=(key_len=[%d,%d],device_hit_len=[%d,%d],"
                "tail_pages=[%d,%d])",
                rid,
                key_len,
                device_hit_len,
                tail_pages,
                int(preflight[0]),
                -int(preflight[1]),
                int(preflight[2]),
                -int(preflight[3]),
                int(preflight[4]),
                -int(preflight[5]),
            )
            raise DisaggregationCollectiveError(
                "external KV match geometry differs across ranks"
            )
        if (
            not mixed_valid
            and not all_valid
            and int(preflight[0]) == int(preflight[2])
            and int(preflight[4]) == 0
        ):
            # Every rank has a complete local hit with no tail pages to load.
            # This is a normal no-op after preflight: all ranks agree and the
            # device cache already holds the full prefix, so no L3 transfer is
            # needed.  Log at DEBUG to avoid flooding the log at high concurrency
            # where every cached request would produce a WARNING.
            logger.debug(
                "External KV match preflight: complete local hit, no L3 load "
                "needed rid=%s key_len=%d device_hit=%d",
                rid,
                key_len,
                int(preflight[2]),
            )
            return False, 0
        if mixed_valid or not all_valid:
            logger.warning(
                "External KV match preflight rejected: rid=%s key_len=%d "
                "device_hit=[%d,%d] tail_pages=[%d,%d] valid=%s",
                rid,
                key_len,
                int(preflight[2]),
                -int(preflight[3]),
                int(preflight[4]),
                -int(preflight[5]),
                all_valid,
            )
            return False, 0
        return True, int(preflight[4])

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
        if tail_len == 0:
            return []
        return get_hash_str(
            key[device_hit_len : device_hit_len + tail_len],
            last_hash,
            page_size=page,
        )

    # ---- init_load_back: remote -> request-private device slots ----

    def load_back(self, req: Req) -> tuple[torch.Tensor, NodeId]:
        cache = self.cache
        empty_indices = cache.tree_core.empty_match_result.device_indices
        hit = self.hit_markers.pop(req.rid, None)
        if hit is None:
            return empty_indices, req.last_node

        device_hit_len = hit.device_hit_len
        tail_hashes = hit.tail_hashes
        prefix_len = device_hit_len + len(tail_hashes) * cache.page_size

        # Build per-component linker transfers.
        component_transfers: list[tuple[TreeComponent, PoolTransfer]] = []
        for component in cache._components_tuple:
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.LOAD, None, tail_hashes
            )
            if transfer is None:
                self._abort_component_transfers(
                    req, component_transfers, device_hit_len, prefix_len
                )
                self._clear_private_slot_ownership(req)
                return empty_indices, req.last_node
            component_transfers.append((component, transfer))

        # Keys can be evicted remotely (master memory-watermark eviction)
        # between the match-time lookup and this load-back; a stale hit would
        # then fail the async layer-wise session fatally. Re-check existence
        # here so the request degrades to a plain cache miss instead.
        revalidate = getattr(self.cache_linker, "revalidate_load", None)
        if revalidate is not None and not revalidate(
            [transfer for _, transfer in component_transfers]
        ):
            self._abort_component_transfers(
                req, component_transfers, device_hit_len, prefix_len
            )
            self._clear_private_slot_ownership(req)
            return empty_indices, req.last_node

        full_transfer = component_transfers[0][1]
        assert full_transfer.name == PoolName.KV
        self._remember_private_slot_range(req, device_hit_len, prefix_len)
        # Prepare component metadata before handing the transfer to the
        # backend worker.  A worker may consume a newly queued request
        # immediately; preparing afterwards would let an exception race with
        # an in-flight DMA and free request-private slots while the worker was
        # still writing them.
        try:
            self._update_load(
                ExternalLinkerLoadPhase.PREPARE,
                req,
                component_transfers,
                prefix_len,
            )
        except BaseException:
            self._abort_component_transfers(
                req, component_transfers, device_hit_len, prefix_len
            )
            raise

        # Queue the prepared transfer without publishing any tree state.  The
        # load owns only request-private slots until every layer and rank
        # reports success; the normal request result path performs the single
        # radix insertion after the forward.  This is the #40896 transaction
        # boundary.
        try:
            queued = self.cache_linker.load(
                req.rid, [transfer for _, transfer in component_transfers]
            )
        except BaseException:
            # A backend is allowed to enqueue before reporting an error.  Drop
            # that not-yet-started entry first; only then return the private
            # slots to the allocators.
            cancelled = self.cache_linker.cancel_queued_load(req.rid)
            if cancelled:
                # The backend confirmed that no worker owns the request, so
                # the component slots are safe to return immediately.
                self._abort_component_transfers(
                    req, component_transfers, device_hit_len, prefix_len
                )
            else:
                # A false cancellation result means the backend may already
                # have handed the request to a worker. Retain the private
                # descriptor until completion/failure is published; do not
                # release a buffer while native RDMA can still write it.
                self.private_loads[req.rid] = _PrivateLoad(
                    req=req,
                    transfers=component_transfers,
                    prev_prefix_len=device_hit_len,
                    prefix_len=prefix_len,
                    device_indices=full_transfer.device_indices,
                )
                logger.warning(
                    "Retaining private external load after cancellation failed: "
                    "rid=%s; waiting for backend terminal status",
                    req.rid,
                )
            raise
        if not queued:
            self._abort_component_transfers(
                req, component_transfers, device_hit_len, prefix_len
            )
            self._clear_private_slot_ownership(req)
            return empty_indices, req.last_node

        self.private_loads[req.rid] = _PrivateLoad(
            req=req,
            transfers=component_transfers,
            prev_prefix_len=device_hit_len,
            prefix_len=prefix_len,
            device_indices=full_transfer.device_indices,
        )
        return full_transfer.device_indices, req.last_node

    def _queue_load(
        self,
        rid: str,
        node_id: NodeId,
        transfers: list[PoolTransfer],
        anchor: NodeId,
    ) -> None:
        """Legacy publish-then-rollback queue; external loads do not use it."""
        if not transfers:
            return
        assert rid not in self.pending_loads
        lock_params = self.cache.inc_lock_ref(node_id).to_dec_params()
        try:
            queued = self.cache_linker.load(rid, transfers)
        except BaseException:
            self.cache.dec_lock_ref(node_id, lock_params)
            raise
        if not queued:
            self.cache.dec_lock_ref(node_id, lock_params)
            raise RuntimeError(f"Failed to queue the linker load for rid={rid!r}.")
        self.pending_loads[rid] = (node_id, lock_params, anchor)

    def _update_load(
        self,
        phase: ExternalLinkerLoadPhase,
        req: Req,
        component_transfers: list[tuple[TreeComponent, PoolTransfer]],
        prefix_len: int,
        *,
        insert_result=None,
        canonical_full: torch.Tensor | None = None,
    ) -> list[PoolTransfer]:
        if not component_transfers:
            return []
        full = component_transfers[0][1]
        result = []
        transfers = (
            reversed(component_transfers)
            if phase == ExternalLinkerLoadPhase.ABORT
            else component_transfers
        )
        for component, transfer in transfers:
            component_canonical = canonical_full
            if phase == ExternalLinkerLoadPhase.COMMIT:
                assert insert_result is not None and insert_result.adopted_ranges is not None
                coverage_start = prefix_len - len(transfer.device_indices)
                ranges = [
                    (max(start, coverage_start), min(end, prefix_len))
                    for start, end in insert_result.adopted_ranges.get(
                        component.component_type, ()
                    )
                    if max(start, coverage_start) < min(end, prefix_len)
                ]
                indices, keys = self._select_adopted_pages(
                    transfer.device_indices,
                    ranges,
                    prefix_len,
                    transfer.keys,
                )
                if not keys:
                    continue
                transfer.device_indices = indices
                transfer.keys = keys
                component_canonical, _ = self._select_adopted_pages(
                    canonical_full, ranges, prefix_len
                )
            transfer = component.update_external_linker_load(
                phase,
                req,
                full,
                transfer,
                prefix_len,
                insert_result=insert_result,
                canonical_full=component_canonical,
            )
            if transfer is not None:
                result.append(transfer)
        return result

    def _select_adopted_pages(
        self,
        indices: torch.Tensor,
        ranges: Sequence[tuple[int, int]],
        prefix_len: int,
        keys: Sequence[str] | None = None,
    ) -> tuple[torch.Tensor, list[str]]:
        """Legacy filtering for callers that still use publish-then-commit.

        The external-linker ``load_back`` path no longer calls this helper; it
        is retained for the legacy queue API and its focused compatibility
        tests while that API is being retired.
        """
        page = self.cache.page_size
        coverage_start = prefix_len - len(indices)
        pages = indices.reshape(-1, page)
        if keys is not None:
            assert len(keys) == len(pages)

        chunks = []
        selected_keys = []
        for start, end in ranges:
            start = max(start, coverage_start)
            end = min(end, prefix_len)
            if start >= end:
                continue
            assert (start - coverage_start) % page == 0
            assert (end - coverage_start) % page == 0
            first = (start - coverage_start) // page
            last = (end - coverage_start) // page
            chunks.append(pages[first:last].reshape(-1))
            if keys is not None:
                selected_keys.extend(keys[first:last])

        if not chunks:
            return indices[:0], selected_keys
        selected = chunks[0] if len(chunks) == 1 else torch.cat(chunks)
        return selected, selected_keys

    # ---- offload: device -> remote, driven by the write-through chain ----

    def offload_nodes(self, node_ids: Sequence[NodeId]) -> None:
        """Persist a write-through chain.

        Skips nodes the store already has. A detached node holds no KV and
        writing it would put corruption in the store under the legitimate
        content hash for that token sequence -- but it is also off the tree,
        so the insert walk that named these ids cannot have reached one.
        Assert rather than skip: one turning up here means the tree is
        corrupt.
        """
        for node_id in node_ids:
            node = self.cache.resolve_node_handle(node_id)
            assert not node.detached, f"write-through named detached node {node_id}"
            if not node.external_cache_stored:
                self._offload_node(node_id)

    def _offload_node(self, node_id: NodeId) -> None:
        cache = self.cache
        node = cache.resolve_node_handle(node_id)
        transfers = []
        for component in cache._components_tuple:
            transfer = component.build_external_linker_transfer(
                LinkerTransferPhase.OFFLOAD, node, None
            )
            if transfer is not None:
                transfers.append(transfer)

        lock_params = cache.inc_lock_ref(node_id).to_dec_params()
        try:
            queued = self.cache_linker.offload(transfers)
        except BaseException:
            cache.dec_lock_ref(node_id, lock_params)
            raise
        if not queued:
            cache.dec_lock_ref(node_id, lock_params)
            return

        cache.tree_core.mark_write_through_pending(node_id)
        node.external_cache_stored = True
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

    def take_completed_loads(self, finish_count: int) -> list[bool]:
        """Pop finished load batches, reporting the local verdict only.

        Split from the commit half so the caller can MIN-reduce the verdict
        across the attention group first. A load failure is rank-local -- one
        rank's batch_get can miss a key the others read fine -- so a rank that
        poisons and aborts on its own verdict diverges from the ranks that do
        not: the request is aborted on some ranks and served on the rest, and
        the rank owning the output stream emits KV that never arrived.

        This mirrors take_completed_offloads / commit_completed_offloads, which
        already reduce their verdict for the same reason.
        """
        return [success for _rids, success in self.take_completed_load_batches(finish_count)]

    def take_completed_load_batches(
        self, finish_count: int
    ) -> list[tuple[list[str], bool]]:
        """Pop completed batches while retaining their RID order.

        The scheduler performs a fixed-size cross-rank fingerprint before it
        reduces the success bits.  Positional success reduction without this
        identity check can apply rank A's result to rank B's request.
        """
        taken = [self.cache_linker.pop_completed_load() for _ in range(finish_count)]
        self.taken_loads.extend(taken)
        return taken

    def finish_loads(
        self, finish_count: int | None = None
    ) -> list[tuple[list[str], bool]]:
        """Move completed loads into the rank-consensus commit phase.

        The completion queue is local to a linker worker.  This method only
        takes descriptors and preserves their RID order; callers must run the
        fixed-size RID preflight before passing verdicts to
        :meth:`commit_completed_loads`.  Keeping the handoff separate makes it
        impossible for a rank to clean up its private slots before peers have
        identified the same batch.
        """
        if finish_count is None:
            finish_count = self.num_completed_loads()
        return self.take_completed_load_batches(finish_count)

    def commit_completed_loads(self, successes: Sequence[bool]) -> list[str]:
        """Apply the reduced verdict; return the rids of the failed batches.

        External linker loads use request-private slots and therefore have no
        failed radix chain to detach.  The old pending_loads branch is kept for
        legacy callers only; external load_back never populates it.
        """
        assert len(successes) <= len(self.taken_loads)
        failed: list[str] = []
        for success in successes:
            rids, _local_success = self.taken_loads.pop(0)
            for rid in rids:
                private = self.private_loads.pop(rid, None)
                if private is not None:
                    if not success:
                        self._abort_private_load(rid, private)
                        failed.append(rid)
                    else:
                        # The request result path now owns these slots and is
                        # the only path allowed to insert/free them.
                        self._handoff_private_slots(private.req)
                    continue
                entry = self.pending_loads.pop(rid, None)
                if entry is None:
                    # Released early by release_request while queued.
                    continue
                node_id, lock_params, anchor = entry
                if not success:
                    self._detach_failed_chain(rid, node_id, anchor)
                    failed.append(rid)
                self.cache.dec_lock_ref(node_id, lock_params)
        return failed

    @staticmethod
    def _remember_private_slot_range(
        req: Req, prev_prefix_len: int, prefix_len: int
    ) -> None:
        """Record the request-row interval owned by an external load.

        The interval is derived from request geometry, never from a
        component's device-index count.  FULL and SWA components can expose
        different physical index counts for the same request, while both
        components share the request-row coordinate space used by cleanup.
        """
        end = max(0, int(prefix_len))
        start = max(0, min(int(prev_prefix_len), end))
        req.external_kv_private_slots_start = start
        req.external_kv_private_slots_end = end
        req.external_kv_private_slots_released = False
        req.external_kv_private_swa_slots_released = False

    @staticmethod
    def _mark_private_slots_released(req: Req) -> None:
        """Transfer release ownership to the linker exactly once."""
        if getattr(req, "external_kv_private_slots_released", False):
            logger.debug(
                "External linker private slots already released: rid=%s", req.rid
            )
            return
        req.external_kv_private_slots_released = True

    @staticmethod
    def _clear_private_slot_ownership(req: Req) -> None:
        """Forget a synchronously aborted load before cache-miss fallback.

        ``load_back`` returns empty indices for a pre-queue miss and the
        scheduler immediately recomputes the prefix. The old private range is
        no longer present in the request row, so retaining its exclusion marker
        would make later cleanup skip newly allocated request-owned slots at
        the same row positions.
        """
        req.external_kv_private_slots_released = False
        req.external_kv_private_swa_slots_released = False
        req.external_kv_private_slots_start = None
        req.external_kv_private_slots_end = None

    def _abort_component_transfers(
        self,
        req: Req,
        component_transfers: list[tuple[TreeComponent, PoolTransfer]],
        prev_prefix_len: int,
        prefix_len: int,
    ) -> None:
        """Abort transfers before their indices are attached to the request."""
        # Even a pre-queue failure can reach the normal request cleanup path.
        # Claim the full private interval before releasing component buffers so
        # cache_finished_req/release_kv_cache skip it on the next pass.
        for _component, transfer in component_transfers:
            if transfer.name == PoolName.KV and transfer.device_indices is not None:
                self._remember_private_slot_range(
                    req, prev_prefix_len, prefix_len
                )
                self._mark_private_slots_released(req)
                break
        self._update_load(
            ExternalLinkerLoadPhase.ABORT,
            req,
            component_transfers,
            prefix_len,
        )

    @staticmethod
    def _handoff_private_slots(req: Req) -> None:
        """Make the normal request/radix path the sole owner after success."""
        req.external_kv_private_slots_released = False
        req.external_kv_private_swa_slots_released = False
        req.external_kv_private_slots_start = None
        req.external_kv_private_slots_end = None

    def _abort_private_load(
        self, rid: str, private: _PrivateLoad | None = None
    ) -> bool:
        """Abort one request-private load, with a single ownership handoff.

        Callers remove the entry from ``private_loads`` before invoking this
        helper.  That makes scheduler failure scans, request cleanup and
        backend completion handling idempotent: only the caller that owns the
        popped entry may free its component slots.  The backend completion
        queue is published only after its layer worker and storage session
        have returned, so this helper is never used to free an in-flight DMA.
        """
        if private is None:
            private = self.private_loads.pop(rid, None)
        if private is None:
            return False
        # Mark before calling component abort. If a component raises after
        # partially freeing, request cleanup must not retry the same range.
        if getattr(private.req, "external_kv_private_slots_start", None) is None:
            self._remember_private_slot_range(
                private.req, private.prev_prefix_len, private.prefix_len
            )
        self._mark_private_slots_released(private.req)
        try:
            self._update_load(
                ExternalLinkerLoadPhase.ABORT,
                private.req,
                private.transfers,
                private.prefix_len,
            )
        except BaseException:
            # The ownership is intentionally not reinserted: retrying an
            # allocator free can double-release a partially freed component.
            # Keep the failure visible for invariant checks and process-level
            # shutdown diagnostics instead.
            logger.exception(
                "Failed to release private external load rid=%r after "
                "backend completion; slots may require process teardown",
                rid,
            )
        return True

    def _detach_failed_chain(self, rid: str, node_id: NodeId, anchor: NodeId) -> None:
        """Cut the chain this load published out of the tree, endpoint first.

        ``external_cache_stored`` is left alone: the chain really did come from
        the store, and clearing it is what makes the unfilled pages eligible
        for write-through -- the opposite of what is wanted.

        Detaching, rather than only flagging, is what keeps the chain from
        being served: match_prefix reads no such flag, so a merely flagged
        chain stayed matchable until the purge managed to drop it, and the
        first request that matched it gave it a device child and blocked the
        reclaim for good.

        The whole chain is filed, not just its endpoint. Deleting the endpoint
        does not cascade: ``_iteratively_delete_tombstone_leaf`` stops at the
        first ancestor still holding a device value, and every node this load
        just filled has one. Endpoint-first order matters too -- a parent only
        becomes a device leaf once its child is gone.

        Filed under the rid because that request's ``cache_finished_req`` is
        the one point where the whole chain becomes reclaimable: Full is a
        path-unlock, so its single ``dec_lock_ref`` drops ``lock_ref`` on every
        node of the chain at once.
        """
        self.failed_chains.setdefault(rid, []).extend(
            self.cache.tree_core.detach_external_load_chain(node_id, anchor)
        )

    def take_failed_chain(self, rid: str) -> list[NodeId]:
        """Pop the chain filed for this rid, if a load failed for it."""
        return self.failed_chains.pop(rid, [])

    def take_completed_offloads(self, finish_count: int) -> list[bool]:
        assert finish_count <= len(self.pending_offloads)
        return [self.cache_linker.pop_completed_offload() for _ in range(finish_count)]

    def commit_completed_offloads(self, successes: Sequence[bool]) -> None:
        assert len(successes) <= len(self.pending_offloads)
        for success in successes:
            pending = self.pending_offloads.pop(0)
            for node_id in pending.publish_node_ids:
                node = self.cache.resolve_node_handle(node_id)
                if node.write_through_pending_id == pending.lock_node_id:
                    node.write_through_pending_id = None
                node.external_cache_stored = success
            self.cache.dec_lock_ref(pending.lock_node_id, pending.lock_params)

    def start_layer_wise_loading(self) -> int:
        return self.cache_linker.start_layer_wise_loading()

    # ---- lifecycle ----

    def reset(self) -> None:
        self.cache_linker.reset()
        self.hit_markers.clear()
        self._release_pending_locks()

    def _release_pending_locks(self) -> None:
        for rid in list(self.private_loads):
            private = self.private_loads.pop(rid, None)
            self._abort_private_load(rid, private)
        for node_id, lock_params, _anchor in self.pending_loads.values():
            self.cache.dec_lock_ref(node_id, lock_params)
        self.pending_loads.clear()
        for pending in self.pending_offloads:
            for node_id in pending.publish_node_ids:
                node = self.cache.resolve_node_handle(node_id)
                if node.write_through_pending_id == pending.lock_node_id:
                    node.write_through_pending_id = None
                node.external_cache_stored = False
            self.cache.dec_lock_ref(pending.lock_node_id, pending.lock_params)
        self.pending_offloads.clear()
        self.failed_chains.clear()
        self.taken_loads.clear()

    def release_request(self, rid: str) -> None:
        # failed_chains is deliberately untouched: the chain outlives the
        # request's linker state, and cache_finished_req is what frees it.
        self.hit_markers.pop(rid, None)
        # Only a load that has not started can be cancelled here. One already
        # in flight keeps its pending_loads entry and its lock until
        # commit_completed_loads retires the batch, which the linkers guarantee
        # happens even when the transfer fails.
        if self.cache_linker.cancel_queued_load(rid):
            private = self.private_loads.pop(rid, None)
            if private is not None:
                self._abort_private_load(rid, private)
            elif rid in self.pending_loads:
                node_id, lock_params, _anchor = self.pending_loads.pop(rid)
                self.cache.dec_lock_ref(node_id, lock_params)

    def close(self) -> None:
        self.cache_linker.close()
        self._release_pending_locks()
