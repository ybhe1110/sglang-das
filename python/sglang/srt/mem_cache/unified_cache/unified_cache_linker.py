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

A load lands in device slots private to the loading request and joins the tree
only when that request inserts it like any computed KV. A failed load therefore
reaches nothing but its own requests, which the scheduler aborts.
"""

from __future__ import annotations

import hashlib
import logging
import threading
import zlib
from dataclasses import replace
from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence
from concurrent.futures import Future
from contextlib import contextmanager
from copy import copy
from typing import TYPE_CHECKING, NamedTuple, Optional

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
    TreeComponent,
)
from sglang.srt.mem_cache.utils import get_storage_hash_str

if TYPE_CHECKING:
    from sglang.srt.managers.schedule_batch import Req
    from sglang.srt.mem_cache.unified_cache.unified_tree_core_interface import NodeId
    from sglang.srt.mem_cache.unified_radix_cache import UnifiedRadixCache

logger = logging.getLogger(__name__)

_EXTERNAL_LINKER_SUPPORTED_COMPONENTS = frozenset(
    {
        ComponentType.FULL,
        ComponentType.SWA,
        ComponentType.MAMBA,
    }
)


class LayerWiseLoadCounter:
    """CPU completion counter compatible with KV pools' layer wait hook.

    A failed layer does not raise into the forward that waits on it: the
    forward runs on whatever the slots hold, and :meth:`finish` reports the
    failure afterwards so the scheduler can abort the loading requests before
    their output is used.
    """

    def __init__(
        self,
        num_layers: int,
        on_layer_ready: Optional[Callable[[int, int], None]] = None,
    ):
        self.num_layers = num_layers
        self.on_layer_ready = on_layer_ready
        self.producer_index = -1
        self.consumer_index = -1
        self.futures: dict[int, list[Future]] = {}

    def update_producer(self) -> int:
        self.producer_index += 1
        self.futures[self.producer_index] = [Future() for _ in range(self.num_layers)]
        return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def complete(self, index: int, layer: int) -> None:
        self.futures[index][layer].set_result(None)

    def fail(self, index: int, error: BaseException) -> None:
        for future in self.futures.get(index, ()):
            if not future.done():
                future.set_exception(error)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        futures = self.futures.get(index)
        if futures is None:
            return
        # Blocks until the layer resolves; a failure is reported by finish().
        futures[threshold].exception()
        # Runs on failure too: an MLA dedup source must still issue the layer
        # broadcast its peers are waiting for.
        if self.on_layer_ready is not None:
            self.on_layer_ready(index, threshold)

    def finish(self, index: int) -> bool:
        """Wait for every layer of batch ``index``; True when all of them landed."""
        futures = self.futures.get(index)
        if futures is None:
            return True
        # Wait before dropping the entry: the load thread still completes the
        # layers the forward never waited on through it.
        errors = [future.exception() for future in futures]
        landed = all(error is None for error in errors)
        del self.futures[index]
        return landed

    def reset(self) -> None:
        self.producer_index = -1
        self.consumer_index = -1
        self.futures.clear()


class ExternalLinkerLoadError(RuntimeError):
    """A recoverable external-cache load failure for the current batch."""


class ExternalLinkerCleanupError(RuntimeError):
    """Allocator ownership is uncertain; only process teardown is safe."""


_INT63_MASK = (1 << 63) - 1
_MATCH_PHASE = 0x4D41544348  # "MATCH"
_LOAD_PHASE = 0x4C4F4144  # "LOAD"


def _collective_error(message: str) -> RuntimeError:
    # Imported lazily: disaggregation.utils reaches server_args, which the
    # cache layer must not import at module load.
    from sglang.srt.disaggregation.utils import DisaggregationCollectiveError

    return DisaggregationCollectiveError(message)


def _rid_digest(rid) -> tuple[int, int]:
    digest = hashlib.blake2b(str(rid).encode("utf-8"), digest_size=16).digest()
    return (
        int.from_bytes(digest[:8], "little") & _INT63_MASK,
        int.from_bytes(digest[8:], "little") & _INT63_MASK,
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
    def finish_layer_wise_loading(self, counter_index: int) -> bool:
        """Wait until the batch started as ``counter_index`` stops writing to
        device memory; True when every layer landed."""

    @abstractmethod
    def cancel_queued_load(self, rid: str) -> bool:
        """Cancel a load that has not started yet."""

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

    def waiting_queue_prefetch_enabled(self) -> bool:
        """Whether waiting-queue host prefetch is enabled on every rank."""
        return False

    def disable_waiting_queue_prefetch(self) -> None:
        """Turn waiting-queue host prefetch off before any job is submitted."""

    def submit_host_prefetch(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        """Queue a request-scoped host prefetch without allocating device pages."""
        return False

    def get_host_prefetch_status(self, rid: str) -> str | None:
        """Return the backend host-prefetch state, or ``None`` if absent."""
        return None

    def revalidate_host_prefetch(self, rid: str) -> bool:
        """Refresh the prepared session without copying into device memory."""
        return False

    def revalidate_no_prefetch_needed(self, rid: str) -> bool:
        """Check that a prefetch finished without DFS reads is still loadable."""
        return False

    def claim_ready_host_prefetch(self, rid: str) -> bool:
        """Transfer a READY speculative session to the normal load path."""
        return False

    def cancel_host_prefetch(self, rid: str) -> None:
        """Cancel or retire a request-scoped host prefetch."""

    def abort_prepared_load(self, rid: str) -> None:
        """Release a session claimed by the normal load path."""


class ExternalCacheHitMarker(NamedTuple):
    """What ``match`` found in the external store, consumed by ``load_back``.

    ``tail_hashes`` are the per-page storage hashes of the restorable tail,
    starting at ``device_hit_len``.
    """

    tail_hashes: list[str]
    device_hit_len: int
    prefix_key: Optional[RadixKey] = None


class PreparedHostPrefetch(NamedTuple):
    """Immutable request-local work for the scheduler's background prefetch round."""

    rid: str
    locally_eligible: bool
    transfers: tuple[PoolTransfer, ...]
    cache_linker: UnifiedCacheLinker
    cancelled: threading.Event


class _PendingOffload(NamedTuple):
    lock_node_id: NodeId
    lock_params: DecLockRefParams
    publish_node_ids: list[NodeId]


class _PrivateLoad(NamedTuple):
    """A queued or in-flight load whose device slots belong to one request.

    The slots are request-private from ``load_back`` on. ``bound`` is set only
    after ``alloc_for_extend`` has written the prefix into the request's KV
    row; until then the request's normal release cannot reach the slots, so a
    rollback has to free them through the component ABORT phase instead.
    """

    req: Req
    prev_prefix_len: int
    prefix_len: int
    transfers: list[tuple[TreeComponent, PoolTransfer]]
    bound: bool = False


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
                "External cache linker supports only Full, SWA and Mamba tree "
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
        self.host_prefetch_hits: dict[str, ExternalCacheHitMarker] = {}
        self.pending_host_prefetch_submissions: dict[
            str, tuple[PreparedHostPrefetch, ExternalCacheHitMarker]
        ] = {}
        # Requests loading in the batch being built, then its counter index once
        # the batch starts; finish_loads consumes both after its forward.
        self.inflight_load_rids: list[str] = []
        # Loaded tails stay outside the tree until their forward completes.
        self.private_load_indices: dict[str, torch.Tensor] = {}
        # The component transfers behind each private tail: needed to roll a
        # load back when a later admission gate rejects its request, before
        # any KV row owns the slots.
        self.private_loads: dict[str, _PrivateLoad] = {}
        # A component ABORT that raised may have freed only part of a pool.
        # Never retry it; keep the descriptors so teardown can report them.
        self.cleanup_quarantine: dict[str, dict] = {}
        self._match_sequence = 0
        self.inflight_load_index = -1
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
        # Private loads retain request ownership through the reduced verdict;
        # offload entries retain tree locks through completion.
        return bool(
            self.host_prefetch_hits
            or self.pending_host_prefetch_submissions
            or self.inflight_load_rids
            or self.private_load_indices
            or self.pending_offloads
            or self.cleanup_quarantine
        )

    def _raise_if_cleanup_quarantined(self) -> None:
        if self.cleanup_quarantine:
            raise ExternalLinkerCleanupError(
                "external KV cleanup quarantined; process teardown required: "
                f"rids={list(self.cleanup_quarantine)}"
            )

    @contextmanager
    def load_back_admission(self, req: Req, admitted_reqs: list[Req]):
        """Roll a private load back if a later admission gate rejects ``req``.

        ``init_load_back`` queues the load and extends the request's prefix
        before the re-selection and shard gates run. A request those gates
        reject stays in the waiting queue while the backend is about to write
        into its slots, so the load is cancelled here. Only a positive cancel
        result proves the slots are untouched; otherwise the descriptor is
        retained and the scheduler must stop.
        """
        self._raise_if_cleanup_quarantined()
        if req.rid in self.private_loads:
            raise ExternalLinkerCleanupError(
                f"external KV admission still owns a private load: rid={req.rid}"
            )
        prefix_len = req.prefix_len
        last_node = req.last_node
        host_loaded_length = getattr(req, "host_loaded_length", 0)
        swa_branching_seqlen = getattr(req, "swa_branching_seqlen", None)
        kv = copy(req.kv) if getattr(req, "kv", None) is not None else None
        try:
            yield
        finally:
            admitted = any(candidate is req for candidate in admitted_reqs)
            private = None if admitted else self.private_loads.get(req.rid)
            if private is not None:
                self._raise_if_cleanup_quarantined()
                if private.bound or not self.cache_linker.cancel_queued_load(
                    req.rid
                ):
                    # A worker may already be writing; neither a rematch nor
                    # a timed wait is a fence. Keep the descriptor and slots.
                    raise ExternalLinkerCleanupError(
                        "cannot cancel unadmitted external KV load; "
                        f"private slots retained: rid={req.rid}"
                    )
                self.private_loads.pop(req.rid)
                if req.rid in self.inflight_load_rids:
                    self.inflight_load_rids.remove(req.rid)
                self._abort_private_load(req.rid, private)
                # PREPARE may have created ReqKvInfo or advanced its SWA
                # boundary, and admission advanced the prefix. Restore the
                # request to its pre-load view for the next scheduling round.
                req.prefix_len = prefix_len
                req.last_node = last_node
                req.host_loaded_length = host_loaded_length
                req.swa_branching_seqlen = swa_branching_seqlen
                req.kv = kv
                self.hit_markers.pop(req.rid, None)

    def bind_private_load(self, req: Req) -> None:
        """Record the request-row owner once its prefix indices are written."""
        private = self.private_loads.get(req.rid)
        if private is None:
            return
        kv = getattr(req, "kv", None)
        if (
            private.req is not req
            or kv is None
            or kv.req_pool_idx is None
            or req.prefix_len < private.prefix_len
            or kv.kv_allocated_len < private.prefix_len
        ):
            raise ExternalLinkerCleanupError(
                f"external KV request-row handoff is incomplete: rid={req.rid}"
            )
        self.private_loads[req.rid] = private._replace(bound=True)

    # ---- match: probe the remote store and report host_hit_length ----

    def match(self, key: RadixKey, req: Req, result: MatchResult) -> MatchResult:
        cache = self.cache
        page = cache.page_size
        device_hit_len = result.device_prefix_len
        self.hit_markers.pop(req.rid, None)
        prefetched_hit = self.host_prefetch_hits.get(req.rid)

        known_hit_len = req.external_cache_hit_length if cache.pp_size > 1 else None
        skip_lookup = False
        if known_hit_len is None and cache.pp_size > 1:
            skip_lookup = cache.pp_rank != 0
            # PP0 owns the absolute boundary, including a complete L1 hit.
            req.external_cache_hit_length = 0 if skip_lookup else device_hit_len

        # Local facts only, with no early return: a full local hit, a missing
        # hash anchor, a component that cannot build a lookup transfer and a
        # reusable prefetch are all rank-local, while every rank of the
        # attention group must reach the collectives below in the same order.
        # The fixed-size preflight carries those facts to the group first.
        tail_hashes = []
        lookup_transfers = []
        local_valid = False
        reusable = False
        local_error = False
        try:
            self._raise_if_cleanup_quarantined()
            private = self.private_loads.get(req.rid)
            if private is not None and not private.bound:
                raise ExternalLinkerCleanupError(
                    f"cannot rematch unbound external KV slots: rid={req.rid}"
                )
            key, _ = key.maybe_to_bigram_view(cache.tree_core.is_eagle)
            if known_hit_len is not None:
                key = key[:known_hit_len]
            tail_hashes = (
                self._tail_hashes(key, result, device_hit_len)
                if not skip_lookup and device_hit_len < len(key)
                else []
            )
            local_valid = bool(tail_hashes)
            if local_valid:
                for component in self._components:
                    transfer = component.build_external_linker_transfer(
                        LinkerTransferPhase.LOOKUP, None, tail_hashes
                    )
                    if transfer is None:
                        local_valid = False
                        lookup_transfers = []
                        break
                    lookup_transfers.append(transfer)

            if prefetched_hit is not None and local_valid:
                expected_len = (
                    prefetched_hit.device_hit_len
                    + len(prefetched_hit.tail_hashes) * page
                )
                current_prefix = key[:expected_len]
                same_prefix = (
                    current_prefix.extra_key == prefetched_hit.prefix_key.extra_key
                    and current_prefix.cache_salt
                    == prefetched_hit.prefix_key.cache_salt
                    and current_prefix.is_bigram == prefetched_hit.prefix_key.is_bigram
                    and current_prefix.raw_token_ids()
                    == prefetched_hit.prefix_key.raw_token_ids()
                )
                reusable = (
                    device_hit_len == prefetched_hit.device_hit_len
                    and list(tail_hashes[: len(prefetched_hit.tail_hashes)])
                    == prefetched_hit.tail_hashes
                    and same_prefix
                    and (
                        known_hit_len is None
                        or len(prefetched_hit.tail_hashes) == len(tail_hashes)
                    )
                )
        except Exception:
            local_error = True
            logger.exception("External KV match preparation failed: rid=%s", req.rid)

        valid, common_tail_pages, reusable = self._match_preflight(
            req.rid,
            key_len=len(key),
            device_hit_len=device_hit_len,
            tail_pages=len(tail_hashes),
            local_valid=local_valid,
            reusable=reusable,
            boundary_mode=2 if skip_lookup else int(known_hit_len is not None),
            prefetch_pages=len(prefetched_hit.tail_hashes) if reusable else 0,
            local_error=local_error,
        )
        if not valid:
            if prefetched_hit is not None:
                self.cancel_waiting_queue_prefetch(req.rid)
            if known_hit_len is not None and device_hit_len < len(key):
                raise _collective_error(
                    "cannot restore the prefix boundary already agreed by PP0"
                )
            return result
        tail_hashes = tail_hashes[:common_tail_pages]
        by_pool = {transfer.name: transfer for transfer in lookup_transfers}

        if prefetched_hit is not None:
            if reusable:
                hit_pages = len(prefetched_hit.tail_hashes)
                hit_tokens = hit_pages * page
                if cache.pp_size > 1:
                    req.external_cache_hit_length = device_hit_len + hit_tokens
                swa_transfer = by_pool.get(PoolName.SWA)
                self.hit_markers[req.rid] = prefetched_hit
                return result._replace(
                    last_host_node=result.best_match_node,
                    host_hit_length=hit_tokens,
                    swa_host_hit_length=max(
                        result.swa_host_hit_length,
                        min(len(swa_transfer.keys), hit_pages) * page
                        if swa_transfer is not None
                        else 0,
                    ),
                    mamba_host_hit_length=max(
                        result.mamba_host_hit_length,
                        1 if PoolName.MAMBA in by_pool else 0,
                    ),
                )

            if known_hit_len is not None:
                self.cancel_waiting_queue_prefetch(req.rid)

        # Tail-relative: page 0 of `tail_hashes` is the first uncached page.
        if known_hit_len is None:
            restorable = []
            lookup_failed = False
            try:
                if prefetched_hit is not None:
                    self.cancel_waiting_queue_prefetch(req.rid)
                restorable = list(self.cache_linker.lookup(req.rid, lookup_transfers))
            except Exception:
                lookup_failed = True
                logger.exception("External KV lookup failed: rid=%s", req.rid)
            hit_pages = self._sync_restorable_prefix(
                restorable,
                num_pages=len(tail_hashes),
                device_hit_pages=0,
                lookup_failed=lookup_failed,
            )
            if cache.pp_size > 1:
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
        reusable: bool,
        boundary_mode: int = 0,
        prefetch_pages: int = 0,
        local_error: bool = False,
    ) -> tuple[bool, int, bool]:
        """Agree on the match geometry before any variable-size collective.

        One MIN reduction over value/negation pairs yields the group minimum
        and maximum of every field. Geometry (key length, device hit, tail
        pages) and identity (RID digest, match sequence, phase, PP mode) must
        agree: a mismatch means the ranks are matching different prefixes or different
        requests, and the forward collectives would then consume different KV
        layouts, so it is fatal. Validity and prefetch reuse may differ: the
        invalid lookups are skipped on every rank, costing a recompute.
        Non-reusable prefetches fall back to a fresh lookup on every rank.

        Prefetch reuse also requires the same cached tail length on every rank.
        Returns ``(valid, tail_pages, reusable)``, identical on every rank.
        """
        sequence = self._match_sequence
        self._match_sequence = sequence + 1
        digest_lo, digest_hi = _rid_digest(rid)
        values = [
            key_len,
            device_hit_len,
            tail_pages,
            int(local_valid),
            int(reusable),
            digest_lo,
            digest_hi,
            sequence,
            _MATCH_PHASE,
            boundary_mode,
            prefetch_pages,
            int(local_error),
        ]
        payload = torch.tensor(
            [item for value in values for item in (value, -value)], dtype=torch.int64
        )
        self.cache._all_reduce_attn_groups(payload, torch.distributed.ReduceOp.MIN)
        lo = payload[0::2].tolist()
        hi = [-value for value in payload[1::2].tolist()]

        if any(lo[i] != hi[i] for i in (5, 6, 7, 8, 9)):
            logger.error(
                "External KV match identity mismatch: rid=%s sequence=%d min=%s max=%s",
                rid,
                sequence,
                lo[5:],
                hi[5:],
            )
            raise _collective_error(
                "external KV match RID/sequence/phase/PP mode differs across ranks"
            )
        if hi[11]:
            raise _collective_error("external KV match preparation failed on a rank")
        if any(lo[i] != hi[i] for i in (0, 1, 2)):
            logger.error(
                "External KV match geometry mismatch: rid=%s "
                "local=(key_len=%d,device_hit_len=%d,tail_pages=%d) "
                "group=(key_len=[%d,%d],device_hit_len=[%d,%d],tail_pages=[%d,%d])",
                rid,
                key_len,
                device_hit_len,
                tail_pages,
                lo[0],
                hi[0],
                lo[1],
                hi[1],
                lo[2],
                hi[2],
            )
            raise _collective_error("external KV match geometry differs across ranks")

        if lo[3] != 1:
            if lo[2] == 0:
                # Every rank has a complete local hit or no page-aligned tail:
                # a normal no-op, but only after the preflight; returning
                # earlier would desynchronize the peers.
                return False, 0, False
            if hi[3] == 1:
                logger.warning(
                    "External KV match preflight rejected: rid=%s key_len=%d "
                    "device_hit_len=%d tail_pages=%d: valid on some ranks only",
                    rid,
                    key_len,
                    device_hit_len,
                    tail_pages,
                )
            else:
                logger.debug(
                    "External KV match skipped on every rank: rid=%s tail_pages=%d",
                    rid,
                    tail_pages,
                )
            return False, 0, False
        return True, lo[2], lo[4] == 1 and lo[10] == hi[10]

    def _sync_restorable_prefix(
        self,
        restorable: list[int],
        *,
        num_pages: int,
        device_hit_pages: int,
        lookup_failed: bool = False,
    ) -> int:
        """Intersect the per-rank sets of restorable prefix lengths and return the
        longest one, or 0 when the ranks share none beyond the device prefix.

        A rank's set is sparse, so reducing per-rank maxima could land on a
        length that only some ranks can restore. On a 0/1 mask MIN is AND, which
        makes the reduction an intersection.
        """
        # This helper owns the variable-size collective, so validate its shape
        # here too. Lookup failures must join this fixed-size vote rather than
        # leave peers waiting in the mask reduction. A mismatch is fatal: a
        # shorter mask could select KV for different token prefixes.
        geometry = torch.tensor(
            [
                num_pages,
                -num_pages,
                device_hit_pages,
                -device_hit_pages,
                int(not lookup_failed),
            ],
            dtype=torch.int64,
        )
        self.cache._all_reduce_attn_groups(geometry, torch.distributed.ReduceOp.MIN)
        if int(geometry[4]) == 0:
            raise _collective_error("external KV lookup failed on a rank")
        if int(geometry[0]) != -int(geometry[1]) or int(geometry[2]) != -int(
            geometry[3]
        ):
            raise _collective_error(
                "external KV restorable mask geometry differs across ranks"
            )
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
        return get_storage_hash_str(
            key[device_hit_len : device_hit_len + tail_len],
            last_hash,
            page_size=page,
        )

    # ---- init_load_back: remote -> request-private device slots ----

    def load_back(self, req: Req) -> tuple[int, NodeId]:
        self._raise_if_cleanup_quarantined()
        if req.rid in self.private_loads:
            raise ExternalLinkerCleanupError(
                f"external KV load already has an owner: rid={req.rid}"
            )
        cache = self.cache
        hit = self.hit_markers.pop(req.rid, None)
        if hit is None:
            return 0, req.last_node
        # A new load: the SWA ABORT ownership bit of any earlier load of this
        # request is stale, and a stale True would skip the next ABORT's free.
        req.external_kv_private_swa_slots_released = False

        prepared_from_host_prefetch = False
        prefetched = self.host_prefetch_hits.pop(req.rid, None) is not None
        if prefetched:
            status = self.cache_linker.get_host_prefetch_status(req.rid)
            locally_complete = status in {"dfs_prefetched", "no_prefetch_needed"}
            ready = torch.tensor(int(locally_complete), dtype=torch.int)
            cache._all_reduce_attn_groups(ready, torch.distributed.ReduceOp.MIN)
            if int(ready.item()) == 0:
                self.cache_linker.cancel_host_prefetch(req.rid)
                self.hit_markers[req.rid] = hit
                return self.load_back(req)

            # Every rank must reach the reduction below, so any local failure,
            # including an exception, becomes a local False.
            try:
                if status == "no_prefetch_needed":
                    # Its session was released while queued; re-check that
                    # the keys survived so eviction falls back to the normal
                    # path instead of failing the later load.
                    locally_valid = self.cache_linker.revalidate_no_prefetch_needed(
                        req.rid
                    )
                else:
                    locally_valid = (
                        status == "dfs_prefetched"
                        and self.cache_linker.revalidate_host_prefetch(req.rid)
                    )
            except BaseException:
                locally_valid = False
            valid = torch.tensor(int(locally_valid), dtype=torch.int)
            cache._all_reduce_attn_groups(valid, torch.distributed.ReduceOp.MIN)
            if int(valid.item()) == 0:
                self.cache_linker.cancel_host_prefetch(req.rid)
                self.hit_markers[req.rid] = hit
                return self.load_back(req)

        device_hit_len = hit.device_hit_len
        tail_hashes = hit.tail_hashes
        prefix_len = device_hit_len + len(tail_hashes) * cache.page_size

        if prefetched:
            # Build before claiming so that the claim reduction also carries
            # the construction verdict: one reduction instead of two.
            component_transfers, constructed, build_error = self._build_load_transfers(
                req, tail_hashes
            )
            claim_error: BaseException | None = None
            try:
                if status == "no_prefetch_needed":
                    self.cache_linker.cancel_host_prefetch(req.rid)
                    claimed = True
                else:
                    claimed = self.cache_linker.claim_ready_host_prefetch(req.rid)
            except BaseException as error:
                # Still join the reduction below, then re-raise after cleanup.
                # Reporting "not constructed" sends every rank to the miss exit
                # rather than to a normal-path retry this rank would not join.
                claimed, constructed, claim_error = False, False, error
            verdict = torch.tensor([int(claimed), int(constructed)], dtype=torch.int)
            cache._all_reduce_attn_groups(verdict, torch.distributed.ReduceOp.MIN)
            claimed_all, constructed_all = (int(value) for value in verdict)
            if not (claimed_all and constructed_all):
                # Free the device slots, then cancel (which also rolls a
                # locally claimed session back); a build or claim error is
                # raised on its own rank only after both were attempted.
                if build_error is not None and claim_error is not None:
                    logger.error(
                        "Claiming the prefetch of rid=%s also failed",
                        req.rid,
                        exc_info=claim_error,
                    )
                self._abort_disagreed_load(
                    req,
                    component_transfers,
                    prefix_len,
                    cancel_prefetch=True,
                    pending=build_error if build_error is not None else claim_error,
                )
                if not constructed_all:
                    # Building failed somewhere: a miss, as on the normal path.
                    return 0, req.last_node
                # Every rank built but a claim failed: retry the normal path,
                # which builds again and agrees on its own construction.
                self.hit_markers[req.rid] = hit
                return self.load_back(req)
            prepared_from_host_prefetch = True
        else:
            component_transfers, constructed, build_error = self._build_load_transfers(
                req, tail_hashes
            )
            all_constructed = torch.tensor(int(constructed), dtype=torch.int)
            cache._all_reduce_attn_groups(
                all_constructed, torch.distributed.ReduceOp.MIN
            )
            if int(all_constructed.item()) == 0:
                # Each rank frees what it built; a component frees its own
                # slots when it fails after allocating.
                self._abort_disagreed_load(
                    req,
                    component_transfers,
                    prefix_len,
                    cancel_prefetch=False,
                    pending=build_error,
                )
                return 0, req.last_node

        # Keys can be evicted remotely (master memory-watermark eviction)
        # between the match-time lookup and this load-back; a stale hit would
        # then fail the async layer-wise session fatally. Re-check existence
        # here so the request degrades to a plain cache miss instead.
        revalidate = getattr(self.cache_linker, "revalidate_load", None)
        if (
            not prepared_from_host_prefetch
            and revalidate is not None
            and not revalidate([transfer for _, transfer in component_transfers])
        ):
            self._update_load(
                ExternalLinkerLoadPhase.ABORT,
                req,
                component_transfers,
                prefix_len,
            )
            return 0, req.last_node

        full_transfer = component_transfers[0][1]
        assert full_transfer.name == PoolName.KV
        # Queue before PREPARE touches the request, so a refused load only has
        # the allocations to undo before falling back to recompute. The result
        # depends on the transfers alone, so every rank takes the same branch.
        try:
            queued = self.cache_linker.load(
                req.rid, [transfer for _, transfer in component_transfers]
            )
        except BaseException:
            self.cache_linker.abort_prepared_load(req.rid)
            self._update_load(
                ExternalLinkerLoadPhase.ABORT, req, component_transfers, prefix_len
            )
            raise
        if not queued:
            self.cache_linker.abort_prepared_load(req.rid)
            self._update_load(
                ExternalLinkerLoadPhase.ABORT, req, component_transfers, prefix_len
            )
            return 0, req.last_node
        self._update_load(
            ExternalLinkerLoadPhase.PREPARE,
            req,
            component_transfers,
            prefix_len,
        )

        # Components omitted from the linker do not run their PREPARE hook.
        # Keep a non-restorable SWA range as tombstones instead of rebuilding
        # it from an uninitialized FULL-to-SWA mapping when the tail is inserted.
        if self._skip_swa:
            if req.kv is None:
                from sglang.srt.managers.schedule_batch import ReqKvInfo

                req.kv = ReqKvInfo(kv_allocated_len=prefix_len)
            req.kv.set_evicted_seqlen(
                ComponentType.SWA,
                max(req.kv.get_evicted_seqlen(ComponentType.SWA), prefix_len),
            )

        # match placed the SWA branch point before this tail was known; a point
        # the tail covers is gone, since the request inserts the whole tail.
        if (
            req.swa_branching_seqlen is not None
            and req.swa_branching_seqlen <= prefix_len
        ):
            req.swa_branching_seqlen = None
        self.private_load_indices[req.rid] = full_transfer.device_indices
        self.private_loads[req.rid] = _PrivateLoad(
            req, device_hit_len, prefix_len, component_transfers
        )
        self.inflight_load_rids.append(req.rid)
        return len(full_transfer.device_indices), req.last_node

    def _update_load(
        self,
        phase: ExternalLinkerLoadPhase,
        req: Req,
        component_transfers: list[tuple[TreeComponent, PoolTransfer]],
        prefix_len: int,
    ) -> list[PoolTransfer]:
        if not component_transfers:
            return []
        if phase == ExternalLinkerLoadPhase.ABORT:
            return self._abort_transfers_with_quarantine(
                req, component_transfers, prefix_len
            )
        full = component_transfers[0][1]
        result = []
        for component, transfer in component_transfers:
            transfer = component.update_external_linker_load(
                phase, req, full, transfer, prefix_len
            )
            if transfer is not None:
                result.append(transfer)
        return result

    def _abort_transfers_with_quarantine(
        self,
        req: Req,
        component_transfers: list[tuple[TreeComponent, PoolTransfer]],
        prefix_len: int,
    ) -> list[PoolTransfer]:
        """Free the components in reverse order; quarantine an ambiguous free.

        A component that raises mid-ABORT may have returned part of its range
        to the allocator. Retrying would double-free it, so the descriptors
        are retained with each component's outcome and every later linker
        operation refuses to run until the process is torn down.
        """
        self._raise_if_cleanup_quarantined()
        full = component_transfers[0][1]
        states = ["not_attempted"] * len(component_transfers)
        for index in reversed(range(len(component_transfers))):
            component, transfer = component_transfers[index]
            states[index] = "uncertain"
            try:
                component.update_external_linker_load(
                    ExternalLinkerLoadPhase.ABORT, req, full, transfer, prefix_len
                )
            except BaseException as error:
                # Keep the tensor references alive; never retry, even on close.
                self.cleanup_quarantine[req.rid] = {
                    "req": req,
                    "transfers": component_transfers,
                    "prefix_len": prefix_len,
                    "states": tuple(states),
                }
                component_slots = [
                    (
                        str(t.name),
                        int(t.device_indices.numel())
                        if t.device_indices is not None
                        else 0,
                        states[i],
                    )
                    for i, (_, t) in enumerate(component_transfers)
                ]
                logger.exception(
                    "External KV release quarantined: rid=%s prefix_len=%d "
                    "components=(pool,slots,state)=%s; process teardown required",
                    req.rid,
                    prefix_len,
                    component_slots,
                )
                raise ExternalLinkerCleanupError(
                    f"external KV release uncertain for rid={req.rid}"
                ) from error
            states[index] = "released"
        return []

    def _abort_private_load(self, rid: str, private: _PrivateLoad) -> None:
        """Free a cancelled load's slots: no KV row references them yet."""
        self.private_load_indices.pop(rid, None)
        self._update_load(
            ExternalLinkerLoadPhase.ABORT,
            private.req,
            private.transfers,
            private.prefix_len,
        )

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
        unbound = [
            rid for rid, private in self.private_loads.items() if not private.bound
        ]
        if unbound:
            # The scheduler writes every admitted request's KV row before
            # starting the batch's loads; a load with no row owner here is
            # one the admission rollback missed.
            raise ExternalLinkerCleanupError(
                f"cannot start external KV loads without a row owner: rids={unbound}"
            )
        self.inflight_load_index = self.cache_linker.start_layer_wise_loading()
        return self.inflight_load_index

    def _preflight_inflight_loads(self, rids: Sequence[str]) -> None:
        """Agree on which requests the next verdict is for.

        The backend completion is local and carries no identity, so two ranks
        could reduce a success bit that belongs to different requests. A fixed
        eight-value payload (count, ordered digest, XOR digest, phase, each
        with its negation) catches a count, order or membership difference
        before the variable request state is committed.
        """
        encoded = [zlib.crc32(str(rid).encode("utf-8")) & 0x7FFFFFFF for rid in rids]
        ordered = 0
        xor = 0
        for value in encoded:
            ordered = ((ordered * 16777619) ^ value) & 0x7FFFFFFF
            xor ^= value
        values = [len(encoded), ordered, xor, _LOAD_PHASE]
        payload = torch.tensor(
            [item for value in values for item in (value, -value)], dtype=torch.int64
        )
        self.cache._all_reduce_attn_groups(payload, torch.distributed.ReduceOp.MIN)
        lo = payload[0::2].tolist()
        hi = [-value for value in payload[1::2].tolist()]
        if lo != hi:
            logger.error(
                "External linker load batch differs across ranks: local=%s "
                "reduced_min=%s reduced_max=%s",
                list(rids),
                lo,
                hi,
            )
            raise _collective_error(
                "external linker load batch RID/order differs across ranks"
            )

    def finish_loads(self) -> list[str]:
        """Wait for the loads of the batch that just ran; return the rids whose
        KV did not land, identically on every rank.

        Runs after that batch's forward, so no slot it wrote to has been freed
        or inserted yet. Every rank holds the same ``inflight_load_rids``
        because ``match`` agrees on the hit across ranks, so either all ranks
        or none enter the reductions; the empty case therefore stays local,
        which keeps decode steps free of collectives. The verdict is reduced
        because an MLA dedup peer only receives the broadcast and never sees
        the source's error.
        """
        rids, self.inflight_load_rids = self.inflight_load_rids, []
        if not rids:
            return []
        index, self.inflight_load_index = self.inflight_load_index, -1
        self._preflight_inflight_loads(rids)
        landed = torch.tensor(
            [int(self.cache_linker.finish_layer_wise_loading(index))],
            dtype=torch.int,
        )
        self.cache._all_reduce_attn_groups(landed, torch.distributed.ReduceOp.MIN)
        unbound = []
        for rid in rids:
            self.private_load_indices.pop(rid, None)
            private = self.private_loads.pop(rid, None)
            if private is not None and not private.bound:
                unbound.append(rid)
        if unbound:
            # Its slots were read into but no KV row ever claimed them, so the
            # normal request release cannot return them.
            raise ExternalLinkerCleanupError(
                f"completed external KV load has no request-row owner: rids={unbound}"
            )
        if landed.item():
            return []
        logger.error(
            "External linker load failed; aborting %d request(s): %s", len(rids), rids
        )
        return rids

    # ---- lifecycle ----

    def reset(self) -> None:
        self._raise_if_cleanup_quarantined()
        self.cache_linker.reset()
        for rid in list(self.pending_host_prefetch_submissions):
            self.cancel_waiting_queue_prefetch(rid)
        self.host_prefetch_hits.clear()
        self.hit_markers.clear()
        self.inflight_load_rids.clear()
        self.private_load_indices.clear()
        self.private_loads.clear()
        self.inflight_load_index = -1
        self._release_pending_offloads()

    def _release_pending_offloads(self) -> None:
        for pending in self.pending_offloads:
            self.cache.tree_core.finish_external_linker_offload(
                pending.publish_node_ids, pending.lock_node_id, False
            )
            self.cache.dec_lock_ref(pending.lock_node_id, pending.lock_params)
        self.pending_offloads.clear()

    def release_request(self, rid: str) -> None:
        self._raise_if_cleanup_quarantined()
        self.cancel_waiting_queue_prefetch(rid)
        # A queued load targets slots the request owns and is about to free.
        # A started one is finished by finish_loads before anything is freed.
        if rid in self.inflight_load_rids and self.cache_linker.cancel_queued_load(rid):
            self.inflight_load_rids.remove(rid)
            private = self.private_loads.pop(rid, None)
            if private is not None and not private.bound:
                # No KV row references these slots yet, so the request's own
                # release cannot return them; free them here instead.
                self._abort_private_load(rid, private)
            self.private_load_indices.pop(rid, None)

    def close(self) -> None:
        if self.cleanup_quarantine:
            logger.error(
                "Closing the external linker with quarantined releases: rids=%s; "
                "their device slots were never returned",
                list(self.cleanup_quarantine),
            )
        self.cache_linker.close()
        self.inflight_load_rids.clear()
        self.private_load_indices.clear()
        self.private_loads.clear()
        self._release_pending_offloads()

    def get_host_prefetch_admission_state(self, rid: str) -> str:
        if rid in self.pending_host_prefetch_submissions:
            return "pending"
        tracked = rid in self.host_prefetch_hits
        status = self.cache_linker.get_host_prefetch_status(rid)
        if status in {"queued", "preparing", "reading"}:
            return "pending"
        if tracked and status == "dfs_prefetched":
            return "dfs_prefetched"
        if tracked and status == "no_prefetch_needed":
            return "no_prefetch_needed"
        if not tracked and status is None:
            return "not_tracked"
        return "terminal"

    def cancel_waiting_queue_prefetch(self, rid: str) -> None:
        self.hit_markers.pop(rid, None)
        self.host_prefetch_hits.pop(rid, None)
        pending = self.pending_host_prefetch_submissions.pop(rid, None)
        if pending is not None:
            pending[0].cancelled.set()
        self.cache_linker.cancel_host_prefetch(rid)

    def prepare_host_prefetch(self, req: Req) -> PreparedHostPrefetch:
        """Build a request-local snapshot without issuing any collectives."""
        hit = self.hit_markers.get(req.rid)
        transfers = []
        locally_eligible = hit is not None
        if hit is not None:
            try:
                for component in self._components:
                    transfer = component.build_external_linker_transfer(
                        LinkerTransferPhase.LOOKUP, None, hit.tail_hashes
                    )
                    if transfer is None:
                        locally_eligible = False
                        break
                    # LOOKUP transfers are CPU metadata only. Copy their
                    # mutable key lists so the worker owns its snapshot.
                    transfers.append(
                        replace(
                            transfer,
                            keys=list(transfer.keys)
                            if transfer.keys is not None
                            else None,
                        )
                    )
            except Exception:
                locally_eligible = False

        job = PreparedHostPrefetch(
            rid=req.rid,
            locally_eligible=locally_eligible,
            transfers=tuple(transfers),
            cache_linker=self.cache_linker,
            cancelled=threading.Event(),
        )
        if hit is not None:
            self.pending_host_prefetch_submissions[req.rid] = (job, hit)
        return job

    def complete_host_prefetch_submission(
        self, job: PreparedHostPrefetch, submitted: bool
    ) -> bool:
        """Commit a background result if this is still the live request."""
        pending = self.pending_host_prefetch_submissions.get(job.rid)
        if pending is None or pending[0] is not job:
            if submitted:
                job.cache_linker.cancel_host_prefetch(job.rid)
            return False

        del self.pending_host_prefetch_submissions[job.rid]
        hit = pending[1]
        if submitted and not job.cancelled.is_set():
            self.host_prefetch_hits[job.rid] = hit
            return True
        if submitted:
            job.cache_linker.cancel_host_prefetch(job.rid)
        return False

    def prefetch_to_host(self, req: Req) -> bool:
        hit = self.hit_markers.get(req.rid)
        transfers = []
        locally_eligible = hit is not None
        if hit is not None:
            try:
                for component in self._components:
                    transfer = component.build_external_linker_transfer(
                        LinkerTransferPhase.LOOKUP, None, hit.tail_hashes
                    )
                    if transfer is None:
                        locally_eligible = False
                        break
                    transfers.append(transfer)
            except Exception:
                locally_eligible = False

        eligible = torch.tensor(int(locally_eligible), dtype=torch.int)
        self.cache._all_reduce_attn_groups(eligible, torch.distributed.ReduceOp.MIN)
        if int(eligible.item()) == 0:
            return False

        try:
            submitted = self.cache_linker.submit_host_prefetch(req.rid, transfers)
        except BaseException:
            submitted = False

        globally_submitted = torch.tensor(int(submitted), dtype=torch.int)
        self.cache._all_reduce_attn_groups(
            globally_submitted, torch.distributed.ReduceOp.MIN
        )
        if int(globally_submitted.item()) == 0:
            self.cache_linker.cancel_host_prefetch(req.rid)
            return False

        self.host_prefetch_hits[req.rid] = hit
        return True

    def _build_load_transfers(
        self, req: Req, tail_hashes
    ) -> tuple[list[tuple[TreeComponent, PoolTransfer]], bool, BaseException | None]:
        """Build every component's LOAD transfer on this rank only.

        Building allocates device slots and can fail on one rank alone, so the
        caller must agree on the returned flag with the other ranks before any
        later collective, and on failure ABORT the transfers returned here (a
        component frees slots it could not hand over). A component returning
        None (not enough room) is an expected miss. An unexpected error may
        have left the tree or an allocator partly changed, so it is returned
        rather than raised: this rank still joins that agreement, and the
        caller re-raises it after cleanup instead of serving on.
        """

        component_transfers: list[tuple[TreeComponent, PoolTransfer]] = []
        try:
            for component in self._components:
                transfer = component.build_external_linker_transfer(
                    LinkerTransferPhase.LOAD, None, tail_hashes
                )
                if transfer is None:
                    return component_transfers, False, None
                component_transfers.append((component, transfer))
        except BaseException as error:
            return component_transfers, False, error
        return component_transfers, True, None

    def _abort_disagreed_load(
        self,
        req: Req,
        component_transfers: list[tuple[TreeComponent, PoolTransfer]],
        prefix_len: int,
        *,
        cancel_prefetch: bool,
        pending: BaseException | None,
    ) -> None:
        """Clean up after the ranks agreed a load cannot proceed.

        Frees this rank's transfers, then, on the prefetched path, cancels the
        prefetch (rolling a claimed session back) even if freeing raised.
        Raises the pending build or claim error, which stays the primary one;
        a cleanup error is logged then, and raised otherwise. Freeing stops at
        the first component that raises, so a failed cleanup does not promise
        that every slot was released.
        """
        cleanup_error: BaseException | None = None
        try:
            self._update_load(
                ExternalLinkerLoadPhase.ABORT, req, component_transfers, prefix_len
            )
        except BaseException as error:
            cleanup_error = error
        if cancel_prefetch:
            try:
                self.cache_linker.cancel_host_prefetch(req.rid)
            except BaseException as error:
                if cleanup_error is None:
                    cleanup_error = error
                else:
                    logger.error(
                        "Cancelling the prefetch of rid=%s also failed",
                        req.rid,
                        exc_info=error,
                    )
        if pending is not None:
            if cleanup_error is not None:
                logger.error(
                    "Cleaning up the failed load of rid=%s also failed",
                    req.rid,
                    exc_info=cleanup_error,
                )
            raise pending
        if cleanup_error is not None:
            raise cleanup_error
