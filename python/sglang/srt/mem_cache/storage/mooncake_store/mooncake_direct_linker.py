from __future__ import annotations

import logging
import os
import threading
import time
from concurrent.futures import Future
from queue import Empty, Queue

import torch

from sglang.srt.mem_cache.cache_init_params import CacheInitParams
from sglang.srt.mem_cache.hicache_storage import (
    HiCacheStorageConfig,
    PoolName,
    PoolTransfer,
)
from sglang.srt.mem_cache.hybrid_cache.hybrid_cache_controller import (
    HybridCacheController,
)
from sglang.srt.mem_cache.hybrid_cache.linker_pool_assembler import (
    resolve_hybrid_device_pool_group,
)
from sglang.srt.mem_cache.storage.mooncake_store.master_availability import (
    MasterAvailability,
    StoreUnavailable,
    checked_exists,
)
from sglang.srt.mem_cache.storage.mooncake_store.native_read_guard import (
    NativeReadGuard,
    read_failure_contract,
)
from sglang.srt.mem_cache.unified_cache.linker_fault_injection import (
    arm_load_failure_injection,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import (
    UnifiedCacheLinker,
)
from sglang.srt.observability.metrics_collector import (
    STAT_LOGGER_ROLE_STORAGE,
    StorageMetricsCollector,
    resolve_collector_class,
)
from sglang.srt.runtime_context import get_memory, get_model
from sglang.srt.utils import freeze_gc, get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


def _get_mooncake_storage_metrics_dp_rank(server_args, params) -> int:
    if getattr(server_args, "enable_dp_attention", False):
        from sglang.srt.layers.dp_attention import get_attention_dp_rank

        return get_attention_dp_rank()
    return getattr(params, "dp_rank", None) or 0


def _storage_suffix(
    *, rank_replicated: bool, tp_rank: int, attn_cp_rank: int, pp_rank: int
) -> str:
    parts = []
    if not rank_replicated:
        parts.append(f"tp{tp_rank}")
    parts.extend((f"cp{attn_cp_rank}", f"pp{pp_rank}"))
    return "_".join(parts)


class LayerWiseLoadCounter:
    """CPU completion counter compatible with KV pools' layer wait hook."""

    def __init__(self, num_layers: int):
        self.num_layers = num_layers
        self.producer_index = -1
        self.consumer_index = -1
        self.futures: dict[int, list[Future]] = {}
        # Batch indices already logged: one line per failed load, not per layer.
        self.reported: set[int] = set()

    def update_producer(self) -> int:
        self.producer_index += 1
        self.futures[self.producer_index] = [Future() for _ in range(self.num_layers)]
        return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def complete(self, index: int, layer: int) -> None:
        self.futures[index][layer].set_result(None)

    def fail(self, index: int, error: BaseException) -> None:
        # A private copy, because wait_until clears the traceback of what it
        # catches and the caller still needs its own exception for the
        # logger.exception() that follows.
        failure = RuntimeError(f"{type(error).__name__}: {error}")
        for future in self.futures.get(index, ()):
            if not future.done():
                future.set_exception(failure)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        futures = self.futures.get(index)
        if futures is None:
            return
        try:
            futures[threshold].result()
        except BaseException as error:
            # Never raise: this runs inside the model forward, where nothing
            # catches before the scheduler's top-level handler and the whole
            # engine goes down. The failure travels out through
            # pop_completed_load(), which aborts the affected requests.
            if index not in self.reported:
                self.reported.add(index)
                logger.error(
                    "Mooncake layer-wise KV load failed for batch %d; affected "
                    "requests will be aborted after this forward: %s",
                    index,
                    error,
                )
            # Every layer re-raises this same exception object, and each
            # appended traceback entry roots a frame chain that pins that
            # layer's activations: +31.9 GiB on one faulted 61-layer forward at
            # --chunked-prefill-size 2048, until the engine OOMed. Nothing
            # reads the traceback.
            error.__traceback__ = None
        finally:
            if threshold == self.num_layers - 1:
                self.futures.pop(index, None)
                self.reported.discard(index)

    def finish(self, index: int) -> bool:
        """Wait for every layer before publishing the batch completion.

        A failed early layer must not short-circuit this barrier while a later
        layer is still unresolved.  The model-forward hook is fail-soft, but
        the completion queue is the ownership boundary for request-private
        slots and therefore may only be published after every Future reaches a
        terminal state.
        """
        futures = self.futures.get(index)
        if futures is None:
            return True
        success = True
        for future in futures:
            try:
                future.result()
            except BaseException as error:
                success = False
                error.__traceback__ = None
        self.futures.pop(index, None)
        self.reported.discard(index)
        return success

    def reset(self) -> None:
        self.producer_index = -1
        self.consumer_index = -1
        self.futures.clear()
        self.reported.clear()


class ReadPlanLoadCounter:
    """Publish one Mooncake ReadPlan; wait for each layer without holding the GIL."""

    def __init__(self, num_layers: int, on_progress=None):
        self.num_layers = num_layers
        self.on_progress = on_progress
        self.producer_index = self.consumer_index = -1
        self.plans: dict[int, Future] = {}
        # Batch indices already logged: one line per failed plan, not per layer.
        self.reported: set[int] = set()

    def update_producer(self) -> int:
        self.producer_index += 1
        self.plans[self.producer_index] = Future()
        return self.producer_index

    def set_consumer(self, index: int) -> None:
        self.consumer_index = index

    def bind(self, index: int, plan) -> None:
        self.plans[index].set_result(plan)

    def fail(self, index: int, error: BaseException) -> None:
        future = self.plans.get(index)
        if future is not None and not future.done():
            future.set_exception(error)

    def wait_until(self, threshold: int) -> None:
        index = self.consumer_index
        future = self.plans.get(index)
        if future is None:
            return
        try:
            future.result().wait(threshold)
            if self.on_progress is not None:
                self.on_progress(index)
        except BaseException as error:
            # Do not raise from the model forward. A rank-local failure here can
            # strand peer TP/CP ranks in a later model collective and brings the
            # scheduler down. load_layer_wise reports False through the linker
            # completion queue; the cache MIN-reduces that verdict across the
            # attention group and aborts the affected requests after forward.
            if index not in self.reported:
                self.reported.add(index)
                logger.error(
                    "Mooncake ReadPlan KV load failed for batch %d; affected "
                    "requests will be aborted after this forward: %s",
                    index,
                    error,
                )
            # Repeated waits otherwise grow a traceback chain that keeps each
            # layer's activations alive until the failed plan is retired.
            error.__traceback__ = None
        finally:
            if threshold == self.num_layers - 1:
                self.plans.pop(index, None)
                self.reported.discard(index)

    def finish(self, index: int) -> bool:
        """Wait for the read plan before publishing batch completion."""
        future = self.plans.get(index)
        if future is None:
            return True
        success = True
        try:
            future.result()
        except BaseException as error:
            success = False
            error.__traceback__ = None
        self.plans.pop(index, None)
        self.reported.discard(index)
        return success

    def reset(self) -> None:
        self.producer_index = self.consumer_index = -1
        self.plans.clear()
        self.reported.clear()


class MooncakeDirectLinker(UnifiedCacheLinker):
    def __init__(
        self,
        server_args,
        params: CacheInitParams,
        *,
        components,
        storage=None,
    ):
        self.page_size = params.page_size
        self.page_wise_load_threshold = server_args.mooncake_page_wise_load_threshold
        self.enable_page_wise_load = server_args.mooncake_enable_page_wise_load
        if self.page_wise_load_threshold <= 0:
            raise ValueError(
                "--mooncake-page-wise-load-threshold must be positive, got "
                f"{self.page_wise_load_threshold}."
            )
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=self.page_size,
            params=params,
            components=components,
        )
        self.pools = self.pool_group.entry_map
        self.num_layers = self.pool_group.num_layers
        self.pp_size = params.pp_size
        # Revalidation must agree even without a LayerSplit storage layout.
        self._load_revalidation_groups = tuple(
            group
            for group in (params.attn_cp_cache_group, params.attn_tp_cache_group)
            if group is not None and torch.distributed.get_world_size(group=group) > 1
        )

        tp_rank = 0
        tp_size = server_args.tp_size
        tp_group = params.attn_tp_cache_group or params.tp_cache_group
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_rank = torch.distributed.get_rank(group=tp_group)
            tp_size = torch.distributed.get_world_size(group=tp_group)
        rank_replicated = self.pool_group.rank_replicated
        self.offload_owner = not rank_replicated or tp_rank == 0
        self.tp_rank = tp_rank
        extra_config, *_ = HybridCacheController.parse_storage_backend_extra_config(
            get_memory().hicache_storage_backend_extra_config
        )
        storage_config = HiCacheStorageConfig(
            tp_rank=tp_rank,
            tp_size=tp_size,
            pp_rank=params.pp_rank,
            pp_size=params.pp_size,
            attn_cp_rank=params.attn_cp_rank,
            attn_cp_size=params.attn_cp_size,
            is_mla_model=rank_replicated,
            enable_storage_metrics=False,
            is_page_first_layout=False,
            model_name=get_model().model_path,
            extra_config=extra_config,
            dp_rank=getattr(params, "dp_rank", None),
        )
        self.l1_fallback_enabled = (
            os.environ.get("SGLANG_MOONCAKE_L1_FALLBACK", "1") == "1"
        )
        contract_path = os.environ.get("SGLANG_MOONCAKE_NATIVE_READ_CONTRACT")
        binary = None
        if contract_path:
            from importlib.util import find_spec

            spec = find_spec("mooncake.store")
            binary = spec.origin if spec is not None else None
        contract = read_failure_contract(contract_path, binary)
        self._native_read_guard = NativeReadGuard(
            self._storage_failed,
            stall_s=float(os.environ.get("SGLANG_MOONCAKE_READ_STALL_S", "5")),
            min_bytes_per_s=float(os.environ.get("SGLANG_MOONCAKE_READ_MIN_MBPS", "64"))
            * 1024**2,
            contract=contract,
        )
        logger.info(
            "Mooncake native read safety: stall_s=%s failure_drained=%s "
            "read_plan_failure_drained=%s; stall detection only stops admission, "
            "it does not cancel DMA",
            self._native_read_guard.stall_s,
            contract.get("range_get_failure_drained", False),
            contract.get("read_plan_failure_drained", False),
        )
        self.read_plan_enabled = os.environ.get("SGLANG_MOONCAKE_READ_PLAN", "0") == "1"
        self.read_plan_reuse_ranges = (
            os.environ.get("SGLANG_MOONCAKE_READ_PLAN_REUSE_RANGES", "0") == "1"
        )
        if self.read_plan_reuse_ranges and not self.read_plan_enabled:
            raise ValueError(
                "SGLANG_MOONCAKE_READ_PLAN_REUSE_RANGES requires SGLANG_MOONCAKE_READ_PLAN=1"
            )
        storage_suffix = _storage_suffix(
            rank_replicated=rank_replicated,
            tp_rank=tp_rank,
            attn_cp_rank=params.attn_cp_rank,
            pp_rank=params.pp_rank,
        )
        if self.pool_group.storage_layout_tag:
            storage_suffix = f"{self.pool_group.storage_layout_tag}_{storage_suffix}"
        pool_device = getattr(kvcache, "device", None)
        storage_device_module = None
        storage_device_index = None
        if pool_device is not None:
            resolved_device = torch.device(pool_device)
            if resolved_device.type != "cpu":
                storage_device_module = torch.get_device_module(resolved_device)
                storage_device_index = resolved_device.index
                if storage_device_index is None:
                    # Capture this rank's selected device on the initializing
                    # thread; a new worker may have a different default device.
                    storage_device_index = storage_device_module.current_device()

        self._metadata_context = threading.local()
        self._availability = None
        self.storage = None

        def create_storage():
            # Restore the captured device before registration, including retries.
            if storage_device_module is not None:
                storage_device_module.set_device(storage_device_index)
            created = storage
            if created is None:
                from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                    MooncakeStore,
                )

                created = MooncakeStore(
                    storage_config,
                    mem_pool=None,
                    enable_client_http_server=params.enable_metrics,
                )
            created.mem_pool_host = self.pool_group
            created.registered_pools = self.pools
            created.mla_suffix = created.mha_suffix = storage_suffix
            if self.read_plan_enabled and not callable(
                getattr(created.store, "create_read_plan", None)
            ):
                raise RuntimeError(
                    "Mooncake ReadPlan requires a ReadPlan-enabled Mooncake package"
                )
            self.register_buffers(storage=created)
            if self._availability is not None:
                # All existence queries, including batch_set_v2's check, receive
                # a bounded metadata-only wait. Data transfers remain synchronous.
                raw_exists = created._batch_exist

                def guarded_exists(keys):
                    keys = list(keys)
                    return self._availability.query(
                        lambda: checked_exists(raw_exists(keys), len(keys)),
                        operation_source=getattr(
                            self._metadata_context, "source", "unknown"
                        ),
                        num_keys=len(keys),
                    )

                created._batch_exist = guarded_exists
            return created

        if self.l1_fallback_enabled:
            self._availability = MasterAvailability(
                create_storage,
                # A missing probe key (0) proves RPC health just as a hit does.
                lambda st: checked_exists(
                    st.store.batch_is_exist(["__sglang_master_health_probe__"]), 1
                ),
                timeout_s=float(
                    os.environ.get("SGLANG_MOONCAKE_MASTER_TIMEOUT_S", "2")
                ),
                retry_s=float(os.environ.get("SGLANG_MOONCAKE_MASTER_RETRY_S", "10")),
            )
            self._availability.initialize()
            self.storage = self._availability.storage
        else:
            self.storage = create_storage()
        logger.info(
            "Mooncake direct linker: rank_replicated=%s tp=%d/%d owner=%s suffix=%s L1_fallback=%s",
            rank_replicated,
            tp_rank,
            tp_size,
            self.offload_owner,
            storage_suffix,
            self.l1_fallback_enabled,
        )

        self.storage_metrics_collector = None
        if params.enable_metrics:
            labels = {
                "storage_backend": "mooncake_direct",
                "tp_rank": tp_rank,
                "dp_rank": _get_mooncake_storage_metrics_dp_rank(server_args, params),
                "pp_rank": params.pp_rank,
                "pp_size": params.pp_size,
                "attn_cp_rank": params.attn_cp_rank,
                "attn_cp_size": params.attn_cp_size,
            }
            if server_args.extra_metric_labels:
                labels.update(server_args.extra_metric_labels)
            collector_cls = resolve_collector_class(
                server_args, STAT_LOGGER_ROLE_STORAGE, StorageMetricsCollector
            )
            self.storage_metrics_collector = collector_cls(labels=labels)

        if self.read_plan_enabled:
            self.layer_done_counter = ReadPlanLoadCounter(
                self.num_layers, on_progress=self._native_read_guard.progress
            )
        else:
            self.layer_done_counter = LayerWiseLoadCounter(self.num_layers)
        if PoolName.MAMBA in self.pools:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )
        self.pending_loads: dict[str, list[PoolTransfer]] = {}
        self.pending_load_tokens: dict[str, int] = {}
        self.gc_frozen = False
        self.load_queue: Queue[
            tuple[int, dict[str, list[PoolTransfer]], object] | None
        ] = Queue()
        # (rids, success) per started load batch. Pushed from a finally so a
        # failed batch still releases the tree-side locks it pinned.
        self.completed_loads: Queue[tuple[list[str], bool]] = Queue()
        self._session_cleanup_error: str | None = None
        self._uncertain_sessions: list[tuple[str, ...]] = []
        self.offload_queue: Queue[
            tuple[list[PoolTransfer], int, float, object] | None
        ] = Queue()
        self.offload_results: Queue[bool] = Queue()
        self.stats = {"lookup": 0, "load": 0, "offload": 0}
        self.load_thread = threading.Thread(
            target=self.load_thread_func,
            daemon=True,
            name=f"mooncake-load-tp{tp_rank}",
        )
        self.load_thread.start()
        self.offload_thread = threading.Thread(
            target=self.offload_thread_func,
            daemon=True,
            name=f"mooncake-offload-tp{tp_rank}",
        )
        self.offload_thread.start()

    def _storage_metadata_call(self, source, operation, *args, **kwargs):
        # Scheduler lookups and background write-backs can run concurrently.
        # Capture this thread's source in guarded_exists before queueing the RPC.
        if getattr(self, "_availability", None) is None:
            return operation(*args, **kwargs)
        context = self._metadata_context
        previous = getattr(context, "source", "unknown")
        context.source = source
        try:
            return operation(*args, **kwargs)
        finally:
            context.source = previous

    def _storage_ready(self) -> bool:
        guard = getattr(self, "_native_read_guard", None)
        if guard is not None and guard.admission_blocked():
            return False
        availability = getattr(self, "_availability", None)
        if availability is None:
            return True
        ready = availability.ready()
        self.storage = availability.storage
        return ready

    def _storage_failed(self, error) -> None:
        availability = getattr(self, "_availability", None)
        if availability is not None and not isinstance(error, StoreUnavailable):
            availability.failed(error)

    def register_buffers(self, storage=None) -> None:
        storage = self.storage if storage is None else storage
        seen = set()
        for pool in self.pools.values():
            for buffer in pool.get_hybrid_pool_buffer():
                buffer_storage = buffer.untyped_storage()
                allocation = (
                    int(buffer_storage.data_ptr()),
                    int(buffer_storage.nbytes()),
                )
                if allocation in seen:
                    continue
                seen.add(allocation)
                result = storage.store.register_buffer(*allocation)
                if result not in (0, None):
                    raise RuntimeError(
                        f"Failed to register GPU KV buffer with Mooncake, error code: {result}."
                    )

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        expanded = self.pool_group.resolve_transfers(transfers)
        if not expanded:
            return []
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        page_keys = list(kv.keys)
        if not page_keys:
            return []
        # PP0 queries every PP shard; the wrapper's existing TP/CP reduction
        # then selects a boundary that all ranks can restore.
        if not self._storage_ready():
            return []
        try:
            result = self._storage_metadata_call(
                "lookup",
                self.storage.batch_exists_v2,
                page_keys,
                expanded,
                query_all_pp=True,
            )
        except Exception as error:
            if not getattr(self, "l1_fallback_enabled", True):
                raise
            self._storage_failed(error)
            # The wrapper still performs its TP/CP reduction with this empty
            # result; never skip that collective on a locally failed lookup.
            return []
        restorable = result.restorable_prefix_pages or []
        self.stats["lookup"] += 1
        if restorable:
            logger.info(
                "Mooncake direct linker lookup hit: rid=%s pages=%d candidates=%d",
                rid,
                restorable[-1],
                len(restorable),
            )
        return restorable

    def revalidate_load(self, transfers: list[PoolTransfer]) -> bool:
        valid = self._revalidate_load_local(transfers)
        groups = getattr(self, "_load_revalidation_groups", ())
        if groups:
            # LayerSplit shards can be evicted independently after lookup.
            # All CP ranks must either restore or recompute the same prefix.
            verdict = torch.tensor([int(valid)], dtype=torch.int)
            for group in groups:
                torch.distributed.all_reduce(
                    verdict, op=torch.distributed.ReduceOp.MIN, group=group
                )
            valid = bool(verdict.item())
        if not valid and getattr(self, "pp_size", 1) > 1:
            # PP0 has already distributed the prefix boundary. Preserve its
            # shape; the queued load reports failure via the existing PP abort
            # path. Future PP0 lookups while down return no external hit.
            return True
        return valid

    def _revalidate_load_local(self, transfers: list[PoolTransfer]) -> bool:
        """Re-check remote existence before the scheduler commits device
        slots to the async layer-wise load.

        The match-time lookup and ``batch_get_session_start`` can be minutes
        apart; master-side memory-watermark eviction in that window expires
        replicas and would fail the layer-wise load fatally. Re-checking here
        lets the caller degrade the request to a plain cache miss instead.

        Connection failures become a cache miss before device slots are
        published. PP uses its already-agreed boundary and aborts on load failure.
        """
        if not self._storage_ready():
            return False
        try:
            resolved = self.pool_group.resolve_transfers(
                transfers, allow_partial=True, allow_missing_kv=True
            )
            if not resolved:
                return True
            key_strs: list[str] = []
            for transfer in resolved:
                component_keys, _ = self.storage._get_hybrid_page_component_keys(
                    list(transfer.keys), transfer
                )
                key_strs.extend(self.storage._tag_keys(component_keys))
            if not key_strs:
                return True
            exist = checked_exists(
                self._storage_metadata_call(
                    "revalidate", self.storage._batch_exist, key_strs
                ),
                len(key_strs),
            )
        except Exception as error:
            self._storage_failed(error)
            return False
        missing = [key for key, state in zip(key_strs, exist) if state != 1]
        if missing:
            logger.warning(
                "Mooncake direct linker load revalidation failed: "
                "missing=%d/%d keys (master eviction race), "
                "degrading request to cache miss",
                len(missing),
                len(key_strs),
            )
            failed_get_cache = getattr(self.storage, "failed_get_cache", None)
            if failed_get_cache is not None:
                failed_get_cache.update_batch([], missing)
            return False
        return True

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        self._raise_if_session_cleanup_uncertain()
        # Query establishes a boundary at which every component is restorable;
        # insert then removes pages already resident in L1. Loading is therefore
        # intentionally partial and may contain only a side pool such as SWA.
        expanded = self.pool_group.resolve_transfers(
            transfers, allow_partial=True, allow_missing_kv=True
        )
        if not expanded:
            return False
        if rid in self.pending_loads:
            raise RuntimeError(f"Mooncake load for rid={rid} is already queued.")
        self.pending_loads[rid] = expanded
        logical_pages = {
            page_key for transfer in expanded for page_key in transfer.keys
        }
        pending_load_tokens = getattr(self, "pending_load_tokens", None)
        if pending_load_tokens is None:
            pending_load_tokens = self.pending_load_tokens = {}
        pending_load_tokens[rid] = len(logical_pages) * self.page_size
        return True

    def cancel_queued_load(self, rid: str) -> bool:
        # External linker loads target request-private slots, so a load that
        # has not been handed to the worker can be removed safely.  The old
        # publish-then-rollback path always returned False here, which left a
        # queued DMA writing into slots already released by an abort.
        if rid not in self.pending_loads:
            return False
        self.pending_loads.pop(rid, None)
        getattr(self, "pending_load_tokens", {}).pop(rid, None)
        return True

    def _raise_if_session_cleanup_uncertain(self) -> None:
        guard = getattr(self, "_native_read_guard", None)
        if guard is not None:
            guard.check()
        error = getattr(self, "_session_cleanup_error", None)
        if error is not None:
            raise RuntimeError(error)

    def _run_native_read(self, operation, **kwargs):
        guard = getattr(self, "_native_read_guard", None)
        if guard is None:
            # Also fail closed for callers constructing a linker without __init__.
            guard = self._native_read_guard = NativeReadGuard(self._storage_failed)
        return guard.run(operation, **kwargs)

    def has_unresolved_transfers(self) -> bool:
        guard = getattr(self, "_native_read_guard", None)
        state = guard.snapshot() if guard is not None else {}
        return bool(
            state.get("active_reads", 0)
            or state.get("uncertain_reads", 0)
            or getattr(self, "_session_cleanup_error", None)
            or self.load_queue.unfinished_tasks
            or self.offload_queue.unfinished_tasks
        )

    def _quarantine_sessions(self, keys: list[str], reason: str) -> None:
        # An exception does not tell us whether native acquired/released some
        # sessions. Do not guess by retrying or publish a reclaimable completion.
        if not hasattr(self, "_uncertain_sessions"):
            self._uncertain_sessions = []
        self._uncertain_sessions.append(tuple(keys))
        self._session_cleanup_error = (
            "Mooncake session ownership uncertain; process teardown required: "
            f"keys={len(keys)} reason={reason}"
        )
        logger.error(self._session_cleanup_error)

    def _start_load_sessions(self, keys: list[str], started: list[list[str]]) -> None:
        """Track successful keys immediately; retry only explicit failures once."""
        remaining = list(keys)
        for attempt in range(2):
            try:
                result = list(self.storage.store.batch_get_session_start(remaining))
                if len(result) != len(remaining):
                    raise ValueError("session start returned incomplete key statuses")
            except BaseException as error:
                self._quarantine_sessions(
                    remaining, f"session start: {type(error).__name__}"
                )
                raise
            acquired = [key for key, code in zip(remaining, result) if code == 0]
            if acquired:
                started.append(acquired)
            remaining = [key for key, code in zip(remaining, result) if code != 0]
            if not remaining:
                return
            if attempt == 0:
                logger.warning(
                    "Mooncake session start partial failure: failed=%d; "
                    "retrying only failed keys",
                    len(remaining),
                )
        failed_get_cache = getattr(self.storage, "failed_get_cache", None)
        if failed_get_cache is not None:
            failed_get_cache.update_batch([], remaining)
        raise RuntimeError(
            f"Mooncake get session start failed: failed={len(remaining)}"
        )

    def _end_load_sessions(self, started: list[list[str]]) -> bool:
        """Consume cleanup attempts once; a partially successful end is not retried."""
        success = True
        while started:
            keys = started.pop(0)
            try:
                # The deployed native ABI returns an integer status. A normal
                # Python return alone does not establish that cleanup succeeded.
                status = self.storage.store.batch_get_session_end(keys)
                if type(status) is not int or status != 0:
                    raise RuntimeError(f"Mooncake get session end failed: {status!r}")
            except BaseException as error:
                self._quarantine_sessions(keys, f"session end: {type(error).__name__}")
                self._storage_failed(error)
                logger.exception("Mooncake layer-wise load session cleanup failed")
                success = False
        return success

    def num_completed_loads(self) -> int:
        self._raise_if_session_cleanup_uncertain()
        return self.completed_loads.qsize()

    def pop_completed_load(self) -> tuple[list[str], bool]:
        return self.completed_loads.get_nowait()

    def freeze_gc_once(self) -> None:
        if self.gc_frozen:
            return
        # Transfer metadata creates many short-lived lists. Keep the mature
        # model graph out of cyclic GC scans before load or offload traffic.
        freeze_gc("Mooncake direct linker")
        self.gc_frozen = True

    def start_layer_wise_loading(self) -> int:
        self._raise_if_session_cleanup_uncertain()
        if not self.pending_loads:
            return -1
        self.freeze_gc_once()
        pending = self.pending_loads
        self.pending_loads = {}

        counter_index = self.layer_done_counter.update_producer()
        ready_event = device_module.Event()
        ready_event.record()
        self.load_queue.put((counter_index, pending, ready_event))
        self.stats["load"] += len(pending)
        return counter_index

    def load_thread_func(self) -> None:
        while True:
            task = self.load_queue.get()
            try:
                if task is None:
                    return
                counter_index, pending, ready_event = task
                success = False
                try:
                    ready_event.synchronize()
                    success = self.load_layer_wise(
                        counter_index, list(pending.values())
                    )
                except BaseException as error:
                    self.layer_done_counter.fail(counter_index, error)
                    logger.exception("Mooncake layer-wise load batch failed")
                finally:
                    success = self.layer_done_counter.finish(counter_index) and success
                    self._finish_prefetch_metrics(list(pending), success is True)
                    guard = getattr(self, "_native_read_guard", None)
                    native_uncertain = (
                        guard is not None and guard.snapshot()["uncertain_reads"]
                    )
                    if (
                        getattr(self, "_session_cleanup_error", None) is None
                        and not native_uncertain
                    ):
                        self.completed_loads.put((list(pending), success))
                    else:
                        logger.error(
                            "Withholding external load completion: batch=%d rids=%s; "
                            "session/native ownership uncertain, "
                            "private slots retained",
                            counter_index,
                            list(pending),
                        )
            finally:
                self.load_queue.task_done()

    def load_layer_wise(
        self, counter_index: int, request_transfers: list[list[PoolTransfer]]
    ) -> bool:
        started = []
        success = False
        maybe_fail = arm_load_failure_injection(self.tp_rank)
        try:
            # Do not reject load() after the wrapper published its pages. A load
            # admitted just before the outage must produce a failed completion.
            if not self._storage_ready():
                raise StoreUnavailable(
                    "Mooncake unavailable before queued load started"
                )
            if getattr(self, "read_plan_enabled", False):
                self.load_with_read_plan(counter_index, request_transfers)
                return True
            batches: dict[PoolName, tuple[list[str], list[int]]] = {}
            for transfers in request_transfers:
                for transfer in transfers:
                    keys, locations = batches.setdefault(transfer.name, ([], []))
                    component_keys, _ = self.storage._get_hybrid_page_component_keys(
                        list(transfer.keys), transfer
                    )
                    keys.extend(self.storage._tag_keys(component_keys))
                    locations.extend(
                        self.pools[transfer.name].prepare_locations(
                            transfer.host_indices
                        )
                    )
            for keys, _ in batches.values():
                self._start_load_sessions(keys, started)

            if self.enable_page_wise_load and any(
                len(keys) >= self.page_wise_load_threshold
                for keys, _ in batches.values()
            ):
                self._load_page_wise(counter_index, batches, started, maybe_fail)
                success = True
                return success

            for layer in range(self.num_layers):
                for name, (keys, locations) in batches.items():
                    meta = self.pools[name].get_prepared_layer_range_meta(
                        locations, layer
                    )
                    if meta is None:
                        continue
                    ptrs, sizes, offsets = meta
                    maybe_fail(name, f"layer={layer}")
                    expected = [sum(item) for item in sizes]
                    self._run_native_read(
                        lambda: self.storage.store.batch_get_into_multi_buffer_ranges(
                            keys, ptrs, sizes, offsets
                        ),
                        batch=counter_index,
                        stage=f"pool={name},layer={layer}",
                        expected=expected,
                    )
                self.layer_done_counter.complete(counter_index, layer)
            success = True
        except BaseException as error:
            self._storage_failed(error)
            self.layer_done_counter.fail(counter_index, error)
            logger.exception("Mooncake layer-wise load batch failed")
        finally:
            guard = getattr(self, "_native_read_guard", None)
            if guard is not None and guard.snapshot()["uncertain_reads"]:
                # Ending cached metadata is not cancellation. Retain all known
                # sessions and private slots until process teardown.
                self._quarantine_sessions(
                    [key for keys in started for key in keys],
                    "native read failure has no verified DMA drain contract",
                )
                success = False
            elif not self._end_load_sessions(started):
                self.layer_done_counter.fail(
                    counter_index, RuntimeError(self._session_cleanup_error)
                )
                success = False
        return success

    def _finish_prefetch_metrics(self, rids: list[str], success: bool) -> None:
        pending_load_tokens = getattr(self, "pending_load_tokens", {})
        tokens = sum(pending_load_tokens.pop(rid, 0) for rid in rids)
        if not success or tokens <= 0:
            return
        self._log_prefetched_tokens(tokens)
        recorder = getattr(
            getattr(self.storage, "store", None), "record_prefetched_tokens", None
        )
        if recorder is not None:
            try:
                recorder(tokens)
            except BaseException:
                logger.warning(
                    "Failed to record Mooncake prefetched token metric.",
                    exc_info=True,
                )

    def _log_prefetched_tokens(self, tokens: int) -> None:
        collector = getattr(self, "storage_metrics_collector", None)
        if collector is None or tokens <= 0:
            return
        try:
            collector.log_prefetched_tokens(tokens)
        except BaseException:
            logger.warning(
                "Failed to record SGLang direct-storage prefetch metrics.",
                exc_info=True,
            )

    def _load_page_wise(self, counter_index: int, batches, started, maybe_fail) -> None:
        """Load all layer ranges for each page before exposing the data."""
        all_keys: list[str] = []
        all_ptrs: list[list[int]] = []
        all_sizes: list[list[int]] = []
        all_offsets: list[list[int]] = []
        for name, (keys, locations) in batches.items():
            ptrs: list[list[int]] = [[] for _ in keys]
            sizes: list[list[int]] = [[] for _ in keys]
            offsets: list[list[int]] = [[] for _ in keys]

            for layer in range(self.num_layers):
                meta = self.pools[name].get_prepared_layer_range_meta(locations, layer)
                if meta is None:
                    continue
                layer_ptrs, layer_sizes, layer_offsets = meta
                if not (
                    len(layer_ptrs)
                    == len(layer_sizes)
                    == len(layer_offsets)
                    == len(keys)
                ):
                    raise ValueError(
                        f"Mooncake pool={name} layer={layer} produced "
                        f"{len(layer_ptrs)} range entries for {len(keys)} keys."
                    )
                for index in range(len(keys)):
                    ptrs[index].extend(layer_ptrs[index])
                    sizes[index].extend(layer_sizes[index])
                    offsets[index].extend(layer_offsets[index])

            all_keys.extend(keys)
            all_ptrs.extend(ptrs)
            all_sizes.extend(sizes)
            all_offsets.extend(offsets)

        lengths = {
            "keys": len(all_keys),
            "ptrs": len(all_ptrs),
            "sizes": len(all_sizes),
            "offsets": len(all_offsets),
        }
        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Mooncake page-wise aggregated metadata mismatch: {lengths}."
            )

        # Mooncake's range API is key-major and does not take a pool argument,
        # so differently suffixed physical-pool objects can share one call.
        maybe_fail("aggregated", "complete_page")
        expected = [sum(item) for item in all_sizes]
        self._run_native_read(
            lambda: self.storage.store.batch_get_into_multi_buffer_ranges(
                all_keys, all_ptrs, all_sizes, all_offsets
            ),
            batch=counter_index,
            stage="complete_page",
            expected=expected,
        )

        # Page-wise loading gives up layer overlap. Release the read sessions
        # only after every complete page is loaded, and before any layer becomes
        # visible to the model. Ambiguous cleanup is never retried in finally.
        if not self._end_load_sessions(started):
            self._raise_if_session_cleanup_uncertain()
        for layer in range(self.num_layers):
            self.layer_done_counter.complete(counter_index, layer)

    def _prepare_read_plan_layouts(self, request_transfers: list[list[PoolTransfer]]):
        # Consolidate index copies once per pool, preserving request/key order.
        # Each component describes (base, row stride, byte count, source offset).
        # Locations may be non-contiguous; Mooncake expands addresses in C++.
        batches = {}
        for transfers in request_transfers:
            for transfer in transfers:
                keys, indices = batches.setdefault(transfer.name, ([], []))
                component_keys, _ = self.storage._get_hybrid_page_component_keys(
                    list(transfer.keys), transfer
                )
                keys.extend(self.storage._tag_keys(component_keys))
                indices.append(transfer.host_indices)

        layouts = []
        for name, (keys, indices) in batches.items():
            pool = self.pools[name]
            joined = indices[0] if len(indices) == 1 else torch.cat(indices)
            locations = pool.prepare_locations(joined)
            layout = []
            for layer in range(self.num_layers):
                mapped = pool.layer_mapping.get(layer)
                buffer_indices = (
                    ()
                    if mapped is None
                    else (mapped,) if isinstance(mapped, int) else tuple(mapped)
                )
                layout.append(
                    [
                        (*component[buffer_index], offsets[buffer_index])
                        for component, offsets in zip(
                            pool.buffer_meta, pool._component_offsets
                        )
                        for buffer_index in buffer_indices
                    ]
                )
            layouts.append((keys, locations, pool.packed, layout))

        return layouts

    def load_with_read_plan(
        self, counter_index: int, request_transfers: list[list[PoolTransfer]]
    ) -> None:
        layouts = self._prepare_read_plan_layouts(request_transfers)
        plan = self.storage.store.create_read_plan(
            layouts,
            self.num_layers,
            reuse_ranges=self.read_plan_reuse_ranges,
            buffer_owners=self.pools,
        )
        self.layer_done_counter.bind(counter_index, plan)
        # run() and wait() release the GIL. Each layer becomes visible only after
        # every pool's bytes have been checked; the last wait includes cleanup.
        byte_count = sum(
            len(locations) * sum(item[2] for layer in layout for item in layer)
            for _, locations, _, layout in layouts
        )
        self._run_native_read(
            plan.run,
            batch=counter_index,
            stage="read_plan",
            read_plan=True,
            byte_count=byte_count,
        )

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        expanded = self.pool_group.resolve_transfers(transfers, allow_partial=True)
        if not expanded:
            return False
        self.freeze_gc_once()
        if not self.offload_owner:
            self.offload_results.put(True)
            return True
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        tokens = len(kv.keys) * self.page_size
        ready_event = device_module.Event()
        ready_event.record()
        self.offload_queue.put((expanded, tokens, time.perf_counter(), ready_event))
        return True

    def offload_thread_func(self) -> None:
        while True:
            task = self.offload_queue.get()
            metric_recorded = False
            try:
                if task is None:
                    return
                expanded, tokens, started, ready_event = task
                ready_event.synchronize()
                if not self._storage_ready():
                    raise StoreUnavailable(
                        "Mooncake unavailable; skipping queued write-back"
                    )
                results = self._storage_metadata_call(
                    "writeback-exists", self.storage.batch_set_v2, expanded
                )
                success = all(all(pool_results) for pool_results in results.values())
                if not success:
                    self._storage_failed(RuntimeError("Mooncake write-back failed"))
                self._log_l4_backup_metric(
                    tokens, time.perf_counter() - started, success
                )
                metric_recorded = True
                if success:
                    self.stats["offload"] += 1
                    if self.stats["offload"] == 1:
                        logger.info("Mooncake direct linker offload: tokens=%d", tokens)
                self.offload_results.put(success)
            except BaseException as error:
                self._storage_failed(error)
                if not isinstance(error, StoreUnavailable):
                    logger.exception("Mooncake offload failed")
                if task is not None and not metric_recorded:
                    _, tokens, started, _ = task
                    self._log_l4_backup_metric(
                        tokens, time.perf_counter() - started, False
                    )
                self.offload_results.put(False)
            finally:
                self.offload_queue.task_done()

    def _log_l4_backup_metric(
        self, tokens: int, duration: float, success: bool
    ) -> None:
        collector = getattr(self, "storage_metrics_collector", None)
        if collector is None:
            return
        try:
            # This branch intentionally does not guess DFS versus local disk.
            # Its Mooncake configuration owns that routing decision.
            collector.log_l4_backup("mooncake", tokens, duration, success)
        except BaseException:
            logger.warning(
                "Failed to record SGLang L4 backup metrics.",
                exc_info=True,
            )

    def num_completed_offloads(self) -> int:
        return self.offload_results.qsize()

    def pop_completed_offload(self) -> bool:
        return self.offload_results.get_nowait()

    def reset(self) -> None:
        self._raise_if_session_cleanup_uncertain()
        self.pending_loads.clear()
        getattr(self, "pending_load_tokens", {}).clear()
        self.load_queue.join()
        self.offload_queue.join()
        self._raise_if_session_cleanup_uncertain()
        while True:
            try:
                self.offload_results.get_nowait()
            except Empty:
                break
        while True:
            try:
                self.completed_loads.get_nowait()
            except Empty:
                break
        self.layer_done_counter.reset()

    def close(self) -> None:
        self.reset()
        guard = getattr(self, "_native_read_guard", None)
        if guard is not None:
            guard.close()
        self.load_queue.put(None)
        self.offload_queue.put(None)
        self.load_thread.join()
        self.offload_thread.join()
        logger.info("Mooncake direct linker stats: %s", self.stats)
        if self._availability is not None:
            self._availability.close()
        if self.storage is not None:
            self.storage.close()
