from __future__ import annotations

import logging
import threading
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
from sglang.srt.mem_cache.unified_cache.linker_lifecycle import (
    LayerWiseLoadCounter,
    all_reduce_with_timeout,
    identity_words,
    wait_queue_drained,
)
from sglang.srt.mem_cache.unified_cache.linker_mla_dedup import (
    LinkerMLADedupBroadcaster,
)
from sglang.srt.mem_cache.unified_cache.unified_cache_linker import UnifiedCacheLinker
from sglang.srt.runtime_context import (
    get_memory,
    get_model,
    get_parallel,
)
from sglang.srt.utils import freeze_gc, get_device_module

logger = logging.getLogger(__name__)
device_module = get_device_module()


def _storage_suffix(
    *, rank_replicated: bool, tp_rank: int, attn_cp_rank: int, pp_rank: int
) -> str:
    parts = []
    if not rank_replicated:
        parts.append(f"tp{tp_rank}")
    parts.extend((f"cp{attn_cp_rank}", f"pp{pp_rank}"))
    return "_".join(parts)


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
        kvcache = params.token_to_kv_pool_allocator.get_kvcache()
        self.pool_group = resolve_hybrid_device_pool_group(
            kvcache=kvcache,
            page_size=self.page_size,
            params=params,
            components=components,
        )
        self.pools = self.pool_group.entry_map
        self.num_layers = self.pool_group.num_layers

        tp_rank = 0
        tp_size = get_parallel().tp_size
        tp_group = params.attn_tp_cache_group or params.tp_cache_group
        if torch.distributed.is_available() and torch.distributed.is_initialized():
            tp_rank = torch.distributed.get_rank(group=tp_group)
            tp_size = torch.distributed.get_world_size(group=tp_group)
        rank_replicated = self.pool_group.rank_replicated
        self.offload_owner = not rank_replicated or tp_rank == 0
        extra_config, *_ = HybridCacheController.parse_storage_backend_extra_config(
            get_memory().hicache_storage_backend_extra_config
        )
        # TODO: unlike `HiCacheController._generate_storage_config`, this sets no
        # `dp_rank`. Aligning the storage-config producers changes persisted
        # storage keys, so it waits for a storage-key format change.
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
        )
        storage_suffix = _storage_suffix(
            rank_replicated=rank_replicated,
            tp_rank=tp_rank,
            attn_cp_rank=params.attn_cp_rank,
            pp_rank=params.pp_rank,
        )
        # Capture the scheduler's concrete device before metadata/load workers start.
        self.device = torch.device(kvcache.device)
        if self.device.index is None and self.device.type != "cpu":
            self.device = torch.device(self.device.type, device_module.current_device())
        self._closed = False

        def create_storage():
            if self.device.type != "cpu":
                device_module.set_device(self.device)
            created = storage
            if created is None:
                from sglang.srt.mem_cache.storage.mooncake_store.mooncake_store import (
                    MooncakeStore,
                )

                created = MooncakeStore(storage_config, mem_pool=None)
            created.mem_pool_host = self.pool_group
            created.registered_pools = self.pools
            created.mla_suffix = created.mha_suffix = storage_suffix
            try:
                self.register_buffers(created)
            except BaseException:
                # A failed registration must not leak a partially initialized
                # store on every recovery attempt. No load has been admitted yet.
                created.close()
                raise
            return created

        self._availability = MasterAvailability(
            create_storage,
            lambda st: checked_exists(
                st.store.batch_is_exist(["__sglang_master_health_probe__"]), 1
            ),
        )
        self._availability.initialize()
        self.storage = self._availability.storage
        logger.info(
            "Mooncake direct linker storage topology: "
            "rank_replicated=%s, tp_rank=%d/%d, offload_owner=%s, suffix=%s",
            rank_replicated,
            tp_rank,
            tp_size,
            self.offload_owner,
            storage_suffix,
        )

        self.mla_broadcaster = None
        self.tp_group = tp_group
        self.broadcast_loads = {}
        self.broadcast_events = []
        if server_args.enable_linker_mla_dedup and rank_replicated and tp_size > 1:
            self.mla_broadcaster = LinkerMLADedupBroadcaster.build(
                self.pool_group, params.tp_cache_group, params.attn_tp_cache_group
            )
            logger.info(
                "MLA linker rank-0 loading enabled: tp_rank=%d/%d", tp_rank, tp_size
            )
        self.layer_done_counter = LayerWiseLoadCounter(
            self.num_layers,
            self._broadcast_loaded_layers if self.mla_broadcaster else None,
        )
        if PoolName.MAMBA in self.pools:
            params.req_to_token_pool.register_layer_transfer_counter(
                self.layer_done_counter
            )
        self.batch_rids = {}
        self.pending_loads: dict[str, list[PoolTransfer]] = {}
        self.gc_frozen = False
        self.load_queue: Queue[
            tuple[int, dict[str, list[PoolTransfer]], object] | None
        ] = Queue()
        self.completed_loads: Queue[list[tuple[str, bool]]] = Queue()
        self.offload_queue: Queue[tuple[list[PoolTransfer], int, object] | None] = (
            Queue()
        )
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

    def _storage_ready(self):
        if self._closed:
            return False
        ready = self._availability.ready()
        self.storage = self._availability.storage
        return ready

    def register_buffers(self, backend=None) -> None:
        backend = self.storage if backend is None else backend
        seen = set()
        for pool in self.pools.values():
            for buffer in pool.get_hybrid_pool_buffer():
                storage = buffer.untyped_storage()
                allocation = (int(storage.data_ptr()), int(storage.nbytes()))
                if allocation in seen:
                    continue
                seen.add(allocation)
                result = backend.store.register_buffer(*allocation)
                if result not in (0, None):
                    raise RuntimeError(
                        "Failed to register GPU KV buffer with Mooncake, "
                        f"error code: {result}."
                    )

    def lookup(self, rid: str, transfers: list[PoolTransfer]) -> list[int]:
        if not self._storage_ready():
            return []
        expanded = self.pool_group.resolve_transfers(transfers)
        if not expanded:
            return []
        kv = next(transfer for transfer in transfers if transfer.name == PoolName.KV)
        page_keys = list(kv.keys)
        if not page_keys:
            return []
        try:
            result = self._availability.query(
                lambda: self.storage.batch_exists_v2(page_keys, expanded),
                operation_source="lookup",
                num_keys=len(page_keys),
            )
        except StoreUnavailable:
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

    def load(self, rid: str, transfers: list[PoolTransfer]) -> bool:
        if not self._storage_ready():
            return False
        # Every tail page is private; no dedup or partial publication before I/O.
        expanded = self.pool_group.resolve_transfers(transfers)
        if not expanded:
            return False
        if rid in self.pending_loads:
            raise RuntimeError(f"Mooncake load for rid={rid} is already queued.")
        self.pending_loads[rid] = expanded
        return True

    def cancel_queued_load(self, rid: str) -> bool:
        return self.pending_loads.pop(rid, None) is not None

    def num_completed_loads(self) -> int:
        return self.completed_loads.qsize()

    def pop_completed_load(self) -> list[tuple[str, bool]]:
        return self.completed_loads.get_nowait()

    def finish_loads(self, consumer_index):
        success = self.layer_done_counter.finish(consumer_index)
        # Broadcast receives are GPU writes too; wait before transferring ownership.
        for event, _ in self.broadcast_events:
            event.synchronize()
        self.broadcast_events.clear()
        rids = self.batch_rids.pop(consumer_index)
        self.completed_loads.put([(rid, success) for rid in rids])

    def freeze_gc_once(self) -> None:
        if self.gc_frozen:
            return
        # Transfer metadata creates many short-lived lists. Keep the mature
        # model graph out of cyclic GC scans before load or offload traffic.
        freeze_gc("Mooncake direct linker")
        self.gc_frozen = True

    def start_layer_wise_loading(self) -> int:
        broadcaster = self.mla_broadcaster
        if broadcaster is not None:
            # Compare logical load keys (not rank-local private slots),
            # including empty batches,
            # before deciding collectively whether this batch can broadcast.
            plan = [
                (rid, [(t.name, list(t.keys)) for t in transfers])
                for rid, transfers in self.pending_loads.items()
            ]
            digest = identity_words(plan)
            payload = torch.tensor(digest + [-v for v in digest], dtype=torch.int64)
            all_reduce_with_timeout(
                payload, torch.distributed.ReduceOp.MIN, self.tp_group
            )
            if payload[:4].tolist() != (-payload[4:]).tolist():
                logger.warning("MLA linker load plans differ; using all-rank reads.")
                broadcaster = None
        if not self.pending_loads:
            return -1
        self.freeze_gc_once()
        pending = self.pending_loads
        self.pending_loads = {}

        counter_index = self.layer_done_counter.update_producer()
        self.batch_rids[counter_index] = list(pending)
        if broadcaster is not None:
            indices = {}
            for transfers in pending.values():
                for transfer in transfers:
                    indices.setdefault(transfer.name, []).append(transfer.host_indices)
            prepared = broadcaster.prepare_broadcast(
                {name: torch.cat(parts) for name, parts in indices.items()},
                device_module.current_stream(),
            )
            if not broadcaster.is_src:
                for layer in range(self.num_layers):
                    self.layer_done_counter.complete(counter_index, layer)
        ready_event = device_module.Event()
        ready_event.record()
        if broadcaster is not None:
            self.broadcast_loads[counter_index] = (
                0,
                prepared,
                list(pending),
                ready_event,
            )
        if broadcaster is None or broadcaster.is_src:
            self.load_queue.put((counter_index, pending, ready_event))
        self.stats["load"] += len(pending)
        return counter_index

    def _broadcast_loaded_layers(self, index: int, threshold: int) -> None:
        batch = self.broadcast_loads.get(index)
        if batch is None:
            return
        first, prepared, rids, ready_event = batch
        if first == 0:
            device_module.current_stream().wait_event(ready_event)
        # KV access may wait several times per layer, or skip sparse layers.
        # Launch each broadcast exactly once, in order, on the forward stream.
        for layer in range(first, threshold + 1):
            self.mla_broadcaster.broadcast_loaded_layer(layer, prepared)
        if threshold == self.num_layers - 1:
            event = device_module.Event()
            event.record()
            self.broadcast_events.append((event, rids))
            if not self.mla_broadcaster.is_src:
                # Event remains separately fenced in finish_loads.
                self.layer_done_counter.publish_terminal(index)
            del self.broadcast_loads[index]
        else:
            self.broadcast_loads[index] = (
                max(first, threshold + 1),
                prepared,
                rids,
                ready_event,
            )

    def load_thread_func(self) -> None:
        device_module.set_device(self.device)
        while True:
            task = self.load_queue.get()
            try:
                if task is None:
                    return
                counter_index, pending, ready_event = task
                try:
                    ready_event.synchronize()
                    self.load_layer_wise(counter_index, list(pending.values()))
                except BaseException as error:
                    self.layer_done_counter.fail(counter_index, error)
                    self._availability.failed(error)
                    logger.exception("Mooncake layer-wise load batch failed")
                finally:
                    self.layer_done_counter.publish_terminal(counter_index)
            finally:
                self.load_queue.task_done()

    def load_layer_wise(
        self, counter_index: int, request_transfers: list[list[PoolTransfer]]
    ) -> None:
        started = []
        try:
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
                result = self.storage.store.batch_get_session_start(keys)
                # Retain successful starts even if a later key failed.
                if result is not None and not isinstance(result, int):
                    started.append(
                        [key for key, code in zip(keys, result) if code == 0]
                    )
                if (
                    result is None
                    or isinstance(result, int)
                    or list(result) != [0] * len(keys)
                ):
                    raise RuntimeError(
                        f"Mooncake get session start failed: keys={len(keys)}, "
                        f"results={result}"
                    )

            for layer in range(self.num_layers):
                for name, (keys, locations) in batches.items():
                    meta = self.pools[name].get_prepared_layer_range_meta(
                        locations, layer
                    )
                    if meta is None:
                        continue
                    ptrs, sizes, offsets = meta
                    result = self.storage.store.batch_get_into_multi_buffer_ranges(
                        keys,
                        ptrs,
                        sizes,
                        offsets,
                    )
                    expected = [sum(item) for item in sizes]
                    if (
                        result is None
                        or isinstance(result, int)
                        or list(result) != expected
                    ):
                        raise RuntimeError(
                            f"Mooncake range get failed for pool={name}, "
                            f"layer={layer}: transferred={result}, "
                            f"expected={expected}"
                        )
                self.layer_done_counter.complete(counter_index, layer)
        except BaseException as error:
            self.layer_done_counter.fail(counter_index, error)
            self._availability.failed(error)
            logger.exception("Mooncake layer-wise load batch failed")
        finally:
            for keys in started:
                try:
                    self.storage.store.batch_get_session_end(keys)
                except BaseException as error:
                    self.layer_done_counter.fail(counter_index, error)
                    self._availability.failed(error)
                    logger.exception("Mooncake layer-wise load session cleanup failed")

    def offload(self, transfers: list[PoolTransfer]) -> bool:
        if not self._storage_ready():
            return False
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
        self.offload_queue.put((expanded, tokens, ready_event))
        return True

    def offload_thread_func(self) -> None:
        device_module.set_device(self.device)
        while True:
            task = self.offload_queue.get()
            try:
                if task is None:
                    return
                expanded, tokens, ready_event = task
                ready_event.synchronize()
                results = self.storage.batch_set_v2(expanded)
                success = all(all(pool_results) for pool_results in results.values())
                if success:
                    self.stats["offload"] += 1
                    if self.stats["offload"] == 1:
                        logger.info("Mooncake direct linker offload: tokens=%d", tokens)
                if not success:
                    self._availability.failed("Mooncake write-back failed")
                self.offload_results.put(success)
            except BaseException as error:
                self._availability.failed(error)
                logger.exception("Mooncake offload failed")
                self.offload_results.put(False)
            finally:
                self.offload_queue.task_done()

    def num_completed_offloads(self) -> int:
        return self.offload_results.qsize()

    def pop_completed_offload(self) -> bool:
        return self.offload_results.get_nowait()

    def reset(self) -> None:
        self.pending_loads.clear()
        wait_queue_drained(self.load_queue)
        wait_queue_drained(self.offload_queue)
        for event, _ in self.broadcast_events:
            event.synchronize()
        self.broadcast_events.clear()
        self.broadcast_loads.clear()
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
        self.batch_rids.clear()

    def close(self) -> None:
        if getattr(self, "_shutdown_complete", False):
            return
        self._closed = True
        if not self._availability.wait_closed(timeout_s=2):
            raise RuntimeError(
                "Mooncake metadata still active; retaining storage/buffers"
            )
        self.reset()
        self.load_queue.put(None)
        self.offload_queue.put(None)
        self.load_thread.join()
        self.offload_thread.join()
        logger.info("Mooncake direct linker stats: %s", self.stats)
        # Initialization may have completed after the caller's timeout.
        self.storage = self._availability.storage
        if self.storage is not None:
            self.storage.close()
        if self.mla_broadcaster is not None:
            self.mla_broadcaster.destroy()
        self._shutdown_complete = True
