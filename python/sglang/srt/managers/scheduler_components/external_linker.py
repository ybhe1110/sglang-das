"""Settle direct-cache loads before any request result is published."""

from http import HTTPStatus


def finish_external_linker_loads(scheduler, batch, result):
    cache = scheduler.tree_cache
    if getattr(cache, "linker", None) is None:
        return
    failed = cache.finish_external_linker_loads(batch.hicache_consumer_index)
    if not failed:
        return

    # Overlap may already have launched another chunk of the same request.
    # Drain all submitted GPU consumers before normal abort frees their rows.
    scheduler.forward_stream.synchronize()
    if getattr(result, "copy_done", None) is not None:
        result.copy_done.synchronize()
    from sglang.srt.disaggregation.utils import prepare_abort

    for req in failed:
        prepare_abort(
            req, "External KV load failed", status_code=HTTPStatus.SERVICE_UNAVAILABLE
        )
        req.skip_radix_cache_insert = True
        if scheduler.chunked_req is req:
            scheduler.chunked_req = None
        if getattr(scheduler, "_pending_chunked_abort_req", None) is req:
            scheduler._pending_chunked_abort_req = None
