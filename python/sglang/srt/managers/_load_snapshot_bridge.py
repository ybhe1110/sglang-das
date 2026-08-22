"""Bridge GetLoadsReqOutput (本分支的 IPC 类型) to LoadSnapshot (reporter 期望的类型)."""

from __future__ import annotations

from sglang.srt.managers.io_struct import GetLoadsReqOutput
from sglang.srt.managers.load_snapshot import LoadSnapshot


def get_loads_output_to_snapshot(output: GetLoadsReqOutput) -> LoadSnapshot:
    """Convert GetLoadsReqOutput from本分支's IPC to LoadSnapshot for reporter.
    
    本分支's scheduler returns GetLoadsReqOutput via the get_loads IPC channel.
    The load reporter expects LoadSnapshot. This function bridges the gap.
    
    num_waiting_uncached_tokens is hardcoded to 0 because本分支's scheduler
    does not expose it and lacks the accessors (get_waiting_queue, get_chunked_req,
    req.num_matched_prefix_tokens) needed to compute it. This is safe for:
    - Non-disaggregated deployments
    - Deployments where cache-hit rate is high enough that uncached waiting tokens
      are negligible for load-balancing purposes.
    """
    return LoadSnapshot(
        timestamp=output.timestamp,
        dp_rank=output.dp_rank,
        num_running_reqs=output.num_running_reqs,
        num_waiting_reqs=output.num_waiting_reqs,
        num_waiting_uncached_tokens=0,  # hardcoded: see docstring
        num_used_tokens=output.num_used_tokens,
        num_total_tokens=output.num_total_tokens,
        num_active_tokens=output.num_total_tokens,  # assume no disagg kv-transfer lag
        max_total_num_tokens=output.max_total_num_tokens,
        max_running_requests=output.max_running_requests,
        token_usage=output.token_usage,
        gen_throughput=output.gen_throughput,
        cache_hit_rate=output.cache_hit_rate,
        utilization=output.utilization,
        # Cumulative counters: omitted (reporter doesn't validate them)
        # Nested sections (memory/spec/lora/disagg/queues): omitted for now
    )
