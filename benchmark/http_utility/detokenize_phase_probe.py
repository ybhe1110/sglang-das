"""Pin independent HTTP/1.1 sockets and compare GET delays with server decode spans."""

import argparse
import asyncio
import hashlib
import json
import math
import os
import random
import statistics
import time
from collections import defaultdict
from contextlib import AsyncExitStack
from datetime import datetime, timezone
from pathlib import Path

import httpx

GETS = ("/v1/models", "/v1/loads")
POST = "/v1/detokenize"
HEADER = "x-sglang-phase-probe"
MS = 1_000_000
LONG_MARKS = (
    "asgi_enter",
    "body_read",
    "json_begin",
    "json_end",
    "submitted",
    "worker_begin",
    "items_checked",
    "normalize_begin",
    "decode_begin",
    "decode_end",
    "serialize_begin",
    "serialize_end",
    "headers_ready",
)
GET_MARKS = ("asgi_enter", "handler_begin", "handler_end", "headers_ready")


async def request(client, path, body=None, pinned=None):
    start = time.perf_counter_ns()
    r = await client.request("POST" if body is not None else "GET", path, content=body)
    end = time.perf_counter_ns()
    r.raise_for_status()
    if HEADER not in r.headers:
        raise RuntimeError(
            f"{path} missing timing header; check deployment and SGLANG_ENABLE_UTILITY_PHASE_PROBE"
        )
    trace = json.loads(r.headers[HEADER])
    sock = r.extensions["network_stream"].get_extra_info("socket")
    identity = (trace["instance"], trace["pid"], sock.getsockname()[1])
    if pinned is not None and identity != pinned:
        raise RuntimeError(
            f"{path}: connection/worker changed from {pinned} to {identity}"
        )
    marks = trace["ns"]
    times = [marks[key] for key in (LONG_MARKS if body is not None else GET_MARKS)]
    if times != sorted(times):
        raise RuntimeError(f"{path}: out-of-order stage timestamps")
    event = dict(path=path, worker=identity, start=start, end=end, ns=marks)
    if body is not None:
        event["response_sha256"] = await asyncio.to_thread(
            lambda: hashlib.sha256(r.content).hexdigest()
        )
    return event


def rtt(event):
    return (event["end"] - event["start"]) / MS


def clock_bounds(events):
    # If server_time = client_time + offset, causality bounds that offset.
    # No wall-clock synchronization or symmetric network path is assumed.
    low = max(e["ns"]["headers_ready"] - e["end"] for e in events)
    high = min(e["ns"]["asgi_enter"] - e["start"] for e in events)
    if low > high:
        raise RuntimeError("Inconsistent client/server clock calibration")
    return low, high


def overlap(a, b, c, d):
    return max(0, min(b, d) - max(a, c)) / MS


def analyze(tool, get, quiet, bounds):
    low, high = bounds
    g, t = get["ns"], tool["ns"]
    decode = t["decode_begin"], t["decode_end"]
    baseline = statistics.median(rtt(e) for e in quiet)
    excess = max(0, rtt(get) - baseline)
    # The mapped send time is an interval. Both endpoints are needed for
    # pre-ASGI delay: ASGI cannot stamp a request while its loop is stalled.
    early, late = get["start"] + low, get["start"] + high
    # Intersect/union the possible RTT windows. Include response delivery:
    # the HTTP loop can stall after preparing headers and before sending bytes.
    lo = overlap(late, get["end"] + low, *decode)
    hi = overlap(early, get["end"] + high, *decode)
    return dict(
        path=get["path"],
        rtt_ms=rtt(get),
        idle_ms=baseline,
        excess_ms=excess,
        actual_offset_ms=(get["start"] - tool["start"]) / MS,
        clock_uncertainty_ms=(high - low) / MS,
        send_to_asgi_ms=[(g["asgi_enter"] - late) / MS, (g["asgi_enter"] - early) / MS],
        headers_to_client_ms=[
            (get["end"] + low - g["headers_ready"]) / MS,
            (get["end"] + high - g["headers_ready"]) / MS,
        ],
        asgi_to_handler_ms=(g["handler_begin"] - g["asgi_enter"]) / MS,
        handler_ms=(g["handler_end"] - g["handler_begin"]) / MS,
        decode_ms=(decode[1] - decode[0]) / MS,
        decode_overlap_ms=[lo, hi],
        pre_handler_decode_ms=[
            overlap(late, g["handler_begin"], *decode),
            overlap(early, g["handler_begin"], *decode),
        ],
        # Up to the entire ordinary baseline could itself overlap decode.
        excess_in_decode_ms=[min(excess, max(0, lo - baseline)), min(excess, hi)],
    )


def summarize(rows, slow_ms):
    result = {}
    for path in GETS:
        route = [r for r in rows if r["path"] == path]
        slow = [r for r in route if r["excess_ms"] >= slow_ms]
        total = sum(r["excess_ms"] for r in slow)
        share = (
            [sum(r["excess_in_decode_ms"][i] for r in slow) / total for i in (0, 1)]
            if total
            else None
        )
        verdict = "inconclusive"
        if share and len({r["round"] for r in slow}) >= 3:
            if share[0] > 0.5:
                verdict = "majority_overlaps_decode"
            elif share[1] < 0.5:
                verdict = "majority_outside_decode"
        offsets = {}
        for offset in sorted({r["offset_ms"] for r in route}):
            group = [r for r in route if r["offset_ms"] == offset]
            offsets[offset] = {
                k: statistics.median(r[k] for r in group)
                for k in (
                    "actual_offset_ms",
                    "rtt_ms",
                    "idle_ms",
                    "excess_ms",
                    "asgi_to_handler_ms",
                    "handler_ms",
                    "decode_ms",
                )
            }
            for k in ("send_to_asgi_ms", "decode_overlap_ms", "pre_handler_decode_ms"):
                offsets[offset][k] = [
                    statistics.median(r[k][i] for r in group) for i in (0, 1)
                ]
            if "control_rtt_ms" in group[0]:
                offsets[offset]["control_rtt_ms"] = statistics.median(
                    r["control_rtt_ms"] for r in group
                )
                offsets[offset]["control_idle_ms"] = statistics.median(
                    r["control_idle_ms"] for r in group
                )
        result[path] = dict(
            verdict=verdict,
            slow_samples=len(slow),
            samples=len(route),
            decode_share_bounds=share,
            median_by_offset_ms=offsets,
        )
    return result


async def run(args, emit):
    body = args.body.read_bytes()
    ids = json.loads(body).get("tokens")
    if not isinstance(ids, list) or not ids or not all(type(i) is int for i in ids):
        raise ValueError("--body must contain a nonempty flat integer tokens list")
    emit(
        "config",
        base_url=args.base_url,
        ids=len(ids),
        body_sha256=hashlib.sha256(body).hexdigest(),
        rounds=args.rounds,
        offsets_ms=args.offsets_ms,
        slow_ms=args.slow_ms,
        seed=args.seed,
    )
    headers = {HEADER: "1", "content-type": "application/json"}
    if key := os.environ.get("SGLANG_API_KEY"):
        headers["authorization"] = f"Bearer {key}"
    groups = defaultdict(list)
    async with AsyncExitStack() as stack:
        for _ in range(args.connections):
            c = await stack.enter_async_context(
                httpx.AsyncClient(
                    base_url=args.base_url,
                    headers=headers,
                    http2=False,
                    trust_env=False,
                    timeout=args.timeout,
                    limits=httpx.Limits(
                        max_connections=1,
                        max_keepalive_connections=1,
                        keepalive_expiry=None,
                    ),
                )
            )
            event = await request(c, GETS[0])
            identity = event["worker"]
            groups[identity[:2]].append((c, identity))
            ranked = sorted(groups.values(), key=len, reverse=True)
            if len(ranked[0]) >= 3 and (
                args.single_worker or (len(ranked) > 1 and len(ranked[1]) >= 2)
            ):
                break
        else:
            raise RuntimeError(
                f"Insufficient pinned connections: {[len(v) for v in groups.values()]}; increase --connections"
            )
        tool, *same = ranked[0][:3]
        control = [] if args.single_worker else ranked[1][:2]
        probes = list(zip(same, GETS)) + list(zip(control, GETS))
        emit(
            "pinned",
            long_worker=tool[1],
            probe_connections=[(v[1], p) for v, p in probes],
        )

        async def get_pair():
            return await asyncio.gather(
                *(request(c, path, pinned=pin) for (c, pin), path in probes)
            )

        warm = await request(tool[0], POST, body, tool[1])
        expected = warm["response_sha256"]
        emit("warmup", event=warm)
        rows = []
        rng = random.Random(args.seed)
        for rep in range(1, args.rounds + 1):
            offsets = args.offsets_ms.copy()
            rng.shuffle(offsets)
            for offset in offsets:
                before = await get_pair()
                # One long request per offset, with independent same/other PID sockets.
                origin = time.perf_counter_ns()

                async def delayed():
                    await asyncio.sleep(
                        max(0, (origin + offset * MS - time.perf_counter_ns()) / 1e9)
                    )
                    return await get_pair()

                long, measured = await asyncio.gather(
                    request(tool[0], POST, body, tool[1]), delayed()
                )
                after = await get_pair()
                stages = {
                    f"{a}_to_{b}": (long["ns"][b] - long["ns"][a]) / MS
                    for a, b in zip(LONG_MARKS, LONG_MARKS[1:])
                }
                emit(
                    "trial",
                    round=rep,
                    offset_ms=offset,
                    long=long,
                    stages_ms=stages,
                    measured=measured,
                    idle_before=before,
                    idle_after=after,
                )
                if long["response_sha256"] != expected:
                    raise RuntimeError("Detokenize response differs from warmup")
                bounds = clock_bounds([long, *before[:2], *measured[:2], *after[:2]])
                for i in range(2):
                    row = analyze(long, measured[i], [before[i], after[i]], bounds)
                    row.update(round=rep, offset_ms=offset)
                    if control:
                        row["control_rtt_ms"] = rtt(measured[i + 2])
                        row["control_idle_ms"] = statistics.median(
                            [rtt(before[i + 2]), rtt(after[i + 2])]
                        )
                    emit("sample", **row)
                    rows.append(row)
        return summarize(rows, args.slow_ms)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", default="http://10.113.129.13:30000")
    parser.add_argument("--body", required=True, type=Path)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path(
            datetime.now(timezone.utc).strftime("detok-phase-%Y%m%dT%H%M%SZ.jsonl")
        ),
    )
    parser.add_argument(
        "--offsets-ms",
        type=float,
        nargs="+",
        default=[0, 25, 50, 75, 100, 125, 150, 200, 250],
    )
    parser.add_argument("--rounds", type=int, default=5)
    parser.add_argument(
        "--slow-ms", type=float, default=10, help="Excess over per-connection idle RTT"
    )
    parser.add_argument("--seed", type=int, default=1916)
    parser.add_argument("--timeout", type=float, default=30)
    parser.add_argument("--connections", type=int, default=64)
    parser.add_argument(
        "--single-worker", action="store_true", help="Omit the other-PID control"
    )
    args = parser.parse_args()
    if (
        args.rounds < 1
        or args.connections < 3
        or not math.isfinite(args.slow_ms)
        or args.slow_ms <= 0
        or not math.isfinite(args.timeout)
        or args.timeout <= 0
        or any(not math.isfinite(o) or o < 0 for o in args.offsets_ms)
        or len(set(args.offsets_ms)) != len(args.offsets_ms)
    ):
        parser.error("invalid rounds, connections, timeout, slow threshold or offsets")
    with args.output.open("x", encoding="utf-8") as f:

        def emit(kind, **data):
            print(json.dumps({"kind": kind, **data}), file=f, flush=True)

        try:
            summary = asyncio.run(run(args, emit))
        except Exception as exc:
            emit("error", message=str(exc))
            raise
        emit("summary", routes=summary)
        print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
