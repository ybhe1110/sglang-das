"""Mixed HTTP/SSE interference probe. Requires httpx; no model/GPU dependency.

Measures client-observed events, not GPU token latency. It does not measure
server queue depth or prove which process handled a request. Add server metrics
and PID/rid tracing separately. Never use untrusted payloads on a shared server.
"""
import argparse
import asyncio
import json
import math
import os
import time
from pathlib import Path

import httpx


def read_body(path):
    if not path:
        return None
    raw = Path(path).read_bytes()
    json.loads(raw.decode("utf-8-sig"))
    return raw.decode("utf-8-sig").encode("utf-8")


def percentile(values, p):
    if not values:
        return None
    values = sorted(values)
    return values[max(0, math.ceil(len(values) * p) - 1)]


async def run(args):
    tool_body = read_body(args.tool_body)
    short_body = read_body(args.short_body)
    stream_body = read_body(args.stream_body)
    if stream_body is not None:
        body = json.loads(stream_body)
        if body.get("stream") is not True:
            raise ValueError("stream body must contain stream: true")
    if args.tool_count and tool_body is None:
        raise ValueError("--tool-body is required when --tool-count > 0")
    headers = {"Content-Type": "application/json"}
    key = os.environ.get(args.api_key_env)
    if key:
        headers["Authorization"] = "Bearer " + key
    paths = args.probe_path or ["/v1/models", "/v1/loads"]
    # No body/header content is copied into the output log.
    events = []
    errors = []
    start = time.perf_counter()
    trigger = asyncio.Event()
    pending = set()
    probe_inflight = 0

    def record(kind, **data):
        if len(events) >= args.max_records:
            raise RuntimeError("max-records reached; shorten the run")
        events.append({"kind": kind, "t_ms": (time.perf_counter()-start)*1000, **data})

    def own(coro):
        task = asyncio.create_task(coro)
        pending.add(task)
        def done(t):
            pending.discard(t)
            if not t.cancelled() and t.exception() is not None:
                errors.append(repr(t.exception()))
        task.add_done_callback(done)
        return task

    common = dict(base_url=args.base_url.rstrip("/"), headers=headers,
                  timeout=httpx.Timeout(args.timeout), trust_env=False)
    # Three independent connection pools prevent tool/SSE connections taking
    # the lightweight probe client's connection slots.
    async with (
        httpx.AsyncClient(**common, limits=httpx.Limits(
            max_connections=args.probe_capacity,
            max_keepalive_connections=args.probe_capacity)) as probes,
        httpx.AsyncClient(**common, limits=httpx.Limits(
            max_connections=args.tool_concurrency,
            max_keepalive_connections=args.tool_concurrency)) as tools,
        httpx.AsyncClient(**common, limits=httpx.Limits(
            max_connections=1, max_keepalive_connections=1)) as streams,
    ):
        async def one_probe(method, path, body, due):
            nonlocal probe_inflight
            actual = time.perf_counter()
            result = {"path": path, "scheduled_ms": (due-start)*1000,
                      "dispatch_lag_ms": max(0, (actual-due)*1000)}
            try:
                response = await probes.request(method, path, content=body)
                result.update(status=response.status_code,
                              response_bytes=len(response.content),
                              worker_pid=response.headers.get("x-worker-pid"),
                              request_id=response.headers.get("x-request-id"))
            except asyncio.CancelledError:
                result["error"] = "CancelledAtTestEnd"
                raise
            except Exception as exc:
                result["error"] = type(exc).__name__
            finally:
                probe_inflight -= 1
                record("probe", elapsed_ms=(time.perf_counter()-actual)*1000, **result)

        async def probe_clock():
            nonlocal probe_inflight
            routes = [("GET", p, None) for p in paths]
            if short_body is not None:
                routes.append(("POST", "/v1/tokenize", short_body))
            interval = args.interval_ms / 1000
            tick = 0
            while True:
                due = start + tick * interval
                if due >= start + args.duration:
                    break
                await asyncio.sleep(max(0, due-time.perf_counter()))
                # Keep the schedule open-loop but bound client outstanding work.
                # Missed ticks are reported and not replayed as a catch-up burst.
                now = time.perf_counter()
                if now-due >= interval:
                    missed = int((now-due)//interval)
                    record("client_missed_ticks", count=missed)
                    tick += missed
                    due = start + tick * interval
                if due >= start + args.duration:
                    break
                for method, path, body in routes:
                    if probe_inflight >= args.probe_capacity:
                        record("probe_skipped", path=path, reason="client_capacity")
                        continue
                    probe_inflight += 1
                    own(one_probe(method, path, body, due))
                tick += 1

        async def consume_stream():
            count, previous = 0, None
            data_lines = []
            try:
                async with streams.stream("POST", args.stream_path,
                                          content=stream_body) as response:
                    record("stream_headers", status=response.status_code,
                           content_type=response.headers.get("content-type"))
                    response.raise_for_status()
                    if "text/event-stream" not in response.headers.get("content-type", ""):
                        raise ValueError("response is not an SSE stream")
                    async for line in response.aiter_lines():
                        if line.startswith("data:"):
                            data_lines.append(line[5:].lstrip())
                        elif line == "" and data_lines:
                            data = "\n".join(data_lines)
                            data_lines.clear()
                            if data.strip() == "[DONE]":
                                record("stream_done", data_events=count)
                                return
                            now = time.perf_counter()
                            count += 1
                            record("sse_data", sequence=count,
                                   gap_ms=None if previous is None else (now-previous)*1000)
                            previous = now
                            if count == args.inject_after_events:
                                trigger.set()
                    record("stream_eof", data_events=count)
            except asyncio.CancelledError:
                record("stream_client_closed", reason="test_duration_end", data_events=count)
                raise
            except Exception as exc:
                record("stream_error", error=type(exc).__name__)

        async def inject():
            if stream_body is None:
                await asyncio.sleep(min(1.0, args.duration/4))
            else:
                try:
                    await asyncio.wait_for(trigger.wait(), min(20, args.duration*0.7))
                except asyncio.TimeoutError:
                    record("invalid_run", reason="stream_did_not_reach_injection_event")
                    return
            record("injection_start", tool_count=args.tool_count)
            indices = iter(range(args.tool_count))
            async def tool_worker():
                for index in indices:
                    began = time.perf_counter()
                    record("tool_start", index=index, path=args.tool_path)
                    try:
                        response = await tools.post(args.tool_path, content=tool_body)
                        record("tool_end", index=index, status=response.status_code,
                               elapsed_ms=(time.perf_counter()-began)*1000,
                               response_bytes=len(response.content))
                    except asyncio.CancelledError:
                        record("tool_client_closed", index=index)
                        raise
                    except Exception as exc:
                        record("tool_error", index=index, error=type(exc).__name__,
                               elapsed_ms=(time.perf_counter()-began)*1000)
            workers = [asyncio.create_task(tool_worker()) for _ in range(
                min(args.tool_concurrency, args.tool_count))]
            try:
                await asyncio.gather(*workers)
            finally:
                for task in workers:
                    if not task.done():
                        task.cancel()
                await asyncio.gather(*workers, return_exceptions=True)
            record("injection_end")

        # Warm up probe connections before measured workload.
        for path in paths:
            response = await probes.get(path)
            if not response.is_success:
                raise RuntimeError(f"Probe preflight failed: {path} -> {response.status_code}; use --probe-path")
        start = time.perf_counter()
        record("start", duration=args.duration, interval_ms=args.interval_ms,
               base_url=args.base_url, probe_paths=paths, tool_path=args.tool_path)
        clock_task = own(probe_clock())
        inject_task = own(inject())
        stream_task = own(consume_stream()) if stream_body is not None else None
        try:
            await asyncio.sleep(args.duration)
            record("measurement_window_end")
        finally:
            # This intentionally closes clients at the end. It does not prove
            # server-side cancellation. Observe server release metrics separately.
            for task in list(pending):
                task.cancel()
            await asyncio.gather(*list(pending), return_exceptions=True)

    summary = {"probes": {}, "client_task_errors": errors,
               "invalid_run": any(e["kind"] in ("invalid_run", "stream_error") for e in events),
               "tool_completed": sum(e["kind"] == "tool_end" for e in events),
               "tool_client_closed": sum(e["kind"] == "tool_client_closed" for e in events),
               "tool_errors": sum(e["kind"] == "tool_error" for e in events),
               "probe_skipped": sum(e["kind"] == "probe_skipped" for e in events)}
    for path in set(e["path"] for e in events if e["kind"] == "probe"):
        subset = [e for e in events if e["kind"] == "probe" and e["path"] == path]
        successful = [e["elapsed_ms"] for e in subset if 200 <= e.get("status", 0) < 300]
        summary["probes"][path] = {"count": len(subset), "successful": len(successful),
            "p50_ms": percentile(successful, .5), "p95_ms": percentile(successful, .95),
            "p99_ms": percentile(successful, .99),
            "max_dispatch_lag_ms": max(e["dispatch_lag_ms"] for e in subset)}
    gaps = [e["gap_ms"] for e in events if e["kind"] == "sse_data" and e["gap_ms"] is not None]
    summary["sse"] = {"gaps": len(gaps), "p99_ms": percentile(gaps,.99),
                      "max_ms": max(gaps, default=None)}
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("".join(json.dumps(e, ensure_ascii=False)+"\n" for e in events), encoding="utf-8")
    output.with_suffix(".summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 1 if errors or summary["invalid_run"] else 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-url", required=True)
    parser.add_argument("--tool-path", default="/v1/tokenize")
    parser.add_argument("--tool-body")
    parser.add_argument("--short-body")
    parser.add_argument("--stream-path", default="/v1/chat/completions")
    parser.add_argument("--stream-body")
    parser.add_argument("--inject-after-events", type=int, default=10)
    parser.add_argument("--tool-count", type=int, default=3)
    parser.add_argument("--tool-concurrency", type=int, default=3)
    parser.add_argument("--duration", type=float, default=20)
    parser.add_argument("--interval-ms", type=float, default=20)
    parser.add_argument("--timeout", type=float, default=60)
    parser.add_argument("--probe-capacity", type=int, default=80)
    parser.add_argument("--probe-path", action="append")
    parser.add_argument("--api-key-env", default="SGLANG_API_KEY")
    parser.add_argument("--max-records", type=int, default=100000)
    parser.add_argument("--output", default="mixed_probe.jsonl")
    args = parser.parse_args()
    if min(args.duration, args.interval_ms, args.timeout, args.tool_concurrency,
           args.probe_capacity, args.inject_after_events, args.max_records) <= 0 or args.tool_count < 0:
        parser.error("limits must be positive; tool-count may be zero")
    raise SystemExit(asyncio.run(run(args)))


if __name__ == "__main__":
    main()
