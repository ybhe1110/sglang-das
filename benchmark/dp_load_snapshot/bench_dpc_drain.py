"""Benchmark the DPC snapshot drain and its serving-side effects.

This benchmark has three independent modes:

``control``
    Measures the CPU cost of the 20 ms DPC drain check and the number of
    owner polls it performs during an idle event loop.
``dispatch``
    Measures the local DPC refresh-plus-routing path without a GPU or network
    dependency.  It is useful for detecting dispatch overhead regressions.
``transport``
    Sends real snapshots through the ZMQ writer and owner reader and measures
    source-timestamp to SHM visibility latency.  ``--drain-mode dispatch``
    emulates the pre-fix behavior, while ``periodic`` exercises the fix.
``decode``
    Sends streaming OpenAI-compatible completion requests to a running SGLang
    server and reports aggregate output tok/s, TTFT, and inter-token timing.
    Run it against the same server configuration before and after the change.

The benchmark intentionally does not launch a model server.  Decode results
must be collected on the target hardware with the same model, prompts, and
parallelism configuration.
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import os
import statistics
import sys
import tempfile
import time
import urllib.error
import urllib.request
from types import SimpleNamespace
from typing import Any, Iterable
from unittest.mock import patch

_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_PYTHON_ROOT = os.path.join(_REPO_ROOT, "python")
if _PYTHON_ROOT not in sys.path:
    sys.path.insert(0, _PYTHON_ROOT)


def _percentile(values: Iterable[float], percentile: float) -> float:
    values = sorted(values)
    if not values:
        return float("nan")
    index = min(len(values) - 1, int(round((percentile / 100) * (len(values) - 1))))
    return values[index]


def _summary(values: list[float], unit: str) -> dict[str, Any]:
    return {
        "unit": unit,
        "count": len(values),
        "mean": statistics.mean(values) if values else float("nan"),
        "p50": _percentile(values, 50),
        "p95": _percentile(values, 95),
        "p99": _percentile(values, 99),
        "max": max(values) if values else float("nan"),
    }


class _NoopPollReader:
    def __init__(self):
        self.poll_count = 0

    def poll(self):
        self.poll_count += 1


class _ReadAllReader:
    def read_all(self):
        return []


class _Worker:
    def __init__(self):
        self.send_count = 0

    def send(self, _payload, flags=0):
        del flags
        self.send_count += 1


def _make_dpc_for_drain():
    from sglang.srt.managers.data_parallel_controller import DataParallelController

    controller = DataParallelController.__new__(DataParallelController)
    controller.load_snapshot_reader = _NoopPollReader()
    controller._last_load_snapshot_drain_time = 0.0
    return controller


def run_control(duration_s: float) -> dict[str, Any]:
    """Measure the idle-loop CPU cost of the periodic drain check."""
    controller = _make_dpc_for_drain()

    def spin(callback=None):
        start = time.perf_counter()
        deadline = start + duration_s
        iterations = 0
        while time.perf_counter() < deadline:
            if callback is not None:
                callback()
            iterations += 1
        return iterations

    baseline_cpu_start = time.process_time()
    baseline_iterations = spin()
    baseline_cpu = time.process_time() - baseline_cpu_start

    drain_cpu_start = time.process_time()
    drain_iterations = spin(controller._drain_load_snapshot_if_due)
    drain_cpu = time.process_time() - drain_cpu_start

    return {
        "mode": "control",
        "duration_s": duration_s,
        "baseline_iterations": baseline_iterations,
        "drain_iterations": drain_iterations,
        "polls": controller.load_snapshot_reader.poll_count,
        "baseline_cpu_pct": baseline_cpu / duration_s * 100,
        "drain_cpu_pct": drain_cpu / duration_s * 100,
        "incremental_cpu_ms": max(0.0, drain_cpu - baseline_cpu) * 1000,
        "incremental_cpu_ms_per_s": max(0.0, drain_cpu - baseline_cpu)
        / duration_s
        * 1000,
    }


def run_dispatch(samples: int, dp_size: int, input_tokens: int) -> dict[str, Any]:
    """Measure DPC refresh plus total-token routing with local fakes."""
    from sglang.srt.managers.data_parallel_controller import (
        DPBudget,
        DataParallelController,
    )

    controller = DataParallelController.__new__(DataParallelController)
    controller.load_snapshot_reader = _ReadAllReader()
    controller._last_refresh_time = 0.0
    controller.dp_budget = DPBudget(dp_size)
    controller.workers = [_Worker() for _ in range(dp_size)]
    controller.status = [True] * dp_size
    controller._active_workers = list(range(dp_size))

    latencies_us = []
    # Keep this benchmark focused on DPC refresh and rank selection.  IPC
    # serialization is measured separately by the serving/transport modes.
    with patch("sglang.srt.managers.data_parallel_controller.sock_send"):
        for _ in range(samples):
            controller._last_refresh_time = 0.0
            req = SimpleNamespace(
                routed_dp_rank=None,
                input_ids=[0] * input_tokens,
                dp_budget_ticket=None,
            )
            start = time.perf_counter()
            controller.refresh_load_budget()
            controller.total_tokens_scheduler(req)
            latencies_us.append((time.perf_counter() - start) * 1e6)

    return {
        "mode": "dispatch",
        "dp_size": dp_size,
        "input_tokens": input_tokens,
        "latency": _summary(latencies_us, "us"),
    }


def _ipc_endpoint() -> str:
    fd, path = tempfile.mkstemp(prefix="sglang_dpc_bench_", suffix=".sock")
    os.close(fd)
    os.unlink(path)
    return f"ipc://{path}"


def run_transport(
    samples: int,
    dp_size: int,
    drain_mode: str,
    drain_interval_s: float,
    timeout_s: float,
) -> dict[str, Any]:
    """Measure ZMQ source timestamp to SHM visibility latency."""
    from sglang.srt.managers.load_snapshot import (
        LoadSnapshot,
        ZmqLoadSnapshotWriter,
        ZmqShmLoadSnapshotReader,
    )

    endpoint = _ipc_endpoint()
    shm_fd, shm_path = tempfile.mkstemp(prefix="sglang_dpc_bench_shm_")
    os.close(shm_fd)
    os.unlink(shm_path)
    reader = ZmqShmLoadSnapshotReader(endpoint, shm_path, dp_size)
    writers = [
        ZmqLoadSnapshotWriter(endpoint, dp_size, dp_rank) for dp_rank in range(dp_size)
    ]
    latencies_ms = []

    try:
        # Let the PUSH socket establish its connection before collecting data.
        for _ in range(20):
            for writer in writers:
                writer.write(LoadSnapshot(dp_rank=writer.dp_rank, timestamp=-1.0))
            reader.poll()
            if all(
                reader._shm_reader.read(rank) is not None for rank in range(dp_size)
            ):
                break
            time.sleep(0.01)

        for _ in range(samples):
            source_timestamp = time.time()
            for writer in writers:
                writer.write(
                    LoadSnapshot(
                        dp_rank=writer.dp_rank,
                        timestamp=source_timestamp,
                        num_running_reqs=1,
                    )
                )
            last_poll = time.perf_counter()
            deadline = last_poll + timeout_s
            visible_ranks = []
            while time.perf_counter() < deadline:
                now = time.perf_counter()
                if now - last_poll >= drain_interval_s:
                    reader.poll()
                    last_poll = now
                visible_ranks = [
                    rank
                    for rank in range(dp_size)
                    if (
                        (visible := reader._shm_reader.read(rank)) is not None
                        and visible.timestamp == source_timestamp
                    )
                ]
                if len(visible_ranks) == dp_size:
                    latencies_ms.append((time.time() - source_timestamp) * 1000)
                    break
                time.sleep(0.0005)
            else:
                raise TimeoutError(
                    f"snapshot was not visible in SHM within {timeout_s}s: "
                    f"{source_timestamp=} {visible_ranks=}"
                )
    finally:
        reader.close()
        for writer in writers:
            writer.close()
        if os.path.exists(shm_path):
            os.unlink(shm_path)

    return {
        "mode": "transport",
        "drain_mode": drain_mode,
        "samples": samples,
        "dp_size": dp_size,
        "drain_interval_s": drain_interval_s,
        "latency": _summary(latencies_ms, "ms"),
    }


def _get_model(base_url: str, timeout_s: float) -> str:
    request = urllib.request.Request(f"{base_url}/v1/models")
    with urllib.request.urlopen(request, timeout=timeout_s) as response:
        models = json.loads(response.read()).get("data", [])
    if not models:
        raise RuntimeError("/v1/models returned no models; pass --model explicitly")
    return models[0]["id"]


def _decode_request(
    base_url: str,
    model: str,
    prompt: str,
    output_tokens: int,
    timeout_s: float,
) -> dict[str, Any]:
    payload = json.dumps(
        {
            "model": model,
            "prompt": prompt,
            "max_tokens": output_tokens,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
    ).encode()
    request = urllib.request.Request(
        f"{base_url}/v1/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    started = time.perf_counter()
    first_token = None
    chunk_times = []
    completion_tokens = None
    try:
        with urllib.request.urlopen(request, timeout=timeout_s) as response:
            for raw_line in response:
                line = raw_line.decode("utf-8").strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                chunk = json.loads(data)
                usage = chunk.get("usage") or {}
                if usage.get("completion_tokens") is not None:
                    completion_tokens = usage["completion_tokens"]
                choices = chunk.get("choices") or []
                if choices and choices[0].get("text"):
                    now = time.perf_counter()
                    first_token = first_token or now
                    chunk_times.append(now)
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(
            f"Decode request failed with HTTP {exc.code}: {body}"
        ) from exc

    finished = time.perf_counter()
    if completion_tokens is None:
        completion_tokens = len(chunk_times)
        token_count_source = "chunks_estimate"
    else:
        token_count_source = "usage"
    decode_time = (finished - (first_token or started)) or 1e-9
    return {
        "latency_s": finished - started,
        "ttft_ms": ((first_token or finished) - started) * 1000,
        "completion_tokens": completion_tokens,
        "decode_tok_s": completion_tokens / decode_time,
        "itl_ms": [
            (right - left) * 1000 for left, right in zip(chunk_times, chunk_times[1:])
        ],
        "token_count_source": token_count_source,
    }


def run_decode(
    host: str,
    port: int,
    model: str | None,
    prompt: str,
    output_tokens: int,
    requests: int,
    concurrency: int,
    timeout_s: float,
) -> dict[str, Any]:
    base_url = f"http://{host}:{port}"
    model = model or _get_model(base_url, timeout_s)
    started = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as executor:
        futures = [
            executor.submit(
                _decode_request,
                base_url,
                model,
                prompt,
                output_tokens,
                timeout_s,
            )
            for _ in range(requests)
        ]
        results = [future.result() for future in futures]
    elapsed = time.perf_counter() - started
    output_tokens_total = sum(result["completion_tokens"] for result in results)
    itl = [value for result in results for value in result["itl_ms"]]
    return {
        "mode": "decode",
        "model": model,
        "requests": requests,
        "concurrency": concurrency,
        "output_tokens_target": output_tokens,
        "output_tokens_total": output_tokens_total,
        "aggregate_output_tok_s": output_tokens_total / elapsed,
        "ttft": _summary([result["ttft_ms"] for result in results], "ms"),
        "per_request_decode_tok_s": _summary(
            [result["decode_tok_s"] for result in results], "tok/s"
        ),
        "itl": _summary(itl, "ms"),
        "token_count_source": sorted(
            {result["token_count_source"] for result in results}
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--mode", choices=("control", "dispatch", "transport", "decode"), required=True
    )
    parser.add_argument("--duration", type=float, default=10.0)
    parser.add_argument("--samples", type=int, default=100)
    parser.add_argument("--dp-size", type=int, default=16)
    parser.add_argument("--input-tokens", type=int, default=128)
    parser.add_argument(
        "--drain-mode", choices=("periodic", "dispatch"), default="periodic"
    )
    parser.add_argument("--drain-interval", type=float, default=0.02)
    parser.add_argument("--dispatch-interval", type=float, default=1.0)
    parser.add_argument("--timeout", type=float, default=30.0)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--model")
    parser.add_argument(
        "--prompt", default="Explain the scheduling tradeoff in one paragraph."
    )
    parser.add_argument("--output-tokens", type=int, default=128)
    parser.add_argument("--requests", type=int, default=32)
    parser.add_argument("--concurrency", type=int, default=16)
    parser.add_argument("--output-json")
    args = parser.parse_args()

    if args.mode == "control":
        result = run_control(args.duration)
    elif args.mode == "dispatch":
        result = run_dispatch(args.samples, args.dp_size, args.input_tokens)
    elif args.mode == "transport":
        interval = (
            args.drain_interval
            if args.drain_mode == "periodic"
            else args.dispatch_interval
        )
        result = run_transport(
            args.samples,
            args.dp_size,
            args.drain_mode,
            interval,
            args.timeout,
        )
    else:
        result = run_decode(
            args.host,
            args.port,
            args.model,
            args.prompt,
            args.output_tokens,
            args.requests,
            args.concurrency,
            args.timeout,
        )

    encoded = json.dumps(result, ensure_ascii=False, indent=2)
    print(encoded)
    if args.output_json:
        with open(args.output_json, "w", encoding="utf-8") as output_file:
            output_file.write(encoded)
            output_file.write("\n")


if __name__ == "__main__":
    main()
