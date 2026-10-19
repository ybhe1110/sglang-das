#!/usr/bin/env python3
"""High-load long-context P/D smoke test.

This is intentionally manual: it talks to an already running Decode HTTP
endpoint, which routes requests to Prefill in a real P/D deployment.  It does
not restart Pods or change Mooncake state.

Example:
  PYTHONPATH=python python test/manual/disaggregation/long_context_mooncake_stress.py \
    --url http://10.113.44.22:30000 \
    --requests 24 --concurrency 4 --input-tokens 1048576 \
    --log-file /data1/log/sglang_server.log
"""

from __future__ import annotations

import argparse
import concurrent.futures
import json
import re
import sys
import time
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


def http_json(url: str, payload: dict | None, timeout_s: float):
    data = None if payload is None else json.dumps(payload).encode("utf-8")
    request = Request(
        url,
        data=data,
        headers={"Content-Type": "application/json"} if data is not None else {},
        method="POST" if data is not None else "GET",
    )
    started = time.monotonic()
    try:
        with urlopen(request, timeout=timeout_s) as response:
            body = response.read()
            return response.status, time.monotonic() - started, body[:512]
    except HTTPError as exc:
        return exc.code, time.monotonic() - started, exc.read(512)
    except (TimeoutError, URLError, OSError) as exc:
        return None, time.monotonic() - started, repr(exc).encode()


def make_prompt(input_tokens: int) -> str:
    # ``tok`` is intentionally simple and stable; actual tokenizer counts can
    # differ by model, so the script reports the requested approximation.
    return "tok " * input_tokens


def run_one(url: str, prompt: str, output_tokens: int, timeout_s: float, index: int):
    payload = {
        "text": prompt,
        "sampling_params": {
            "temperature": 0.0,
            "max_new_tokens": output_tokens,
        },
    }
    status, elapsed, body = http_json(url, payload, timeout_s)
    return {
        "index": index,
        "status": status,
        "elapsed_s": round(elapsed, 3),
        "ok": status == 200,
        "body_prefix": body.decode("utf-8", errors="replace"),
    }


def scan_logs(paths: list[str]) -> dict:
    patterns = {
        "collective_timeout": re.compile(r"collective.*(?:timeout|failed)|DisaggregationCollectiveError", re.I),
        "chunk_timeout": re.compile(r"SourceTransferTimeout|Synchronous transfer did not complete|MC_TRANSFER_TIMEOUT", re.I),
        "source_drain": re.compile(r"source drain|source-drain|retaining buffers|retaining resources", re.I),
        "session_growth": re.compile(
            r"sessions=\[|peer session|session .*destroy|peer sessions tracked: count=(\d+)",
            re.I,
        ),
        "restart_or_watchdog": re.compile(r"watchdog|SIGQUIT|SIGKILL|OOM|liveness", re.I),
    }
    counts = {key: 0 for key in patterns}
    for name in paths:
        path = Path(name)
        if not path.exists():
            continue
        for line in path.read_text(errors="replace").splitlines():
            for key, pattern in patterns.items():
                counts[key] += bool(pattern.search(line))
    return counts


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--url", required=True, help="Decode HTTP URL, e.g. http://D_IP:30000/generate")
    parser.add_argument("--requests", type=int, default=16)
    parser.add_argument("--concurrency", type=int, default=4)
    parser.add_argument("--input-tokens", type=int, default=1_048_576)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--timeout", type=float, default=900.0)
    parser.add_argument("--health-url", default=None)
    parser.add_argument("--log-file", action="append", default=[])
    parser.add_argument(
        "--expected-peer-sessions",
        type=int,
        default=None,
        help="Optional upper bound for 'peer sessions tracked: count=N' log entries",
    )
    parser.add_argument("--result-file", default=None)
    args = parser.parse_args()

    if args.input_tokens < 1 or args.requests < 1 or args.concurrency < 1:
        parser.error("input-tokens, requests and concurrency must be positive")
    health_url = args.health_url or args.url.rsplit("/", 1)[0] + "/health"
    status, elapsed, body = http_json(health_url, None, 30.0)
    if status != 200:
        print(f"health check failed: status={status} body={body!r}", file=sys.stderr)
        return 2

    prompt = make_prompt(args.input_tokens)
    started = time.monotonic()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        futures = [
            pool.submit(
                run_one,
                args.url,
                prompt,
                args.output_tokens,
                args.timeout,
                index,
            )
            for index in range(args.requests)
        ]
        results = [future.result() for future in futures]
    elapsed_total = time.monotonic() - started

    final_status, _, final_body = http_json(health_url, None, 30.0)
    report = {
        "input_tokens_requested": args.input_tokens,
        "chunk_size_assumed": 32768,
        "chunk_count_assumed": (args.input_tokens + 32767) // 32768,
        "requests": args.requests,
        "concurrency": args.concurrency,
        "elapsed_total_s": round(elapsed_total, 3),
        "successes": sum(item["ok"] for item in results),
        "failures": sum(not item["ok"] for item in results),
        "final_health_status": final_status,
        "final_health_body": final_body.decode("utf-8", errors="replace"),
        "results": results,
        "log_signals": scan_logs(args.log_file),
    }
    session_counts = []
    session_pattern = re.compile(r"peer sessions tracked: count=(\d+)", re.I)
    for name in args.log_file:
        path = Path(name)
        if path.exists():
            session_counts.extend(
                int(value)
                for value in session_pattern.findall(path.read_text(errors="replace"))
            )
    report["peer_session_counts"] = session_counts
    report["peer_session_max"] = max(session_counts, default=None)
    if args.expected_peer_sessions is not None and session_counts:
        report["peer_session_bound_ok"] = (
            report["peer_session_max"] <= args.expected_peer_sessions
        )
    else:
        report["peer_session_bound_ok"] = None
    print(json.dumps(report, ensure_ascii=False, indent=2))
    if args.result_file:
        Path(args.result_file).write_text(
            json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    session_ok = report["peer_session_bound_ok"] is not False
    return 0 if report["failures"] == 0 and final_status == 200 and session_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
