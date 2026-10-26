"""Bounded P/D cache validation with exact input lengths and replayable seeds.

Run on the test cluster gateway; requires requests. Both endpoints receive the
same RID and bootstrap room. This sends real traffic but injects no faults.
Example:
  python paired_l3_validation.py --prefill 10.113.129.9 \
    --decode 10.113.129.7 --length 65536 --concurrency 2 --requests 4 \
    --seed 888 --out results.jsonl
For 1M context use --length 1048320, leaving room for output. Repeat a seed on
another Prefill to exercise L3; confirm lookup/load and pool logs separately.
Within a run each request uses seed + index; use --seed-step 0 for identical
prompts. Health checks catch immediate idle-pool crashes, but Pod restart counts,
all-rank logs and native resource snapshots must also be checked separately.
"""

import argparse
import concurrent.futures
import json
import random
import secrets
import threading
import time
from pathlib import Path

import requests


def request(endpoint, payload, timeout):
    started = time.monotonic()
    first = None
    last = {}
    with requests.post(
        endpoint + "/generate", json=payload, stream=True, timeout=(15, timeout)
    ) as response:
        response.raise_for_status()
        for line in response.iter_lines():
            if not line.startswith(b"data:"):
                continue
            data = line[5:].strip()
            if data == b"[DONE]":
                break
            last = json.loads(data)
            meta = last.get("meta_info", {})
            reason = meta.get("finish_reason") or {}
            if "error" in last or reason.get("type") == "abort":
                raise RuntimeError(str(last)[:2000])
            if first is None and (last.get("text") or last.get("output_ids")):
                first = time.monotonic() - started
    if not last.get("meta_info", {}).get("finish_reason"):
        raise RuntimeError("Stream ended without a terminal result")
    return {
        "elapsed": time.monotonic() - started,
        "ttft": first,
        "meta": last["meta_info"],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--prefill", required=True)
    parser.add_argument("--decode", required=True)
    parser.add_argument("--port", type=int, default=30000)
    parser.add_argument("--bootstrap-port", type=int, default=8998)
    parser.add_argument("--length", type=int, default=4096)
    parser.add_argument("--output-tokens", type=int, default=8)
    parser.add_argument("--context-length", type=int, default=1048576)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--requests", type=int, default=1)
    parser.add_argument("--seed", type=int, default=888)
    parser.add_argument("--seed-step", type=int, default=1)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--settle-seconds", type=float, default=5)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (0 < args.length < args.context_length - args.output_tokens):
        parser.error("Input plus output must remain below the context limit")
    if min(args.concurrency, args.requests, args.output_tokens) <= 0:
        parser.error("Counts must be positive")
    p = f"http://{args.prefill}:{args.port}"
    d = f"http://{args.decode}:{args.port}"
    lock = threading.Lock()
    failed = threading.Event()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def one(index):
        if failed.is_set():
            return False
        seed = args.seed + index * args.seed_step
        rng = random.Random(seed)
        rid = "l3-validation-" + secrets.token_hex(8)
        payload = {
            "rid": rid,
            "input_ids": [rng.randrange(1000, 20000) for _ in range(args.length)],
            "sampling_params": {
                "temperature": 0,
                "max_new_tokens": args.output_tokens,
                "ignore_eos": True,
            },
            "bootstrap_host": args.prefill,
            "bootstrap_port": args.bootstrap_port,
            "bootstrap_room": secrets.randbits(63),
            "stream": True,
        }
        row = dict(
            rid=rid,
            seed=seed,
            length=args.length,
            prefill=p,
            decode=d,
            room=payload["bootstrap_room"],
            started=time.time(),
            ok=False,
        )
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                fp = pool.submit(request, p, payload, args.timeout)
                fd = pool.submit(request, d, payload, args.timeout)
                row["p"], row["d"] = fp.result(), fd.result()
            assert row["d"]["meta"]["prompt_tokens"] == args.length
            assert row["d"]["meta"]["completion_tokens"] == args.output_tokens
            # An HTTP 200 can precede a fatal idle-pool check by several seconds.
            time.sleep(args.settle_seconds)
            for endpoint in (p, d):
                requests.get(endpoint + "/health", timeout=30).raise_for_status()
            row["ok"] = True
        except Exception as error:
            failed.set()
            row["error"] = repr(error)
            for endpoint in (p, d):
                try:
                    requests.post(
                        endpoint + "/abort_request", json={"rid": rid}, timeout=5
                    )
                except requests.RequestException:
                    pass
        with lock:
            with args.out.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        return row["ok"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(one, range(args.requests)))
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
