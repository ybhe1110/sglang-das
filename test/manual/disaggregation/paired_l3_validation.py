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

Load-failure case (TEST SERVICES ONLY):
1. Seed L3 on a healthy Prefill using a new --seed, --length 65536. Confirm the
   store/offload finished before replay. Keep that prompt/seed for recovery.
2. On a different Prefill with cold L1, set these launch environment variables
   for its ENTIRE TP/CP group, then wait for readiness:
     SGLANG_TEST_LINKER_LOAD_FAILURE_PROB=1
     SGLANG_TEST_LINKER_LOAD_FAILURE_RANKS=0
   This v4.1 hook exercises the layer-wise loader. With MLA rank-0 loading,
   only the source rank runs native reads; select rank 0 (or disable dedup for
   arbitrary per-rank injection). Verify the deployed configuration first.
3. Replay the seed/length on that Prefill with --expect-abort
   --followup-length 4096. Both P and D must return exactly one abort terminal;
   P must identify an external KV load failure. A transport timeout, generic
   HTTP 5xx, successful generation or EOF without a terminal FAILS the test.
   The followup uses a new RID and different prompt and must generate normally.
4. Also test RANKS=all (a separate rollout), 4K input, and a 64K+ input whose
   uncached suffix actually exceeds chunked_prefill_size. To force the latter,
   seed only 4K and replay the same seed with --length 65536. Confirm chunked
   execution in rank logs; a long input mostly served by L3 is not proof of it.
5. Remove the injection variables, roll the whole group and wait for readiness.
   Replay the failed seed/length without --expect-abort to verify L3 recovery.

The script NEVER modifies deployment settings or enables faults itself. Its
JSONL scope is HTTP terminals + health, NOT allocator/native verification.
Correlate the failure RID with L3 lookup and injected failure in fresh per-rank
logs. Require no unintended Pod restart, no stranded inflight requests, and
balanced FULL/SWA pools after loads/forwards/transfers drain. Collect QP/MR and
session baselines separately. A passing HTTP test alone proves none of these.
Wrap the entire command in `timeout 1800s` on Linux: requests' timeout is an
inactivity timeout, not an absolute wall-clock deadline for a streaming peer.
"""

import argparse
import concurrent.futures
import json
import random
import secrets
import threading
import time
from pathlib import Path


def validate_terminal(last, terminal_count, expect_abort=False, required_message=None):
    meta = last.get("meta_info", {})
    reason = meta.get("finish_reason") or {}
    if terminal_count != 1:
        raise RuntimeError(
            f"Expected exactly one terminal result, got {terminal_count}"
        )
    if not isinstance(reason, dict) or not reason.get("type"):
        raise RuntimeError("Stream ended without a structured terminal result")
    aborted = reason["type"] == "abort"
    if aborted != expect_abort:
        raise RuntimeError(f"Unexpected terminal (expect_abort={expect_abort}): {last}")
    if not expect_abort and (
        "error" in last or reason["type"] not in ("length", "stop")
    ):
        raise RuntimeError(f"Not a successful generation: {last}")
    if required_message and required_message.lower() not in str(reason).lower():
        raise RuntimeError(f"Abort does not identify the expected cause: {reason}")
    return meta


def request(endpoint, payload, timeout, expect_abort=False, required_message=None):
    import requests

    started = time.monotonic()
    first = None
    last = {}
    terminal_count = 0
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
            if reason:
                terminal_count += 1
            if "error" in last and not reason:
                raise RuntimeError(str(last)[:2000])
            if first is None and (last.get("text") or last.get("output_ids")):
                first = time.monotonic() - started
    meta = validate_terminal(last, terminal_count, expect_abort, required_message)
    return {
        "elapsed": time.monotonic() - started,
        "ttft": first,
        "meta": meta,
        "terminal_count": terminal_count,
    }


def main():
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
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
    parser.add_argument("--expect-abort", action="store_true")
    parser.add_argument("--followup-length", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (0 < args.length < args.context_length - args.output_tokens):
        parser.error("Input plus output must remain below the context limit")
    if min(args.concurrency, args.requests, args.output_tokens) <= 0:
        parser.error("Counts must be positive")
    if args.timeout <= 0 or args.settle_seconds < 0:
        parser.error("Timeout must be positive and settle time nonnegative")
    if (
        args.followup_length < 0
        or args.followup_length >= args.context_length - args.output_tokens
    ):
        parser.error("Followup input plus output must remain below the context limit")
    if args.followup_length and not args.expect_abort:
        parser.error("--followup-length is for --expect-abort cases")
    import requests

    p = f"http://{args.prefill}:{args.port}"
    d = f"http://{args.decode}:{args.port}"
    lock = threading.Lock()
    failed = threading.Event()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def one(index, *, followup=False):
        if failed.is_set():
            return False
        seed = args.seed + index * args.seed_step + (10000000 if followup else 0)
        length = args.followup_length if followup else args.length
        expect_abort = args.expect_abort and not followup
        rng = random.Random(seed)
        rid = "l3-validation-" + secrets.token_hex(8)
        payload = {
            "rid": rid,
            "input_ids": [rng.randrange(1000, 20000) for _ in range(length)],
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
            length=length,
            prefill=p,
            decode=d,
            room=payload["bootstrap_room"],
            started=time.time(),
            ok=False,
            phase="followup" if followup else "abort" if expect_abort else "success",
            verification_scope="http_terminals_and_health_only",
        )
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = {
                    "p": pool.submit(
                        request,
                        p,
                        payload,
                        args.timeout,
                        expect_abort,
                        "external KV" if expect_abort else None,
                    ),
                    "d": pool.submit(request, d, payload, args.timeout, expect_abort),
                }
                errors = {}
                for side, future in futures.items():
                    try:
                        row[side] = future.result()
                    except Exception as error:
                        errors[side] = repr(error)
                if errors:
                    row["endpoint_errors"] = errors
                    raise RuntimeError(f"P/D terminal validation failed: {errors}")
            if not expect_abort:
                assert row["d"]["meta"]["prompt_tokens"] == length
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
    if all(results) and args.followup_length:
        results.append(one(0, followup=True))
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
