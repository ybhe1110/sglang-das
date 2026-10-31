"""Bounded P/D cache validation with exact input lengths and replayable seeds.

Run on the test cluster gateway; requires requests. Both endpoints receive the
same RID, bootstrap room and fresh bootstrap_pair_key. This sends real traffic
but injects no faults. Use --decode-dp-size to spread requests across Decode ranks.
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
     SGLANG_TEST_LINKER_LOAD_FAILURE_RANKS=all
   Use the layer-wise or page-wise loader: the read-plan path does not invoke
   this injection hook. Verify the deployed configuration before testing.
   RANKS selects the linker's attention TP rank, not necessarily the global
   scheduler rank: under TP8/CP8, RANKS=0 can select every CP rank. Use all for
   an explicit whole-group failure; verify rank mapping for asymmetric tests.
3. Replay the seed/length on that Prefill with --expect-abort
   --followup-length 4096. Both P and D must return exactly one abort terminal;
   P must identify an external KV load failure. A transport timeout, generic
   HTTP 5xx, successful generation or EOF without a terminal FAILS the test.
   The followup uses a new RID and different prompt and must generate normally.
4. After verifying the rank mapping, test an asymmetric failure separately.
   Also test 4K input and a 64K+ input whose
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
This direct P/D client does not emulate a gateway cancelling D when P aborts:
it deliberately requires each engine to produce a terminal. On failure it
cancels only its own RID on both sides and stops further rounds. Keep a missing
D terminal as a failed result; do not count the subsequent cleanup as a pass.
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
    if expect_abort is not None and aborted != expect_abort:
        raise RuntimeError(f"Unexpected terminal (expect_abort={expect_abort}): {last}")
    if not aborted and ("error" in last or reason["type"] not in ("length", "stop")):
        raise RuntimeError(f"Not a successful generation: {last}")
    if (
        aborted
        and required_message
        and required_message.lower() not in str(reason).lower()
    ):
        raise RuntimeError(f"Abort does not identify the expected cause: {reason}")
    return meta


def paired_payloads(payload, decode_rank):
    """Bind P/D to one attempt without relying on matching local counters."""
    shared = dict(payload, bootstrap_pair_key=secrets.token_hex(24))
    return (
        dict(shared, routed_dp_rank=0, disagg_prefill_dp_rank=0),
        dict(shared, routed_dp_rank=decode_rank, disagg_prefill_dp_rank=0),
    )


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
    outcomes = parser.add_mutually_exclusive_group()
    outcomes.add_argument("--expect-abort", action="store_true")
    outcomes.add_argument(
        "--allow-load-aborts",
        action="store_true",
        help="Mixed fault stress: allow load aborts and safe L1 fallback; require at least one abort.",
    )
    parser.add_argument("--decode-rank", type=int, default=0)
    parser.add_argument("--decode-dp-size", type=int, default=1)
    parser.add_argument("--followup-length", type=int, default=0)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if not (0 < args.length < args.context_length - args.output_tokens):
        parser.error("Input plus output must remain below the context limit")
    if min(args.concurrency, args.requests, args.output_tokens) <= 0:
        parser.error("Counts must be positive")
    if args.timeout <= 0 or args.settle_seconds < 0:
        parser.error("Timeout must be positive and settle time nonnegative")
    if args.decode_dp_size <= 0 or not 0 <= args.decode_rank < args.decode_dp_size:
        parser.error("Decode rank must be within decode-dp-size")
    if (
        args.followup_length < 0
        or args.followup_length >= args.context_length - args.output_tokens
    ):
        parser.error("Followup input plus output must remain below the context limit")
    if args.followup_length and not (args.expect_abort or args.allow_load_aborts):
        parser.error("--followup-length requires a fault case")
    import requests

    p = f"http://{args.prefill}:{args.port}"
    d = f"http://{args.decode}:{args.port}"
    lock = threading.Lock()
    failed = threading.Event()
    aborted_pairs = []
    args.out.parent.mkdir(parents=True, exist_ok=True)

    def one(index, *, followup=False):
        if failed.is_set():
            return False
        seed = args.seed + index * args.seed_step + (10000000 if followup else 0)
        length = args.followup_length if followup else args.length
        expect_abort = (
            False if followup else None if args.allow_load_aborts else args.expect_abort
        )
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
        decode_rank = (args.decode_rank + index) % args.decode_dp_size
        p_payload, d_payload = paired_payloads(payload, decode_rank)
        row = dict(
            rid=rid,
            seed=seed,
            length=length,
            prefill=p,
            decode=d,
            room=payload["bootstrap_room"],
            started=time.time(),
            ok=False,
            phase=(
                "followup"
                if followup
                else (
                    "mixed"
                    if expect_abort is None
                    else "abort" if expect_abort else "success"
                )
            ),
            decode_rank=decode_rank,
            verification_scope="http_terminals_and_health_only",
        )
        try:
            with concurrent.futures.ThreadPoolExecutor(max_workers=2) as pool:
                futures = {
                    "p": pool.submit(
                        request,
                        p,
                        p_payload,
                        args.timeout,
                        expect_abort,
                        "external KV" if expect_abort is not False else None,
                    ),
                    "d": pool.submit(request, d, d_payload, args.timeout, expect_abort),
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
            p_abort = row["p"]["meta"]["finish_reason"]["type"] == "abort"
            d_abort = row["d"]["meta"]["finish_reason"]["type"] == "abort"
            if p_abort != d_abort:
                raise RuntimeError("P/D disagree on the request terminal state")
            row["aborted"] = p_abort
            if not p_abort:
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
            if row.get("ok") and row.get("aborted"):
                aborted_pairs.append(rid)
            with args.out.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(row) + "\n")
            print(json.dumps(row), flush=True)
        return row["ok"]

    with concurrent.futures.ThreadPoolExecutor(max_workers=args.concurrency) as pool:
        results = list(pool.map(one, range(args.requests)))
    if all(results) and args.followup_length:
        results.append(one(0, followup=True))
    if args.allow_load_aborts and not aborted_pairs:
        print(
            "FAIL: no validated load abort observed; failure coverage is unproven",
            flush=True,
        )
        raise SystemExit(1)
    raise SystemExit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
