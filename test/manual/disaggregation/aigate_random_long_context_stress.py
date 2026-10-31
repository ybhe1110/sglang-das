"""Bounded chat stress through aigate, with reproducible random prompt sizes.

The word vocabulary was calibrated against the deployed tokenizer. Requested
lengths remain targets: report usage.prompt_tokens, truncation and context-limit
rejections separately. A successful HTTP status without an SSE terminal is NOT
a successful request. No automatic retries, engine-direct inference or faults.
"""

import argparse
import concurrent.futures
import http.client
import json
import random
import signal
import socket
import threading
import time
import uuid
from array import array
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from urllib.parse import urlsplit

WORDS = "apple river mountain cloud copper garden window forest paper orange table lamp stone green blue train summer winter".split()


def now():
    return datetime.now(timezone.utc).isoformat()


def append(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def save(path, row):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(row, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


class PromptBank:
    def __init__(self, seed, maximum, count=8):
        self.entries = []
        for bank in range(count):
            rng = random.Random(seed + bank)
            words = [rng.choice(WORDS) for _ in range(maximum)]
            offsets = array("I", [0])
            size = 0
            for word in words:
                size += len(word) + 1
                offsets.append(size)
            self.entries.append((" ".join(words) + " ", offsets))

    def prompt(self, bank, words):
        text, offsets = self.entries[bank]
        code = f"MARKER_{bank:06d}"
        content = (
            f"Synthetic cache validation document {bank}. The validation code in this header is {code}.\n"
            + text[: offsets[words]].rstrip()
            + "\nReturn only the exact validation code from the header. Do not explain."
        )
        return content, code


def classify(row, context_length):
    usage = row.get("usage") or {}
    actual = usage.get("prompt_tokens")
    row["actual_prompt_tokens"] = actual
    row["over_context_target"] = (
        row["target_tokens"] + row["max_tokens"] > context_length
    )
    row["possible_truncation"] = (
        actual is not None and actual + 256 < row["target_tokens"]
    )
    row["ok"] = (
        row.get("status") == 200
        and row.get("done") is True
        and len(row.get("terminals", [])) == 1
        and row["terminals"][0] in ("stop", "length")
        and not row.get("stream_error")
        and not row.get("exception")
    )
    error = str(row.get("error_body", "")) + str(row.get("stream_error", ""))
    context_error = row.get("status") in (400, 413, 422) and any(
        term in error.lower()
        for term in ("context", "too long", "maximum", "max_tokens", "token limit")
    )
    row["outcome"] = (
        "ok"
        if row["ok"]
        else (
            "context_rejected"
            if context_error
            else "client_deadline" if row.get("deadline_fired") else "request_failed"
        )
    )
    row["content_ok"] = row.get("text", "").strip() == row["expected"]
    return row


def request(
    url,
    model,
    bank,
    target,
    word_overhead,
    output_tokens,
    timeout_s,
    index,
    context_length,
    bank_id,
):
    prompt, expected = bank.prompt(bank_id, max(1, target - word_overhead))
    payload = json.dumps(
        {
            "model": model,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": output_tokens,
            "temperature": 0,
            "stream": True,
        }
    ).encode()
    del prompt
    rid = "random-long-" + uuid.uuid4().hex
    row = dict(
        index=index,
        rid=rid,
        bank=bank_id,
        target_tokens=target,
        max_tokens=output_tokens,
        start=now(),
        expected=expected,
        text="",
        reasoning="",
        terminals=[],
        done=False,
        usage=None,
        ttft=None,
        request_bytes=len(payload),
    )
    started = time.monotonic()
    deadline_fired = threading.Event()
    parsed = urlsplit(url)
    cls = (
        http.client.HTTPSConnection
        if parsed.scheme == "https"
        else http.client.HTTPConnection
    )
    connection = cls(parsed.hostname, parsed.port, timeout=10)
    timer = None
    try:
        connection.connect()
        sock = connection.sock

        def expire():
            deadline_fired.set()
            try:
                sock.shutdown(socket.SHUT_RDWR)
            except OSError:
                pass

        timer = threading.Timer(
            max(0.001, timeout_s - (time.monotonic() - started)), expire
        )
        timer.daemon = True
        timer.start()
        connection.request(
            "POST",
            parsed.path or "/",
            body=payload,
            headers={"Content-Type": "application/json", "X-Request-ID": rid},
        )
        del payload
        sock.settimeout(timeout_s)
        response = connection.getresponse()
        row["status"] = response.status
        if response.status != 200:
            row["error_body"] = response.read(16000).decode("utf-8", errors="replace")
        else:
            received = 0
            while True:
                line = response.readline(65537)
                received += len(line)
                if len(line) > 65536 or received > 2 * 1024**2:
                    raise ValueError("unexpectedly large response")
                if not line:
                    break
                if deadline_fired.is_set():
                    raise TimeoutError("absolute request deadline")
                if not line.startswith(b"data:"):
                    continue
                data = line[5:].strip()
                if data == b"[DONE]":
                    row["done"] = True
                    break
                item = json.loads(data)
                if "error" in item:
                    row["stream_error"] = item["error"]
                if item.get("usage"):
                    row["usage"] = item["usage"]
                for choice in item.get("choices", []):
                    delta = choice.get("delta") or {}
                    content = delta.get("content") or ""
                    reasoning = delta.get("reasoning_content") or ""
                    row["text"] += content
                    row["reasoning"] += reasoning
                    if (content or reasoning) and row["ttft"] is None:
                        row["ttft"] = round(time.monotonic() - started, 3)
                    if choice.get("finish_reason"):
                        row["terminals"].append(choice["finish_reason"])
    except Exception as error:
        row["exception"] = f"{type(error).__name__}: {str(error)[:2000]}"
    finally:
        if timer is not None:
            timer.cancel()
        connection.close()
    row.update(
        end=now(),
        elapsed=round(time.monotonic() - started, 3),
        deadline_fired=deadline_fired.is_set(),
    )
    return classify(row, context_length)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--duration", type=float, default=2700)
    parser.add_argument("--timeout", type=float, default=900)
    parser.add_argument("--seed", type=int, default=10311430)
    parser.add_argument("--minimum", type=int, default=32768)
    parser.add_argument("--maximum", type=int, default=1100000)
    parser.add_argument("--min-concurrency", type=int, default=16)
    parser.add_argument("--max-concurrency", type=int, default=32)
    parser.add_argument("--change-seconds", type=float, default=60)
    parser.add_argument("--max-tokens", type=int, default=32)
    parser.add_argument("--context-length", type=int, default=1048576)
    parser.add_argument("--word-overhead", type=int, default=41)
    parser.add_argument("--calibrate-only", action="store_true")
    args = parser.parse_args()
    if not (
        0 < args.minimum <= args.maximum
        and 0 < args.min_concurrency <= args.max_concurrency <= 128
    ):
        parser.error("invalid lengths or concurrency bounds")
    if min(args.duration, args.timeout, args.change_seconds, args.max_tokens) <= 0:
        parser.error("durations and output budget must be positive")
    args.out.mkdir(parents=True, exist_ok=True)
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *unused: stop.set())
    bank = PromptBank(
        args.seed,
        args.minimum if args.calibrate_only else args.maximum,
        count=1 if args.calibrate_only else 8,
    )
    if args.calibrate_only:
        row = request(
            args.url,
            args.model,
            bank,
            args.minimum,
            args.word_overhead,
            args.max_tokens,
            args.timeout,
            0,
            args.context_length,
            0,
        )
        save(args.out / "calibration.json", row)
        print(json.dumps(row), flush=True)
        return 0 if row["ok"] else 1
    rng = random.Random(args.seed)
    began = time.monotonic()
    config = vars(args).copy()
    config["out"] = str(config["out"])
    config.update(
        start=now(),
        planned_submission_end_unix=time.time() + args.duration,
        planned_requests_end_unix=time.time() + args.duration + args.timeout,
        length_distribution="uniform integer; first wave includes fixed boundary cases",
        cache_policy="8 repeatable prompt banks with varying prefix lengths; no fault injection",
    )
    save(args.out / "run-config.json", config)
    rows, active, counts = [], {}, Counter()
    target = rng.randint(args.min_concurrency, args.max_concurrency)
    boundaries = [
        n
        for n in (
            32768,
            65536,
            131072,
            262144,
            524288,
            786432,
            1048000,
            1048576,
            1100000,
        )
        if args.minimum <= n <= args.maximum
    ]
    rng.shuffle(boundaries)
    changed, last_sent, last_saved = -args.change_seconds, -1, -1
    index = peak = 0
    pool = concurrent.futures.ThreadPoolExecutor(max_workers=args.max_concurrency)
    try:
        while True:
            elapsed = time.monotonic() - began
            halted = stop.is_set() or (args.out / "STOP_SUBMISSIONS").exists()
            submitting = elapsed < args.duration and not halted
            for future in list(active):
                if future.done():
                    row = future.result()
                    rows.append(row)
                    counts[row["outcome"]] += 1
                    append(args.out / "requests.jsonl", row)
                    del active[future]
            if submitting and elapsed - changed >= args.change_seconds:
                target = rng.randint(args.min_concurrency, args.max_concurrency)
                append(
                    args.out / "concurrency.jsonl",
                    dict(time=now(), target=target, active=len(active)),
                )
                changed = elapsed
            if submitting and len(active) < target and elapsed - last_sent >= 0.25:
                length = (
                    boundaries.pop()
                    if boundaries
                    else rng.randint(args.minimum, args.maximum)
                )
                bank_id = rng.randrange(8)
                deadline = min(
                    args.timeout, max(0.01, args.duration + args.timeout - elapsed)
                )
                future = pool.submit(
                    request,
                    args.url,
                    args.model,
                    bank,
                    length,
                    args.word_overhead,
                    args.max_tokens,
                    deadline,
                    index,
                    args.context_length,
                    bank_id,
                )
                active[future] = index
                append(
                    args.out / "submitted.jsonl",
                    dict(
                        time=now(),
                        index=index,
                        target_tokens=length,
                        bank=bank_id,
                        active=len(active),
                        target_concurrency=target,
                    ),
                )
                index += 1
                peak = max(peak, len(active))
                last_sent = elapsed
            if elapsed - last_saved >= 5 or (not submitting and not active):
                save(
                    args.out / "load-status.json",
                    dict(
                        time=now(),
                        elapsed=round(elapsed, 1),
                        submitting=submitting,
                        halted=halted,
                        active=len(active),
                        target_concurrency=target,
                        peak=peak,
                        submitted=index,
                        completed=len(rows),
                        outcomes=dict(counts),
                    ),
                )
                last_saved = elapsed
            if not submitting and not active:
                break
            time.sleep(0.05)
    finally:
        pool.shutdown(wait=True)
    actual = [
        r["actual_prompt_tokens"]
        for r in rows
        if r.get("actual_prompt_tokens") is not None
    ]
    summary = dict(
        start=config["start"],
        end=now(),
        submitted=index,
        completed=len(rows),
        peak=peak,
        elapsed=round(time.monotonic() - began, 1),
        outcomes=dict(counts),
        halted=halted,
        actual_prompt_tokens_min=min(actual, default=None),
        actual_prompt_tokens_max=max(actual, default=None),
        possible_truncations=sum(r["possible_truncation"] for r in rows),
        content_matches=sum(r["content_ok"] for r in rows),
        note="HTTP success is not proof of pool balance, L3 coverage or absence of native DMA hazards",
    )
    save(args.out / "load-done.json", summary)
    print(json.dumps(summary), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
