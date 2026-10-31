"""Require positive FULL/SWA balance evidence after a specified drain boundary.

Collect each current container's logs separately after stopping test traffic.
Use a unique POD-UID/container-start label (not a reused Pod name). Include all
P and D Pods; do not concatenate previous-container logs. Record UID, startedAt
and restartCount before/after collection separately: this parser cannot query
Kubernetes or certify that a label actually identifies the supplied container.

Example (8 scheduler processes per supplied Pod):
  python verify_drained_kv_audit.py --since 2026-10-31T06:00:00Z \
    --expected-processes 8 --log P0-UID/start=P0.log --log D0-UID/start=D0.log

No audit, missing ranks/processes, any imbalance, or a failed audit after the
boundary fails acceptance. Wait at least 30 seconds after full drain for
rate-limited success records. Review fault/quarantine logs separately; a passed
balance alone does not certify native DMA safety or end-to-end request success.
No estimates are added to the pool accounting.
"""

import argparse
import json
import re
from datetime import datetime
from pathlib import Path

AUDIT = re.compile(
    r"Idle KV pool audit: status=(pass|fail) drained=true pid=(\d+) "
    r"audit_time=(\S+) (.*)"
)
POOL = re.compile(r"\[(full|swa)\] ([^|]+)")
FIELD = re.compile(
    r"\b(total|available|evictable|protected|session_held|uncached|accounted|delta)=(-?\d+)"
)


def timestamp(value):
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        raise ValueError("timestamp must include a timezone")
    return parsed


def verify(lines, *, since, expected_processes):
    if expected_processes <= 0:
        raise ValueError("expected_processes must be positive")
    latest, errors = {}, []
    for number, line in enumerate(lines, 1):
        match = AUDIT.search(line)
        if not match:
            continue
        status, pid, when, body = match.groups()
        when = timestamp(when)
        if when < since:
            continue
        pools = {
            name: {k: int(v) for k, v in FIELD.findall(text)}
            for name, text in POOL.findall(body)
        }
        if status != "pass":
            errors.append(f"line {number}: pid={pid} failed audit")
        for name in ("full", "swa"):
            fields = pools.get(name, {})
            required = {
                "total",
                "available",
                "evictable",
                "protected",
                "session_held",
                "uncached",
                "accounted",
                "delta",
            }
            if not required <= fields.keys():
                errors.append(f"line {number}: pid={pid} missing {name} accounting")
                continue
            accounted = sum(
                fields[key]
                for key in (
                    "available",
                    "evictable",
                    "protected",
                    "session_held",
                    "uncached",
                )
            )
            if (
                accounted != fields["accounted"]
                or accounted != fields["total"]
                or fields["delta"] != 0
                or any(fields[k] < 0 for k in required - {"delta"})
            ):
                errors.append(f"line {number}: pid={pid} unbalanced {name}: {fields}")
        if pid not in latest or when >= latest[pid]["parsed_time"]:
            latest[pid] = {
                "parsed_time": when,
                "audit_time": when.isoformat(),
                "pools": pools,
            }
    if len(latest) != expected_processes:
        errors.append(
            f"expected {expected_processes} processes, found {len(latest)}; missing proof or mixed container generations"
        )
    return {
        "passed": not errors,
        "errors": errors,
        "processes": {
            pid: {k: v for k, v in value.items() if k != "parsed_time"}
            for pid, value in latest.items()
        },
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--since", required=True, type=timestamp)
    parser.add_argument("--expected-processes", required=True, type=int)
    parser.add_argument(
        "--log", required=True, action="append", metavar="POD-UID/START=PATH"
    )
    args = parser.parse_args()
    results = {}
    for spec in args.log:
        label, sep, path = spec.partition("=")
        if not sep or not label or label in results:
            parser.error("each --log needs a distinct POD-UID/START=PATH")
        with Path(path).open(encoding="utf-8", errors="replace") as stream:
            results[label] = verify(
                stream, since=args.since, expected_processes=args.expected_processes
            )
    print(json.dumps(results, indent=2, ensure_ascii=False))
    return 0 if all(result["passed"] for result in results.values()) else 1


if __name__ == "__main__":
    raise SystemExit(main())
