"""Bounded read-only P/D evidence collector for the accompanying aigate stress.

Writes STOP_SUBMISSIONS only in this test's output directory on a new Pod
generation/restart or sustained loss of all Prefills. Never changes Kubernetes,
signals service processes, injects faults or starts inference requests itself.
"""

import argparse
import concurrent.futures
import hashlib
import json
import re
import subprocess
import time
import urllib.request
from collections import Counter
from datetime import datetime, timedelta, timezone
from pathlib import Path

PATTERNS = {
    "process_exit": r"Scheduler hit an exception|Fatal Python|Subprocess.*crashed|SIGQUIT|SIGKILL|exit code -[0-9]|SIGTERM received",
    "health_failed": r"Health check failed|liveness probe failed",
    "collective": r"DisaggregationCollectiveError|gloo::EnforceNotMet|collective.*(?:mismatch|timed out)|Connection reset by peer",
    "pool": r"pool memory leak|Pool invariant detail|Idle KV pool audit: status=fail",
    "quarantine": r"quarantin|ownership uncertain|Withholding external load completion",
    "source_drain": r"source.?drain.*(?:unknown|timeout|timed out)|SourceTransferTimeout",
    "load_fail": r"External linker load failed|External KV load failed|Mooncake.*load.*fail",
    "cq": r"CQ.*timeout",
    "fallback": r"using L1 cache|falling back|fallback.*L1|L3.*restored|store.*recovered",
    "truncate": r"Truncating the input|Truncating max_new_tokens|context length",
    "lookup": r"direct linker lookup hit",
}


def utc():
    return datetime.now(timezone.utc)


def write(path, value):
    temporary = path.with_suffix(".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    temporary.replace(path)


def command(args, timeout=25):
    try:
        result = subprocess.run(args, capture_output=True, text=True, timeout=timeout)
        return result.stdout, result.stderr, result.returncode
    except Exception as error:
        return "", repr(error), -1


def health(ip):
    try:
        with urllib.request.urlopen(f"http://{ip}:30000/health", timeout=5) as response:
            return {"status": response.status}
    except Exception as error:
        return {"error": repr(error)}


def identity(pod):
    return (
        pod["metadata"]["uid"],
        tuple(
            (
                c["name"],
                c.get("restartCount", 0),
                c.get("state", {}).get("running", {}).get("startedAt"),
            )
            for c in pod["status"].get("containerStatuses", [])
        ),
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--prefix", default="dsv4flash-pd-node-ybhe2-ly-l1-l3-2-")
    parser.add_argument("--namespace", default="aiservice")
    parser.add_argument("--gateway-prefix", default="aigate-frontend-")
    parser.add_argument("--duration", type=float, default=4200)
    parser.add_argument("--interval", type=float, default=30)
    args = parser.parse_args()
    root = args.out
    root.mkdir(parents=True, exist_ok=True)
    started, baseline, previous, cursors, hashes = time.monotonic(), {}, {}, {}, {}
    totals, changes, gaps = Counter(), [], []
    all_p_down_since = done_since = None
    iteration = 0

    def stop(reason):
        path = root / "STOP_SUBMISSIONS"
        if not path.exists():
            path.write_text(utc().isoformat() + " " + reason, encoding="utf-8")

    while time.monotonic() - started < args.duration:
        tick = time.monotonic()
        now = utc()
        stamp = now.strftime("%Y%m%dT%H%M%SZ")
        out, err, rc = command(
            ["kubectl", "get", "pods", "-n", args.namespace, "-o", "json"]
        )
        summary = {
            "time": now.isoformat(),
            "elapsed": round(tick - started),
            "pods": [],
            "logs": {},
            "errors": [],
        }
        if rc:
            gaps.append({"time": now.isoformat(), "operation": "pods", "error": err})
            summary["errors"].append(err)
        else:
            pods = [
                p
                for p in json.loads(out)["items"]
                if p["metadata"]["name"].startswith((args.prefix, args.gateway_prefix))
            ]
            write(root / (stamp + "-pods.json"), pods)
            if iteration == 0:
                baseline = {p["metadata"]["name"]: identity(p) for p in pods}
                write(root / "pods-before.json", pods)
            names = {p["metadata"]["name"] for p in pods}
            if set(baseline) - names:
                stop("target Pod disappeared during stress")

            def collect(pod):
                name = pod["metadata"]["name"]
                ident = identity(pod)
                status = pod["status"]
                ready = any(
                    c.get("type") == "Ready" and c.get("status") == "True"
                    for c in status.get("conditions", [])
                )
                row = {
                    "name": name,
                    "uid": ident[0],
                    "ip": status.get("podIP"),
                    "ready": ready,
                    "phase": status.get("phase"),
                    "containers": status.get("containerStatuses", []),
                }
                changed = name in baseline and ident != baseline[name]
                event = None
                if changed and previous.get(name) != ident:
                    event = {
                        "time": now.isoformat(),
                        "name": name,
                        "before": previous.get(name, baseline[name]),
                        "after": ident,
                    }
                    before_log, pe, pr = command(
                        [
                            "kubectl",
                            "logs",
                            "-n",
                            args.namespace,
                            name,
                            "--previous",
                            "--timestamps",
                        ],
                        45,
                    )
                    (root / (stamp + "-" + name + "-previous.log")).write_text(
                        before_log, encoding="utf-8"
                    )
                    event["previous_log_error"] = pe
                    event["previous_log_rc"] = pr
                    events, ee, er = command(
                        [
                            "kubectl",
                            "get",
                            "events",
                            "-n",
                            args.namespace,
                            "--field-selector=involvedObject.name=" + name,
                            "-o",
                            "json",
                        ]
                    )
                    (root / (stamp + "-" + name + "-events.json")).write_text(
                        events, encoding="utf-8"
                    )
                previous[name] = ident
                key = repr(ident)
                cursor = cursors.get(key, now - timedelta(minutes=2))
                logs, le, lr = command(
                    [
                        "kubectl",
                        "logs",
                        "-n",
                        args.namespace,
                        name,
                        "--since-time=" + cursor.isoformat().replace("+00:00", "Z"),
                        "--timestamps",
                    ]
                )
                logpath = root / (stamp + "-" + name + ".log")
                logpath.write_text(logs, encoding="utf-8")
                counts, samples = Counter(), []
                seen = hashes.setdefault(key, set())
                for line in logs.splitlines():
                    matched = [
                        kind
                        for kind, pattern in PATTERNS.items()
                        if re.search(pattern, line, re.I)
                    ]
                    if not matched:
                        continue
                    digest = hashlib.sha256(line.encode()).digest()
                    if digest in seen:
                        continue
                    seen.add(digest)
                    counts.update(matched)
                    if (
                        any(k not in ("lookup", "truncate") for k in matched)
                        and len(samples) < 15
                    ):
                        samples.append(line[:3000])
                if lr == 0:
                    cursors[key] = now - timedelta(seconds=2)
                details = {
                    "path": str(logpath),
                    "counts": dict(counts),
                    "samples": samples,
                    "rc": lr,
                    "error": le,
                }
                # Decode worker has no HTTP server; probe only P and D leader.
                if row["ip"] and ("-prefill-" in name or name.endswith("decode-0-0")):
                    row["health"] = health(row["ip"])
                    if iteration % 4 == 0:
                        try:
                            with urllib.request.urlopen(
                                f"http://{row['ip']}:30000/get_load", timeout=5
                            ) as response:
                                body = response.read().decode()
                            (root / (stamp + "-" + name + "-load.json")).write_text(
                                body, encoding="utf-8"
                            )
                        except Exception as error:
                            row["get_load_error"] = repr(error)
                return row, details, event

            with concurrent.futures.ThreadPoolExecutor(max_workers=6) as pool:
                for row, details, event in pool.map(collect, pods):
                    summary["pods"].append(row)
                    summary["logs"][row["name"]] = details
                    if iteration > 0:
                        totals.update(details["counts"])
                    if details["rc"]:
                        gaps.append(
                            {
                                "time": now.isoformat(),
                                "operation": "logs",
                                "pod": row["name"],
                                "error": details["error"],
                            }
                        )
                    if event:
                        changes.append(event)
                        stop("new target Pod generation/restart: " + row["name"])
            prefills = [p for p in summary["pods"] if "-prefill-" in p["name"]]
            if prefills and all(
                not p["ready"] or p.get("health", {}).get("status") != 200
                for p in prefills
            ):
                all_p_down_since = all_p_down_since or tick
                if tick - all_p_down_since >= 120:
                    stop("all Prefills unhealthy for at least 120 seconds")
            else:
                all_p_down_since = None
        summary.update(
            cumulative_counts=dict(totals),
            changes=changes,
            coverage_gaps=gaps,
            stopped_submissions=(root / "STOP_SUBMISSIONS").exists(),
        )
        write(root / (stamp + "-summary.json"), summary)
        write(root / "monitor-status.json", summary)
        if (root / "load-done.json").exists():
            done_since = done_since or tick
            if tick - done_since >= 120:
                break
        iteration += 1
        time.sleep(max(1, args.interval - (time.monotonic() - tick)))
    write(root / "monitor-done.json", summary)


if __name__ == "__main__":
    main()
