"""Read-only DP16 discovery/metrics acceptance, optionally compare around traffic.

Requires prometheus-client for text parsing. Does not submit inference requests.
Pass --baseline from a snapshot taken before ordinary AND streaming Chat traffic.
"""

import argparse
import json
import math
import time
import urllib.parse
import urllib.request
from pathlib import Path

from reconcile_aigate_dp16 import atomic_json

GENERATION_METRICS = ("finished", "ttft", "e2e", "ttft_sum", "e2e_sum")


def get(url, timeout=10):
    with urllib.request.urlopen(url, timeout=timeout) as response:
        return response.read().decode()


def check_registry(plan, models):
    expected = {i["address"] for i in plan["ingresses"]}
    if {i["rank"] for i in plan["ingresses"]} != set(range(16)) or len(expected) != 16:
        raise ValueError("plan must have exactly 16 distinct D ingresses, ranks 0..15")
    replicas = [
        r
        for m in models["models"]
        if m["model"] == plan["model"]
        for r in m["replicas"]
    ]
    actual = [r["addr"] for r in replicas if r["role"] == "decode"]
    if len(actual) != 16 or set(actual) != expected:
        raise ValueError(
            f"Aigate D registry mismatch: missing={sorted(expected - set(actual))}, unexpected={sorted(set(actual) - expected)}, count={len(actual)}"
        )
    expected_p = {
        t["targets"][0] for t in plan["targets"] if t["labels"]["role"] == "prefill"
    }
    actual_p = [r["addr"] for r in replicas if r["role"] == "prefill"]
    if (
        not expected_p
        or len(actual_p) != len(expected_p)
        or set(actual_p) != expected_p
    ):
        raise ValueError("Aigate P registry does not match current engine Pods")


def check_prometheus_targets(plan, active, job):
    selected = [t for t in active if t["labels"].get("job") == job]
    expected = {t["targets"][0]: t["labels"]["role"] for t in plan["targets"]}
    if len(selected) != len(expected) or {
        urllib.parse.urlsplit(t["scrapeUrl"]).netloc for t in selected
    } != set(expected):
        raise ValueError(
            "Prometheus targets differ from plan: check stale IPs and duplicate rank scrapes"
        )
    for target in selected:
        url = urllib.parse.urlsplit(target["scrapeUrl"])
        if target["health"] != "up" or target.get("lastError"):
            raise ValueError(
                f"Prometheus target down: {target['scrapeUrl']} {target.get('lastError')}"
            )
        if (
            url.path != "/metrics"
            or target["labels"].get("role") != expected[url.netloc]
        ):
            raise ValueError(
                f"Prometheus path/role mismatch: {target['scrapeUrl']} {target['labels']}"
            )


def read_metrics(text, model, role):
    from prometheus_client.parser import text_string_to_metric_families

    samples = [
        s
        for f in text_string_to_metric_families(text)
        for s in f.samples
        if s.labels.get("model_name") == model
    ]
    if not samples or any(s.labels.get("engine_type") != role for s in samples):
        raise ValueError(f"metrics model/engine_type mismatch for {model}/{role}")
    result = {"ranks": {}, **{k: {} for k in GENERATION_METRICS}}
    names = {
        "sglang:num_requests_total": "finished",
        "sglang:time_to_first_token_seconds_count": "ttft",
        "sglang:e2e_request_latency_seconds_count": "e2e",
        "sglang:time_to_first_token_seconds_sum": "ttft_sum",
        "sglang:e2e_request_latency_seconds_sum": "e2e_sum",
    }
    for sample in samples:
        if sample.name in names:
            stream = sample.labels["is_streaming"]
            values = result[names[sample.name]]
            values[stream] = values.get(stream, 0) + sample.value
        if (
            sample.name == "sglang:per_stage_req_latency_seconds_count"
            and sample.labels.get("stage") == "decode_transferred"
        ):
            rank = sample.labels["dp_rank"]
            if rank in result["ranks"]:
                raise ValueError(f"duplicate rank {rank} transfer counters")
            result["ranks"][rank] = sample.value
    # Check the terminal bucket for every observed TTFT/E2E histogram, including
    # custom labels. Missing traffic may legitimately mean no series yet.
    buckets = {
        (
            s.name,
            tuple(sorted((k, v) for k, v in s.labels.items() if k != "le")),
        ): s.value
        for s in samples
        if s.name.endswith("_bucket") and s.labels.get("le") == "+Inf"
    }
    for sample in samples:
        if names.get(sample.name) in ("ttft", "e2e"):
            key = (
                sample.name.removesuffix("_count") + "_bucket",
                tuple(sorted(sample.labels.items())),
            )
            if buckets.get(key) != sample.value:
                raise ValueError(f"histogram +Inf/count mismatch: {sample.name}")
    return result


def compare(before, after):
    if before["engines"] != after["engines"] or before["model"] != after["model"]:
        raise ValueError("engine identity changed; take a new baseline")
    deltas = {}
    for rank in map(str, range(16)):
        delta = after["ranks"].get(rank, 0) - before["ranks"].get(rank, 0)
        if not math.isfinite(delta) or delta <= 0:
            raise ValueError(
                f"rank {rank} has no new KV-transferred requests (or counter reset): {delta}"
            )
        deltas[rank] = delta
    for metric in GENERATION_METRICS:
        for stream in ("false", "true"):
            delta = after[metric].get(stream, 0) - before[metric].get(stream, 0)
            if not math.isfinite(delta) or delta <= 0:
                raise ValueError(
                    f"no positive {metric} delta for is_streaming={stream}: {delta}"
                )
    return deltas


def capture(plan, gateway, prometheus, job):
    check_registry(plan, json.loads(get(gateway.rstrip("/") + "/v1/models")))
    for item in plan["ingresses"]:
        load = json.loads(get("http://" + item["address"] + "/v1/loads"))
        if (
            load.get("global_dp_rank") != item["rank"]
            or load.get("global_dp_size") != 16
        ):
            raise ValueError(f"wrong listener identity at {item['address']}")
    active = json.loads(get(prometheus.rstrip("/") + "/api/v1/targets"))["data"][
        "activeTargets"
    ]
    check_prometheus_targets(plan, active, job)
    result = {
        "time": time.time(),
        "model": plan["model"],
        "engines": plan["engine_epochs"],
        "ranks": {},
        **{k: {} for k in GENERATION_METRICS},
    }
    for target in plan["targets"]:
        role = target["labels"]["role"]
        metrics = read_metrics(
            get("http://" + target["targets"][0] + "/metrics"), plan["model"], role
        )
        if role != "decode":
            continue
        if result["ranks"].keys() & metrics["ranks"].keys():
            raise ValueError("same rank reported by multiple scrape targets")
        result["ranks"].update(metrics["ranks"])
        for key in GENERATION_METRICS:
            for stream, value in metrics[key].items():
                result[key][stream] = result[key].get(stream, 0) + value
    # Prove the real Prometheus ingests the frontend series as well as returning
    # successful HTTP scrapes. No equality with the raw scrape: traffic may run
    # concurrently and Prometheus can be one scrape interval behind.
    result["prometheus"] = {}
    for name in (
        "num_requests_total",
        "time_to_first_token_seconds_count",
        "e2e_request_latency_seconds_count",
        "time_to_first_token_seconds_sum",
        "e2e_request_latency_seconds_sum",
    ):
        query = f'sglang:{name}{{job={json.dumps(job)},model_name={json.dumps(plan["model"])},engine_type="decode"}}'
        data = json.loads(
            get(
                prometheus.rstrip("/")
                + "/api/v1/query?"
                + urllib.parse.urlencode({"query": query})
            )
        )
        values = {}
        for series in data["data"]["result"]:
            labels = series["metric"]
            if labels.get("role") != "decode":
                raise ValueError(f"stored metric has wrong role: {labels}")
            stream = labels["is_streaming"]
            values[stream] = values.get(stream, 0) + float(series["value"][1])
        result["prometheus"][name] = values
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--gateway", required=True)
    parser.add_argument("--prometheus", required=True)
    parser.add_argument("--job", default="sglang")
    parser.add_argument("--snapshot", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    plan = json.loads(args.plan.read_text(encoding="utf-8"))
    after = capture(plan, args.gateway, args.prometheus, args.job)
    atomic_json(args.snapshot, after)
    if args.baseline:
        before = json.loads(args.baseline.read_text(encoding="utf-8"))
        delta = compare(before, after)
        for name, counts in after["prometheus"].items():
            for stream in ("false", "true"):
                change = counts.get(stream, 0) - before["prometheus"][name].get(
                    stream, 0
                )
                if not math.isfinite(change) or change <= 0:
                    raise ValueError(
                        f"Prometheus has not observed new {name}/{stream}; wait at least two scrape intervals"
                    )
        print(json.dumps({"rank_transferred_deltas": delta}, indent=2))
    print(
        "PASS: 16 D registrations, rank identities, unique healthy scrape targets and role labels"
    )


if __name__ == "__main__":
    main()
