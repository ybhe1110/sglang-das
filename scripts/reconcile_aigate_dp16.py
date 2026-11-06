"""Maintain Aigate registrations and Prometheus file_sd for one RBG P/DP16 group.

Kubernetes is read-only without --apply; output files are always written. Run
on the Prometheus host with kubectl and Python 3.10+. See
examples/monitoring/aigate-dp16/README.md for the deployment.
"""

import argparse
import hashlib
import ipaddress
import json
import logging
import os
import re
import subprocess
import tempfile
import time
import urllib.request
from pathlib import Path
from types import SimpleNamespace

from render_aigate_dp16_registrations import registration_pod

AIGATE = "aigate.iflytek.com/"
RBG = "rbg.workloads.x-k8s.io/"
MANAGED = "sglang.io/dp16-discovery"
logger = logging.getLogger(__name__)


def address(ip, port):
    parsed = ipaddress.ip_address(ip)
    return f"[{ip}]:{port}" if parsed.version == 6 else f"{ip}:{port}"


def ready(pod):
    return (
        not pod["metadata"].get("deletionTimestamp")
        and pod.get("status", {}).get("phase") == "Running"
        and any(
            c["type"] == "Ready" and c["status"] == "True"
            for c in pod.get("status", {}).get("conditions", [])
        )
    )


def make_plan(
    pods,
    *,
    namespace,
    engine_group,
    name_prefix,
    router_group,
    transfer_group,
    model,
    pause_image,
    base_port=30000,
):
    """Pure desired-state calculation. Missing/not-ready engines are withdrawn.

    Supports the explicit topology: three (or more/fewer) single-node P DP1
    replicas and one two-node D TP16/DP16 engine, eight listeners per node.
    Reads RBG labels because the D worker may have no Aigate labels or ports.
    """
    if (
        not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", name_prefix)
        or len(name_prefix) > 40
    ):
        raise ValueError("name-prefix must be a DNS label of at most 40 characters")
    if not 1 <= base_port <= 65520:
        raise ValueError("base-port through base-port+15 must be valid ports")
    registrations, targets, ingresses, engine_uids = [], [], [], []
    engine_epochs = {}
    decode_nodes = set()
    decode_hosts = set()
    occupied = set()
    for pod in pods:
        meta = pod["metadata"]
        labels = meta.get("labels", {})
        if (
            meta.get("namespace") != namespace
            or labels.get(RBG + "group-name") != engine_group
        ):
            continue
        if not ready(pod):
            continue
        role = labels.get(RBG + "role-name")
        if role not in ("prefill", "decode"):
            raise ValueError(f"{meta['name']}: expected RBG role-name prefill/decode")
        spec, status = pod["spec"], pod["status"]
        if not spec.get("hostNetwork") or not spec.get("nodeName"):
            raise ValueError(
                f"{meta['name']}: only scheduled hostNetwork engines are supported"
            )
        if status.get("podIP") != status.get("hostIP"):
            raise ValueError(f"{meta['name']}: Pod IP must equal host IP")
        node_index = int(labels.get(RBG + "component-index", "-1"))
        size = labels.get(RBG + "component-size")
        if role == "decode":
            if node_index not in (0, 1) or size != "2" or node_index in decode_nodes:
                raise ValueError(
                    "expected one D engine with unique node indices 0 and 1"
                )
            decode_nodes.add(node_index)
            if spec["nodeName"] in decode_hosts:
                raise ValueError("D node0 and node1 must run on different hosts")
            decode_hosts.add(spec["nodeName"])
            first_rank = node_index * 8
        else:
            if node_index != 0 or size != "1":
                raise ValueError("prefill replicas must each be single-node DP1")
            first_rank = 0
        ip = status["podIP"]
        endpoint = address(ip, base_port + first_rank)
        if endpoint in occupied:
            raise ValueError(f"duplicate engine endpoint {endpoint}")
        occupied.add(endpoint)
        engine_uids.append(meta["uid"])
        engine_epochs[meta["uid"]] = sorted(
            (c["name"], c.get("containerID"), c.get("restartCount", 0))
            for c in status.get("containerStatuses", [])
        )
        targets.append(
            {
                "targets": [endpoint],
                "labels": {
                    "namespace": namespace,
                    "pod": meta["name"],
                    "engine_uid": meta["uid"],
                    "node_name": spec["nodeName"],
                    "role": role,
                    "router_group": router_group,
                },
            }
        )
        expected_labels = {
            AIGATE + "router-group": router_group,
            AIGATE + "model-id": model,
            AIGATE + "role": role,
            AIGATE + "serving-mode": "disaggregated",
            AIGATE + "transfer-group": transfer_group,
        }
        named_ports = [
            port["containerPort"]
            for c in spec["containers"]
            for port in c.get("ports", [])
            if port.get("name") == "sglang-http"
        ]
        discovered = labels.get(AIGATE + "router-group") == router_group
        if role == "prefill" or discovered:
            if any(labels.get(k) != v for k, v in expected_labels.items()):
                raise ValueError(
                    f"{meta['name']}: fix Aigate labels in the engine template"
                )
            if named_ports != [base_port + first_rank]:
                raise ValueError(
                    f"{meta['name']}: sglang-http must advertise {base_port + first_rank}"
                )
        if role == "prefill":
            continue
        args = SimpleNamespace(
            namespace=namespace,
            name_prefix=name_prefix,
            base_port=base_port,
            router_group=router_group,
            transfer_group=transfer_group,
            model=model,
            pause_image=pause_image,
        )
        for rank in range(first_rank, first_rank + 8):
            ingresses.append(
                {
                    "rank": rank,
                    "address": address(ip, base_port + rank),
                    "engine_uid": meta["uid"],
                }
            )
            if discovered and rank == first_rank:
                continue
            # Kubernetes defaults hostPort=containerPort for hostNetwork Pods.
            # Do not create a registration over a port already reserved by an engine.
            if any(
                p.get("hostPort", p["containerPort"]) == base_port + rank
                for c in spec["containers"]
                for p in c.get("ports", [])
            ):
                raise ValueError(
                    f"{meta['name']}: port {base_port + rank} already reserved; fix engine discovery labels"
                )
            item = registration_pod(args, rank, spec["nodeName"])
            suffix = hashlib.sha256(meta["uid"].encode()).hexdigest()[:10]
            item["metadata"]["name"] = f"{name_prefix}-{suffix}-r{rank}"
            item["metadata"]["labels"][MANAGED] = name_prefix
            item["metadata"]["ownerReferences"] = [
                {
                    "apiVersion": "v1",
                    "kind": "Pod",
                    "name": meta["name"],
                    "uid": meta["uid"],
                }
            ]
            # spec.nodeName is an actual node name, not necessarily its hostname label.
            del item["spec"]["nodeSelector"]
            item["spec"]["affinity"] = {
                "nodeAffinity": {
                    "requiredDuringSchedulingIgnoredDuringExecution": {
                        "nodeSelectorTerms": [
                            {
                                "matchFields": [
                                    {
                                        "key": "metadata.name",
                                        "operator": "In",
                                        "values": [spec["nodeName"]],
                                    }
                                ]
                            }
                        ],
                    }
                }
            }
            registrations.append(item)
    return {
        "registrations": {"apiVersion": "v1", "kind": "List", "items": registrations},
        "targets": sorted(targets, key=lambda t: t["targets"]),
        "ingresses": sorted(ingresses, key=lambda i: i["rank"]),
        "engine_uids": sorted(engine_uids),
        "engine_epochs": engine_epochs,
        "model": model,
    }


def atomic_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    content = json.dumps(data, indent=2, sort_keys=True) + "\n"
    if path.exists() and path.read_text(encoding="utf-8") == content:
        return
    fd, temporary = tempfile.mkstemp(dir=path.parent, prefix=".dp16-", suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            f.write(content)
        os.chmod(temporary, 0o644)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def kubectl(namespace, *args, body=None):
    result = subprocess.run(
        ["kubectl", "--request-timeout=20s", "-n", namespace, *args],
        input=json.dumps(body) if body is not None else None,
        text=True,
        capture_output=True,
        timeout=30,
        check=True,
    )
    return result.stdout


def load_identity(endpoint):
    with urllib.request.urlopen(f"http://{endpoint}/v1/loads", timeout=3) as response:
        return json.load(response)


def validate_ingresses(plan, read=load_identity):
    # Applying a Rust layout to a Python deployment would advertise nonexistent
    # listeners. Validate the live protocol before publishing anything.
    for ingress in plan["ingresses"]:
        identity = read(ingress["address"])
        if (
            identity.get("global_dp_rank") != ingress["rank"]
            or identity.get("global_dp_size") != 16
        ):
            raise ValueError(
                f"not the expected Rust DP16 listener: {ingress['address']}"
            )


def reconcile(plan, existing, namespace, name_prefix, run=kubectl):
    """Only delete this reconciler's registration Pods, with UID preconditions."""
    desired = {p["metadata"]["name"]: p for p in plan["registrations"]["items"]}
    current = {
        p["metadata"]["name"]: p
        for p in existing
        if p["metadata"].get("namespace") == namespace
    }
    # Validate every collision before any mutation.
    for name, pod in desired.items():
        if name in current:
            meta = current[name]["metadata"]
            if (
                meta.get("labels", {}).get(MANAGED) != name_prefix
                or meta.get("ownerReferences") != pod["metadata"]["ownerReferences"]
            ):
                raise ValueError(f"registration name collision: {name}")
    for name, pod in current.items():
        meta = pod["metadata"]
        if meta.get("labels", {}).get(MANAGED) != name_prefix or name in desired:
            continue
        owners = meta.get("ownerReferences", [])
        if (
            len(owners) != 1
            or owners[0].get("kind") != "Pod"
            or RBG + "group-name" in meta.get("labels", {})
            or [c["name"] for c in pod["spec"]["containers"]] != ["registration"]
        ):
            raise ValueError(f"refusing to delete a non-registration Pod {name}")
        run(
            namespace,
            "delete",
            "--raw",
            f"/api/v1/namespaces/{namespace}/pods/{name}",
            "-f",
            "-",
            body={
                "apiVersion": "v1",
                "kind": "DeleteOptions",
                "preconditions": {"uid": meta["uid"]},
            },
        )
    for name, pod in desired.items():
        if name in current:
            continue
        run(namespace, "create", "-f", "-", body=pod)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in (
        "namespace",
        "engine-group",
        "name-prefix",
        "router-group",
        "transfer-group",
        "model",
        "pause-image",
        "output-dir",
    ):
        parser.add_argument("--" + name, required=True)
    parser.add_argument("--base-port", type=int, default=30000)
    parser.add_argument(
        "--pods-json", type=Path, help="Offline kubectl get pods -o json snapshot"
    )
    parser.add_argument(
        "--apply",
        action="store_true",
        help="Reconcile registration Pods; never changes engine Pods",
    )
    parser.add_argument(
        "--watch", type=float, default=0, help="Repeat every N seconds (live mode only)"
    )
    args = parser.parse_args()
    if args.watch < 0 or (args.pods_json and (args.apply or args.watch)):
        parser.error(
            "offline snapshots cannot be applied/watched; watch must be nonnegative"
        )
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    options = vars(args).copy()
    for key in ("pods_json", "apply", "watch", "output_dir"):
        options.pop(key)
    output = Path(args.output_dir)
    while True:
        try:
            snapshot = json.loads(
                args.pods_json.read_text(encoding="utf-8")
                if args.pods_json
                else kubectl(args.namespace, "get", "pods", "-o", "json")
            )
            plan = make_plan(snapshot["items"], **options)
            if args.apply:
                validate_ingresses(plan)
            atomic_json(output / "plan.json", plan)
            # Prometheus should read this exact file, not a *.json glob (plan is not file_sd).
            atomic_json(output / "sglang-targets.json", plan["targets"])
            if args.apply:
                reconcile(plan, snapshot["items"], args.namespace, args.name_prefix)
            logger.info(
                "%d D ingresses, %d registrations, %d scrape targets",
                len(plan["ingresses"]),
                len(plan["registrations"]["items"]),
                len(plan["targets"]),
            )
        except (ValueError, KeyError, OSError, subprocess.SubprocessError):
            logger.exception("discovery failed; no further changes in this iteration")
            if not args.watch:
                raise
        if not args.watch:
            return
        time.sleep(args.watch)


if __name__ == "__main__":
    main()
