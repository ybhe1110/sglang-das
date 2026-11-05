"""Print Kubernetes discovery Pods for the 14 additional Rust DP16 listeners."""

import argparse
import json
import re


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--decode-nodes", nargs=2, required=True, metavar=("NODE0", "NODE1"))
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--name-prefix", required=True)
    parser.add_argument("--router-group", required=True)
    parser.add_argument("--transfer-group", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--pause-image", required=True, help="Image whose default process stays idle")
    parser.add_argument("--base-port", type=int, default=30000)
    args = parser.parse_args()
    if len(set(args.decode_nodes)) != 2:
        parser.error("DP16 requires two different decode nodes")
    if not 1 <= args.base_port <= 65535 - 15:
        parser.error("base-port through base-port + 15 must be valid TCP ports")
    if len(args.name_prefix) > 55 or not re.fullmatch(r"[a-z0-9]([-a-z0-9]*[a-z0-9])?", args.name_prefix):
        parser.error("name-prefix must be a DNS label of at most 55 characters")

    pods = []
    for rank in range(16):
        if rank in (0, 8):
            continue  # The two engine Pods advertise their first listener.
        port = args.base_port + rank
        pods.append({
            "apiVersion": "v1",
            "kind": "Pod",
            "metadata": {
                "name": f"{args.name_prefix}-rank-{rank}",
                "namespace": args.namespace,
                "labels": {
                    "aigate.iflytek.com/router-group": args.router_group,
                    "aigate.iflytek.com/model-id": args.model,
                    "aigate.iflytek.com/serving-mode": "disaggregated",
                    "aigate.iflytek.com/role": "decode",
                    "aigate.iflytek.com/transfer-group": args.transfer_group,
                },
            },
            "spec": {
                "hostNetwork": True,
                "nodeSelector": {"kubernetes.io/hostname": args.decode_nodes[rank // 8]},
                "tolerations": [{"key": "dcu", "operator": "Exists", "effect": "NoSchedule"}],
                "automountServiceAccountToken": False,
                "containers": [{
                    "name": "registration",
                    "image": args.pause_image,
                    "ports": [{"name": "sglang-http", "containerPort": port}],
                    "readinessProbe": {
                        "httpGet": {"path": "/health", "port": port},
                        "periodSeconds": 5,
                        "timeoutSeconds": 3,
                    },
                }],
            },
        })
    print(json.dumps({"apiVersion": "v1", "kind": "List", "items": pods}, indent=2))


if __name__ == "__main__":
    main()
