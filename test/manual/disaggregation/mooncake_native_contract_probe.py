"""Read-only inventory; this does NOT establish a DMA stop guarantee.

Run with the deployed wheel's Python, save stdout per image digest and node:
  python mooncake_native_contract_probe.py > native-contract-inventory.json

No Store/TransferEngine instance is created, and no requests are submitted.
To enable SGLANG_MOONCAKE_NATIVE_READ_CONTRACT, a separate reviewed JSON must
contain schema_version=1, this binary_sha256, verification (source revision,
review and isolated test evidence), and range_get_failure_drained=true.
ReadPlan requires its own read_plan_failure_drained=true guarantee.

Required evidence: every negative/short return and exception joins/cancels all
submitted WRs before returning, including subsequent layers and partial failures.
session_end() dropping cached metadata is not this guarantee. In an isolated
process, retain the original buffers during failure injection; correlate native
outstanding WR/CQ state, cancellation completion, canary changes, and QP/MR
reclamation. Observe through reconnect and endpoint teardown. No finite quiet
canary interval alone proves a fence. Never reuse uncertain buffers for testing.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
from datetime import datetime, timezone
from pathlib import Path


def inventory():
    module = importlib.import_module("mooncake.store")
    binary = Path(module.__file__).resolve()
    digest = hashlib.sha256()
    with binary.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    cls = module.MooncakeDistributedStore
    methods = {}
    for name in dir(cls):
        if any(
            word in name.lower()
            for word in (
                "session",
                "cancel",
                "drain",
                "fence",
                "close",
                "shutdown",
                "range",
                "read_plan",
                "outstanding",
            )
        ):
            value = getattr(cls, name)
            if callable(value):
                methods[name] = (getattr(value, "__doc__", "") or "")[:4000]
    try:
        version = importlib.metadata.version("mooncake-transfer-engine")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return {
        "collected_utc": datetime.now(timezone.utc).isoformat(),
        "hostname": platform.node(),
        "wheel_version": version,
        "binary": str(binary),
        "binary_sha256": digest.hexdigest(),
        "methods": methods,
        "configured_environment_only_not_proof_of_native_support": {
            key: os.environ.get(key)
            for key in (
                "MC_TRANSFER_TIMEOUT",
                "MC_SLICE_TIMEOUT",
                "MC_RPC_TIMEOUT_MS",
                "SGLANG_MOONCAKE_READ_STALL_S",
                "SGLANG_MOONCAKE_READ_MIN_MBPS",
                "SGLANG_MOONCAKE_NATIVE_READ_CONTRACT",
            )
        },
        "dma_failure_drain_verified": False,
        "endpoint_teardown_verified": False,
        "warning": "API presence/version/docstrings do not establish a DMA fence",
    }


def main():
    argparse.ArgumentParser(description=__doc__).parse_args()
    print(json.dumps(inventory(), indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
