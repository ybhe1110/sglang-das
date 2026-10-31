"""Admission deadlines are not DMA fences.

A slow native call opens the L1 circuit, but only its calling worker may publish
completion. A failed return is reclaimable only with an audited, binary-pinned
contract. Neither Future.cancel(), session_end(), nor elapsed time proves this.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import threading
import time
from pathlib import Path

logger = logging.getLogger(__name__)


class NativeReadUncertain(RuntimeError):
    """Native writes may still reference the private buffers."""


def read_failure_contract(path: str | None, binary: str | None) -> dict:
    """An operator's reviewed contract is tied to the exact deployed binary.

    This validates provenance, not the hardware guarantee. The latter requires
    vendor source review and an isolated outstanding-WR/buffer-canary test.
    """
    if not path:
        return {}
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if (
        data.get("schema_version") != 1
        or not isinstance(data.get("verification"), str)
        or not data["verification"].strip()
        or data.get("range_get_failure_drained") is not True
    ):
        raise ValueError("Mooncake native read contract is not verified")
    if not binary:
        raise ValueError("Mooncake native read contract has no binary to verify")
    digest = hashlib.sha256()
    with open(binary, "rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    if data.get("binary_sha256") != digest.hexdigest():
        raise ValueError("Mooncake native read contract binary SHA256 mismatch")
    return data


class NativeReadGuard:
    """One watchdog per linker; never abandon or cancel the native caller."""

    def __init__(
        self, on_stall, *, stall_s=5.0, min_bytes_per_s=64 * 1024**2, contract=None
    ):
        if not math.isfinite(stall_s) or stall_s <= 0:
            raise ValueError("Mooncake native read stall deadline must be positive")
        self.stall_s = stall_s
        if not math.isfinite(min_bytes_per_s) or min_bytes_per_s <= 0:
            raise ValueError("Mooncake read minimum throughput must be positive")
        self.min_bytes_per_s = min_bytes_per_s
        self.contract = contract or {}
        self._on_stall = on_stall
        self._condition = threading.Condition()
        self._active = None
        self._uncertain = None
        self._closed = False
        self._thread = None
        self.stalls = 0
        self.calls = 0

    def _watch(self):
        while True:
            with self._condition:
                while not self._closed and (
                    self._active is None or self._active["stalled"]
                ):
                    self._condition.wait()
                if self._closed:
                    return
                operation = self._active
                remaining = operation["deadline"] - time.monotonic()
                if remaining > 0:
                    self._condition.wait(remaining)
                    continue
                operation["stalled"] = True
                self.stalls += 1
            # The callback can take the availability lock; never hold our lock.
            self._notify_stall(operation)

    def _notify_stall(self, operation):
        error = RuntimeError(
            "Mooncake native read stalled: "
            f"batch={operation['batch']} stage={operation['stage']} "
            f"budget_s={operation['budget']:.3f}; L3 admission stopped, "
            "native call still owns buffers (not a DMA fence)"
        )
        logger.warning("%s", error)
        try:
            self._on_stall(error)
        except Exception:
            logger.exception("Failed to open Mooncake availability circuit")

    def check(self):
        with self._condition:
            if self._uncertain is not None:
                raise NativeReadUncertain(self._uncertain)

    def admission_blocked(self):
        with self._condition:
            return self._uncertain is not None or bool(
                self._active is not None and self._active["stalled"]
            )

    def progress(self, batch):
        """ReadPlan layer completion is progress, not a whole-plan DMA fence."""
        with self._condition:
            if self._active is not None and self._active["batch"] == batch:
                self._active["deadline"] = time.monotonic() + self._active["budget"]
                self._condition.notify_all()

    def snapshot(self):
        with self._condition:
            return {
                "active_reads": int(self._active is not None),
                "uncertain_reads": int(self._uncertain is not None),
                "stalls": self.stalls,
                "calls": self.calls,
            }

    def run(
        self,
        operation,
        *,
        batch,
        stage,
        expected=None,
        read_plan=False,
        byte_count=None,
    ):
        self.check()
        byte_count = sum(expected or ()) if byte_count is None else byte_count
        if type(byte_count) is not int or byte_count < 0:
            raise ValueError("native read byte budget must be a non-negative integer")
        with self._condition:
            if self._closed or self._active is not None:
                raise RuntimeError("Mooncake read guard is closed or already active")
            budget = self.stall_s + byte_count / self.min_bytes_per_s
            current = {
                "batch": batch,
                "stage": stage,
                "started": time.monotonic(),
                "budget": budget,
                "deadline": time.monotonic() + budget,
                "stalled": False,
            }
            self._active = current
            self.calls += 1
            if self._thread is None:
                self._thread = threading.Thread(
                    target=self._watch, name="mooncake-read-watch", daemon=True
                )
                self._thread.start()
            self._condition.notify_all()
        failure = None
        late_stall = False
        try:
            result = operation()
            if expected is not None and (
                not isinstance(result, (list, tuple))
                or len(result) != len(expected)
                or any(type(value) is not int for value in result)
                or list(result) != expected
            ):
                # Do not retain huge pointer/size arrays or exception tracebacks.
                sample = result[:8] if isinstance(result, (list, tuple)) else result
                raise RuntimeError(
                    "native range get returned incomplete/failed bytes: "
                    f"expected_keys={len(expected)} result_sample={str(sample)[:160]}"
                )
        except BaseException as error:
            failure = f"{type(error).__name__}: {str(error)[:512]}"
        finally:
            with self._condition:
                # Native code or CPU starvation can delay the Python watchdog.
                # Do not submit another layer just because it missed its wakeup.
                if not current["stalled"] and time.monotonic() >= current["deadline"]:
                    current["stalled"] = late_stall = True
                    self.stalls += 1
                self._active = None
                if failure is not None:
                    field = (
                        "read_plan_failure_drained"
                        if read_plan
                        else "range_get_failure_drained"
                    )
                    if self.contract.get(field) is not True:
                        self._uncertain = (
                            "Mooncake native DMA ownership uncertain: "
                            f"batch={batch} stage={stage}; {failure}; "
                            "completion withheld, private slots retained; "
                            "process teardown required"
                        )
                        self._closed = True
                self._condition.notify_all()
        if late_stall:
            self._notify_stall(current)
        self.check()
        if failure is not None:
            raise RuntimeError(failure)
        # A stalled but eventually successful call has now drained. Fail this
        # batch instead of submitting further layers to the same unhealthy peer.
        if current["stalled"]:
            raise RuntimeError(
                f"Mooncake slow read drained: batch={batch} stage={stage}; abort batch"
            )
        return result

    def close(self):
        self.check()
        with self._condition:
            if self._active is not None:
                raise NativeReadUncertain("Cannot close Mooncake with an active read")
            self._closed = True
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join()
